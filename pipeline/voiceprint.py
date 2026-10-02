"""Speaker identification against enrolled voice profiles.

Diarization says *that* SPEAKER_03 talks for 40 minutes; this module helps a
human say *who* SPEAKER_03 is. Each diarized cluster gets one speaker
embedding (a centroid over its longest clean segments), which is compared by
cosine similarity with the enrolled profiles of one group (e.g. one
municipal council). The output is a *suggestion*: nothing here confirms an
identity, and the server never auto-confirms one.

Embeddings come from pyannote's own speaker-embedding model, reached through
the pyannote.audio install diarization already uses — no new dependency.
Everything torch/pyannote is imported lazily, so the pure helpers (clip
selection, cosine maths, one-to-one assignment) import and test without them.
If pyannote is missing, `VoiceprintUnavailable` says so; callers turn that
into a 503 and the pipeline itself never touches this module.

Audio is read clip by clip with seeks, never whole: a five-hour council
session at 16 kHz is ~2 GB as float64, and identification only needs a
minute of it per speaker.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("gochidubb.voiceprint")

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
SAMPLE_RATE = 16000

# Up to this much of a speaker's speech goes into one embedding.
MAX_EMBED_SECS = 60.0
# A cluster with less speech than this is not compared: an embedding over a
# couple of "sí" and "gracias" matches everyone a little and no one well.
MIN_CLUSTER_SECS = 5.0
# Per-clip bounds. Very short clips are mostly onset/offset; very long ones
# are capped so one monologue cannot be the whole minute.
MIN_CLIP_SECS = 1.0
MAX_CLIP_SECS = 15.0
# Segments whisper itself thought were probably not speech are skipped.
MAX_NO_SPEECH_PROB = 0.5


class VoiceprintUnavailable(RuntimeError):
    """The embedding model cannot run here (pyannote missing or not loadable)."""


# ── Pure helpers ──────────────────────────────────────────────────────────

def speaker_speech_secs(segments: Iterable[dict]) -> Dict[str, float]:
    """Total transcribed speech per speaker, non-speech segments excluded."""
    out: Dict[str, float] = {}
    for s in segments or ():
        if s.get("non_speech"):
            continue
        try:
            d = float(s.get("end", 0.0)) - float(s.get("start", 0.0))
        except (TypeError, ValueError):
            continue
        sp = s.get("speaker") or "SPEAKER_00"
        out[sp] = out.get(sp, 0.0) + max(d, 0.0)
    return out


def select_clips(segments: Iterable[dict], speaker: str,
                 max_total: float = MAX_EMBED_SECS) -> List[Tuple[float, float]]:
    """(start, end) clips of `speaker` to embed: longest clean segments first.

    "Clean" = not marked non-speech, whisper's no_speech_prob not high, and
    at least MIN_CLIP_SECS long. Each clip is capped at MAX_CLIP_SECS and the
    total at `max_total`. Returned in chronological order.
    """
    cands = []
    for s in segments or ():
        if (s.get("speaker") or "SPEAKER_00") != speaker or s.get("non_speech"):
            continue
        nsp = s.get("no_speech_prob")
        if isinstance(nsp, (int, float)) and nsp > MAX_NO_SPEECH_PROB:
            continue
        try:
            start, end = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end - start >= MIN_CLIP_SECS:
            cands.append((start, end))
    cands.sort(key=lambda c: -(c[1] - c[0]))
    picked, total = [], 0.0
    for start, end in cands:
        if total >= max_total:
            break
        d = min(end - start, MAX_CLIP_SECS, max_total - total)
        if d < MIN_CLIP_SECS:
            break
        picked.append((start, start + d))
        total += d
    return sorted(picked)


def normalize(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


def centroid(vectors: Sequence, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """L2-normalized (weighted) mean of L2-normalized vectors."""
    vs = [normalize(v) for v in vectors if v is not None and len(v)]
    if not vs:
        raise ValueError("no vectors to average")
    w = np.ones(len(vs)) if weights is None else np.asarray(weights[:len(vs)], dtype=np.float64)
    return normalize(np.average(np.stack(vs), axis=0, weights=w))


def cosine(a, b) -> float:
    a, b = normalize(a), normalize(b)
    if a.shape != b.shape or not a.size:
        return 0.0
    return float(np.dot(a, b))


def score_profiles(vec, profiles: Sequence[dict]) -> List[dict]:
    """Cosine of `vec` against each profile's centroid, best first."""
    out = []
    for p in profiles:
        embs = p.get("embeddings") or []
        if not embs:
            continue
        try:
            score = cosine(vec, centroid(embs))
        except ValueError:
            continue
        out.append({"profile_id": p["id"], "name": p.get("name") or "",
                    "role": p.get("role") or None, "score": round(score, 4)})
    out.sort(key=lambda m: -m["score"])
    return out


def assign_matches(cluster_vecs: Dict[str, Sequence], profiles: Sequence[dict],
                   threshold: float) -> Dict[str, dict]:
    """Best profile per cluster, one-to-one, at or above `threshold`.

    Greedy over all (cluster, profile) pairs by descending score: the
    strongest pairing is fixed first, then neither side is offered again.
    That is what stops one councillor's profile from being suggested for
    three clusters at once. Clusters left without a pair are absent.
    """
    pairs = []
    for spk, vec in cluster_vecs.items():
        for m in score_profiles(vec, profiles):
            if m["score"] >= threshold:
                pairs.append((m["score"], spk, m))
    pairs.sort(key=lambda t: (-t[0], t[1]))
    taken_spk, taken_prof, out = set(), set(), {}
    for _score, spk, m in pairs:
        if spk in taken_spk or m["profile_id"] in taken_prof:
            continue
        out[spk] = m
        taken_spk.add(spk)
        taken_prof.add(m["profile_id"])
    return out


# ── Model + audio (lazy torch/pyannote) ──────────────────────────────────

_inference = None
_inference_lock = threading.Lock()


def _load_inference(token: str = ""):
    """pyannote embedding Inference(window="whole"), loaded once per process."""
    global _inference
    with _inference_lock:
        if _inference is not None:
            return _inference
        try:
            from pyannote.audio import Inference, Model
        except Exception as e:
            raise VoiceprintUnavailable(
                "Speaker identification needs pyannote.audio (the same "
                "package diarization uses): pip install pyannote.audio"
            ) from e
        model = None
        errors = []
        # pyannote 4.x takes token=, 3.x use_auth_token=; the model is not
        # gated, so no token at all is also worth a try.
        attempts = ([{"token": token}, {"use_auth_token": token}] if token else []) + [{}]
        for kw in attempts:
            try:
                model = Model.from_pretrained(EMBEDDING_MODEL, **kw)
                if model is not None:
                    break
            except TypeError as e:
                errors.append(f"{type(e).__name__}: {e}")
                continue
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
                break
        if model is None:
            raise VoiceprintUnavailable(
                f"Could not load the speaker-embedding model {EMBEDDING_MODEL}"
                + (f": {errors[-1]}" if errors else "")
                + ". Check the network / Hugging Face token (System → Setup).")
        inference = Inference(model, window="whole")
        try:
            import torch
            if torch.cuda.is_available():
                inference.to(torch.device("cuda"))
        except Exception:
            pass
        _inference = inference
        log.info(f"[voiceprint] embedding model loaded: {EMBEDDING_MODEL}")
        return _inference


def _read_clip(f, start: float, end: float) -> np.ndarray:
    sr = f.samplerate
    a = max(int(start * sr), 0)
    n = max(int((end - start) * sr), 0)
    if a >= f.frames or n <= 0:
        return np.zeros(0, dtype=np.float32)
    f.seek(a)
    data = f.read(min(n, f.frames - a), dtype="float32", always_2d=True)
    return data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]


def _embed_waveform(inference, wav: np.ndarray, sr: int) -> np.ndarray:
    import torch
    t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None, :]
    emb = inference({"waveform": t, "sample_rate": sr})
    return np.asarray(emb, dtype=np.float64).reshape(-1)


def embed_clips(audio_path: str, clips: Sequence[Tuple[float, float]],
                token: str = "") -> Optional[np.ndarray]:
    """One centroid embedding over `clips` of `audio_path`, or None.

    Each clip is embedded separately and the results averaged weighted by
    duration — concatenating clips would put artificial joins inside the
    model's window.
    """
    import soundfile as sf
    inference = _load_inference(token)
    vecs, weights = [], []
    with sf.SoundFile(audio_path) as f:
        sr = f.samplerate
        for start, end in clips:
            wav = _read_clip(f, start, end)
            if wav.size < int(MIN_CLIP_SECS * sr * 0.9):
                continue
            try:
                vecs.append(_embed_waveform(inference, wav, sr))
                weights.append(wav.size / sr)
            except Exception as e:
                log.warning(f"[voiceprint] clip {start:.1f}-{end:.1f}s "
                            f"not embeddable: {e}")
    if not vecs:
        return None
    return centroid(vecs, weights)


def embed_file(audio_path: str, token: str = "",
               max_total: float = MAX_EMBED_SECS) -> Optional[np.ndarray]:
    """Embedding of a standalone recording of one voice (enrollment upload).

    The first `max_total` seconds, in MAX_CLIP_SECS windows.
    """
    import soundfile as sf
    with sf.SoundFile(audio_path) as f:
        dur = f.frames / float(f.samplerate or SAMPLE_RATE)
    clips, t = [], 0.0
    while t < min(dur, max_total):
        end = min(t + MAX_CLIP_SECS, dur, max_total)
        if end - t >= MIN_CLIP_SECS:
            clips.append((t, end))
        t = end
    return embed_clips(audio_path, clips, token) if clips else None
