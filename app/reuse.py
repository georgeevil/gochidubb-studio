"""Content-addressed stage reuse (beta).

Measured over 19 real jobs, download + extract + transcribe + diarize are
**59.8% of all pipeline time**, and none of their output depends on the target
language or the translation model. Today every redub into a new language redoes
all of it, because checkpoints are keyed by job id and a new job cannot see
them.

This module keys them by *content* instead: a stage's fingerprint is a hash of
exactly the inputs that determine its output. Two jobs that would compute the
same thing get the same fingerprint, so the second one can reuse the first's
work — across languages, across models, and (later) across users.

Nothing here touches the pipeline. These are pure functions over a context
dict; `app/artifact_store.py` persists them and `server.py` wires them in
behind `cfg.reuse_enabled`, which is off by default.

Two rules keep this honest:

**Everything that changes the output must be in the fingerprint.** Including
STAGE_VERSIONS — bump a stage's version whenever its behaviour changes, or
upgrading faster-whisper silently serves transcripts from the old one forever.
A stale reuse is the dangerous failure here because nothing about it looks
wrong.

**Quality is recorded when the artifact is written, never inferred when it is
read.** At reuse time the evidence — the audio, the QA transcripts — is
expensive to recompute or already gone.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("gochidubb.reuse")

# Bump a stage's version when its behaviour changes: a new model default, a
# changed ffmpeg filter chain, a different prompt. The version is part of the
# fingerprint, so a bump invalidates every cached artifact for that stage
# without deleting anything.
#
#   download  1  yt-dlp / file copy
#   extract   1  demux + optional afftdn denoise + VAD trim + bg split
#   transcribe 1 faster-whisper
#   diarize   1  pyannote + speaker reference cutting
#   translate 2  v2: _rejection_reason validation on both batch and single paths
#   tts       1  VoxCPM2 / edge-tts
STAGE_VERSIONS = {
    "download": 1,
    "extract": 1,
    "transcribe": 1,
    "diarize": 1,
    "translate": 2,
    "tts": 1,
}

# Stages worth caching, and what determines each one's output. A key absent
# from the context is recorded as null rather than skipped, so "the caller
# forgot to pass auto_denoise" can never collide with "auto_denoise was off".
#
# `assemble` and `merge` are deliberately absent: together they are 3.2% of
# pipeline time and they are the stages you actually re-tune (tts_max_stretch),
# so caching them would trade nothing for a stale-output risk.
STAGE_INPUTS = {
    "download": ("source_fingerprint",),
    # VAD decides which audio file ctx["audio_16k"] ends up naming and what is
    # in it, so it belongs here: without it, turning VAD off and re-running
    # silently reuses VAD-trimmed audio the job asked not to have.
    "extract": ("source_fingerprint", "auto_denoise", "keep_bg",
                "vad_enabled", "vad_threshold"),
    # The initial prompt biases spelling and the backend is different code
    # (faster-whisper vs mlx-whisper), so both change the transcript.
    "transcribe": ("audio_fingerprint", "whisper_model", "source_lang",
                   "initial_prompt", "asr_backend"),
    # reference_audio and skip_diarization change the output completely:
    # a job with an uploaded voice must not inherit speaker refs cut from the
    # video. same_language flips whether prompt_text is kept or cleared.
    "diarize": ("audio_fingerprint", "speaker_mode", "diarization_model",
                "reference_audio_fingerprint", "skip_diarization",
                "same_language", "min_speakers", "max_speakers"),
    "translate": ("segments_fingerprint", "target_lang", "model",
                  "context_hint", "glossary_fingerprint"),
    # NOTE: with the default `auto` voice preset and no voice_style,
    # resolve_voice_config derives voice_seed from the job id, so every job
    # gets a unique key and this cache never hits. That is deliberate — the
    # seed genuinely changes the audio — but it means tts reuse only pays off
    # when a preset or style pins the seed. See docs/stage-reuse.md.
    "tts": ("translation_fingerprint", "speaker_refs_fingerprint", "tts_engine",
            "voice_preset", "voice_style", "tts_speed", "voice_seed",
            "target_lang"),
}

CACHEABLE_STAGES = tuple(STAGE_INPUTS)

_HASH_CHUNK = 1 << 20          # 1 MiB


def hash_file(path: str, max_bytes: int = 0) -> str:
    """Content hash of a file, or "" when it cannot be read.

    Identity has to come from the bytes, not the path: `uploads/video.mp4` is
    reused for different uploads, and yt-dlp can hand back a different encode
    of the same URL depending on format selection. Keying on the path would
    serve one video's transcript for another's audio.

    `max_bytes` hashes only a prefix — cheap for very large media, at the cost
    of not distinguishing files that differ only past the cut. Size is always
    mixed in, so a truncation or an append is still caught.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    h = hashlib.sha256()
    h.update(str(size).encode())
    try:
        with open(path, "rb") as f:
            read = 0
            while True:
                want = _HASH_CHUNK
                if max_bytes:
                    if read >= max_bytes:
                        break
                    want = min(want, max_bytes - read)
                chunk = f.read(want)
                if not chunk:
                    break
                h.update(chunk)
                read += len(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def hash_value(value: Any) -> str:
    """Stable hash of any JSON-able value.

    `sort_keys` matters: dict ordering must not change a fingerprint, or the
    same inputs would miss the cache depending on how the context was built.
    """
    try:
        blob = json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
    except (TypeError, ValueError):
        blob = repr(value)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def hash_segments(segments) -> str:
    """Fingerprint of transcript segments — text and timing only.

    Confidence scores, speaker labels and file paths are excluded on purpose:
    two transcripts with identical words at identical times produce identical
    translations, and including the noise would defeat the cache.
    """
    rows = [
        [round(float(s.get("start") or 0.0), 3),
         round(float(s.get("end") or 0.0), 3),
         (s.get("text") or "").strip()]
        for s in (segments or [])
    ]
    return hash_value(rows)


def hash_translations(segments) -> str:
    """Fingerprint of what TTS will actually be asked to speak."""
    rows = [
        [round(float(s.get("start") or 0.0), 3),
         (s.get("translated_text") or s.get("text") or "").strip(),
         s.get("speaker") or ""]
        for s in (segments or [])
    ]
    return hash_value(rows)


def stage_fingerprint(stage: str, ctx: Dict[str, Any]) -> Optional[str]:
    """Fingerprint for one stage, or None when the stage is not cacheable.

    Returns None — never a partial hash — when a required input is missing and
    unknowable, because a fingerprint computed from incomplete inputs would
    collide with a genuinely different job.
    """
    keys = STAGE_INPUTS.get(stage)
    if not keys:
        return None
    payload: Dict[str, Any] = {"__stage": stage,
                               "__version": STAGE_VERSIONS.get(stage, 0)}
    for key in keys:
        value = ctx.get(key)
        # An empty content hash means "we could not read the file". Guessing
        # would key this job's work under a fingerprint that does not describe
        # it, so refuse to cache instead.
        if key.endswith("_fingerprint") and not value:
            return None
        payload[key] = value
    return hash_value(payload)


# ── Quality gates ────────────────────────────────────────────────────
# Recorded when an artifact is produced, compared when it is considered for
# reuse. Defaults are deliberately permissive: the point of a gate is to stop
# reuse of work that was already known to be bad, not to impose a quality bar
# the pipeline does not otherwise enforce.
DEFAULT_GATES = {
    # Mean per-word ASR confidence across the transcript.
    "transcribe_min_word_conf": 0.45,
    # Fraction of segments whisper flagged as probably-not-speech.
    "transcribe_max_no_speech": 0.35,
    # A speaker reference shorter than this clones badly.
    "diarize_min_ref_sec": 1.0,
    # Segments that fell back to their source text, as a fraction.
    "translate_max_fallback": 0.0,
    # Semantic checklist score, when a checklist exists for the pair.
    "translate_min_semantic": 0.67,
    # Fraction of segments whose whisper round-trip failed QA.
    "tts_max_failed_qa": 0.25,
}


def evaluate_quality(stage: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Measure a freshly produced stage's output, for storing alongside it.

    Every figure here is already computed by the pipeline; this only collects
    them. Returns {} for stages with nothing meaningful to measure, which
    gates as a pass.
    """
    segments = ctx.get("segments") or []
    if stage == "transcribe":
        confs = [s["word_conf_mean"] for s in segments
                 if isinstance(s.get("word_conf_mean"), (int, float))]
        no_speech = [s.get("no_speech_prob") for s in segments
                     if isinstance(s.get("no_speech_prob"), (int, float))]
        return {
            "segments": len(segments),
            "word_conf_mean": (round(sum(confs) / len(confs), 4)
                               if confs else None),
            "no_speech_ratio": (
                round(sum(1 for p in no_speech if p > 0.5) / len(no_speech), 4)
                if no_speech else None),
        }
    if stage == "diarize":
        refs = ctx.get("speaker_refs") or {}
        durations = []
        for path in refs.values():
            try:
                import soundfile as sf
                info = sf.info(path)
                durations.append(info.frames / info.samplerate)
            except Exception:
                continue
        return {
            "speakers": len(refs),
            "min_ref_sec": round(min(durations), 3) if durations else None,
        }
    if stage == "translate":
        total = len(segments) or 1
        # Mirrors server._is_untranslated: an EMPTY translation counts too.
        # Counting only "translation == source" let a run where the model
        # returned blanks record a perfect 0.0 fallback ratio and sail
        # through the gate that exists to catch exactly that.
        def _untranslated(seg):
            src = (seg.get("text") or "").strip()
            tgt = (seg.get("translated_text") or "").strip()
            return bool(src) and (not tgt or tgt == src)
        fell_back = sum(1 for s in segments if _untranslated(s))
        return {
            "segments": len(segments),
            "fallback_ratio": round(fell_back / total, 4),
            "semantic_score": ctx.get("semantic_score"),
        }
    if stage == "tts":
        scored = [s for s in segments if isinstance(s.get("qa_score"), (int, float))]
        missing = sum(1 for s in segments if not s.get("audio_path"))
        failed = sum(1 for s in scored if s["qa_score"] > 0.4)
        return {
            "segments": len(segments),
            "missing_audio": missing,
            "failed_qa_ratio": (round(failed / len(scored), 4) if scored else None),
        }
    return {}


def passes_gate(stage: str, quality: Optional[Dict[str, Any]],
                gates: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Whether a stored artifact is good enough to reuse, and why not.

    A `None` measurement is treated as "not measured", which passes: the gate
    exists to reject work already known to be bad, and refusing everything
    unmeasured would disable the cache on the first stage that cannot score
    itself.
    """
    g = dict(DEFAULT_GATES)
    g.update(gates or {})
    q = quality or {}

    def _num(key):
        v = q.get(key)
        return v if isinstance(v, (int, float)) else None

    if stage == "transcribe":
        conf = _num("word_conf_mean")
        if conf is not None and conf < g["transcribe_min_word_conf"]:
            return False, (f"transcript confidence {conf:.2f} below "
                           f"{g['transcribe_min_word_conf']:.2f}")
        ns = _num("no_speech_ratio")
        if ns is not None and ns > g["transcribe_max_no_speech"]:
            return False, f"{ns:.0%} of segments flagged as non-speech"
    elif stage == "diarize":
        ref = _num("min_ref_sec")
        if ref is not None and ref < g["diarize_min_ref_sec"]:
            return False, (f"shortest speaker reference {ref:.2f}s is under "
                           f"{g['diarize_min_ref_sec']:.2f}s")
    elif stage == "translate":
        fb = _num("fallback_ratio")
        if fb is not None and fb > g["translate_max_fallback"]:
            return False, f"{fb:.0%} of segments were left untranslated"
        sem = _num("semantic_score")
        if sem is not None and sem < g["translate_min_semantic"]:
            return False, (f"semantic score {sem:.2f} below "
                           f"{g['translate_min_semantic']:.2f}")
    elif stage == "tts":
        if q.get("missing_audio"):
            return False, f"{q['missing_audio']} segment(s) have no audio"
        qa = _num("failed_qa_ratio")
        if qa is not None and qa > g["tts_max_failed_qa"]:
            return False, f"{qa:.0%} of segments failed synthesis QA"
    return True, ""


def gates_from_config(cfg) -> Dict[str, Any]:
    """Gate thresholds from UserConfig, falling back to the defaults."""
    out = dict(DEFAULT_GATES)
    for key in DEFAULT_GATES:
        value = getattr(cfg, f"reuse_{key}", None)
        if isinstance(value, (int, float)):
            out[key] = value
    return out
