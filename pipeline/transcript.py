"""Speaker-attributed transcript export (JSON / SRT / VTT / TXT).

Pure functions over a pipeline checkpoint plus the job's human-entered
speaker data — no server imports, no job state. The server picks the
checkpoint and hands over three per-speaker maps from the job:

  * `labels`    — display renames from the speaker editor (speaker_labels)
  * `matches`   — voice-profile *suggestions* (speaker_matches); shown, never
                  used as a name
  * `confirmed` — names a human confirmed (speaker_confirmed)

Only a confirmed name is ever rendered as "who said it" ahead of a label;
an unconfirmed voice match stays a suggestion in the JSON. That separation
is the point: a transcript published with officials' names must not carry a
name a model guessed.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from .subtitles import srt_text, vtt_text

log = logging.getLogger("gochidubb.transcript")

FORMATS = ("json", "srt", "vtt", "txt")
DEFAULT_SPEAKER = "SPEAKER_00"


def _num(v) -> Optional[float]:
    """A float suitable for JSON, or None.

    NaN and the infinities are rejected rather than passed through. `float()`
    accepts them happily and `json.dumps` then emits bare NaN/Infinity, which
    is not JSON: FastAPI's encoder raises «Out of range float values are not
    JSON compliant» and the whole export 500s. One such value is enough —
    mlx-whisper returned a single NaN `avg_logprob` in a 1,317-segment
    council session, and that one number made the transcript unreadable.
    """
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clean_match(m: Any) -> Optional[dict]:
    if not isinstance(m, dict) or not m.get("profile_id"):
        return None
    return {"profile_id": str(m["profile_id"]), "name": str(m.get("name") or ""),
            "role": m.get("role") or None,
            "score": round(_num(m.get("score")) or 0.0, 4)}


def _clean_confirmed(c: Any) -> Optional[dict]:
    if not isinstance(c, dict) or not c.get("name"):
        return None
    return {"profile_id": c.get("profile_id") or None, "name": str(c["name"]),
            "role": c.get("role") or None}


def build_transcript(job_id: str, checkpoint: dict, *,
                     source: str = "",
                     duration: Optional[float] = None,
                     labels: Optional[dict] = None,
                     matches: Optional[dict] = None,
                     confirmed: Optional[dict] = None) -> dict:
    """The transcript document (the JSON export's exact shape).

    Segments a human marked non-speech are left out. Segment order is the
    checkpoint's; idx is kept as-is so a line can be referred back to the
    job's editors.
    """
    labels = labels or {}
    matches = matches or {}
    confirmed = confirmed or {}
    cp = checkpoint or {}

    segments: List[dict] = []
    stats: Dict[str, Dict[str, float]] = {}
    for i, s in enumerate(cp.get("segments") or []):
        if s.get("non_speech"):
            continue
        start = _num(s.get("start")) or 0.0
        end = _num(s.get("end"))
        end = start if end is None else end
        spk = str(s.get("speaker") or DEFAULT_SPEAKER)
        try:
            idx = int(s.get("idx", i))
        except (TypeError, ValueError):
            idx = i
        segments.append({
            "idx": idx,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": str(s.get("text") or "").strip(),
            "speaker": spk,
            "avg_logprob": _num(s.get("avg_logprob")),
            "no_speech_prob": _num(s.get("no_speech_prob")),
        })
        st = stats.setdefault(spk, {"talk": 0.0, "n": 0})
        st["talk"] += max(end - start, 0.0)
        st["n"] += 1

    speakers = []
    # Most talk first: in a council session that is the chair, which is the
    # row a reviewer wants at the top.
    for spk in sorted(stats, key=lambda k: (-stats[k]["talk"], k)):
        speakers.append({
            "id": spk,
            "label": labels.get(spk) or None,
            "talk_secs": round(stats[spk]["talk"], 1),
            "segments": int(stats[spk]["n"]),
            "match": _clean_match(matches.get(spk)),
            "confirmed": _clean_confirmed(confirmed.get(spk)),
        })

    asr = cp.get("asr")
    if not (isinstance(asr, dict) and asr.get("backend")):
        asr = None
    else:
        asr = {"backend": str(asr["backend"]), "model": str(asr.get("model") or "")}
    diar = cp.get("diarization")
    if not (isinstance(diar, dict) and diar.get("model")):
        diar = None
    else:
        diar = {"model": str(diar["model"])}

    dur = _num(duration if duration is not None else cp.get("duration"))
    lang = cp.get("effective_src") or cp.get("source_lang_detected") or None
    if lang == "auto":
        lang = cp.get("source_lang_detected") or None
    return {
        "job_id": job_id,
        "source": str(source or cp.get("source") or ""),
        "duration": round(dur, 3) if dur is not None else None,
        "language": lang,
        "asr": asr,
        "diarization": diar,
        "speakers": speakers,
        "segments": segments,
    }


def speaker_display_names(doc: dict) -> Dict[str, str]:
    """speaker id → the name to print: confirmed name, else label, else id.

    A voice-profile match is deliberately NOT in this chain — it is a
    suggestion until a human confirms it.
    """
    out = {}
    for sp in doc.get("speakers") or []:
        conf = sp.get("confirmed") or {}
        out[sp["id"]] = conf.get("name") or sp.get("label") or sp["id"]
    return out


def _cues(doc: dict) -> List[dict]:
    names = speaker_display_names(doc)
    cues = []
    for s in doc.get("segments") or []:
        if not s["text"]:
            continue
        who = names.get(s["speaker"], s["speaker"])
        end = s["end"] if s["end"] > s["start"] else s["start"] + 0.001
        cues.append({"start": s["start"], "end": end,
                     "text": f"{who}: {s['text']}"})
    return cues


def _hms(seconds: float) -> str:
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_txt(doc: dict) -> str:
    lines = [f"Transcript · job {doc['job_id']}"]
    if doc.get("source"):
        lines.append(f"Source: {doc['source']}")
    meta = []
    if doc.get("language"):
        meta.append(f"Language: {doc['language']}")
    if doc.get("asr"):
        meta.append(f"ASR: {doc['asr']['backend']} {doc['asr']['model']}".rstrip())
    if doc.get("diarization"):
        meta.append(f"Diarization: {doc['diarization']['model']}")
    if meta:
        lines.append(" · ".join(meta))
    lines.append("")
    names = speaker_display_names(doc)
    for s in doc.get("segments") or []:
        if not s["text"]:
            continue
        lines.append(f"[{_hms(s['start'])}] "
                     f"{names.get(s['speaker'], s['speaker'])}: {s['text']}")
    return "\n".join(lines) + "\n"


def render(doc: dict, fmt: str) -> str:
    """Render a transcript document as srt / vtt / txt (json is the doc)."""
    if fmt == "srt":
        return srt_text(_cues(doc))
    if fmt == "vtt":
        return vtt_text(_cues(doc))
    if fmt == "txt":
        return render_txt(doc)
    raise ValueError(f"Unknown transcript format {fmt!r} ({', '.join(FORMATS)})")
