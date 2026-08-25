"""Cost-of-edits counting (CLD-271): how many seconds an edit re-synthesizes.

Pure arithmetic over segment dicts — the server route turns the seconds into
money via ``app.billing.marginal_cost`` so the quote always matches the
meter (marginal on the month's real usage, never priced from zero; see
billing's honesty boundary: the minutes are real, the money is an estimate).

The counting rules, from the design:

  * ``segment_text``      — the listed segments' durations.
  * ``speaker_voice``     — every speech segment of that speaker.
  * ``pronunciation``     — every segment whose ``translated_text`` contains
                            the term (word-boundary match — the same matcher
                            the pronunciation seam uses, so the estimate and
                            the effect agree on which segments are touched).
  * ``subtitle_display``  — always 0: display-only, the audio is untouched.

Overlapping edits count a segment ONCE (union, not sum): retyping a line and
recasting its speaker re-synthesizes that segment one time, not two.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from pipeline.flags import _contains_phrase

KINDS = ("segment_text", "speaker_voice", "pronunciation", "subtitle_display")

FREE_REASON_PRE_TTS = ("Text edits before TTS are free — nothing has been "
                       "synthesized yet.")


def _duration(seg: dict) -> float:
    try:
        return max(0.0, float(seg.get("end") or 0.0) - float(seg.get("start") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def edit_seconds(segments: Iterable[dict],
                 edits: List[dict]) -> Tuple[float, List[int], List[Dict[str, Any]]]:
    """(total_seconds, union_idxs, breakdown) for a list of edit dicts.

    Raises ValueError on an unknown ``kind`` or a non-dict edit — the route
    turns that into a 400 rather than quoting a price for a request it did
    not understand.
    """
    by_idx: Dict[int, dict] = {}
    for i, seg in enumerate(segments):
        try:
            idx = int(seg.get("idx", i))
        except (TypeError, ValueError):
            idx = i
        by_idx[idx] = seg

    union: set = set()
    breakdown: List[Dict[str, Any]] = []
    for e in edits:
        if not isinstance(e, dict):
            raise ValueError("each edit must be an object")
        kind = e.get("kind")
        if kind not in KINDS:
            raise ValueError(f"unknown edit kind: {kind!r}")

        idxs: set = set()
        if kind == "segment_text":
            for raw in e.get("idxs") or []:
                try:
                    i = int(raw)
                except (TypeError, ValueError):
                    raise ValueError(f"segment_text idxs must be integers, "
                                     f"got {raw!r}")
                if i in by_idx:
                    idxs.add(i)
        elif kind == "speaker_voice":
            speaker = e.get("speaker")
            if not speaker:
                raise ValueError("speaker_voice edit needs a 'speaker'")
            idxs = {k for k, s in by_idx.items()
                    if s.get("speaker") == speaker and not s.get("non_speech")}
        elif kind == "pronunciation":
            term = str(e.get("term") or "").strip()
            if not term:
                raise ValueError("pronunciation edit needs a 'term'")
            idxs = {k for k, s in by_idx.items()
                    if _contains_phrase(s.get("translated_text") or "", term)}
        elif kind == "subtitle_display":
            # Display-only: named in the breakdown so the UI can show the
            # zero, but it never joins the resynth union.
            listed = []
            for raw in e.get("idxs") or []:
                try:
                    listed.append(int(raw))
                except (TypeError, ValueError):
                    continue
            breakdown.append({"kind": kind, "idxs": sorted(listed),
                              "seconds": 0.0})
            continue

        seconds = sum(_duration(by_idx[k]) for k in idxs)
        breakdown.append({"kind": kind, "idxs": sorted(idxs),
                          "seconds": round(seconds, 2)})
        union |= idxs

    total = sum(_duration(by_idx[k]) for k in union)
    return round(total, 2), sorted(union), breakdown
