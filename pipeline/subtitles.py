"""Subtitle cues: build, validate, auto-fix, export (CLD-269).

Pure functions over segment dicts — no server imports, no job state. The
server builds cues from the newest checkpoint's segments plus the job's
`subtitle_overrides` map and serves them render-ready; the same builder
feeds the subtitles review gate and the final-QC checklist, so the count a
reviewer approves is the count the gate held the job on.

Two texts per cue, deliberately:

  * `text` is the dialogue — `translated_text`, the contract every other
    system reads (TTS, retry-reuse, the emotion-tag heuristic; CLAUDE.md).
  * `display_text` is an optional presentation override — a CPS-shortening
    or a re-wrap that changes what is *shown* without changing what was
    *spoken*, which is why it lives on the job and never in a checkpoint.

Timing nudges (`start_delta`/`end_delta`) are bounded to ±MAX_TIME_DELTA and
clamped so cues never overlap a neighbor, no matter what the caller sends.
"""
from __future__ import annotations

import logging
import re
import textwrap
from typing import Any, Dict, List, Optional

log = logging.getLogger("gochidubb.subtitles")

# Netflix-style defaults; runtime values come from UserConfig via the server.
DEFAULT_LIMITS: Dict[str, Any] = {
    "max_chars_per_line": 42,
    "max_lines": 2,
    "max_cps": 17.0,
    "min_gap_ms": 120,
}

# Bound on per-cue timing nudges, seconds (±500 ms per the design).
MAX_TIME_DELTA = 0.5

# A clamp must never squeeze a cue below this many seconds on screen.
_MIN_CUE_DUR = 0.001


def _limits(limits: Optional[dict]) -> dict:
    out = dict(DEFAULT_LIMITS)
    if limits:
        out.update({k: limits[k] for k in DEFAULT_LIMITS if k in limits})
    return out


def cue_text(cue: dict) -> str:
    """The text a viewer sees: the display override when present, else the
    dialogue."""
    return cue.get("display_text") or cue.get("text") or ""


def build_cues(segments: List[dict], overrides: Optional[dict] = None) -> List[dict]:
    """Segments -> ordered cue dicts {idx, start, end, text[, display_text]}.

    `placed_start`/`placed_end` are preferred over `start`/`end` — same rule
    as the assembler's SRT writer: once a segment has been shifted or
    compressed to fit, the source timestamps are the wrong answer.
    Non-speech segments and empty texts produce no cue. `overrides` is the
    job's `subtitle_overrides` map ({"<idx>": {display_text?, start_delta?,
    end_delta?}}); deltas are clamped to ±MAX_TIME_DELTA and cues are then
    clamped against their neighbors so no override can make them overlap.
    """
    ov: Dict[int, dict] = {}
    for k, v in (overrides or {}).items():
        try:
            ov[int(k)] = v if isinstance(v, dict) else {}
        except (TypeError, ValueError):
            continue

    cues: List[dict] = []
    for i, seg in enumerate(segments):
        if seg.get("non_speech"):
            continue
        text = str(seg.get("translated_text") or seg.get("text") or "").strip()
        if not text:
            continue
        try:
            idx = int(seg.get("idx", i))
        except (TypeError, ValueError):
            idx = i
        try:
            start = float(seg.get("placed_start", seg.get("start", 0.0)))
            end = float(seg.get("placed_end", seg.get("end", start)))
        except (TypeError, ValueError):
            continue
        cue = {"idx": idx, "start": start, "end": end, "text": text}
        o = ov.get(idx)
        if o:
            dt = o.get("display_text")
            if isinstance(dt, str) and dt.strip():
                cue["display_text"] = dt.strip()
            for key, field in (("start_delta", "start"), ("end_delta", "end")):
                try:
                    d = float(o.get(key) or 0.0)
                except (TypeError, ValueError):
                    d = 0.0
                cue[field] += max(-MAX_TIME_DELTA, min(MAX_TIME_DELTA, d))
        cues.append(cue)

    # Neighbor clamps: order and non-overlap survive any override.
    for j, c in enumerate(cues):
        c["start"] = max(0.0, c["start"])
        if j > 0 and c["start"] < cues[j - 1]["end"]:
            c["start"] = cues[j - 1]["end"]
        if c["end"] < c["start"] + _MIN_CUE_DUR:
            c["end"] = c["start"] + _MIN_CUE_DUR
    return cues


def annotate_cues(cues: List[dict], limits: Optional[dict] = None) -> List[dict]:
    """Copy of `cues` with render-ready metrics and per-cue violations.

    Adds: cps, lines, chars_per_line, gap_ms (to the NEXT cue; None on the
    last), violations (this cue's subset of validate_cues). The UI renders
    these directly — no client re-derivation.
    """
    lim = _limits(limits)
    by_idx: Dict[int, List[dict]] = {}
    for v in validate_cues(cues, lim):
        by_idx.setdefault(v["idx"], []).append(v)
    out = []
    for j, c in enumerate(cues):
        eff = cue_text(c)
        lines = eff.split("\n")
        dur = max(0.0, c["end"] - c["start"])
        gap_ms = None
        if j + 1 < len(cues):
            gap_ms = round((cues[j + 1]["start"] - c["end"]) * 1000.0, 1)
        a = dict(c)
        a.update({
            "start": round(c["start"], 3),
            "end": round(c["end"], 3),
            "cps": round(len(eff.replace("\n", "")) / dur, 1) if dur > 0 else None,
            "lines": len(lines),
            "chars_per_line": max(len(ln) for ln in lines),
            "gap_ms": gap_ms,
            "violations": by_idx.get(c["idx"], []),
        })
        out.append(a)
    return out


def validate_cues(cues: List[dict], limits: Optional[dict] = None) -> List[dict]:
    """Typed limit violations: {idx, kind, value, limit}.

    Kinds: line_too_long, too_many_lines, cps_exceeded, gap_too_small.
    A gap violation belongs to the EARLIER cue (its out-time is what the
    auto-fix shaves).
    """
    lim = _limits(limits)
    out: List[dict] = []
    for j, c in enumerate(cues):
        eff = cue_text(c)
        lines = eff.split("\n")
        longest = max(len(ln) for ln in lines)
        if longest > lim["max_chars_per_line"]:
            out.append({"idx": c["idx"], "kind": "line_too_long",
                        "value": longest, "limit": lim["max_chars_per_line"]})
        if len(lines) > lim["max_lines"]:
            out.append({"idx": c["idx"], "kind": "too_many_lines",
                        "value": len(lines), "limit": lim["max_lines"]})
        dur = c["end"] - c["start"]
        if dur > 0:
            cps = len(eff.replace("\n", "")) / dur
            if cps > lim["max_cps"] + 1e-9:
                out.append({"idx": c["idx"], "kind": "cps_exceeded",
                            "value": round(cps, 1), "limit": lim["max_cps"]})
        if j + 1 < len(cues):
            gap_ms = (cues[j + 1]["start"] - c["end"]) * 1000.0
            if gap_ms < lim["min_gap_ms"]:
                out.append({"idx": c["idx"], "kind": "gap_too_small",
                            "value": round(gap_ms, 1), "limit": lim["min_gap_ms"]})
    return out


# ── Auto-fix ─────────────────────────────────────────────────────────────

def _rewrap(text: str, max_chars: int, max_lines: int) -> Optional[str]:
    """Re-wrap `text` inside the line limits; None when no change helps.

    Greedy wrap at spaces first; when that needs too many lines, try breaking
    at clause boundaries (after . ! ? ; : ,); as the last resort, balance the
    words into exactly `max_lines` lines — those may still exceed the char
    limit, and the caller's post-validation decides whether that counted as
    an improvement.
    """
    flat = " ".join(text.split())
    if not flat:
        return None

    lines = textwrap.wrap(flat, width=max_chars, break_long_words=False,
                          break_on_hyphens=False)
    if not lines:
        return None
    if len(lines) > max_lines:
        # Clause-boundary attempt: wrap clause chunks instead of raw words.
        clauses = [p.strip() for p in
                   re.split(r"(?<=[.!?;:,])\s+", flat) if p.strip()]
        if len(clauses) > 1:
            merged: List[str] = []
            for cl in clauses:
                if merged and len(merged[-1]) + 1 + len(cl) <= max_chars:
                    merged[-1] = merged[-1] + " " + cl
                else:
                    merged.append(cl)
            if len(merged) <= max_lines and all(
                    len(ln) <= max_chars for ln in merged):
                lines = merged
    if len(lines) > max_lines:
        # Merge to exactly max_lines, balanced by length.
        words = flat.split()
        lines, target = [], max(1, len(flat) // max_lines)
        cur = ""
        for w in words:
            if cur and len(cur) >= target and len(lines) < max_lines - 1:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
    wrapped = "\n".join(lines)
    return wrapped if wrapped != text else None


def autofix_cues(cues: List[dict], limits: Optional[dict] = None) -> Dict[int, dict]:
    """Compute fixes for the mechanical violations; {} when nothing helps.

    Returns {idx: {display_text?, end_delta?}} where `end_delta` is relative
    to the cue's CURRENT end (the caller folds it into any stored override).
    Rules, per the design:
      1. line too long -> re-wrap at the last space inside the limit
      2. too many lines -> re-break at clause boundary, else merge
      3. CPS over limit -> extend the out-time into the available gap only
         (shortening text is a human edit)
      4. gap under the minimum -> shave the earlier cue's out-time
    The fix set is applied to a scratch copy and re-validated: it is returned
    only when violations strictly decrease, so a fix can never make the cue
    list worse than it found it.
    """
    lim = _limits(limits)
    before = len(validate_cues(cues, lim))
    if not before:
        return {}

    work = [dict(c) for c in cues]
    fixes: Dict[int, dict] = {}

    def fix(idx: int) -> dict:
        return fixes.setdefault(idx, {})

    for j, c in enumerate(work):
        eff = cue_text(c)
        lines = eff.split("\n")
        if (max(len(ln) for ln in lines) > lim["max_chars_per_line"]
                or len(lines) > lim["max_lines"]):
            wrapped = _rewrap(eff, lim["max_chars_per_line"], lim["max_lines"])
            if wrapped:
                c["display_text"] = wrapped
                fix(c["idx"])["display_text"] = wrapped

    min_gap_s = lim["min_gap_ms"] / 1000.0
    for j, c in enumerate(work):
        dur = c["end"] - c["start"]
        if dur <= 0:
            continue
        n = len(cue_text(c).replace("\n", ""))
        if n / dur <= lim["max_cps"] + 1e-9:
            continue
        needed = n / lim["max_cps"] - dur
        room = ((work[j + 1]["start"] - min_gap_s - c["end"])
                if j + 1 < len(work) else MAX_TIME_DELTA)
        delta = round(min(needed, room, MAX_TIME_DELTA), 3)
        if delta > 0.01:
            c["end"] += delta
            fix(c["idx"])["end_delta"] = delta

    for j in range(len(work) - 1):
        c, nxt = work[j], work[j + 1]
        gap = nxt["start"] - c["end"]
        if gap >= min_gap_s:
            continue
        new_end = nxt["start"] - min_gap_s
        delta = round(new_end - c["end"], 3)
        if delta < -MAX_TIME_DELTA or new_end <= c["start"] + 0.2:
            continue  # cannot shave enough without gutting the cue
        c["end"] = new_end
        fix(c["idx"])["end_delta"] = round(
            fix(c["idx"]).get("end_delta", 0.0) + delta, 3)

    fixes = {k: v for k, v in fixes.items() if v}
    if not fixes:
        return {}
    after = len(validate_cues(work, lim))
    if after >= before:
        log.info(f"[autofix] discarded fix set: {before} -> {after} "
                 f"violations (must strictly decrease)")
        return {}
    return fixes


# ── Export ───────────────────────────────────────────────────────────────

def _fmt_time(seconds: float, decimal_sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{decimal_sep}{ms:03d}"


def write_srt_cues(cues: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for n, c in enumerate(cues, 1):
            f.write(f"{n}\n{_fmt_time(c['start'], ',')} --> "
                    f"{_fmt_time(c['end'], ',')}\n{cue_text(c)}\n\n")


def write_vtt_cues(cues: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for n, c in enumerate(cues, 1):
            f.write(f"{n}\n{_fmt_time(c['start'], '.')} --> "
                    f"{_fmt_time(c['end'], '.')}\n{cue_text(c)}\n\n")
