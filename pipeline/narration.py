"""Narration: a written script in, a voice track and subtitles out.

A dub inherits its clock from the source video. Every line has a start and an
end before a word is synthesized, and the assembler's whole job is to fit new
speech into those slots. A narration has no source: the script is the only
input, so the order is reversed. Synthesize first, measure what came back, and
let the audio set the timeline — each slide holds for as long as its passage
takes to say.

That reversal is why this is its own module instead of a mode of the
assembler. `plan_segment_fit` answers "how do I squeeze this clip into the gap
the original speaker left", which is a question a narration never asks.

Everything above the "I/O" rule is pure: script parsing, sentence splitting,
the timeline, subtitle cues, the ffmpeg argv. Those are the parts with
decisions in them, and they are tested without a model, a GPU or ffmpeg. The
two functions below the rule (mixing the clips, rendering the slideshow) do
the work the pure parts planned. No server imports and no job state — the
server owns the job and calls in here, the same arrangement as subtitles.py.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .subtitles import DEFAULT_LIMITS, _rewrap, validate_cues

log = logging.getLogger("gochidubb.narration")

# The one speaker a narration has. Segments carry it so the engines' speaker
# lookup (speaker_refs[speaker]) resolves the narrator's reference the same
# way it resolves SPEAKER_00 for a dub.
NARRATOR = "NARRATOR"

SLIDE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Bounds on a submitted script. 100k characters is roughly two hours of
# speech — past any narration this is for, and a ceiling on how much GPU time
# one request can ask for.
MAX_SCRIPT_CHARS = 100_000
MAX_SLIDES = 500

# One synthesis call per sentence keeps prosody whole; a sentence longer than
# this is cut at a clause boundary, because very long inputs are where the
# cloning engines start dropping or repeating words.
MAX_TTS_CHARS = 220

# Reading speed used ONLY to hold the timeline open for a line whose clip
# failed to synthesize: the slide still stays up long enough to read it.
READING_CPS = 15.0

# Seconds. `slide_lead` is the beat between a slide appearing and the voice
# starting; `slide_gap` is the pause after a slide's last line, before the
# next slide appears. Both exist because a cut that lands exactly on the first
# syllable reads as a mistake.
DEFAULT_TIMING: Dict[str, float] = {
    "lead_in": 0.5,
    "sentence_gap": 0.35,
    "paragraph_gap": 0.7,
    "slide_lead": 0.3,
    "slide_gap": 0.9,
    "tail": 1.2,
    "silent_slide_hold": 3.0,
}
_TIMING_MAX = 30.0

# A subtitle cue shorter than this is a flash, not a subtitle.
_MIN_CUE_SEC = 0.3


class ScriptError(ValueError):
    """The script cannot be narrated as written. The message is user-facing."""


# ═══════════════════════════════════════════════════════════════════════
#  Script parsing
# ═══════════════════════════════════════════════════════════════════════

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_IMAGE = re.compile(r"^!\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)$")
_MD_PAUSE = re.compile(
    r"^<!--\s*pause\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*-->$", re.IGNORECASE)
_MD_COMMENT = re.compile(r"^<!--.*-->$")
_MD_INLINE_COMMENT = re.compile(r"<!--.*?-->")
_MD_LIST = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _strip_inline_markup(text: str) -> str:
    """Markdown emphasis, code ticks and links -> the words a voice says."""
    text = _MD_LINK.sub(r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"\1", text)
    return text.replace("`", "")


def _parse_markdown(text: str) -> dict:
    """Markdown -> the raw document `_normalize` accepts.

    The first `#` heading is the title. Every later heading opens a slide.
    A line that is only an image sets that slide's picture; `<!-- pause: N -->`
    on a line of its own sets the pause after it; any other HTML comment is a
    note to self and is not spoken — on its own line, after the words on a
    line, or running over several lines. A narrator that reads "less than,
    exclamation mark, dash, dash" aloud is the failure this guards against.
    Text before the first slide heading becomes an opening slide.
    """
    title = ""
    slides: List[dict] = []
    cur: Optional[dict] = None
    para: List[str] = []
    in_comment = False

    def flush_para() -> None:
        if para and cur is not None:
            cur["narration"].append(" ".join(para))
        para.clear()

    def open_slide(heading: str) -> dict:
        flush_para()
        s = {"title": heading, "narration": []}
        slides.append(s)
        return s

    for raw in text.splitlines():
        line = raw.strip()
        if in_comment:
            # Inside a comment that opened on an earlier line. Nothing here is
            # spoken until it closes, and a blank line in it is not a
            # paragraph break.
            _, closed, rest = line.partition("-->")
            if not closed:
                continue
            in_comment = False
            line = rest.strip()
            if not line:
                continue
        if not line:
            flush_para()
            continue
        h = _MD_HEADING.match(line)
        if h:
            heading = _strip_inline_markup(h.group(2)).strip()
            if len(h.group(1)) == 1 and not title and not slides:
                title = heading
                continue
            cur = open_slide(heading)
            continue
        if cur is None:
            cur = open_slide(title)
        img = _MD_IMAGE.match(line)
        if img:
            cur.setdefault("image", img.group(1))
            continue
        pause = _MD_PAUSE.match(line)
        if pause:
            cur["pause_after"] = float(pause.group(1))
            continue
        if _MD_COMMENT.match(line):
            continue
        line = _MD_INLINE_COMMENT.sub(" ", line)
        if "<!--" in line:
            line, _, _ = line.partition("<!--")
            in_comment = True
        line = _MD_LIST.sub("", line.strip())
        if line.startswith(">"):
            line = line.lstrip("> ").strip()
        line = _strip_inline_markup(line).strip()
        if line:
            para.append(line)
    flush_para()
    return {"title": title, "slides": slides}


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()[:48]
    return s.strip("-") or fallback


def _paragraphs(value: Any) -> List[str]:
    """A slide's `narration` (one string or a list of them) -> paragraphs."""
    parts = value if isinstance(value, (list, tuple)) else [value]
    out: List[str] = []
    for part in parts:
        for chunk in re.split(r"\n\s*\n", str(part or "")):
            flat = " ".join(chunk.split())
            if flat:
                out.append(flat)
    return out


def _normalize(doc: Any) -> dict:
    if isinstance(doc, list):
        doc = {"slides": doc}
    if not isinstance(doc, dict):
        raise ScriptError("A script is an object with a `slides` list.")
    raw_slides = doc.get("slides")
    if not isinstance(raw_slides, list) or not raw_slides:
        raise ScriptError("The script has no slides.")
    if len(raw_slides) > MAX_SLIDES:
        raise ScriptError(
            f"The script has {len(raw_slides)} slides; the limit is {MAX_SLIDES}.")

    slides: List[dict] = []
    seen: set = set()
    total_chars = 0
    for n, raw in enumerate(raw_slides, 1):
        if isinstance(raw, str):
            raw = {"narration": raw}
        if not isinstance(raw, dict):
            raise ScriptError(f"Slide {n} is not an object.")
        title = " ".join(str(raw.get("title") or "").split())
        sid = str(raw.get("id") or "").strip() or _slug(title, f"slide-{n}")
        if not SLIDE_ID_RE.match(sid):
            raise ScriptError(
                f"Slide {n} has id {sid!r}; ids use letters, digits, '-' and "
                f"'_' only (64 at most).")
        base, k = sid, 2
        while sid in seen:
            sid = f"{base[:60]}-{k}"
            k += 1
        seen.add(sid)

        # `paragraphs` is this function's own output key: accepting it back
        # makes parse_script idempotent, which the server relies on — the
        # route parses to validate, queues the parsed document, and the
        # runner parses what it was handed.
        paras = _paragraphs(raw.get("narration", raw.get(
            "paragraphs", raw.get("text", ""))))
        total_chars += sum(len(p) for p in paras)

        # Only the file NAME survives. The server matches it against the
        # uploaded images, so a path in a script can never point at a file
        # somewhere else on the machine.
        image = os.path.basename(str(raw.get("image") or "").replace("\\", "/"))
        if image and not image.lower().endswith(IMAGE_EXTS):
            raise ScriptError(
                f"Slide {n} names image {image!r}; use "
                f"{', '.join(IMAGE_EXTS)}.")

        slide = {"id": sid, "title": title, "image": image, "paragraphs": paras}
        pause = raw.get("pause_after")
        if pause is not None:
            try:
                slide["pause_after"] = max(0.0, min(_TIMING_MAX, float(pause)))
            except (TypeError, ValueError):
                raise ScriptError(
                    f"Slide {n}: pause_after must be a number of seconds.")
        slides.append(slide)

    if total_chars == 0:
        raise ScriptError("No slide has any narration text.")
    if total_chars > MAX_SCRIPT_CHARS:
        raise ScriptError(
            f"The script is {total_chars:,} characters; the limit is "
            f"{MAX_SCRIPT_CHARS:,}.")
    return {"title": " ".join(str(doc.get("title") or "").split()),
            "slides": slides}


def parse_script(raw: Any) -> dict:
    """A script in any accepted form -> {title, slides: [...]}.

    Accepts a dict, a JSON document, or Markdown. Each returned slide is
    {id, title, image, paragraphs[, pause_after]}; `image` is a bare file name
    or "". Raises ScriptError with a message meant for the person who wrote
    the script.
    """
    if isinstance(raw, (dict, list)):
        return _normalize(raw)
    text = str(raw or "").strip()
    if not text:
        raise ScriptError("The script is empty.")
    if text[0] in "{[":
        try:
            return _normalize(json.loads(text))
        except json.JSONDecodeError as e:
            raise ScriptError(f"The script looks like JSON but does not parse: {e}")
    return _normalize(_parse_markdown(text))


# ═══════════════════════════════════════════════════════════════════════
#  Sentences and chunks
# ═══════════════════════════════════════════════════════════════════════

# A period after one of these does not end a sentence. Erring toward "not a
# boundary" is the safe direction: a missed split gives one longer line; a
# false split cuts "art. 12" into two clips with a pause in the middle.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "sra", "jr", "st", "vs", "etc",
    "no", "nos", "art", "arts", "fig", "vol", "sec", "ch", "pp", "approx",
    "inc", "ltd", "lic", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
    # "11 Dec. 2014" is one date, not two sentences.
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
})
_SENTENCE_END = re.compile(r"([.!?…]+[\"'”’)\]]*)\s+")
_OPENERS = "\"'“‘([¿¡"


def split_sentences(text: str) -> List[str]:
    """Split a paragraph into sentences. Whitespace is collapsed first."""
    flat = " ".join(str(text or "").split())
    out: List[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(flat):
        nxt = flat[m.end():m.end() + 1]
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in _OPENERS):
            continue
        if m.group(1).startswith(".") and not m.group(1).startswith(".."):
            word = flat[start:m.start()].rsplit(" ", 1)[-1]
            word = word.lstrip(_OPENERS).lower()
            if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue
        out.append(flat[start:m.end(1)].strip())
        start = m.end()
    tail = flat[start:].strip()
    if tail:
        out.append(tail)
    return out


def _is_lead_in(piece: str) -> bool:
    """"second," / "and third," / "however," — a short clause that opens
    what follows it and reads as a dangling scrap at the end of a line."""
    return len(piece) <= 14 and piece.endswith(",")


def _greedy_chunks(flat: str, width: int) -> List[str]:
    pieces = [p for p in re.split(r"(?<=[;:,—–])\s+", flat) if p]
    out: List[str] = []
    cur = ""
    for i, piece in enumerate(pieces):
        fits = bool(cur) and len(cur) + 1 + len(piece) <= width
        # A lead-in goes with the clause it introduces, not the one before
        # it — unless the line so far is short enough that it has to.
        strand = (fits and _is_lead_in(piece) and i < len(pieces) - 1
                  and len(cur) >= 0.4 * width)
        if len(piece) > width:
            # Too long for one line on its own: wrap it together with what
            # is pending, so a lead-in flows into it instead of being cut
            # off as a two-word line.
            wrapped = textwrap.wrap(f"{cur} {piece}".strip(), width=width,
                                    break_long_words=False,
                                    break_on_hyphens=False)
            out.extend(wrapped[:-1])
            cur = wrapped[-1] if wrapped else ""
        elif not cur:
            cur = piece
        elif fits and not strand:
            cur = f"{cur} {piece}"
        else:
            out.append(cur)
            cur = piece
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int) -> List[str]:
    """Cut `text` into pieces of at most `max_chars`, at clause boundaries
    where there are any and at spaces otherwise.

    The pieces are balanced: a 100-character line under an 80-character limit
    comes back as two halves, not as 80 characters and a 20-character orphan
    that flashes past as a subtitle or gets its own clipped synthesis call.
    A single word longer than the limit stays whole.
    """
    flat = " ".join(str(text or "").split())
    if not flat:
        return []
    max_chars = max(1, int(max_chars))
    if len(flat) <= max_chars:
        return [flat]
    n = math.ceil(len(flat) / max_chars)
    target = min(max_chars, math.ceil(len(flat) / n) + 15)
    balanced = _greedy_chunks(flat, target)
    if len(balanced) <= n + 1:
        return balanced
    return _greedy_chunks(flat, max_chars)


def build_segments(script: dict, max_chars: int = MAX_TTS_CHARS) -> List[dict]:
    """One segment per synthesis call, in script order.

    Each carries the keys the TTS engines read (`idx`, `text`,
    `translated_text`, `speaker`) plus where it came from (`slide`,
    `slide_index`) and whether it closes a paragraph. `start`/`end` are zero:
    there is no timeline yet, and `build_timeline` writes the real one once
    the clips exist.
    """
    segments: List[dict] = []
    for si, slide in enumerate(script["slides"]):
        for para in slide.get("paragraphs") or []:
            units = [u for s in split_sentences(para)
                     for u in chunk_text(s, max_chars)]
            for ui, unit in enumerate(units):
                segments.append({
                    "idx": len(segments),
                    "start": 0.0, "end": 0.0,
                    "text": unit, "translated_text": unit,
                    "speaker": NARRATOR,
                    "slide": slide["id"], "slide_index": si,
                    "paragraph_end": ui == len(units) - 1,
                })
    return segments


def estimate_duration(segments: Iterable[dict],
                      timing: Optional[dict] = None) -> float:
    """Rough spoken length in seconds, before anything is synthesized."""
    t = resolve_timing(timing)
    segs = list(segments)
    speech = sum(len(s.get("text") or "") for s in segs) / READING_CPS
    return round(speech + len(segs) * t["sentence_gap"], 1)


# ═══════════════════════════════════════════════════════════════════════
#  Timeline
# ═══════════════════════════════════════════════════════════════════════

def resolve_timing(overrides: Optional[dict] = None) -> Dict[str, float]:
    """DEFAULT_TIMING with valid overrides applied, each clamped to 0-30 s.

    Unknown keys and non-numbers are ignored, not errors: a caller sending a
    stale or misspelled knob gets the default pause, not a failed job.
    """
    out = dict(DEFAULT_TIMING)
    for k, v in (overrides or {}).items():
        if k not in DEFAULT_TIMING or v is None:
            continue
        try:
            out[k] = max(0.0, min(_TIMING_MAX, float(v)))
        except (TypeError, ValueError):
            continue
    return out


def build_timeline(script: dict, segments: List[dict],
                   durations: Dict[int, Optional[float]],
                   timing: Optional[dict] = None) -> dict:
    """Place every segment and every slide on one clock.

    `durations` maps segment idx -> measured clip length in seconds. A missing
    or non-positive entry means that line failed to synthesize: it still gets
    a slot, sized from its text at reading speed and marked `missing_audio`,
    so the slide stays up long enough to read the subtitle and the lines
    after it are not pulled forward onto the wrong picture.

    Returns {duration, timing, segments, slides}. Segments gain
    placed_start/placed_end (mirrored into start/end, the keys the subtitle
    and transcript code read). Slides gain start/end and are contiguous: each
    one begins exactly where the previous one ended, so a slideshow built
    from them has no gaps and no overlaps.
    """
    t = resolve_timing(timing)
    by_slide: Dict[int, List[dict]] = {}
    for seg in segments:
        by_slide.setdefault(int(seg.get("slide_index", 0)), []).append(seg)

    clock = 0.0
    placed: List[dict] = []
    slides_out: List[dict] = []
    last = len(script["slides"]) - 1
    for si, slide in enumerate(script["slides"]):
        slide_start = clock
        units = by_slide.get(si, [])
        pause = slide.get("pause_after")
        if not units:
            # A picture with nothing said over it: a title card.
            clock += pause if pause is not None else t["silent_slide_hold"]
        else:
            clock += t["lead_in"] if si == 0 else t["slide_lead"]
            for k, seg in enumerate(units):
                dur = durations.get(seg["idx"])
                missing = not dur or dur <= 0
                if missing:
                    dur = max(1.0, len(seg.get("text") or "") / READING_CPS)
                start, end = clock, clock + float(dur)
                out = dict(seg)
                out.update(start=round(start, 3), end=round(end, 3),
                           placed_start=round(start, 3),
                           placed_end=round(end, 3))
                if missing:
                    out["missing_audio"] = True
                placed.append(out)
                clock = end
                if k < len(units) - 1:
                    clock += (t["paragraph_gap"] if seg.get("paragraph_end")
                              else t["sentence_gap"])
            if pause is not None:
                clock += pause
            else:
                clock += t["tail"] if si == last else t["slide_gap"]
        slides_out.append({
            "id": slide["id"], "index": si, "title": slide.get("title", ""),
            "image": slide.get("image", ""),
            "start": round(slide_start, 3), "end": round(clock, 3),
        })
    return {"duration": round(clock, 3), "timing": t,
            "segments": placed, "slides": slides_out}


# ═══════════════════════════════════════════════════════════════════════
#  Subtitles
# ═══════════════════════════════════════════════════════════════════════

def subtitle_cues(placed_segments: List[dict],
                  limits: Optional[dict] = None) -> List[dict]:
    """Timeline segments -> subtitle cues that fit the line limits.

    A synthesis unit is a whole sentence, which is usually too long for one
    subtitle. Each is cut into display chunks and the unit's time is shared
    between them in proportion to their length. There is no forced alignment
    behind that — within a sentence a cue can lead or trail the voice by a
    few hundred milliseconds. It cannot drift further, because every sentence
    boundary is exact: those times were measured from the clips.

    Cues carry {idx, start, end, text[, display_text], slide, segment}; the
    SRT/VTT writers in subtitles.py take them as they are.
    """
    lim = dict(DEFAULT_LIMITS)
    if limits:
        lim.update({k: limits[k] for k in DEFAULT_LIMITS if k in limits})
    per_line, max_lines = int(lim["max_chars_per_line"]), int(lim["max_lines"])
    # A little under the box: N lines of exactly `per_line` only fit when the
    # words happen to break there, and they never do.
    cap = max(per_line, per_line * max_lines - 6)
    gap = float(lim["min_gap_ms"]) / 1000.0

    cues: List[dict] = []
    for seg in placed_segments:
        text = (seg.get("translated_text") or seg.get("text") or "").strip()
        if not text:
            continue
        chunks = chunk_text(text, cap)
        total = sum(len(c) for c in chunks) or 1
        t0 = float(seg["placed_start"])
        span = max(0.0, float(seg["placed_end"]) - t0)
        for chunk in chunks:
            share = span * len(chunk) / total
            cue = {"idx": len(cues), "start": round(t0, 3),
                   "end": round(t0 + share, 3), "text": chunk,
                   "slide": seg.get("slide", ""), "segment": seg.get("idx")}
            wrapped = _rewrap(chunk, per_line, max_lines)
            if wrapped and "\n" in wrapped:
                cue["display_text"] = wrapped
            cues.append(cue)
            t0 += share

    # Open the minimum gap by shaving the earlier cue, never by moving the
    # later one: a cue's start is tied to the voice, its end is not.
    # The extra millisecond keeps the result on the right side of the
    # validator's `<` after float rounding (19.36 - 19.24 is 0.11999…).
    for cur, nxt in zip(cues, cues[1:]):
        if nxt["start"] - cur["end"] < gap + 0.001:
            new_end = nxt["start"] - gap - 0.001
            if new_end - cur["start"] >= _MIN_CUE_SEC:
                cur["end"] = round(new_end, 3)
    return cues


def cue_violations(cues: List[dict], limits: Optional[dict] = None) -> List[dict]:
    """Line-limit violations left in `cues` (subtitles.validate_cues)."""
    return validate_cues(cues, limits)


# ═══════════════════════════════════════════════════════════════════════
#  Slideshow planning
# ═══════════════════════════════════════════════════════════════════════

def slideshow_spans(slides: List[dict]) -> List[Tuple[str, float]]:
    """Timeline slides -> [(image, seconds)], in order.

    A slide with no picture of its own keeps the previous picture on screen,
    so its time is added to that span. Leading slides with no picture have
    nothing to inherit and raise ScriptError — the caller decides whether
    that means "no video" or "tell the user".
    """
    spans: List[Tuple[str, float]] = []
    for s in slides:
        dur = max(0.0, float(s["end"]) - float(s["start"]))
        image = s.get("image") or ""
        if image and (not spans or spans[-1][0] != image):
            spans.append((image, dur))
        elif spans:
            spans[-1] = (spans[-1][0], spans[-1][1] + dur)
        else:
            raise ScriptError(
                f"Slide {s.get('id')!r} comes first and has no image; a "
                f"slideshow needs a picture to open on.")
    return spans


def _concat_quote(path: str) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "'\\''") + "'"


def concat_script(spans: List[Tuple[str, float]]) -> str:
    """The ffconcat document for `spans` (paths as given).

    A still is one frame, and the demuxer gives each entry's `duration` to
    the gap before the NEXT entry. So the last picture is listed a second
    time: with nothing after it, its duration has no gap to describe and the
    picture would be on screen for a single frame.

    This only places one frame per slide change on the clock. Holding each
    picture until the next one is `slideshow_command`'s job.
    """
    if not spans:
        raise ScriptError("No slides to show.")
    lines = ["ffconcat version 1.0"]
    for path, dur in spans:
        lines.append(f"file {_concat_quote(path)}")
        lines.append(f"duration {max(float(dur), 0.0):.3f}")
    lines.append(f"file {_concat_quote(spans[-1][0])}")
    return "\n".join(lines) + "\n"


def _filter_escape(path: str) -> str:
    """A path as the subtitles filter wants it: forward slashes, and the
    drive colon escaped (the argument is parsed as a filter string)."""
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def slideshow_command(concat_path: str, audio_path: str, output_path: str, *,
                      duration: float, size: Tuple[int, int] = (1920, 1080),
                      fps: int = 30, srt_path: str = "",
                      burn_subs: bool = False, crf: int = 18,
                      preset: str = "medium") -> List[str]:
    """ffmpeg argv: stills + narration (+ subtitles) -> one H.264/AAC MP4.

    Subtitles are a soft track by default — viewers can turn them off, and
    YouTube reads its own caption file anyway. `burn_subs` paints them into
    the picture for players that ignore subtitle tracks (most chat apps).
    Pictures are letterboxed to `size`, never stretched.

    The frame rate is set by the `fps` FILTER, not by `-r`. The input is one
    frame per slide change, and something has to repeat each frame until the
    next. `-r` leaves that to the encoder, which only repeats a frame once
    the following one arrives — and the closing frame sits exactly at the
    end, where `-t` has already dropped it. Measured on a two-slide, 18.3 s
    narration: `-r 30` wrote 220 frames (7.3 s of picture under 18.3 s of
    sound, in a container that still reports 18.3 s — right in ffprobe's
    summary, wrong in a player); `fps=30` wrote all 550. The filter runs
    before `-t` is applied, so it sees the closing frame and fills up to it.

    Order matters too: `fps` comes after the scaling, so two pictures are
    scaled twice and not 550 times, and before `subtitles`, which can only
    paint a cue onto a frame that exists at that moment.
    """
    w, h = int(size[0]), int(size[1])
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
          f"fps={max(1, int(fps))}")
    soft = bool(srt_path) and not burn_subs
    if srt_path and burn_subs:
        vf += f",subtitles='{_filter_escape(srt_path)}'"
    vf += ",format=yuv420p"

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
           "-i", str(audio_path)]
    if soft:
        cmd += ["-i", str(srt_path)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    if soft:
        cmd += ["-map", "2:0"]
    cmd += ["-vf", vf,
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k"]
    if soft:
        cmd += ["-c:s", "mov_text"]
    # Both inputs already run for the timeline's length; -t pins the output
    # to it so a rounding difference between them cannot leave a tail of
    # silence or a frozen last frame.
    cmd += ["-t", f"{max(float(duration), 0.1):.3f}",
            "-movflags", "+faststart", str(output_path)]
    return cmd


# ═══════════════════════════════════════════════════════════════════════
#  I/O — measuring, mixing and rendering what the functions above planned
# ═══════════════════════════════════════════════════════════════════════

def clip_duration(path: str) -> float:
    """Length of an audio file in seconds; 0.0 when it cannot be read."""
    if not path or not os.path.exists(path):
        return 0.0
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate:
            return info.frames / float(info.samplerate)
    except Exception:
        pass  # mp3 on an older libsndfile — ask ffprobe instead
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip() or 0.0)
    except Exception as e:
        log.warning(f"[narration] could not measure {os.path.basename(path)}: {e}")
        return 0.0


def measure_durations(segments: List[dict]) -> Dict[int, float]:
    """{idx: seconds} for every segment that has a readable clip."""
    out: Dict[int, float] = {}
    for seg in segments:
        dur = clip_duration(seg.get("audio_path") or "")
        if dur > 0:
            out[seg["idx"]] = dur
    return out


def _read_mono(path: str, sample_rate: int):
    """A clip as mono float32 at `sample_rate`. Decodes through ffmpeg only
    when soundfile cannot read it or the rate differs (edge-tts writes mp3)."""
    import numpy as np
    import soundfile as sf
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr == sample_rate:
            return data.mean(axis=1)
    except Exception:
        pass
    with tempfile.TemporaryDirectory(prefix="gochidubb_narr_") as tmp:
        wav = os.path.join(tmp, "clip.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(sample_rate),
             "-acodec", "pcm_s16le", wav],
            check=True, capture_output=True, timeout=120)
        data, _sr = sf.read(wav, dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.mean(axis=1))


def mix_narration(placed_segments: List[dict], output_path: str, *,
                  duration: float, sample_rate: int = 48000,
                  loudnorm: bool = True) -> dict:
    """Lay every clip on one track at its placed time and write a WAV.

    Each clip gets a 5 ms fade at both ends: clips are cut at arbitrary
    sample values and butting them against silence clicks. A line with no
    clip leaves silence in its slot. Returns {placed, missing, peak}.

    Loudness normalization is the dub pipeline's own (assembler), so a
    narration and a dub from the same install sit at the same level. It is
    best-effort: a narration at its raw level is still a narration.
    """
    import numpy as np
    import soundfile as sf

    total = max(1, int(math.ceil(float(duration) * sample_rate)))
    track = np.zeros(total, dtype=np.float32)
    fade = max(1, int(0.005 * sample_rate))
    placed = missing = 0
    for seg in placed_segments:
        path = seg.get("audio_path") or ""
        if not path or not os.path.exists(path):
            missing += 1
            continue
        try:
            clip = np.array(_read_mono(path, sample_rate), dtype=np.float32)
        except Exception as e:
            log.warning(f"[narration] unreadable clip for line {seg.get('idx')}: {e}")
            missing += 1
            continue
        if len(clip) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        start = int(round(float(seg["placed_start"]) * sample_rate))
        end = min(start + len(clip), total)
        if end > start:
            track[start:end] += clip[: end - start]
            placed += 1
    peak = float(np.max(np.abs(track))) if total else 0.0
    if peak > 0.99:
        track *= 0.99 / peak
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    sf.write(output_path, track, sample_rate, subtype="PCM_16")

    if loudnorm and placed:
        try:
            from .assembler import _normalize_loudness_inplace
            _normalize_loudness_inplace(output_path)
        except Exception as e:
            log.warning(f"[narration] loudness normalization skipped: {e}")
    return {"placed": placed, "missing": missing, "peak": round(peak, 3)}


def render_slideshow(spans: List[Tuple[str, float]], audio_path: str,
                     output_path: str, *, duration: float,
                     srt_path: str = "", burn_subs: bool = False,
                     size: Tuple[int, int] = (1920, 1080), fps: int = 30) -> str:
    """Encode the slideshow. `spans` hold image paths that exist on disk."""
    from .ffmpeg_run import run_ffmpeg

    work = os.path.dirname(os.path.abspath(output_path)) or "."
    concat_path = os.path.join(work, "narration_slides.ffconcat")
    with open(concat_path, "w", encoding="utf-8") as f:
        f.write(concat_script(spans))
    cmd = slideshow_command(
        concat_path, audio_path, output_path, duration=duration, size=size,
        fps=fps, srt_path=srt_path, burn_subs=burn_subs)
    run_ffmpeg(cmd, desc="narration-slideshow", logger=log)
    return output_path
