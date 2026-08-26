"""Pronunciation respelling + the one seam that writes `tts_text`.

`tts_text` is what VoxCPM speaks when it differs from the dialogue line.
`translated_text` is the line of dialogue: the assemble stage rewrites
subtitles.srt from it, the translation editor shows it, the partial-retry
reuse check compares it, and the assembler's emotion-tag heuristic reads a
leading "(" from it — so instructions to the model (the Voice Design style
prefix) and sound-alike respellings (glossary `say`) must never touch it.

Two writers used to build that string independently; `compose_tts_text` is
now the ONLY place its semantics live. It is deliberately pure — no config,
no file reads — so the estimate route, the TTS stages, and the tests all
agree on what a respelling does.

`tts_text` never enters the checkpoint whitelist (see server.py
`_serialize_segments`): it is recomputed at every TTS run, which is also
what makes a glossary edit apply on retry without touching checkpoints.
Only VoxCPMSynthesizer reads it — edge-tts and F5 never see the key, so a
respelling can never be read out loud by an engine that treats it as text.
"""
import logging
import re
from typing import Dict, Optional

log = logging.getLogger("gochidubb.pronounce")

# The same word-boundary idea as pipeline/flags.py::_contains_phrase: a term
# matches on its own (multi-word terms included), not inside a longer word.
# \b breaks on non-ASCII letters; these lookarounds treat any letter as
# word-forming, which is what a glossary over 65 languages needs.
_BOUNDARY_L = r"(?<![^\W\d_])"
_BOUNDARY_R = r"(?![^\W\d_])"


def build_say_map(glossary: Optional[dict], target_lang: str) -> Dict[str, str]:
    """Flatten the on-disk glossary to {term: respelling} for one language.

    Accepts the domains shape (``{"domains": [{target_lang, terms}]}``).
    Domains scoped to another language are skipped; domains with no
    target_lang apply to every language, same per-lang rule as
    ``flags.glossary_terms``. Term values are plain strings (no `say` —
    skipped here) or ``{dst?, say?}`` objects; only entries with a
    non-empty `say` make it into the map.
    """
    if not isinstance(glossary, dict):
        return {}
    lang = (target_lang or "").strip().lower()[:2]
    out: Dict[str, str] = {}
    for domain in glossary.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        dl = str(domain.get("target_lang") or "").strip().lower()[:2]
        if dl and lang and dl != lang:
            continue
        for term, value in (domain.get("terms") or {}).items():
            if not isinstance(term, str) or not term.strip():
                continue
            say = value.get("say") if isinstance(value, dict) else None
            if isinstance(say, str) and say.strip():
                out[term.strip()] = say.strip()
    return out


def apply_say_map(text: str, say_map: Optional[Dict[str, str]]) -> str:
    """Respell every glossary `say` term in `text`, word-boundary and
    case-insensitive. Longest terms first, so "kubectl apply" wins over
    "kubectl" when both are defined. Returns the text unchanged when the
    map is empty or nothing matches."""
    if not text or not say_map:
        return text
    for term in sorted(say_map, key=len, reverse=True):
        say = say_map[term]
        if not term or not say:
            continue
        pattern = _BOUNDARY_L + re.escape(term) + _BOUNDARY_R
        try:
            # Replacement via callable: a `say` containing "\\1" or "\\g" is
            # someone's respelling, not a group reference.
            text = re.sub(pattern, lambda m: say, text,
                          flags=re.IGNORECASE | re.UNICODE)
        except re.error:  # pragma: no cover - re.escape makes this unreachable
            continue
    return text


def compose_tts_text(seg: dict, style_prefix: str = "",
                     say_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The ONLY writer of ``seg['tts_text']`` semantics.

    Applies word-boundary, case-insensitive respelling of glossary `say`
    terms to the segment's dialogue (translated_text, falling back to text),
    then the Voice-Design ``(style)`` prefix. Returns None when neither
    applies — the caller then leaves tts_text unset, and the engine speaks
    the dialogue line itself.

    The prefix is skipped when the line already opens with "(" — that is an
    emotion tag, and stacking a style instruction in front of it makes
    VoxCPM read one of them out loud. Same rule the inline writers had.
    """
    base = seg.get("translated_text") or seg.get("text") or ""
    if not base:
        return None
    respelled = apply_say_map(base, say_map)
    style = (style_prefix or "").strip().strip("()")
    prefixed = respelled
    if style and not base.lstrip().startswith("("):
        prefixed = f"({style}){respelled}"
    return prefixed if prefixed != base else None
