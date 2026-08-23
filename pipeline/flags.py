"""Which few segments a creator should actually look at before recording.

The review screen is not a transcript editor. It shows a handful of spans —
usually none — where the translator plausibly got something wrong in a way a
human can settle in one click, and it says "everything else is already fine".
That promise is only worth making if this module is *quiet*: a detector that
fires on every other segment has turned the screen back into a transcript
editor with extra steps.

So every detector here is deliberately narrow, and each one exists because a
creator can act on it:

1. ``_inconsistent_renderings`` — the same name came back two different ways.
   A real, observable failure of the batched translator: batches are
   translated independently, so nothing holds a name steady across them. One
   glossary entry fixes every occurrence, which is the whole review loop in a
   single detector. This leads.
2. ``_untransliterated_name`` — a Latin-script name survived verbatim into a
   Cyrillic/CJK line. Cross-script only: within one script a repeated token is
   almost always a name that *should* carry over (see ``quality._source_bleed``
   for the same reasoning).
3. ``_glossary_miss`` — a term the user already decided on was not honoured.
4. ``_low_asr`` — the transcriber itself was unsure. The weakest: there is
   no Keep/Change in it, only an admission, so it is shown at most once and
   only onto an otherwise empty screen.

Findings are ranked in three tiers. Detectors 1 and 3 put a name decision in
front of someone and may fill the screen — a name-dense video that genuinely
needs five confirmations should show five. Detector 2 is capped, and 4 never
dilutes a question the creator can actually answer.

A fifth detector, a length-ratio "this might be a joke" test, was written and
then removed: over the whole corpus it produced two flags, both of them
unanswerable ("Bro, no way."). A flag nobody can act on is worse than no flag.

Signals available here are exactly what survives checkpointing: ``text``,
``translated_text``, ``start``/``end``, ``speaker`` and the ASR aggregates
``avg_logprob`` / ``no_speech_prob`` / ``word_conf_mean`` / ``word_conf_min``.
The per-word array is deliberately dropped before a checkpoint is written, so
per-term spans are reconstructed by string matching here — which is why the
detector that needs no alignment at all is the one that leads.

**Calibrated against 4,417 real segments from 53 finished jobs on this
install, not against intuition.** The first cut returned five flags for half
of them and the weakest detector was 60% of the output. Every threshold and
suppression rule below that cites a measurement earned it by being wrong on
that corpus first — the contraction reduction, the inflection test, the
containment rule and the ASR gate especially. Re-run that sweep before
loosening any of them.

Pure module: no I/O, no model loads, no network, GPU-free.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger("gochidubb.flags")

# ── Tunables ──────────────────────────────────────────────────────────
# How confident each kind of finding is. Ranking is by score, so this is
# also the order the review screen shows them in.
SCORES = {
    "name_inconsistent": 0.95,
    "glossary_miss": 0.80,
    "name_untransliterated": 0.60,
    "low_asr": 0.30,
}

# Two renderings this similar are the same name spelled two ways ("Киото" /
# "Кёто"); anything below is two different words that happen to co-occur.
_VARIANT_SIMILARITY = 0.60

# Occurrences a name needs before two *non-verbatim* renderings count as a
# finding. When the source spelling survives in one line and not another
# (the "Kyoto" / "Киото" shape) two occurrences are enough — the source word
# itself anchors the comparison. When neither line kept it, the only
# evidence is that two tokens happen to appear nowhere else, and on a long
# transcript two occurrences make that a coincidence often enough to matter:
# it is what paired "Michael" with "Вин"/"Карвин".
_MIN_OCCURRENCES_FOR_PAIR = 3

# ASR aggregates below these are worth a "we couldn't hear this clearly".
#
# Measured over 4,232 real segments on this install before these were set.
# word_conf_min is the *minimum* per-word confidence, a low-order statistic
# that drifts down with segment length on its own: its median is 0.69 and
# 22.6% of all segments fall under 0.35. A gate that fires on a fifth of
# everything is not "the transcriber was unsure", it is "whisper has one
# weak word per line" — and it was producing five "we couldn't hear this"
# cards for a seven-segment clip whose transcript was perfectly clean.
#
# So the gate now needs the segment to be mushy *as a whole* (mean) as well
# as somewhere in particular (min). Together they fire on 1.7% of segments.
# avg_logprob is a separate arm and needs no company: below -0.9 is 0.1% of
# real segments, which is the rarity a filler signal should have.
_LOW_WORD_CONF_MEAN = 0.50
_LOW_WORD_CONF_MIN = 0.15
_LOW_AVG_LOGPROB = -0.9
_LOW_ASR_MIN_WORDS = 3

# Ceilings on the two weakest detectors. Each fills space that the findings
# a creator can actually act on left over — but with nothing else on the
# screen they filled all five slots, and five cards nobody can act on read
# as "your video is a mess" rather than "here are two words to confirm".
#
# "we couldn't hear this" is the weaker of the two: there is no Keep/Change
# in it, only an admission, so it is held back entirely unless the screen
# would otherwise be empty. A name kept in Latin at least has an answer.
_MAX_FILLER = 1
# One is deliberate rather than two: measured over the corpus it cut cap-hits
# from 3 checkpoints to 1 and total volume by a fifth, with no change to how
# many checkpoints come back clean. A name left in Latin is nearly always a
# "keep it" decision, so the second and third of them buy the reader little.
_MAX_UNTRANSLITERATED = 1

# Scripts where capitalisation marks a proper noun. Everywhere else the only
# thing that stands out in a target line is a Latin-script run.
_CASED_SCRIPTS = frozenset({"latin", "cyrillic", "greek"})

# ── Lexicons ──────────────────────────────────────────────────────────
# Words that are capitalised for reasons other than being a name. Kept
# generous on purpose: a stopword that slips through costs a false flag,
# and a name wrongly filtered costs nothing but a missed one.
_NOT_NAMES = frozenset("""
    the this that there they then them these those their and but for not you
    your yours yes yeah yep nope well with what when where which why who whom
    whose how here his her hers him its our ours now one two three four five
    six seven eight nine ten just let like look make made more most much new
    next only other over some still such take than thank thanks time very
    want was way were will would should could because before after again all
    also always any are can come did does don down each even every first from
    get got going good great have here hey hello okay actually maybe really
    basically alright anyway know last left right say see she sure think too
    try use watch while about above across against back been being both call
    called does doing done else enough ever few find give had has himself how
    into keep kind long many mean might must need never night nothing off
    once only open own part people place put same said say seen since said
    something sorry sound start stop sword tell thing things told took
    understand until upon used using want watch week whatever whether year
    years young
    god jesus christ damn shit crap hell wow oops ouch yikes phew huh
    guys guy folks everyone everybody nobody somebody someone
    wait stay run hold move listen please holy dead custom team gone real
    true false full half open close free live love hate best worst better
    worse big small long short high low old cool nice fine okay ready done
    welcome again hey oh ah yep nah wow sure
    bro bruh dude man boy girl kid kids mom dad doc boss buddy mate top
    bring take give keep leave send bring put set turn watch dog cat
    relax chill wow hurry breathe focus remember forget imagine guess
    honestly seriously anyway meanwhile suddenly finally
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october
    november december
""".split())

# Global platforms and brands. These reach a creator's audience under the
# same name in every language, so "are you sure about YouTube?" is the
# fastest way to teach someone the review screen is not worth reading. The
# list is deliberately short and only holds names that are genuinely global —
# a product name specific to one channel's subject matter is a real question
# and stays flaggable.
_GLOBAL_BRANDS = frozenset("""
    youtube youtu tiktok instagram facebook twitter twitch reddit discord
    telegram whatsapp snapchat linkedin pinterest tumblr vimeo patreon
    google apple microsoft amazon netflix spotify adobe nvidia intel samsung
    android iphone ipad macbook windows linux ubuntu chrome safari firefox
    minecraft roblox fortnite lego legos nintendo playstation xbox steam
    v-bucks robux twitchcon youtuber
    paypal visa mastercard stripe uber airbnb tesla spacex nasa
    openai chatgpt claude anthropic gemini copilot github gitlab
    wifi bluetooth usb gps ai vpn
""".split())

# Words that are not names in the *source* language. _NOT_NAMES is English,
# which is why a Spanish source dubbed into Russian offered "Hola" as a name
# to confirm. Only the handful that actually turn up capitalised mid-line —
# greetings, fillers, courtesy words — not a real stopword list.
_NOT_NAMES_BY_LANG = {
    "es": "hola gracias adios adiós señor señora bueno vale claro pues oye "
          "amigo amiga chico chica mira dios madre padre",
    "pt": "ola olá obrigado obrigada senhor senhora bom bem claro pois olha "
          "amigo amiga cara gente deus mae mãe pai",
    "fr": "bonjour salut merci monsieur madame bien alors voila voilà écoute "
          "ecoute ami amie dieu mere mère pere père",
    "de": "hallo danke herr frau gut also schau freund freundin gott mutter "
          "vater bitte tschüss tschuss",
    "it": "ciao grazie signore signora bene allora guarda amico amica dio "
          "madre padre prego",
    "uk": "привіт дякую пане пані добре отже дивись друг подруга боже мама тато",
    "ru": "привет спасибо господин госпожа хорошо итак смотри друг подруга "
          "боже мама папа",
}

# A proper-noun candidate in Latin source text. All-caps tokens are filtered
# afterwards: "NASA" and "AI" carry over on purpose and a creator asked to
# "spell it differently" would rightly be confused.
_SOURCE_NAME_RE = re.compile(r"\b[A-Z][A-Za-z’'\-]{2,}\b")

# What comes after the apostrophe in a contraction or a possessive. Both
# reduce to the word in front of it: "Marvin's" is the name Marvin, and
# "I'm" / "That's" / "There's" are not names at all — they only looked like
# ones because the apostrophe made them long enough to pass the length test.
# Real transcripts are full of these, and "how should we spell I'm?" is the
# single worst question this screen could ask anyone.
_CONTRACTION_TAILS = frozenset({"s", "m", "d", "t", "re", "ve", "ll"})


def _name_head(token: str) -> str:
    """"Marvin's" -> "Marvin"; "I'm" -> "I"; "O'Brien" -> "O'Brien"."""
    for apos in ("’", "'"):
        if apos in token:
            head, _, tail = token.rpartition(apos)
            if head and tail.lower() in _CONTRACTION_TAILS:
                return head
    return token

# A word in any script, keeping internal apostrophes and hyphens together.
_WORD_RE = re.compile(r"[^\W\d_]+(?:[’'\-][^\W\d_]+)*", re.UNICODE)


# ── Small helpers ─────────────────────────────────────────────────────

def _target_script(target_lang: str) -> str:
    """Script name for a target code, e.g. "cyrillic". Latin when unknown.

    Imported lazily, exactly as ``quality._source_bleed`` does, so this
    module stays importable (and testable) without pulling the translator's
    HTTP stack in behind it.
    """
    code = (target_lang or "").strip().lower()[:2]
    if not code:
        return "latin"
    try:
        from .translator import _LANG_SCRIPT
    except ImportError:  # pragma: no cover - translator is always present
        return "latin"
    return _LANG_SCRIPT.get(code, "latin")


def _stopwords_for(source_lang: str) -> frozenset:
    """Words that are not names, in English plus the source language."""
    code = (source_lang or "").strip().lower()[:2]
    extra = _NOT_NAMES_BY_LANG.get(code)
    if not extra:
        return _NOT_NAMES
    return _NOT_NAMES | frozenset(extra.split())


def _is_latin(token: str) -> bool:
    """Whether every letter in the token is Latin-script."""
    letters = [c for c in token if c.isalpha()]
    return bool(letters) and all(ord(c) < 0x0250 for c in letters)


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Case-insensitive whole-word search that also handles multi-word terms."""
    if not haystack or not needle:
        return False
    pattern = r"(?<![^\W\d_])" + re.escape(needle.strip()) + r"(?![^\W\d_])"
    try:
        return re.search(pattern, haystack, re.IGNORECASE | re.UNICODE) is not None
    except re.error:  # pragma: no cover - re.escape makes this unreachable
        return needle.lower() in haystack.lower()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# An inflected form keeps the stem and changes the ending: "Кевин" /
# "Кевина", "Дагестан" / "Дагестане", "Ганнер" / "Ганнеру". Russian, Polish,
# Ukrainian and the rest decline names as a matter of grammar, and this was
# the single largest source of false findings on real transcripts — asking a
# creator to pick one spelling of "Kevin" would get them ungrammatical
# Russian if they answered.
#
# The test is a shared stem plus two short tails, not similarity: the true
# pairs it must not swallow ("Киото" / "Кёто", "Чад" / "Чед") differ at the
# second character, so they share no stem worth the name.
_INFLECTION_MAX_TAIL = 3
_INFLECTION_MIN_STEM = 3
_INFLECTION_STEM_SHARE = 0.6


def _same_stem(a: str, b: str) -> bool:
    """Whether two renderings are one name in two grammatical cases.

    Compared on the first hyphen-segment as well as whole: "Шрек-холла" and
    "Шрека" are the same name, once inside a compound noun and once
    declined, and neither is a spelling a creator should be asked to choose
    between.
    """
    if "-" in a or "-" in b:
        ha, hb = a.split("-")[0], b.split("-")[0]
        if (ha, hb) != (a, b) and _same_stem(ha, hb):
            return True
    x, y = a.lower(), b.lower()
    n = 0
    for cx, cy in zip(x, y):
        if cx != cy:
            break
        n += 1
    longer = max(len(x), len(y))
    if n < max(_INFLECTION_MIN_STEM, longer * _INFLECTION_STEM_SHARE):
        return False
    return (len(x) - n) <= _INFLECTION_MAX_TAIL and (len(y) - n) <= _INFLECTION_MAX_TAIL


def _source_names(text: str,
                  stops: frozenset = _NOT_NAMES) -> List[Tuple[str, bool]]:
    """Proper-noun candidates in a source line, each with "was it initial?".

    Every sentence starts capitalised, so a segment-initial token is not
    evidence of a name — but discarding it outright loses real names that
    happen to open a line. Callers get the flag instead and decide: the
    convention here is that a token counts as a name once it has been seen
    *somewhere* away from the start.
    """
    if not text:
        return []
    first = _WORD_RE.search(text)
    first_start = first.start() if first else -1
    out: List[Tuple[str, bool]] = []
    seen: set = set()
    for m in _SOURCE_NAME_RE.finditer(text):
        tok = _name_head(m.group(0))
        if tok.isupper() or len(tok) < 3 or tok.lower() in stops:
            continue
        initial = m.start() == first_start
        if tok in seen:
            continue
        seen.add(tok)
        out.append((tok, initial))
    return out


def _target_candidates(text: str, cased: bool) -> List[Tuple[str, bool]]:
    """Tokens in a translated line that could be a rendered proper noun.

    Same "initial" caveat as ``_source_names`` — in a cased script the first
    word is capitalised whatever it is.
    """
    if not text:
        return []
    out: List[Tuple[str, bool]] = []
    seen: set = set()
    for i, m in enumerate(_WORD_RE.finditer(text)):
        tok = m.group(0)
        if len(tok) < 3 or tok.isupper() or not tok[0].isupper():
            continue
        if not cased and not _is_latin(tok):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append((tok, cased and i == 0))
    return out


def glossary_terms(glossary: Any, target_lang: str) -> Dict[str, str]:
    """Flatten a glossary to ``{source term: target term}`` for one language.

    Accepts either the on-disk shape (``{"domains": [{target_lang, terms}]}``)
    or an already-flat mapping, using the same per-``target_lang`` rule as
    ``translator._load_user_glossary`` so the review screen and the
    translation prompt agree on which terms are in force.
    """
    if not glossary:
        return {}
    lang = (target_lang or "").strip().lower()[:2]
    if isinstance(glossary, dict) and "domains" in glossary:
        terms: Dict[str, str] = {}
        for domain in glossary.get("domains") or []:
            if not isinstance(domain, dict):
                continue
            dl = str(domain.get("target_lang") or "").strip().lower()[:2]
            if dl and lang and dl != lang:
                continue
            for k, v in (domain.get("terms") or {}).items():
                if isinstance(k, str) and isinstance(v, str):
                    terms[k] = v
        return terms
    if isinstance(glossary, dict):
        return {k: v for k, v in glossary.items()
                if isinstance(k, str) and isinstance(v, str)}
    return {}


def _flag(seg: Dict[str, Any], kind: str, reason: str, *,
          source_span: str = "", target_span: str = "",
          variants: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    return {
        "idx": seg.get("idx"),
        "kind": kind,
        "start": float(seg.get("start") or 0.0),
        "end": float(seg.get("end") or 0.0),
        "source_text": seg.get("text") or "",
        "translated_text": seg.get("translated_text") or "",
        "source_span": source_span,
        "target_span": target_span,
        "reason": reason,
        "score": SCORES.get(reason, 0.1),
        "variants": list(variants or []),
    }


def _usable(segments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Segments that have both sides of a translation to compare."""
    out = []
    for s in segments or ():
        if not isinstance(s, dict):
            continue
        if not (s.get("text") or "").strip():
            continue
        out.append(s)
    return out


# ── Detectors ─────────────────────────────────────────────────────────

def _inconsistent_renderings(segments: List[Dict[str, Any]], *,
                             target_lang: str,
                             source_lang: str,
                             skip: frozenset) -> List[Dict[str, Any]]:
    """One source name, two different target renderings.

    The hard part is deciding which target token *is* the rendering, with no
    word alignment to read it off. Similarity to the other rendering is not
    enough on its own — measured against real transcripts, the true pair
    ("Киото" / "Кёто") and the false one ("Потому" / "Почему") score
    identically, so no threshold separates them.

    What does separate them is where else the token appears. A rendering of a
    name shows up only in lines that name it; a word of the language shows up
    everywhere. So a token is treated as a candidate rendering of N only when
    every line it appears in is a line whose source contains N. That single
    containment test removes the whole class of false positives that a
    similarity threshold cannot.
    """
    cased = _target_script(target_lang) in _CASED_SCRIPTS
    stops = _stopwords_for(source_lang)

    occurrences: Dict[str, List[int]] = {}
    confirmed: set = set()
    for pos, seg in enumerate(segments):
        for name, initial in _source_names(seg.get("text") or "", stops):
            if name.lower() in skip:
                continue
            occurrences.setdefault(name, []).append(pos)
            if not initial:
                confirmed.add(name)

    # Every target-side candidate token, and every line it appears in.
    tok_positions: Dict[str, set] = {}
    for pos, seg in enumerate(segments):
        for tok, _initial in _target_candidates(
                seg.get("translated_text") or "", cased):
            tok_positions.setdefault(tok, set()).add(pos)

    flags: List[Dict[str, Any]] = []
    for name, positions in occurrences.items():
        # Seen only ever at the start of a line — that is a capital letter,
        # not evidence of a name.
        if name not in confirmed or len(positions) < 2:
            continue
        translated = {p: (segments[p].get("translated_text") or "")
                      for p in positions}
        # Nothing to compare if some occurrence was never translated.
        if not all(translated[p].strip() for p in positions):
            continue

        named = set(positions)
        verbatim = {p for p in positions
                    if _contains_phrase(translated[p], name)}
        rest = named - verbatim

        renderings: Dict[str, set] = {}
        for tok, seen_in in tok_positions.items():
            if tok.lower() == name.lower() or not seen_in <= named:
                continue
            here = seen_in & rest
            if here:
                renderings[tok] = here

        if verbatim and rest:
            # Kept as-is in one line, replaced in another. Across scripts
            # this is the common shape: "Kyoto" once, "Киото" the next.
            if len(renderings) != 1:
                # No candidate, or several with no way to tell them apart.
                # A card that cannot name the other spelling asks the user a
                # question with no answer in it, so say nothing instead.
                continue
            other = next(iter(renderings))
            flags.append(_flag(
                segments[positions[0]], "name", "name_inconsistent",
                source_span=name,
                target_span=name if positions[0] in verbatim else other,
                variants=[name, other],
            ))
        else:
            # Two different renderings, neither the source spelling. Both
            # must survive containment, and they still have to look like each
            # other — otherwise two names that happen to co-occur pair up.
            if len(positions) < _MIN_OCCURRENCES_FOR_PAIR:
                continue
            pair = _variant_pair(renderings)
            if pair:
                a, b = pair
                flags.append(_flag(
                    segments[positions[0]], "name", "name_inconsistent",
                    source_span=name, target_span=a, variants=[a, b],
                ))
    return flags


def _variant_pair(renderings: Dict[str, set]) -> Optional[Tuple[str, str]]:
    """Two candidate renderings that never share a line and read alike."""
    toks = sorted(renderings)
    for i, a in enumerate(toks):
        for b in toks[i + 1:]:
            if renderings[a] & renderings[b]:
                # Both in one line: two things being named, not one name
                # spelled two ways.
                continue
            if _same_stem(a, b):
                # Declined, not misspelled. Nothing for a human to decide.
                continue
            if _similar(a, b) >= _VARIANT_SIMILARITY:
                return (a, b)
    return None


def _untransliterated_name(segments: List[Dict[str, Any]], *,
                           target_lang: str,
                           source_lang: str,
                           skip: frozenset) -> List[Dict[str, Any]]:
    """A repeated Latin-script name carried verbatim into a non-Latin line.

    Only names that come up more than once: one decision that fixes many
    occurrences is worth making, a passing mention is not — and on real
    transcripts the single mentions were most of the volume.

    Cross-script only. Within one script a token repeated from the source is
    almost always a name that *should* carry over unchanged, and flagging
    those would bury the findings that matter — the same reasoning
    ``quality._source_bleed`` spells out for its own word check.
    """
    target_script = _target_script(target_lang)
    if target_script == "latin" or _target_script(source_lang) == target_script:
        return []
    stops = _stopwords_for(source_lang)
    repeats: Dict[str, int] = {}
    for seg in segments:
        for m in _WORD_RE.finditer(seg.get("translated_text") or ""):
            key = _name_head(m.group(0)).lower()
            repeats[key] = repeats.get(key, 0) + 1
    seen: set = set()
    flags: List[Dict[str, Any]] = []
    for seg in segments:
        src = seg.get("text") or ""
        tgt = seg.get("translated_text") or ""
        if not tgt.strip():
            continue
        for m in _WORD_RE.finditer(tgt):
            # Same reduction as the source side: an English contraction that
            # survived into a Cyrillic line is not a name anyone can respell.
            tok = _name_head(m.group(0))
            key = tok.lower()
            if key in seen or key in skip:
                continue
            # Short tokens and acronyms are noise; a name is 4+ letters and
            # is Capitalised rather than SHOUTED.
            if len(tok) < 4 or tok.isupper() or not tok[0].isupper():
                continue
            if not _is_latin(tok) or key in stops:
                continue
            if key in _GLOBAL_BRANDS or key.rstrip("s") in _GLOBAL_BRANDS:
                # Reaches the audience under this name in every language.
                continue
            if repeats.get(key, 0) < 2:
                continue
            if not _contains_phrase(src, tok):
                continue
            seen.add(key)
            flags.append(_flag(seg, "name", "name_untransliterated",
                               source_span=tok, target_span=tok))
    return flags


def _glossary_miss(segments: List[Dict[str, Any]],
                   terms: Dict[str, str]) -> List[Dict[str, Any]]:
    """A term the user already decided on that the translation did not use."""
    if not terms:
        return []
    seen: set = set()
    flags: List[Dict[str, Any]] = []
    for seg in segments:
        src = seg.get("text") or ""
        tgt = seg.get("translated_text") or ""
        if not tgt.strip():
            continue
        for term, mapped in terms.items():
            key = term.lower()
            if key in seen or not term.strip() or not mapped.strip():
                continue
            if not _contains_phrase(src, term):
                continue
            if _contains_phrase(tgt, mapped):
                continue
            seen.add(key)
            flags.append(_flag(seg, "term", "glossary_miss",
                               source_span=term, target_span=mapped,
                               variants=[mapped]))
    return flags


def _low_asr(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The transcriber itself was unsure it heard this correctly."""
    flags: List[Dict[str, Any]] = []
    for seg in segments:
        src = (seg.get("text") or "").strip()
        if len(src.split()) < _LOW_ASR_MIN_WORDS:
            continue
        wcm = seg.get("word_conf_min")
        wca = seg.get("word_conf_mean")
        alp = seg.get("avg_logprob")
        bad = False
        if (isinstance(wcm, (int, float)) and isinstance(wca, (int, float))
                and wca < _LOW_WORD_CONF_MEAN and wcm < _LOW_WORD_CONF_MIN):
            bad = True
        if isinstance(alp, (int, float)) and alp < _LOW_AVG_LOGPROB:
            bad = True
        if bad:
            flags.append(_flag(seg, "unclear", "low_asr", source_span=src[:80]))
    return flags


# ── Entry point ───────────────────────────────────────────────────────

def flag_segments(segments: Sequence[Dict[str, Any]], *,
                  target_lang: str,
                  source_lang: str = "",
                  glossary: Any = None,
                  max_flags: int = 5) -> List[Dict[str, Any]]:
    """The few spans worth a creator's attention, best first.

    Returns ``[]`` — the expected result for a clean transcript — when
    nothing crosses a detector's bar. At most one flag per segment index,
    at most ``max_flags`` in total.

    ``glossary`` accepts the on-disk ``{"domains": [...]}`` shape or a flat
    ``{term: translation}`` map. Terms already in it are treated as settled:
    they never come back as a name flag, only as a *miss* when the
    translation ignored a decision the user already made.
    """
    segs = _usable(segments)
    cap = max(0, int(max_flags or 0))
    if not segs or cap == 0:
        return []

    terms = glossary_terms(glossary, target_lang)
    settled = frozenset(t.lower() for t in terms)

    # Tier 1 — a name decision. Allowed to fill the screen: a name-dense
    # video that genuinely needs five confirmations should show five.
    strong: List[Dict[str, Any]] = []
    strong += _inconsistent_renderings(segs, target_lang=target_lang,
                                       source_lang=source_lang, skip=settled)
    strong += _glossary_miss(segs, terms)
    ranked = _rank(strong, cap)

    # Tier 2 — a weaker name decision, capped so it cannot become the whole
    # screen on a transcript full of untranslated product names.
    room = min(cap - len(ranked), _MAX_UNTRANSLITERATED)
    if room > 0:
        used = {f["idx"] for f in ranked}
        # Also skip a name tier 1 already asked about: the same word came
        # back once kept in Latin and once transliterated, which is one
        # question, not two. Asking twice on one screen reads as a bug.
        asked = {f["source_span"].lower() for f in ranked if f["source_span"]}
        ranked += _rank(
            [f for f in _untransliterated_name(
                segs, target_lang=target_lang, source_lang=source_lang,
                skip=settled)
             if f["idx"] not in used
             and f["source_span"].lower() not in asked], room)

    # Tier 3 — "have a listen to this bit", and only onto an empty screen.
    # There is no Keep/Change in it: a creator shown a garbled line has no
    # better guess than the model did, so it must never dilute a question
    # they *can* answer. One, or none.
    if not ranked and cap > 0:
        ranked = _rank(_low_asr(segs), min(cap, _MAX_FILLER))
    return ranked


def _rank(flags: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    """Best first, one per segment, truncated to `cap`."""
    ordered = sorted(flags, key=lambda f: (-f["score"], f["start"]))
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for f in ordered:
        if f["idx"] in seen:
            continue
        seen.add(f["idx"])
        out.append(f)
        if len(out) >= cap:
            break
    return out
