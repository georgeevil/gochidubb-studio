"""pipeline/pronounce.py — the one seam allowed to define tts_text.

These are the CLD-266 guarantees: respelling never touches translated_text,
matches whole words case-insensitively across non-ASCII scripts, composes
with the Voice Design style prefix in a fixed order, and stays silent (None)
when it has nothing to say — so the caller leaves tts_text unset and the
engine speaks the dialogue line itself.
"""
import pytest

from pipeline.pronounce import apply_say_map, build_say_map, compose_tts_text


# ── build_say_map ──────────────────────────────────────────────────────

def _glossary(**domain):
    base = {"name": "d", "triggers": [], "target_lang": "es", "terms": {}}
    base.update(domain)
    return {"domains": [base]}


def test_build_say_map_reads_only_object_entries_with_say():
    g = _glossary(terms={
        "GoChi": {"dst": "GoChi", "say": "GOH-chee"},
        "plain": "translated",              # legacy string entry: no say
        "empty": {"dst": "x", "say": "  "}, # blank say is no say
        "sayonly": {"say": "sigh-OH"},
    })
    assert build_say_map(g, "es") == {"GoChi": "GOH-chee",
                                      "sayonly": "sigh-OH"}


def test_build_say_map_scopes_by_language():
    g = _glossary(target_lang="fr", terms={"GoChi": {"say": "GOH-chee"}})
    assert build_say_map(g, "es") == {}
    assert build_say_map(g, "fr") == {"GoChi": "GOH-chee"}
    # region variants collapse to the two-letter code
    assert build_say_map(g, "fr-CA") == {"GoChi": "GOH-chee"}


def test_build_say_map_unscoped_domain_applies_to_every_language():
    g = _glossary(target_lang="", terms={"GoChi": {"say": "GOH-chee"}})
    assert build_say_map(g, "ja") == {"GoChi": "GOH-chee"}


@pytest.mark.parametrize("bad", [None, [], "x", {"domains": "nope"},
                                 {"domains": [42]}])
def test_build_say_map_shrugs_at_malformed_glossaries(bad):
    assert build_say_map(bad, "es") == {}


# ── apply_say_map ──────────────────────────────────────────────────────

def test_respelling_is_word_bounded():
    out = apply_say_map("GoChi and GoChiDUBB", {"GoChi": "GOH-chee"})
    assert out == "GOH-chee and GoChiDUBB"


def test_respelling_is_case_insensitive():
    assert apply_say_map("GOCHI rules", {"GoChi": "GOH-chee"}) == \
        "GOH-chee rules"


def test_longest_term_wins():
    out = apply_say_map(
        "run kubectl apply now",
        {"kubectl": "kube-control", "kubectl apply": "kube-control apply"})
    assert out == "run kube-control apply now"


def test_boundaries_hold_for_cyrillic():
    # \b would split inside Cyrillic words; the lookarounds must not.
    out = apply_say_map("Кёто и Кётоский", {"Кёто": "Киото"})
    assert out == "Киото и Кётоский"


def test_backslashes_in_a_respelling_are_literal():
    assert apply_say_map("say x", {"x": r"ex\1"}) == r"say ex\1"


def test_empty_map_returns_text_unchanged():
    assert apply_say_map("hello", {}) == "hello"
    assert apply_say_map("hello", None) == "hello"


# ── compose_tts_text ───────────────────────────────────────────────────

def _seg(text="Hola GoChi."):
    return {"idx": 0, "translated_text": text}


def test_none_when_nothing_applies():
    assert compose_tts_text(_seg(), style_prefix="", say_map={}) is None
    assert compose_tts_text(_seg(), say_map={"zzz": "sleep"}) is None
    assert compose_tts_text({"translated_text": ""},
                            style_prefix="calm") is None


def test_respelling_alone_produces_tts_text():
    out = compose_tts_text(_seg(), say_map={"GoChi": "GOH-chee"})
    assert out == "Hola GOH-chee."


def test_prefix_alone_produces_tts_text():
    assert compose_tts_text(_seg(), style_prefix="calm narrator") == \
        "(calm narrator)Hola GoChi."


def test_prefix_wraps_the_respelled_text():
    out = compose_tts_text(_seg(), style_prefix="(calm)",
                           say_map={"GoChi": "GOH-chee"})
    assert out == "(calm)Hola GOH-chee."


def test_emotion_tag_suppresses_the_prefix_but_not_the_respelling():
    """A line opening with "(" already carries an emotion tag; stacking a
    style instruction in front makes VoxCPM read one of them aloud."""
    seg = _seg("(whispers) GoChi wins.")
    out = compose_tts_text(seg, style_prefix="calm",
                           say_map={"GoChi": "GOH-chee"})
    assert out == "(whispers) GOH-chee wins."


def test_dialogue_line_is_never_mutated():
    seg = _seg()
    compose_tts_text(seg, style_prefix="calm", say_map={"GoChi": "X"})
    assert seg["translated_text"] == "Hola GoChi."
    assert "tts_text" not in seg  # the seam computes; the caller assigns


def test_falls_back_to_text_when_untranslated():
    out = compose_tts_text({"text": "GoChi here"},
                           say_map={"GoChi": "GOH-chee"})
    assert out == "GOH-chee here"
