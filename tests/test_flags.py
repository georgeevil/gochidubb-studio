"""Tests for the review-screen flag heuristic (pipeline/flags.py).

The property under test is mostly a *negative* one. The review screen
promises "only 2 things to check — everything else is already fine", and
that promise is broken by a detector that fires too readily, not by one that
misses something. So the first and most important test here is that an
ordinary transcript produces nothing at all.
"""
import pytest

from pipeline import flags
from pipeline.flags import flag_segments


def seg(idx, text, translated, **extra):
    s = {"idx": idx, "start": float(idx) * 2, "end": float(idx) * 2 + 2,
         "text": text, "translated_text": translated,
         "speaker": "SPEAKER_00"}
    s.update(extra)
    return s


# A perfectly ordinary en->ru dub: no names, nothing unusual.
CLEAN_RU = [
    seg(0, "Hello everyone welcome to my channel.",
        "Привет всем, добро пожаловать на мой канал."),
    seg(1, "Today we are talking about dubbing.",
        "Сегодня мы поговорим о дубляже."),
    seg(2, "It is really cool technology that lets you change the language.",
        "Это действительно классная технология, которая позволяет менять язык."),
    seg(3, "This is another completely normal segment.",
        "Этот сегмент тоже совершенно обычный."),
    seg(4, "Let me show you how it works step by step.",
        "Позвольте показать вам, как это работает, шаг за шагом."),
]


# ── The common case is zero flags ─────────────────────────────────────

def test_clean_transcript_yields_no_flags():
    assert flag_segments(CLEAN_RU, target_lang="ru", source_lang="en") == []


def test_clean_same_script_transcript_yields_no_flags():
    """en->es keeps names verbatim on purpose; that is not a finding."""
    segs = [
        seg(0, "We flew to Kyoto last spring.",
            "Volamos a Kyoto la primavera pasada."),
        seg(1, "Kyoto was full of tourists then.",
            "Kyoto estaba lleno de turistas entonces."),
        seg(2, "Nakamura showed us the temple.",
            "Nakamura nos mostró el templo."),
    ]
    assert flag_segments(segs, target_lang="es", source_lang="en") == []


def test_capitalised_sentence_openers_are_not_names():
    """"Today", "This", "Amazing" open sentences; none of them is a name."""
    segs = [
        seg(0, "Today we begin.", "Сегодня мы начинаем."),
        seg(1, "Today is different.", "Сегодня всё иначе."),
        seg(2, "Amazing right?", "Потрясающе, да?"),
        seg(3, "Amazing work here.", "Отличная работа здесь."),
    ]
    assert flag_segments(segs, target_lang="ru", source_lang="en") == []


def test_conftest_fixtures_produce_nothing_without_translations(sample_segments):
    """Raw transcription output — no translated_text anywhere — is not a
    review screen's problem, and must not become one."""
    segs = [{**s, "idx": i} for i, s in enumerate(sample_segments)]
    assert flag_segments(segs, target_lang="ru", source_lang="en") == []


def test_empty_input():
    assert flag_segments([], target_lang="ru") == []
    assert flag_segments(None, target_lang="ru") == []


# ── Detector 1: inconsistent renderings ───────────────────────────────

def test_planted_inconsistent_name_yields_exactly_one_flag():
    segs = CLEAN_RU + [
        seg(5, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной."),
        seg(6, "Kyoto was full of tourists then.",
            "Кёто был полон туристов тогда."),
        seg(7, "I would go back to Kyoto tomorrow.",
            "Я бы вернулся в Киото хоть завтра."),
    ]
    found = flag_segments(segs, target_lang="ru", source_lang="en")
    assert len(found) == 1
    f = found[0]
    assert f["kind"] == "name"
    assert f["reason"] == "name_inconsistent"
    assert f["source_span"] == "Kyoto"
    assert set(f["variants"]) == {"Киото", "Кёто"}
    # Flagged where the name first appears, not where the drift was noticed.
    assert f["idx"] == 5


def test_two_transliterations_alone_are_not_enough_evidence():
    """With the source spelling gone from both lines, the only evidence is
    that two tokens appear nowhere else — which on a long transcript is a
    coincidence often enough to matter. It is what paired "Michael" with two
    unrelated Russian tokens on a real 1,815-segment job."""
    segs = CLEAN_RU + [
        seg(5, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной."),
        seg(6, "Kyoto was full of tourists then.",
            "Кёто был полон туристов тогда."),
    ]
    assert flag_segments(segs, target_lang="ru", source_lang="en") == []


def test_the_source_spelling_surviving_anchors_it_at_two_mentions():
    """When one line kept the name as-is, the source word anchors the
    comparison and two occurrences are enough."""
    segs = CLEAN_RU + [
        seg(5, "We flew to Kyoto last spring.",
            "Мы летали в Kyoto прошлой весной."),
        seg(6, "Kyoto was full of tourists then.",
            "Киото был полон туристов тогда."),
    ]
    found = flag_segments(segs, target_lang="ru", source_lang="en")
    assert [f["reason"] for f in found] == ["name_inconsistent"]


def test_name_kept_in_one_line_and_translated_in_another():
    segs = CLEAN_RU + [
        seg(5, "We flew to Kyoto last spring.",
            "Мы летали в Kyoto прошлой весной."),
        seg(6, "Kyoto was full of tourists then.",
            "Киото был полон туристов тогда."),
    ]
    found = flag_segments(segs, target_lang="ru", source_lang="en")
    assert [f["reason"] for f in found] == ["name_inconsistent"]
    assert "Kyoto" in found[0]["variants"]
    assert "Киото" in found[0]["variants"]


def test_a_consistently_rendered_name_is_not_flagged():
    segs = CLEAN_RU + [
        seg(5, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной."),
        seg(6, "Kyoto was full of tourists then.",
            "Киото был полон туристов тогда."),
        seg(7, "I would go back to Kyoto tomorrow.",
            "Я бы вернулся в Киото хоть завтра."),
    ]
    assert flag_segments(segs, target_lang="ru", source_lang="en") == []


def test_a_name_seen_only_once_is_not_flagged():
    segs = CLEAN_RU + [
        seg(5, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной."),
    ]
    assert flag_segments(segs, target_lang="ru", source_lang="en") == []


# ── Detector 2: untransliterated names ────────────────────────────────

def test_latin_name_surviving_into_a_cyrillic_line():
    segs = [
        seg(0, "Our sponsor today is Squarespace.",
            "Наш сегодняшний спонсор — Squarespace."),
        seg(1, "It is a good service.", "Это хороший сервис."),
        seg(2, "I have used Squarespace for years.",
            "Я пользуюсь Squarespace уже много лет."),
    ]
    found = flag_segments(segs, target_lang="ru", source_lang="en")
    assert [f["reason"] for f in found] == ["name_untransliterated"]
    assert found[0]["source_span"] == "Squarespace"


def test_a_name_mentioned_once_is_not_worth_a_decision():
    """One decision that fixes many occurrences is worth making; a passing
    mention is not — and on real transcripts single mentions were most of
    this detector's volume."""
    segs = [
        seg(0, "Our sponsor today is Squarespace.",
            "Наш сегодняшний спонсор — Squarespace."),
        seg(1, "It is a good service.", "Это хороший сервис."),
    ]
    assert flag_segments(segs, target_lang="ru", source_lang="en") == []


def test_acronyms_carry_over_without_being_flagged():
    segs = [
        seg(0, "The NASA launch was yesterday.", "Запуск NASA был вчера."),
        seg(1, "We use AI for this.", "Мы используем AI для этого."),
    ]
    assert flag_segments(segs, target_lang="ru", source_lang="en") == []


def test_same_script_pairs_skip_the_transliteration_check():
    segs = [
        seg(0, "Our sponsor today is Squarespace.",
            "Nuestro patrocinador de hoy es Squarespace."),
    ]
    assert flag_segments(segs, target_lang="es", source_lang="en") == []


# ── Detector 3: glossary ──────────────────────────────────────────────

GLOSSARY = {"domains": [
    {"name": "creator-review", "triggers": [], "target_lang": "ru",
     "terms": {"Kyoto": "Киото"}},
    {"name": "other-language", "triggers": [], "target_lang": "de",
     "terms": {"Squarespace": "Squarespace"}},
]}


def test_a_confirmed_glossary_term_does_not_re_flag():
    """The user already settled "Kyoto" -> "Киото". Honoured everywhere, so
    the review screen must have nothing to say about it."""
    segs = CLEAN_RU + [
        seg(5, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной."),
        seg(6, "Kyoto was full of tourists then.",
            "Киото был полон туристов тогда."),
    ]
    assert flag_segments(segs, target_lang="ru", source_lang="en",
                         glossary=GLOSSARY) == []


def test_a_glossary_term_the_translation_ignored_is_flagged_once():
    segs = CLEAN_RU + [
        seg(5, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной."),
        seg(6, "Kyoto was full of tourists then.",
            "Кёто был полон туристов тогда."),
        seg(7, "I would go back to Kyoto tomorrow.",
            "Я бы вернулся в Киото хоть завтра."),
    ]
    found = flag_segments(segs, target_lang="ru", source_lang="en",
                          glossary=GLOSSARY)
    assert [f["reason"] for f in found] == ["glossary_miss"]
    assert found[0]["source_span"] == "Kyoto"
    assert found[0]["target_span"] == "Киото"


def test_glossary_terms_for_another_language_are_ignored():
    assert flags.glossary_terms(GLOSSARY, "ru") == {"Kyoto": "Киото"}
    assert flags.glossary_terms(GLOSSARY, "de") == {"Squarespace": "Squarespace"}
    assert flags.glossary_terms(GLOSSARY, "fr") == {}


def test_a_flat_glossary_mapping_is_accepted():
    assert flags.glossary_terms({"Kyoto": "Киото"}, "ru") == {"Kyoto": "Киото"}
    assert flags.glossary_terms(None, "ru") == {}


# ── Removed detector: idioms ──────────────────────────────────────────

def test_length_outliers_are_not_flagged_at_all():
    """A "this might be a joke" detector was written and then removed.

    Over the whole real corpus it produced two flags, both unanswerable
    ("Bro, no way."), with an empty target span covering a whole sentence.
    A flag nobody can act on is worse than no flag.
    """
    for src, tgt in [
        ("Oh my god, that is a piece of cake for him. [laughs]", "Ха."),
        ("The quarterly revenue figures were released this morning.", "Sí."),
        ("Bro, no way.", "Нет."),
    ]:
        assert flag_segments([seg(0, src, tgt)],
                             target_lang="es", source_lang="en") == []


# ── Detector 5: low-confidence ASR, filler only ───────────────────────

def test_low_confidence_asr_is_surfaced_when_there_is_room():
    segs = [seg(0, "something something mumbled here", "algo algo murmurado",
                word_conf_mean=0.3, word_conf_min=0.05)]
    found = flag_segments(segs, target_lang="es", source_lang="en")
    assert [f["kind"] for f in found] == ["unclear"]


def test_one_weak_word_in_an_otherwise_clear_line_is_not_a_finding():
    """word_conf_min is a minimum, so it drifts down with length on its own.

    Measured on this install, 22.6% of all real segments fall below the
    threshold this used to fire on — which produced five "we couldn't hear
    this" cards for a seven-segment clip whose transcript was clean. The
    segment now has to be mushy as a whole before it counts.
    """
    segs = [seg(0, "We are empty in Vienna today", "Estamos vacíos en Viena hoy",
                word_conf_mean=0.88, word_conf_min=0.2)]
    assert flag_segments(segs, target_lang="es", source_lang="en") == []


def test_filler_never_fills_the_whole_screen():
    """Five cards a creator cannot act on read as "your video is a mess"."""
    segs = [seg(i, f"mumbled line number {i} here", f"línea {i}",
                word_conf_mean=0.2, word_conf_min=0.02) for i in range(8)]
    found = flag_segments(segs, target_lang="es", source_lang="en", max_flags=5)
    assert len(found) <= 2
    assert all(f["kind"] == "unclear" for f in found)


def test_low_confidence_never_displaces_an_actionable_finding():
    segs = [
        seg(0, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной.",
            word_conf_mean=0.2, word_conf_min=0.02),
        seg(1, "Kyoto was full of tourists then.",
            "Кёто был полон туристов тогда.",
            word_conf_mean=0.2, word_conf_min=0.02),
        seg(2, "I would go back to Kyoto tomorrow.",
            "Я бы вернулся в Киото хоть завтра.",
            word_conf_mean=0.2, word_conf_min=0.02),
    ]
    found = flag_segments(segs, target_lang="ru", source_lang="en", max_flags=1)
    assert [f["reason"] for f in found] == ["name_inconsistent"]


def test_a_two_word_mumble_is_not_worth_anyone_s_time():
    segs = [seg(0, "Okay so", "Vale entonces",
                word_conf_mean=0.2, word_conf_min=0.02)]
    assert flag_segments(segs, target_lang="es", source_lang="en") == []


# ── Ranking and the cap ───────────────────────────────────────────────

def _noisy_transcript(n=12):
    """A transcript engineered to trip several detectors at once."""
    segs = []
    for i in range(n):
        segs.append(seg(i, f"The meeting with Anderson number {i} was unclear.",
                        f"Встреча {i} была неясной.",
                        word_conf_mean=0.2, word_conf_min=0.02))
    return segs


def test_the_cap_holds():
    found = flag_segments(_noisy_transcript(), target_lang="ru",
                          source_lang="en", max_flags=5)
    assert len(found) <= 5


def test_the_cap_is_configurable_and_zero_means_nothing():
    noisy = _noisy_transcript()
    assert len(flag_segments(noisy, target_lang="ru", source_lang="en",
                             max_flags=2)) <= 2
    assert flag_segments(noisy, target_lang="ru", source_lang="en",
                         max_flags=0) == []


def test_at_most_one_flag_per_segment():
    found = flag_segments(_noisy_transcript(), target_lang="ru",
                          source_lang="en")
    idxs = [f["idx"] for f in found]
    assert len(idxs) == len(set(idxs))


def test_flags_are_ordered_best_first():
    segs = [
        seg(0, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной."),
        seg(1, "Kyoto was full of tourists then.",
            "Кёто был полон туристов тогда."),
        seg(3, "I would go back to Kyoto tomorrow.",
            "Я бы вернулся в Киото хоть завтра."),
        seg(2, "The whole trip was mumbled through.",
            "Вся поездка прошла невнятно.", word_conf_mean=0.2, word_conf_min=0.02),
    ]
    found = flag_segments(segs, target_lang="ru", source_lang="en")
    # low_asr is held back entirely: there is a question on screen that the
    # creator can actually answer, and "we couldn't hear this" only dilutes it.
    assert [f["reason"] for f in found] == ["name_inconsistent"]


# ── Shape ─────────────────────────────────────────────────────────────

def test_every_flag_carries_the_full_shape():
    segs = [
        seg(0, "We flew to Kyoto last spring.",
            "Мы летали в Киото прошлой весной."),
        seg(1, "Kyoto was full of tourists then.",
            "Кёто был полон туристов тогда."),
        seg(2, "I would go back to Kyoto tomorrow.",
            "Я бы вернулся в Киото хоть завтра."),
    ]
    f = flag_segments(segs, target_lang="ru", source_lang="en")[0]
    for key in ("idx", "kind", "start", "end", "source_text",
                "translated_text", "source_span", "target_span", "reason",
                "score", "variants"):
        assert key in f, key
    assert isinstance(f["start"], float)
    assert isinstance(f["variants"], list)
    assert f["kind"] in ("name", "term", "joke", "unclear")


def test_untranslated_segments_are_skipped_not_crashed_on():
    segs = [
        seg(0, "We flew to Kyoto last spring.", ""),
        seg(1, "Kyoto was full of tourists then.", None),
        {"idx": 2},
        "not a dict",
    ]
    assert flag_segments(segs, target_lang="ru", source_lang="en") == []


@pytest.mark.parametrize("lang", ["ru", "ja", "zh", "ar", "es", "", "xx"])
def test_no_target_language_breaks_it(lang):
    flag_segments(CLEAN_RU, target_lang=lang, source_lang="en")
