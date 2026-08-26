"""Tests for app/estimate_edits.py (CLD-271 counting rules).

Seconds only — the money is the route's job (billing.marginal_cost, already
tested in test_billing.py). What matters here: each kind selects the right
segments, overlapping edits count a segment once, display edits are free,
and malformed edits raise instead of quoting a price for nonsense.
"""
import pytest

from app.estimate_edits import edit_seconds


@pytest.fixture
def segments():
    return [
        {"idx": 0, "start": 0.0, "end": 2.0, "speaker": "S1",
         "translated_text": "GoChi is a tool."},
        {"idx": 1, "start": 2.0, "end": 5.0, "speaker": "S2",
         "translated_text": "It dubs videos."},
        {"idx": 2, "start": 5.0, "end": 9.0, "speaker": "S1",
         "translated_text": "GoChiDUBB does more."},
        {"idx": 3, "start": 9.0, "end": 10.0, "speaker": "S2",
         "translated_text": "Indeed."},
        {"idx": 4, "start": 10.0, "end": 12.0, "speaker": "S2",
         "translated_text": "(applause)", "non_speech": True},
    ]


def test_segment_text_sums_the_listed_segments(segments):
    total, idxs, breakdown = edit_seconds(
        segments, [{"kind": "segment_text", "idxs": [0, 2]}])
    assert total == 6.0  # 2 + 4
    assert idxs == [0, 2]
    assert breakdown == [{"kind": "segment_text", "idxs": [0, 2],
                          "seconds": 6.0}]


def test_unknown_idxs_are_ignored_not_billed(segments):
    total, idxs, _ = edit_seconds(
        segments, [{"kind": "segment_text", "idxs": [0, 99]}])
    assert total == 2.0 and idxs == [0]


def test_speaker_voice_sums_that_speakers_speech(segments):
    total, idxs, _ = edit_seconds(
        segments, [{"kind": "speaker_voice", "speaker": "S2"}])
    # S2 speaks idx 1 (3 s) and 3 (1 s); the non_speech idx 4 is not speech.
    assert total == 4.0 and idxs == [1, 3]


def test_pronunciation_matches_word_boundaries(segments):
    total, idxs, _ = edit_seconds(
        segments, [{"kind": "pronunciation", "term": "GoChi"}])
    # "GoChi" hits idx 0 but NOT "GoChiDUBB" in idx 2 — the same matcher the
    # pronunciation seam uses, so the estimate and the effect agree.
    assert idxs == [0] and total == 2.0


def test_subtitle_display_is_always_free(segments):
    total, idxs, breakdown = edit_seconds(
        segments, [{"kind": "subtitle_display", "idxs": [0, 1, 2]}])
    assert total == 0.0 and idxs == []
    assert breakdown == [{"kind": "subtitle_display", "idxs": [0, 1, 2],
                          "seconds": 0.0}]


def test_overlapping_edits_union_not_sum(segments):
    total, idxs, breakdown = edit_seconds(segments, [
        {"kind": "segment_text", "idxs": [1]},
        {"kind": "speaker_voice", "speaker": "S2"},   # idx 1 and 3
    ])
    # idx 1 appears in both edits but its 3 s are billed once.
    assert idxs == [1, 3]
    assert total == 4.0
    # The per-edit breakdown still shows each edit's own footprint.
    assert [b["seconds"] for b in breakdown] == [3.0, 4.0]


def test_display_edit_does_not_join_the_union(segments):
    total, idxs, _ = edit_seconds(segments, [
        {"kind": "subtitle_display", "idxs": [0]},
        {"kind": "segment_text", "idxs": [3]},
    ])
    assert idxs == [3] and total == 1.0


def test_empty_edit_list_costs_nothing(segments):
    assert edit_seconds(segments, []) == (0.0, [], [])


@pytest.mark.parametrize("bad", [
    [{"kind": "voodoo"}],
    [{"kind": "segment_text", "idxs": ["one"]}],
    [{"kind": "speaker_voice"}],
    [{"kind": "pronunciation", "term": "  "}],
    ["not-an-object"],
])
def test_malformed_edits_raise(segments, bad):
    with pytest.raises(ValueError):
        edit_seconds(segments, bad)


def test_segments_without_idx_fall_back_to_position():
    segs = [{"start": 0.0, "end": 1.5, "speaker": "S1",
             "translated_text": "hey"},
            {"start": 1.5, "end": 4.0, "speaker": "S1",
             "translated_text": "there"}]
    total, idxs, _ = edit_seconds(
        segs, [{"kind": "speaker_voice", "speaker": "S1"}])
    assert idxs == [0, 1] and total == 4.0
