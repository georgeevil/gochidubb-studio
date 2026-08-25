"""Tests for pipeline/subtitles.py (CLD-269).

The cue builder, validator and auto-fixer are pure functions; the properties
worth pinning are the ones the review workbench relies on: placed-time
preference, override precedence and bounding, typed violations against
crafted limit-breakers, the auto-fix never-worsens guarantee, and the exact
SRT/VTT bytes a player will parse.
"""
import pytest

from pipeline import subtitles as subs


def _cue(idx, start, end, text, display_text=None):
    c = {"idx": idx, "start": float(start), "end": float(end), "text": text}
    if display_text:
        c["display_text"] = display_text
    return c


# ═══════════════════════════════════════════════════════════════════════
#  build_cues
# ═══════════════════════════════════════════════════════════════════════

class TestBuildCues:
    def test_prefers_placed_times_and_translated_text(self):
        segs = [{"idx": 0, "start": 1.0, "end": 2.0, "placed_start": 1.4,
                 "placed_end": 2.6, "text": "src", "translated_text": "dub"}]
        (c,) = subs.build_cues(segs)
        assert (c["start"], c["end"], c["text"]) == (1.4, 2.6, "dub")

    def test_falls_back_to_source_times_and_text(self):
        segs = [{"idx": 3, "start": 1.0, "end": 2.0, "text": "src"}]
        (c,) = subs.build_cues(segs)
        assert (c["idx"], c["start"], c["end"], c["text"]) == (3, 1.0, 2.0, "src")

    def test_skips_non_speech_and_empty_segments(self):
        segs = [
            {"idx": 0, "start": 0, "end": 1, "text": "kept"},
            {"idx": 1, "start": 1, "end": 2, "text": "music", "non_speech": True},
            {"idx": 2, "start": 2, "end": 3, "text": "   "},
        ]
        assert [c["idx"] for c in subs.build_cues(segs)] == [0]

    def test_display_text_override_wins_but_text_is_untouched(self):
        segs = [{"idx": 0, "start": 0, "end": 2, "translated_text": "spoken",
                 "text": "src"}]
        (c,) = subs.build_cues(segs, {"0": {"display_text": "shown"}})
        assert c["text"] == "spoken"          # the dialogue contract
        assert c["display_text"] == "shown"   # what the viewer reads
        assert subs.cue_text(c) == "shown"

    def test_override_keys_accept_int_and_str(self):
        segs = [{"idx": 0, "start": 0, "end": 2, "text": "a"}]
        for key in (0, "0"):
            (c,) = subs.build_cues(segs, {key: {"display_text": "b"}})
            assert c["display_text"] == "b"

    def test_time_deltas_apply_and_are_bounded(self):
        segs = [{"idx": 0, "start": 5.0, "end": 7.0, "text": "a"}]
        (c,) = subs.build_cues(segs, {"0": {"start_delta": 0.2,
                                            "end_delta": -0.3}})
        assert (c["start"], c["end"]) == (5.2, 6.7)
        # Beyond ±MAX_TIME_DELTA is clamped, not honored.
        (c,) = subs.build_cues(segs, {"0": {"end_delta": 4.0}})
        assert c["end"] == pytest.approx(7.0 + subs.MAX_TIME_DELTA)

    def test_overrides_can_never_overlap_neighbors(self):
        segs = [
            {"idx": 0, "start": 0.0, "end": 2.0, "text": "a"},
            {"idx": 1, "start": 2.2, "end": 4.0, "text": "b"},
        ]
        cues = subs.build_cues(segs, {"0": {"end_delta": 0.5},
                                      "1": {"start_delta": -0.5}})
        assert cues[1]["start"] >= cues[0]["end"]
        assert cues[1]["end"] > cues[1]["start"]

    def test_start_never_goes_negative(self):
        segs = [{"idx": 0, "start": 0.1, "end": 1.0, "text": "a"}]
        (c,) = subs.build_cues(segs, {"0": {"start_delta": -0.5}})
        assert c["start"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  validate_cues — one crafted breaker per kind
# ═══════════════════════════════════════════════════════════════════════

class TestValidateCues:
    def test_clean_cues_have_no_violations(self):
        cues = [_cue(0, 0, 2, "Short line."), _cue(1, 2.5, 4.5, "Another.")]
        assert subs.validate_cues(cues) == []

    def test_line_too_long(self):
        text = "x" * 43
        (v,) = subs.validate_cues([_cue(0, 0, 10, text)])
        assert v["kind"] == "line_too_long"
        assert v["value"] == 43 and v["limit"] == 42

    def test_too_many_lines(self):
        (v,) = subs.validate_cues([_cue(0, 0, 10, "a\nb\nc")])
        assert v["kind"] == "too_many_lines"
        assert v["value"] == 3 and v["limit"] == 2

    def test_cps_exceeded(self):
        # 42 chars in 2 s = 21 CPS against the 17 limit.
        (v,) = subs.validate_cues([_cue(0, 0, 2, "x" * 42)])
        assert v["kind"] == "cps_exceeded"
        assert v["value"] == 21.0 and v["limit"] == 17.0

    def test_line_breaks_do_not_count_toward_cps(self):
        # 20+20 chars over 4 s = 10 CPS — the \n itself is not a character
        # the viewer reads.
        cues = [_cue(0, 0, 4, ("y" * 20) + "\n" + ("y" * 20))]
        assert subs.validate_cues(cues) == []

    def test_gap_too_small_belongs_to_the_earlier_cue(self):
        cues = [_cue(0, 0, 5.0, "a"), _cue(1, 5.05, 7.0, "b")]
        (v,) = subs.validate_cues(cues)
        assert v["kind"] == "gap_too_small" and v["idx"] == 0
        assert v["value"] == pytest.approx(50.0) and v["limit"] == 120

    def test_limits_are_configurable(self):
        cues = [_cue(0, 0, 10, "x" * 30)]
        assert subs.validate_cues(cues) == []
        vs = subs.validate_cues(cues, {"max_chars_per_line": 20})
        assert [v["kind"] for v in vs] == ["line_too_long"]


# ═══════════════════════════════════════════════════════════════════════
#  annotate_cues
# ═══════════════════════════════════════════════════════════════════════

class TestAnnotateCues:
    def test_metrics_are_render_ready(self):
        cues = [_cue(0, 0.0, 2.0, "ab\ncdef"), _cue(1, 2.5, 4.0, "xyz")]
        a, b = subs.annotate_cues(cues)
        assert a["lines"] == 2 and a["chars_per_line"] == 4
        assert a["cps"] == pytest.approx(3.0)  # 6 chars / 2 s
        assert a["gap_ms"] == pytest.approx(500.0)
        assert b["gap_ms"] is None  # last cue has no next

    def test_violations_are_attached_per_cue(self):
        cues = [_cue(0, 0, 2, "x" * 42), _cue(1, 3, 5, "fine")]
        a, b = subs.annotate_cues(cues)
        assert [v["kind"] for v in a["violations"]] == ["cps_exceeded"]
        assert b["violations"] == []


# ═══════════════════════════════════════════════════════════════════════
#  autofix_cues
# ═══════════════════════════════════════════════════════════════════════

def _apply(cues, fixes):
    """What the server route does with a fix set, minus persistence."""
    out = [dict(c) for c in cues]
    for c in out:
        f = fixes.get(c["idx"])
        if not f:
            continue
        if "display_text" in f:
            c["display_text"] = f["display_text"]
        if "end_delta" in f:
            c["end"] += f["end_delta"]
    return out


class TestAutofixCues:
    def test_no_violations_means_no_fixes(self):
        assert subs.autofix_cues([_cue(0, 0, 2, "Fine.")]) == {}

    def test_rewraps_a_long_line_at_a_space(self):
        text = "This sentence is deliberately much longer than the line limit"
        cues = [_cue(0, 0, 10, text)]
        fixes = subs.autofix_cues(cues)
        wrapped = fixes[0]["display_text"]
        assert "\n" in wrapped
        assert all(len(ln) <= 42 for ln in wrapped.split("\n"))
        assert wrapped.replace("\n", " ") == text  # words survive intact
        assert subs.validate_cues(_apply(cues, fixes)) == []

    def test_rebreaks_three_lines_to_two(self):
        cues = [_cue(0, 0, 10, "First clause here,\nsecond clause here,\nthird.")]
        fixes = subs.autofix_cues(cues)
        assert len(fixes[0]["display_text"].split("\n")) <= 2
        assert subs.validate_cues(_apply(cues, fixes)) == []

    def test_cps_fix_extends_the_out_time_into_the_gap_only(self):
        # 42 chars / 2 s = 21 CPS; needs 42/17 ≈ 2.47 s. Plenty of gap.
        cues = [_cue(0, 0.0, 2.0, "x" * 42), _cue(1, 8.0, 9.0, "ok")]
        fixes = subs.autofix_cues(cues)
        assert 0.4 < fixes[0]["end_delta"] <= subs.MAX_TIME_DELTA
        assert "display_text" not in fixes[0]  # shortening text is a human edit
        assert subs.validate_cues(_apply(cues, fixes)) == []

    def test_cps_fix_never_eats_the_minimum_gap(self):
        cues = [_cue(0, 0.0, 2.0, "x" * 42), _cue(1, 2.3, 3.0, "ok")]
        fixes = subs.autofix_cues(cues)
        if fixes:
            fixed = _apply(cues, fixes)
            assert fixed[1]["start"] - fixed[0]["end"] >= 0.120 - 1e-9

    def test_gap_fix_shaves_the_earlier_cue(self):
        cues = [_cue(0, 0.0, 5.0, "a"), _cue(1, 5.05, 7.0, "b")]
        fixes = subs.autofix_cues(cues)
        # The 50 ms gap needs 70 ms more, taken from the earlier cue's end.
        assert fixes[0]["end_delta"] == pytest.approx(-0.07, abs=0.01)
        fixed = _apply(cues, fixes)
        assert fixed[1]["start"] - fixed[0]["end"] == pytest.approx(0.12, abs=0.005)
        assert subs.validate_cues(fixed) == []

    def test_unfixable_cue_is_left_flagged_not_mangled(self):
        # One unbreakable 60-char word: wrapping cannot help, and there is
        # no gap to extend into. The right answer is no fix at all.
        cues = [_cue(0, 0.0, 2.0, "w" * 60), _cue(1, 2.12, 3.0, "ok")]
        assert subs.autofix_cues(cues) == {}

    def test_fix_sets_strictly_decrease_violations(self):
        scenarios = [
            [_cue(0, 0, 10, "word " * 20)],
            [_cue(0, 0.0, 2.0, "x" * 42), _cue(1, 8.0, 9.0, "ok")],
            [_cue(0, 0.0, 5.0, "a"), _cue(1, 5.05, 7.0, "b")],
            [_cue(0, 0, 8, "one clause here,\nand another,\nand a third one")],
        ]
        for cues in scenarios:
            fixes = subs.autofix_cues(cues)
            if fixes:
                assert (len(subs.validate_cues(_apply(cues, fixes)))
                        < len(subs.validate_cues(cues)))


# ═══════════════════════════════════════════════════════════════════════
#  Export — exact bytes
# ═══════════════════════════════════════════════════════════════════════

class TestWriters:
    CUES = [
        _cue(0, 0.5, 2.25, "Hello"),
        _cue(1, 3.0, 4.5, "spoken text", display_text="World\nAgain"),
    ]

    def test_srt_golden(self, tmp_path):
        p = tmp_path / "out.srt"
        subs.write_srt_cues(self.CUES, str(p))
        assert p.read_text(encoding="utf-8") == (
            "1\n00:00:00,500 --> 00:00:02,250\nHello\n\n"
            "2\n00:00:03,000 --> 00:00:04,500\nWorld\nAgain\n\n"
        )

    def test_vtt_golden(self, tmp_path):
        p = tmp_path / "out.vtt"
        subs.write_vtt_cues(self.CUES, str(p))
        assert p.read_text(encoding="utf-8") == (
            "WEBVTT\n\n"
            "1\n00:00:00.500 --> 00:00:02.250\nHello\n\n"
            "2\n00:00:03.000 --> 00:00:04.500\nWorld\nAgain\n\n"
        )

    def test_writers_show_the_display_override(self, tmp_path):
        p = tmp_path / "out.srt"
        subs.write_srt_cues(self.CUES, str(p))
        content = p.read_text(encoding="utf-8")
        assert "World" in content and "spoken text" not in content
