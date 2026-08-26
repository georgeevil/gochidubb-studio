"""Tests for pipeline/assembler.py — SRT formatting, writing, and audio assembly."""
import os
import tempfile
from types import SimpleNamespace

import pytest

from pipeline.assembler import (
    MIN_SEGMENT_GAP,
    build_audio_mix_filter,
    format_srt_time,
    merge_audio_video,
    plan_segment_fit,
    write_srt,
)


class TestFormatSrtTime:
    """format_srt_time() converts float seconds to SRT timestamp format."""

    def test_zero(self):
        assert format_srt_time(0.0) == "00:00:00,000"

    def test_simple_seconds(self):
        assert format_srt_time(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert format_srt_time(65.0) == "00:01:05,000"

    def test_hours(self):
        assert format_srt_time(3661.0) == "01:01:01,000"

    def test_milliseconds_rounding(self):
        # 0.12345 seconds → 123 ms
        result = format_srt_time(0.12345)
        assert result == "00:00:00,123" or result == "00:00:00,123"

    def test_large_value(self):
        assert format_srt_time(10000.0) == "02:46:40,000"

    def test_fractional_seconds(self):
        result = format_srt_time(12.345)
        assert result.endswith("345")
        assert result.startswith("00:00:12")


class TestWriteSrt:
    """write_srt() writes standard .srt format."""

    def test_writes_segments(self, sample_segments):
        """Writes valid SRT content with sequential numbering."""
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt.srt")
        try:
            write_srt(sample_segments, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()

            # Check structure: index, timestamp, text, blank line
            assert "1" in content
            assert "00:00:00,500" in content or "00:00:00,500" in content
            assert "Hello everyone" in content
            # Should have segment count entries
            assert content.count("-->") == len(sample_segments)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_uses_translated_text(self):
        """If translated_text exists, use it instead of text."""
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "translated_text": "Hola"},
        ]
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_translated.srt")
        try:
            write_srt(segs, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert "Hola" in content
            assert "Hello" not in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_empty_segments(self):
        """Empty segment list should produce empty SRT."""
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_empty.srt")
        try:
            write_srt([], tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert content == ""
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_output_is_utf8(self):
        """SRT should be UTF-8 encoded for non-ASCII support."""
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Привет мир"},
        ]
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_utf8.srt")
        try:
            write_srt(segs, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert "Привет мир" in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_prefers_dubbed_timings_when_present(self):
        """Subtitles must follow the audio, not the source it was made from.

        The SRT is first written before assembly, so a segment that had to be
        shifted or compressed leaves the file out of sync by exactly that
        amount — a 10s drift shipped this way.
        """
        segs = [{
            "start": 1.0, "end": 2.0, "text": "Hello",
            "placed_start": 3.5, "placed_end": 5.25,
        }]
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_placed.srt")
        try:
            write_srt(segs, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert "00:00:03,500 --> 00:00:05,250" in content
            assert "00:00:01,000" not in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_falls_back_to_source_timings_when_unplaced(self):
        """A transcript that was never assembled has no placements."""
        segs = [{"start": 1.0, "end": 2.0, "text": "Hello"}]
        tmp = os.path.join(tempfile.gettempdir(), "_test_write_srt_unplaced.srt")
        try:
            write_srt(segs, tmp)
            with open(tmp, encoding="utf-8") as f:
                content = f.read()
            assert "00:00:01,000 --> 00:00:02,000" in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


class TestPlanSegmentFit:
    """plan_segment_fit() decides placement and time-compression together."""

    def test_leaves_a_fitting_segment_alone(self):
        start, speed = plan_segment_fit(
            seg_start=5.0, next_start=10.0, tts_dur=3.0, current_end=4.0)
        assert start == 5.0
        assert speed == 1.0

    def test_uses_the_pause_after_a_segment_before_compressing(self):
        """A 6s clip in a 4s slot followed by 4s of silence needs no stretch.

        Budgeting to the segment's own `end` would have compressed this;
        budgeting to the next segment's start is what makes the overrun free.
        """
        start, speed = plan_segment_fit(
            seg_start=1.0, next_start=9.0, tts_dur=6.0, current_end=0.0)
        assert start == 1.0
        assert speed == 1.0

    def test_compresses_to_exactly_fit_when_the_ceiling_allows(self):
        room = 5.0 - MIN_SEGMENT_GAP - 1.0
        start, speed = plan_segment_fit(
            seg_start=1.0, next_start=5.0, tts_dur=5.0, current_end=0.0)
        assert start == 1.0
        assert speed > 1.15
        assert 5.0 / speed == pytest.approx(room)

    def test_compression_stops_at_the_ceiling_and_spills_the_remainder(self):
        """Quality wins over sync once the stretch would be audible.

        The leftover isn't lost: the next segment starts late, so its own
        budget shrinks and it absorbs what's left rather than passing it on.
        """
        room = 5.0 - MIN_SEGMENT_GAP - 1.0
        _, speed = plan_segment_fit(
            seg_start=1.0, next_start=5.0, tts_dur=6.0, current_end=0.0,
            max_stretch=1.4)
        assert speed == pytest.approx(1.4)
        assert 6.0 / speed > room

    def test_never_stretches_short_audio_out(self):
        _, speed = plan_segment_fit(
            seg_start=1.0, next_start=20.0, tts_dur=0.5, current_end=0.0)
        assert speed == 1.0

    def test_pushes_a_segment_that_would_overlap_the_previous_one(self):
        start, _ = plan_segment_fit(
            seg_start=5.0, next_start=12.0, tts_dur=1.0, current_end=6.0)
        assert start == pytest.approx(6.0 + MIN_SEGMENT_GAP)

    def test_a_late_segment_compresses_harder_to_catch_up(self):
        """The drift is paid off by the segment that inherited it.

        This is the whole fix: previously a late segment kept its full
        duration and handed the delay to the next one, so the error grew
        monotonically instead of being absorbed.
        """
        on_time = plan_segment_fit(
            seg_start=5.0, next_start=10.0, tts_dur=6.0, current_end=0.0)[1]
        late = plan_segment_fit(
            seg_start=5.0, next_start=10.0, tts_dur=6.0, current_end=7.0)[1]
        assert late > on_time

    def test_respects_the_stretch_ceiling(self):
        _, speed = plan_segment_fit(
            seg_start=1.0, next_start=2.0, tts_dur=30.0, current_end=0.0,
            max_stretch=1.4)
        assert speed == pytest.approx(1.4)


class TestPlanSegmentFitPadSlack:
    """pad_slack (CLD-268): spend inter-segment silence on an early start
    instead of compressing harder — bounded so it can never overlap."""

    def test_starts_early_by_up_to_the_slack(self):
        start, _ = plan_segment_fit(
            seg_start=10.0, next_start=15.0, tts_dur=1.0, current_end=5.0,
            pad_slack=0.5)
        assert start == pytest.approx(9.5)

    def test_early_start_is_bounded_by_the_previous_segment(self):
        start, _ = plan_segment_fit(
            seg_start=10.0, next_start=15.0, tts_dur=1.0, current_end=9.8,
            pad_slack=0.5)
        assert start == pytest.approx(9.8 + MIN_SEGMENT_GAP)

    def test_early_start_never_goes_negative(self):
        start, _ = plan_segment_fit(
            seg_start=0.2, next_start=5.0, tts_dur=1.0, current_end=0.0,
            pad_slack=0.5)
        assert start == 0.0

    def test_slack_grows_the_budget_and_softens_the_compression(self):
        no_pad = plan_segment_fit(
            seg_start=10.0, next_start=12.0, tts_dur=2.5, current_end=5.0)
        padded = plan_segment_fit(
            seg_start=10.0, next_start=12.0, tts_dur=2.5, current_end=5.0,
            pad_slack=0.5)
        assert padded[0] < no_pad[0]
        assert padded[1] < no_pad[1]

    def test_zero_slack_is_the_old_behavior(self):
        for kwargs in (
            dict(seg_start=5.0, next_start=10.0, tts_dur=3.0, current_end=4.0),
            dict(seg_start=1.0, next_start=5.0, tts_dur=6.0, current_end=0.0),
            dict(seg_start=5.0, next_start=12.0, tts_dur=1.0, current_end=6.0),
        ):
            assert (plan_segment_fit(**kwargs)
                    == plan_segment_fit(pad_slack=0.0, **kwargs))

    def test_never_overlaps_the_previous_segment(self):
        for slack in (0.0, 0.3, 1.0, 5.0):
            start, _ = plan_segment_fit(
                seg_start=10.0, next_start=15.0, tts_dur=1.0,
                current_end=9.95, pad_slack=slack)
            assert start >= 9.95 + MIN_SEGMENT_GAP - 1e-9


class TestAssembleFitOverrides:
    """assemble_dubbed_audio honours the per-segment sync-fit plan."""

    def _segments(self, wav):
        return [
            {"idx": 0, "start": 1.0, "end": 1.4, "audio_path": wav,
             "translated_text": "one"},
            {"idx": 1, "start": 5.0, "end": 5.4, "audio_path": wav,
             "translated_text": "two"},
        ]

    def test_pad_ms_places_the_segment_early(self, temp_audio_file, tmp_path):
        from pipeline.assembler import assemble_dubbed_audio
        segs = self._segments(temp_audio_file)
        assemble_dubbed_audio(
            segs, 10.0, str(tmp_path / "mix.wav"), apply_loudnorm=False,
            fit_overrides={1: {"pad_ms": 300}})
        assert segs[1]["placed_start"] == pytest.approx(4.7, abs=0.01)

    def test_string_keys_work_like_json_round_tripped_ones(
            self, temp_audio_file, tmp_path):
        from pipeline.assembler import assemble_dubbed_audio
        segs = self._segments(temp_audio_file)
        assemble_dubbed_audio(
            segs, 10.0, str(tmp_path / "mix.wav"), apply_loudnorm=False,
            fit_overrides={"1": {"pad_ms": 300}})
        assert segs[1]["placed_start"] == pytest.approx(4.7, abs=0.01)

    def test_no_overrides_changes_nothing(self, temp_audio_file, tmp_path):
        from pipeline.assembler import assemble_dubbed_audio
        plain = self._segments(temp_audio_file)
        assemble_dubbed_audio(plain, 10.0, str(tmp_path / "a.wav"),
                              apply_loudnorm=False)
        overridden = self._segments(temp_audio_file)
        assemble_dubbed_audio(overridden, 10.0, str(tmp_path / "b.wav"),
                              apply_loudnorm=False, fit_overrides={})
        assert ([s.get("placed_start") for s in plain]
                == [s.get("placed_start") for s in overridden])

    def test_does_not_compress_into_a_vanished_slot(self):
        """Past the next segment's start there is no room to budget against.

        Dividing by that would demand an absurd speed; the segment is simply
        allowed to run long and the ceiling handles the rest.
        """
        start, speed = plan_segment_fit(
            seg_start=5.0, next_start=6.0, tts_dur=4.0, current_end=8.0)
        assert start == pytest.approx(8.0 + MIN_SEGMENT_GAP)
        assert speed == 1.0

    def test_drift_does_not_accumulate_across_a_natural_pause(self):
        """A real gap in the source lets the dub resynchronise for free."""
        _, speed = plan_segment_fit(
            seg_start=30.0, next_start=36.0, tts_dur=4.0, current_end=25.0)
        assert speed == 1.0


class TestBuildAudioMixFilter:
    """build_audio_mix_filter() renders the dub+bed filter_complex graph."""

    def test_legacy_flat_mix_is_byte_identical(self):
        # ducking=False must reproduce the pre-ducking mix exactly, so the
        # legacy toggle really is "the old behaviour" and not a near-miss.
        assert build_audio_mix_filter(0.15, ducking=False) == (
            "[1:a]volume=1.0[dub];[2:a]volume=0.15[bg];"
            "[dub][bg]amix=inputs=2:duration=first:normalize=0[out]")

    def test_ducking_graph_shape(self):
        f = build_audio_mix_filter(0.5, ducking=True)
        assert "asplit=2" in f
        assert ("sidechaincompress=threshold=0.03:ratio=8:"
                "attack=20:release=400") in f
        assert f.endswith("[out]")
        # The bed ceiling applies BEFORE the compressor, on the bed input.
        assert "[2:a]volume=0.5[bgin]" in f
        assert f.index("volume=0.5") < f.index("sidechaincompress")

    def test_dub_is_first_amix_input(self):
        # duration=first keys the mix length off the dub in both modes.
        flat = build_audio_mix_filter(0.3, ducking=False)
        assert "[dub][bg]amix" in flat
        ducked = build_audio_mix_filter(0.3, ducking=True)
        assert "[dubm][bgduck]amix" in ducked

    def test_bed_ceiling_follows_bg_volume(self):
        assert "volume=0.0" in build_audio_mix_filter(0.0, ducking=True)


class TestMergeUsesMixFilter:
    """merge_audio_video() hands the mix graph to ffmpeg unchanged."""

    def _merge(self, monkeypatch, tmp_path, bg_path, **kwargs):
        calls = []
        monkeypatch.setattr("pipeline.assembler._run",
                            lambda cmd, desc="", timeout=None: calls.append(cmd))
        monkeypatch.setattr("pipeline.assembler._probe_video_stream",
                            lambda path: ("h264", "yuv420p"))
        merge_audio_video("fake_video.mp4", "fake_dub.wav",
                          str(tmp_path / "out.mp4"),
                          background_audio_path=bg_path, **kwargs)
        assert len(calls) == 1
        return calls[0]

    def _filter_complex(self, cmd):
        return cmd[cmd.index("-filter_complex") + 1]

    def test_ducking_filter_reaches_ffmpeg(self, monkeypatch, tmp_path,
                                           temp_audio_file):
        cmd = self._merge(monkeypatch, tmp_path, temp_audio_file,
                          bg_volume=0.5, bg_ducking=True)
        assert self._filter_complex(cmd) == build_audio_mix_filter(0.5, True)

    def test_flat_filter_reaches_ffmpeg(self, monkeypatch, tmp_path,
                                        temp_audio_file):
        cmd = self._merge(monkeypatch, tmp_path, temp_audio_file,
                          bg_volume=0.2, bg_ducking=False)
        assert self._filter_complex(cmd) == build_audio_mix_filter(0.2, False)

    def test_no_bg_mix_without_background_path(self, monkeypatch, tmp_path):
        cmd = self._merge(monkeypatch, tmp_path, "",
                          bg_volume=0.5, bg_ducking=True)
        assert "-filter_complex" not in cmd
        assert "amix" not in " ".join(cmd)

    def test_extend_branch_keeps_video_prefix_and_maps(self, monkeypatch,
                                                       tmp_path,
                                                       temp_audio_file):
        # Audio longer than video → the tpad branch: the mix graph is
        # appended after the [0:v]…[v] video filter, and both labels map out.
        import subprocess

        durations = {"fake_video.mp4": "2.0\n", "fake_dub.wav": "10.0\n"}

        def fake_ffprobe(cmd, **kw):
            return SimpleNamespace(stdout=durations[cmd[-1]])

        monkeypatch.setattr(subprocess, "run", fake_ffprobe)
        cmd = self._merge(monkeypatch, tmp_path, temp_audio_file,
                          bg_volume=0.5, bg_ducking=True)
        fc = self._filter_complex(cmd)
        assert fc.startswith("[0:v]tpad=stop_mode=clone:stop_duration=")
        assert fc.endswith("[v];" + build_audio_mix_filter(0.5, True))
        maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
        assert maps == ["[v]", "[out]"]
