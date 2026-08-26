"""The background bed's level, and being able to change it after the fact.

Two complaints, one cause. "The background is quieter than in the original"
is a mix decision, and the only place it could be set was a global preference
in Settings → Output — which does nothing to a video that has already been
rendered. The level belongs to one dub, and the stage that applies it is a
mux that takes seconds, so it should be a per-job override on a re-run of
that stage rather than a preference for next time.
"""
import pytest

import server
from pipeline.assembler import build_audio_mix_filter


class TestTheLevelReachesFfmpeg:
    def test_the_bed_level_is_what_was_asked_for(self):
        assert "volume=0.8" in build_audio_mix_filter(0.8, ducking=True)
        assert "volume=0.5" in build_audio_mix_filter(0.5, ducking=True)

    def test_the_level_applies_before_the_ducker(self):
        """bg_volume is a ceiling — the level when nobody speaks — so it has
        to be set before sidechaincompress pulls the bed down under speech.
        After it, the same number would mean something else entirely."""
        f = build_audio_mix_filter(0.8, ducking=True)
        assert f.index("volume=0.8") < f.index("sidechaincompress")

    def test_ducking_off_is_a_flat_mix_at_that_level(self):
        f = build_audio_mix_filter(0.8, ducking=False)
        assert "volume=0.8" in f
        assert "sidechaincompress" not in f

    def test_the_dub_stays_the_first_amix_input(self):
        """amix uses duration=first, so the dub has to be input one or the
        mix runs as long as the music bed instead of the speech."""
        for duck in (True, False):
            f = build_audio_mix_filter(0.5, ducking=duck)
            assert "amix=inputs=2:duration=first" in f


class TestItCanBeChangedOnAFinishedDub:
    def test_the_level_is_overridable_on_a_retry(self):
        assert "bg_volume" in server._RETRY_OVERRIDE_KEYS
        assert "bg_ducking" in server._RETRY_OVERRIDE_KEYS

    def test_the_level_is_coerced_to_a_number(self):
        """Overrides arrive as JSON from a form field. A level that stayed a
        string would reach ffmpeg as volume=0.8 by luck and break the moment
        anything did arithmetic on it."""
        assert "bg_volume" in server._RETRY_NUMERIC_KEYS

    def test_the_merge_stage_offers_both_controls(self):
        keys = {o["key"] for o in server.STAGE_RETRY_OPTIONS["merge"]}
        assert {"bg_volume", "bg_ducking"} <= keys

    def test_the_level_control_is_bounded_like_the_setting(self):
        from app.config import FIELD_SPECS
        opt = next(o for o in server.STAGE_RETRY_OPTIONS["merge"]
                   if o["key"] == "bg_volume")
        _, lo, hi = FIELD_SPECS["background_volume"]
        assert (opt["min"], opt["max"]) == (lo, hi), (
            "the retry control and the global setting must agree on range, or "
            "one of them accepts a value the other refuses")

    def test_merge_is_the_last_stage_so_redoing_it_redoes_nothing_else(self):
        """This is what makes the control cheap enough to be a slider: a
        re-mix re-runs the mux and stops. If merge stopped being last, an
        'Apply' would quietly re-run whatever came after it."""
        assert server.STAGE_ORDER[-1] == "merge"

    def test_the_dub_track_is_not_rebuilt_by_a_remix(self):
        """merge reads dubbed_wav from the assemble checkpoint, so the voices
        — the expensive part — survive a level change untouched."""
        idx = server.STAGE_ORDER.index("merge")
        assert server.PIPELINE_STAGES[idx - 1]["id"] == "assemble"
        assert "dubbed_wav" in server.PIPELINE_STAGES[idx - 1]["artifacts"]


class TestWhatAFinishedJobOffers:
    @pytest.fixture
    def job(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
        monkeypatch.setitem(server.jobs, "j1",
                            {"id": "j1", "status": "complete",
                             "target_lang": "ru", "keep_bg": True})
        work = tmp_path / "j1"
        work.mkdir()
        for name in ("dubbed_video.mp4", "subtitles.srt", "background.wav"):
            (work / name).write_bytes(b"x" * 2048)
        return work

    def test_only_files_that_exist_are_offered(self, job):
        import asyncio
        d = asyncio.run(server.list_job_files("j1"))
        keys = {f["key"] for f in d["files"]}
        assert {"video", "subtitles", "background"} == keys
        assert "dub_audio" not in keys, (
            "offered a download for a file that is not there")

    def test_an_empty_file_is_not_offered(self, job):
        """A zero-byte artifact is a failed write, not a download."""
        import asyncio
        (job / "vocals.wav").write_bytes(b"")
        d = asyncio.run(server.list_job_files("j1"))
        assert "vocals" not in {f["key"] for f in d["files"]}

    def test_every_entry_carries_what_a_link_needs(self, job):
        import asyncio
        d = asyncio.run(server.list_job_files("j1"))
        for f in d["files"]:
            assert f["url"].startswith("/outputs/j1/")
            assert f["label"] and f["filename"]
            # A kilobyte-sized subtitle file must not advertise itself as
            # "0.0 MB", which reads as an empty download.
            assert f["size_label"] and not f["size_label"].startswith("0.0")


class TestBedBalanceMeasurement:
    """The programmatic answer to "is the music as loud as the original?" —
    windows come from placements, so speech never contaminates the reading."""

    def _placements(self):
        return [
            {"idx": 0, "dub_start": 2.0, "dub_end": 4.0},
            {"idx": 1, "dub_start": 20.0, "dub_end": 22.0},
        ]

    def test_longest_speech_free_span_wins(self):
        from pipeline.quality import bed_measure_window
        w = bed_measure_window(self._placements(), total_duration=60.0)
        # 22..60 (38s) beats 4..20 (16s) and the 2s lead-in; edges trimmed.
        assert w is not None
        start, dur = w
        assert start == pytest.approx(22.5)
        assert dur == pytest.approx(37.0)

    def test_short_gaps_measure_nothing(self):
        from pipeline.quality import bed_measure_window
        tight = [{"idx": i, "dub_start": i * 3.0, "dub_end": i * 3.0 + 2.0}
                 for i in range(10)]
        assert bed_measure_window(tight, total_duration=30.0) is None

    def test_missing_file_degrades_to_none(self):
        from pipeline.quality import measure_window_lufs
        assert measure_window_lufs("/nope/nothing.wav", 0.0, 3.0) is None


class TestRetryOverrideCoercion:
    """bg_volume used to skip validation entirely — coerce_field found no
    'bg_volume' spec and passed the raw JSON through to the filter string."""

    def test_out_of_range_bed_level_is_rejected(self, monkeypatch):
        from fastapi.testclient import TestClient
        import time as _t
        server.jobs["bgj"] = {"id": "bgj", "status": "complete",
                              "created": _t.time()}
        monkeypatch.setattr(server, "_load_checkpoint",
                            lambda *a: {"stage": "tts_done", "segments": []})
        try:
            c = TestClient(server.app)
            r = c.post("/api/dub/bgj/retry_stage/merge",
                       data={"overrides": '{"bg_volume": 99}'})
            assert r.status_code == 400
            assert "background_volume" in r.json()["error"]
        finally:
            server.jobs.pop("bgj", None)
