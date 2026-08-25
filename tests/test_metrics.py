"""Tests for pipeline/metrics.py — stage timing and resource sampling."""
import time

import pytest

from pipeline.metrics import (
    METRICS_FILENAME,
    ResourceSampler,
    format_resources,
    gpu_backend,
    load_metrics,
    record_stage,
    stage_timer,
)


class TestResourceSampler:
    def test_collects_at_least_one_sample_for_a_short_stage(self):
        s = ResourceSampler(interval=5.0).start()
        summary = s.stop()
        assert summary["samples"] >= 1, "a fast stage must still yield a data point"
        assert summary["cpu_cores"] >= 1

    def test_reports_avg_and_peak_together(self):
        s = ResourceSampler(interval=0.25).start()
        time.sleep(0.6)
        summary = s.stop()
        if "cpu_proc_pct_avg" in summary:
            assert summary["cpu_proc_pct_peak"] >= summary["cpu_proc_pct_avg"]

    def test_cpu_is_normalized_to_a_single_percentage(self):
        """psutil sums per-core (800% on 8 cores); we divide so 100% means
        'every core saturated' and the number stays readable."""
        s = ResourceSampler(interval=0.25).start()
        time.sleep(0.4)
        summary = s.stop()
        if "cpu_proc_pct_peak" in summary:
            assert summary["cpu_proc_pct_peak"] <= 100.5

    def test_a_reading_taken_too_soon_is_skipped_rather_than_recorded(self):
        """The CI failure this guards, made deterministic.

        cpu_percent(None) divides the CPU-time delta by the wall-clock delta
        since the previous call. start() primes the counters and the sampler
        thread reads them immediately after, so on a fast runner that divisor
        is microseconds and the quotient is enormous — CI produced a 988%
        "peak" on a normalized scale whose ceiling is 100, and it was being
        averaged into metrics.json.

        Driven by the clock rather than by racing a thread: the bug only
        appears when thread-start latency happens to be short, which is
        exactly why it passed on a laptop and failed on a runner.
        """
        s = ResourceSampler(interval=5.0)

        s._last_cpu_at = time.monotonic()          # as if just read
        s._sample_once()
        assert s._cpu_proc == [], (
            "recorded a CPU rate measured over a window of ~0 seconds")
        assert len(s._rss_mb) == 1, (
            "skipping the CPU rate cost the memory reading taken with it — "
            "RSS is a reading, not a rate, and is valid at any interval")

        s._last_cpu_at = time.monotonic() - 1.0    # a real window
        s._sample_once()
        assert len(s._cpu_proc) == 1
        assert len(s._rss_mb) == 2

    def test_a_stage_with_no_cpu_reading_still_reports_being_sampled(self):
        """`samples` counts sampling passes. Deriving it from the CPU list
        would make a stage too short to time a CPU rate look unsampled."""
        s = ResourceSampler(interval=5.0)
        s._last_cpu_at = time.monotonic()
        s._sample_once()
        summary = s.summary()
        assert summary["samples"] == 1
        assert "cpu_proc_pct_avg" not in summary
        assert summary["rss_mb_peak"] > 0

    def test_a_started_sampler_never_reports_an_impossible_percentage(self):
        summary = ResourceSampler(interval=5.0).start().stop()
        assert summary["samples"] >= 1
        if "cpu_proc_pct_peak" in summary:
            assert summary["cpu_proc_pct_peak"] <= 100.5, (
                f"reported {summary['cpu_proc_pct_peak']}% on a normalized "
                f"scale that tops out at 100")

    def test_stop_is_idempotent(self):
        s = ResourceSampler(interval=0.25).start()
        first = s.stop()
        assert s.stop()["samples"] == first["samples"]


class TestFormatResources:
    def test_handles_empty_samples(self):
        assert format_resources({}) == "no resource samples"

    def test_renders_cpu_and_gpu(self):
        out = format_resources({
            "cpu_proc_pct_avg": 40.0, "cpu_proc_pct_peak": 92.0,
            "gpu_util_pct_avg": 70.0, "gpu_util_pct_peak": 99.0,
            "gpu_mem_mb_peak": 6144.0, "gpu_mem_total_mb": 12288.0,
            "rss_mb_peak": 2048.0,
        })
        assert "cpu 40%avg/92%peak" in out
        assert "gpu 70%avg/99%peak" in out
        assert "6144MB/12288MB" in out


class TestRecordStage:
    def test_writes_and_reloads(self, tmp_path):
        record_stage(tmp_path, "job1", "transcribe", "ok", 12.5, 1000.0,
                     resources={"cpu_proc_pct_peak": 88.0}, detail={"segments": 3})
        data = load_metrics(tmp_path)
        entry = data["stages"]["transcribe"]
        assert entry["duration_sec"] == 12.5
        assert entry["finished_at"] == 1012.5
        assert entry["attempt"] == 1
        assert entry["detail"] == {"segments": 3}
        assert (tmp_path / METRICS_FILENAME).exists()

    def test_attempt_counter_increments_per_stage(self, tmp_path):
        for _ in range(3):
            record_stage(tmp_path, "job1", "translate", "ok", 1.0, 0.0)
        record_stage(tmp_path, "job1", "tts", "ok", 1.0, 0.0)
        stages = load_metrics(tmp_path)["stages"]
        assert stages["translate"]["attempt"] == 3
        assert stages["tts"]["attempt"] == 1

    def test_history_keeps_every_attempt(self, tmp_path):
        record_stage(tmp_path, "job1", "translate", "error", 1.0, 0.0, error="boom")
        record_stage(tmp_path, "job1", "translate", "ok", 2.0, 0.0)
        data = load_metrics(tmp_path)
        assert len(data["history"]) == 2
        assert data["history"][0]["error"] == "boom"
        # `stages` always holds the latest attempt
        assert data["stages"]["translate"]["status"] == "ok"

    def test_history_is_capped(self, tmp_path):
        for i in range(210):
            record_stage(tmp_path, "job1", "tts", "ok", 0.1, float(i))
        assert len(load_metrics(tmp_path)["history"]) == 200

    def test_error_text_is_truncated(self, tmp_path):
        record_stage(tmp_path, "job1", "tts", "error", 0.1, 0.0, error="x" * 5000)
        assert len(load_metrics(tmp_path)["stages"]["tts"]["error"]) == 2000

    def test_corrupt_metrics_file_does_not_raise(self, tmp_path):
        (tmp_path / METRICS_FILENAME).write_text("{not json")
        assert load_metrics(tmp_path) == {"stages": {}, "history": []}
        record_stage(tmp_path, "job1", "tts", "ok", 1.0, 0.0)
        assert "tts" in load_metrics(tmp_path)["stages"]

    def test_missing_dir_is_created(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist"
        record_stage(target, "job1", "tts", "ok", 1.0, 0.0)
        assert (target / METRICS_FILENAME).exists()


class TestStageTimer:
    def test_records_success_with_duration(self, tmp_path):
        with stage_timer(tmp_path, "job1", "extract", interval=5.0) as perf:
            perf["denoise"] = True
        entry = load_metrics(tmp_path)["stages"]["extract"]
        assert entry["status"] == "ok"
        assert entry["duration_sec"] >= 0
        assert entry["detail"] == {"denoise": True}

    def test_records_failure_and_reraises(self, tmp_path):
        """A stage that dies still reports how long it ran before dying."""
        with pytest.raises(ValueError):
            with stage_timer(tmp_path, "job1", "diarize", interval=5.0):
                raise ValueError("pyannote exploded")
        entry = load_metrics(tmp_path)["stages"]["diarize"]
        assert entry["status"] == "error"
        assert "pyannote exploded" in entry["error"]

    def test_cancellation_is_not_recorded_as_an_error(self, tmp_path):
        class JobCancelled(Exception):
            pass

        with pytest.raises(JobCancelled):
            with stage_timer(tmp_path, "job1", "tts", interval=5.0):
                raise JobCancelled("user cancelled")
        assert load_metrics(tmp_path)["stages"]["tts"]["status"] == "cancelled"

    def test_detail_survives_an_exception(self, tmp_path):
        """Whatever the stage managed to measure before failing is kept."""
        with pytest.raises(RuntimeError):
            with stage_timer(tmp_path, "job1", "translate", interval=5.0) as perf:
                perf["model"] = "qwen2.5:14b"
                raise RuntimeError("ollama down")
        assert load_metrics(tmp_path)["stages"]["translate"]["detail"]["model"] == "qwen2.5:14b"

    def test_repeated_runs_accumulate_attempts(self, tmp_path):
        for _ in range(2):
            with stage_timer(tmp_path, "job1", "merge", interval=5.0):
                pass
        assert load_metrics(tmp_path)["stages"]["merge"]["attempt"] == 2


class TestGpuProbe:
    def test_backend_is_a_known_value_or_none(self):
        assert gpu_backend() in (None, "pynvml", "torch.cuda", "torch.mps", "nvidia-smi")

    def test_probing_is_cached(self):
        assert gpu_backend() == gpu_backend()
