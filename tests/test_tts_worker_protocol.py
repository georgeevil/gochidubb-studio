"""One pipe, many jobs — telling whose output you are reading.

The daemon TTS worker loads VoxCPM once and then serves job after job over a
single stdout pipe, with `{"event": "job_done"}` marking a boundary. That
works until a parent stops reading mid-job: its tail stays in the pipe, and
the next call reads those events as its own.

It happened. Job 6d7e4eda:

    [1/8] tier=2 qa=0.07 ...
    TTS worker done: 8/8 tiers={'2': 8} qa_regens=0
    ERROR Pipeline failed at stage 'tts':
          All 8 TTS segments failed - check model/GPU

Every segment rendered, tier 2, QA between 0.00 and 0.07 — and the stage
raised "all failed" and sent whoever read it to go and look at the GPU. The
next run gave it away: its progress counter ran 9/8, 10/8 ... 16/8, because
it was consuming the previous job's eight events before its own.

Every event carries the token of the job it belongs to now, so a foreign tail
is recognisable instead of silently becoming this run's result.
"""
import json

import pytest

from pipeline.synthesizer import VoxCPMSynthesizer


class _FakePipe:
    """Stands in for the worker's stdout: hands back pre-canned lines."""

    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


class _FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass


class _FakeProc:
    def __init__(self, lines):
        self.stdout = _FakePipe(lines)
        self.stdin = _FakeStdin()

    def poll(self):
        return None            # alive, so the caller reuses it as a daemon


def _events(token, n, ok=True):
    """A well-formed job: a segment event per index, then done + job_done."""
    out = [{"event": "segment", "idx": i, "ok": ok, "tier": 2,
            "qa_score": 0.02, "token": token} for i in range(n)]
    out.append({"event": "done", "ok": n if ok else 0, "total": n,
                "tier_stats": {"2": n}, "token": token})
    out.append({"event": "job_done", "token": token})
    return out


def _run(engine, segments, lines, tmp_path):
    engine._worker_proc = _FakeProc([json.dumps(e) + "\n" for e in lines])
    # The engine writes job.json and reads the token back out of it, so let
    # the real code path run and only fake the pipe.
    return engine.synthesize_segments(
        segments, str(tmp_path), speaker_refs={}, speaker_transcripts={},
        voice_seed=1, tts_speed="fast", target_lang="ru")


@pytest.fixture
def segments():
    return [{"idx": i, "speaker": "S", "translated_text": f"line {i}"}
            for i in range(3)]


def _token_written(engine):
    """The token the engine put in the job file it just wrote."""
    path = engine._worker_proc.stdin.written[-1].strip()
    with open(path, encoding="utf-8") as f:
        return json.load(f)["job_token"]


class TestAForeignTailIsNotMistakenForThisRun:
    def test_a_clean_run_attaches_every_segment(self, segments, tmp_path):
        """Baseline: with no interference the results land as before."""
        eng = VoxCPMSynthesizer()
        out = _run_with_token_echo(eng, segments, tmp_path, n=3)
        assert all(s["audio_path"] for s in out)
        assert [s["qa_score"] for s in out] == [0.02] * 3

    def test_a_previous_jobs_tail_is_discarded(self, segments, tmp_path):
        """The regression. A stale job_done used to end the read loop before
        this run's own segments arrived, and every segment came back with no
        audio_path — which the stage reports as 'all segments failed'."""
        eng = VoxCPMSynthesizer()
        # The incident's exact shape: what was left in the pipe was the tail
        # of a previous job — its closing job_done. The reader took that as
        # the end of *this* job and stopped before a single one of its own
        # segments arrived.
        out = _run_with_token_echo(
            eng, segments, tmp_path, n=3,
            stale=[{"event": "job_done", "token": "OLDJOB"}])
        assert all(s["audio_path"] for s in out), (
            "a leftover job_done from an earlier job ended this run's read "
            "loop before its own segments arrived — every segment came back "
            "with no audio, which the stage reports as 'all segments failed'")

    def test_a_stale_tail_is_reported_not_absorbed(self, segments, tmp_path, caplog):
        """Recovering is right; recovering quietly is not. Something stopped
        reading a previous job mid-flight and that is worth finding."""
        eng = VoxCPMSynthesizer()
        with caplog.at_level("WARNING"):
            _run_with_token_echo(
                eng, segments, tmp_path, n=3,
                stale=[{"event": "done", "ok": 8, "total": 8, "token": "OLD"},
                       {"event": "job_done", "token": "OLD"}])
        assert any("left in the worker pipe" in r.message for r in caplog.records)

    def test_an_untokenized_event_is_still_honoured(self, segments, tmp_path):
        """A worker mid-upgrade, or an error emitted before a job's token was
        set, must not be filtered out — the check only rejects a token that
        is present and different."""
        eng = VoxCPMSynthesizer()
        out = _run_with_token_echo(eng, segments, tmp_path, n=3, strip_token=True)
        assert all(s["audio_path"] for s in out)


class TestFailureIsNamedHonestly:
    def test_a_desync_says_so_instead_of_blaming_the_gpu(self, segments, tmp_path):
        """If it ever happens again, the message should point at the pipe.
        'check model/GPU' cost real time on a run where the GPU was fine."""
        eng = VoxCPMSynthesizer()
        # done says 8 rendered; not one segment event carries this token.
        stale = [{"event": "done", "ok": 8, "total": 8, "tier_stats": {"2": 8}},
                 {"event": "job_done"}]
        _run(eng, segments, stale, tmp_path)
        detail = eng.last_failure_detail()
        assert "desynchronized" in detail and "8" in detail

    def test_genuine_segment_failures_still_report_their_own_error(
            self, segments, tmp_path):
        eng = VoxCPMSynthesizer()
        lines = [{"event": "segment", "idx": 0, "ok": False,
                  "error": "CUDA out of memory"},
                 {"event": "done", "ok": 0, "total": 3},
                 {"event": "job_done"}]
        _run(eng, segments, lines, tmp_path)
        assert "CUDA out of memory" in eng.last_failure_detail()


def _run_with_token_echo(engine, segments, tmp_path, *, n,
                         stale=None, strip_token=False):
    """Drive one synthesize_segments where the fake worker echoes the token
    the engine actually generated — which is the whole point of the scheme,
    and cannot be pre-canned because the engine picks it at call time."""
    captured = {}

    class _TokenLearningProc(_FakeProc):
        def __init__(self):
            super().__init__([])

        def poll(self):
            return None

    proc = _TokenLearningProc()

    class _Stdin(_FakeStdin):
        def write(self, s):
            super().write(s)
            with open(s.strip(), encoding="utf-8") as f:
                # .get, so this helper still runs against a build that does
                # not write a token — which is how it demonstrates the bug.
                captured["token"] = json.load(f).get("job_token", "")
            events = list(stale or [])
            mine = _events(captured["token"], n)
            if strip_token:
                mine = [{k: v for k, v in e.items() if k != "token"}
                        for e in mine]
            proc.stdout._lines = [json.dumps(e) + "\n" for e in events + mine]

    proc.stdin = _Stdin()
    engine._worker_proc = proc
    return engine.synthesize_segments(
        segments, str(tmp_path), speaker_refs={}, speaker_transcripts={},
        voice_seed=1, tts_speed="fast", target_lang="ru")
