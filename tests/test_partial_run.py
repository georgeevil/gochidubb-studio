"""A submitted job can be told to stop after a stage (CLD partial runs).

The driver has always been able to stop early — `run_pipeline_stages` takes
`stop_after` and its tail lands the job on 'paused'. Until now only
/retry_stage could ask for it, so the only way to end a *new* job before the
dub was a review gate, which is a different thing: a gate waits for a human
to approve, and approving runs everything after it. These tests cover the
submit side, where "stop here and keep what you have" is the whole request.

Hermetic the same way tests/test_creator_routes.py is (read its header):
TestClient without the context manager, so no lifespan, no real DB. The two
things start_dub reaches for on its own — the Ollama catalogue and the queue
— are patched out, and OUTPUT_DIR is redirected at tmp_path so the job's work
directory is not created under the real outputs/.
"""
import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    saved = dict(server.jobs)
    server.jobs.clear()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)

    async def _no_ollama():
        return False, []

    enqueued = []

    async def _enqueue(job_id, args):
        enqueued.append((job_id, args))

    monkeypatch.setattr(server, "check_ollama", _no_ollama)
    monkeypatch.setattr(server, "enqueue_job", _enqueue)
    c = TestClient(server.app)
    c.enqueued = enqueued
    try:
        yield c
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


# ═══════════════════════════════════════════════════════════════════════
#  _earlier_stage — two ceilings, the lower one wins
# ═══════════════════════════════════════════════════════════════════════

def test_earlier_stage_picks_the_earlier_of_two():
    assert server._earlier_stage("diarize", "merge") == "diarize"
    assert server._earlier_stage("merge", "diarize") == "diarize"
    assert server._earlier_stage("tts", "tts") == "tts"


def test_earlier_stage_treats_empty_as_no_ceiling():
    # "" means "run to the end", so it must never win against a real stage —
    # otherwise a reupload job asked to stop after download would run the
    # whole dub.
    assert server._earlier_stage("", "download") == "download"
    assert server._earlier_stage("download", "") == "download"
    assert server._earlier_stage("", "") == ""


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/dub — stop_after
# ═══════════════════════════════════════════════════════════════════════

def _submit(client, **extra):
    form = {"source": "/tmp/not-a-url.mp4", "target_lang": "es"}
    form.update(extra)
    return client.post("/api/dub", data=form)


def test_unknown_stop_after_is_refused_by_name(client):
    r = _submit(client, stop_after="diarise")  # British spelling, not a stage
    assert r.status_code == 400
    # The error names the stage list rather than just saying "invalid", so a
    # caller's typo is one read away from the fix.
    assert "diarise" in r.json()["error"]
    assert "diarize" in r.json()["error"]
    assert not client.enqueued


def test_stop_after_rides_onto_the_job_and_the_queue(client):
    r = _submit(client, stop_after="diarize")
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert server.jobs[job_id]["stop_after"] == "diarize"
    # The queue carries it too: the job dict is for display, the enqueued
    # args are what the driver actually runs with.
    assert client.enqueued[0][0] == job_id
    assert client.enqueued[0][1]["stop_after"] == "diarize"


def test_a_full_run_is_still_the_default(client):
    r = _submit(client)
    job_id = r.json()["job_id"]
    assert server.jobs[job_id]["stop_after"] == ""
    assert client.enqueued[0][1]["stop_after"] == ""


def test_every_pipeline_stage_is_accepted(client):
    for stage in server.STAGE_ORDER:
        r = _submit(client, stop_after=stage)
        assert r.status_code == 200, (stage, r.text)
        assert server.jobs[r.json()["job_id"]]["stop_after"] == stage
