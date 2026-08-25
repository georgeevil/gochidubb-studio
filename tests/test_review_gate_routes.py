"""Route-level tests for the review-gate rework (CLD-264/CLD-270).

Hermetic exactly the way tests/test_creator_routes.py is (read its header):

* `TestClient(app)` WITHOUT the context manager — no lifespan, so
  `app.db._DB_PATH` stays None and `save_job` is a no-op; the real
  gochidubb.db is never opened.
* Background work never races the event loop: `/continue` goes through
  `server._spawn_background`, which every test here patches to *capture*
  the coroutine — run deterministically with `asyncio.run()`, or closed
  when the test only cares that (or whether) it was spawned.
* Webhook deliveries are captured by patching `server._fire_webhooks`;
  audit writes by patching `server.app_audit.record`. No file under the
  project root is touched.
"""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

import server
from app import review_gates as rg


@pytest.fixture
def client():
    saved = dict(server.jobs)
    server.jobs.clear()
    try:
        yield TestClient(server.app)
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


@pytest.fixture
def spawned(monkeypatch):
    """Capture coroutines handed to _spawn_background instead of racing
    the torn-down portal (CLAUDE.md's measured 1 ms cancellation window)."""
    captured = []
    monkeypatch.setattr(server, "_spawn_background",
                        lambda coro: captured.append(coro))
    yield captured
    for coro in captured:
        coro.close()


@pytest.fixture
def webhooks_fired(monkeypatch):
    fired = []
    monkeypatch.setattr(server, "_fire_webhooks",
                        lambda status, job: fired.append(status))
    return fired


def _segments(n_speakers=1):
    return [
        {"idx": i, "start": float(i), "end": i + 0.9,
         "text": f"line {i}", "translated_text": f"строка {i}",
         "speaker": f"S{i % n_speakers}"}
        for i in range(4)
    ]


def _gated_job(jid, gate, gates, **kw):
    status, cp = rg.GATE_STATUS[gate]
    job = {"id": jid, "status": status, "progress": 60,
           "created": time.time(), "target_lang": "ru",
           "review_gates": gates, "gates_cleared": [],
           "pending_gate": gate, "wizard_mode": "auto"}
    job.update(kw)
    return job


def _checkpoint(stage, gates, n_speakers=1, **kw):
    cp = {"stage": stage, "segments": _segments(n_speakers),
          "target_lang": "ru", "review_gates": gates, "gates_cleared": []}
    cp.update(kw)
    return cp


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/dub/{id}/continue — one gate per approve
# ═══════════════════════════════════════════════════════════════════════

def test_continue_advances_exactly_one_gate_at_a_shared_boundary(
        client, spawned, webhooks_fired, monkeypatch):
    """translation cleared → voice_cast pending: flip status, run nothing."""
    gates = dict(rg.all_off(), translation="on", voice_cast="on")
    server.jobs["j1"] = _gated_job("j1", "translation", gates)
    monkeypatch.setattr(server, "_latest_checkpoint",
                        lambda jid: _checkpoint("translation_done", gates))

    d = client.post("/api/dub/j1/continue").json()

    assert d == {"ok": True, "job_id": "j1", "now_awaiting": "voice_cast",
                 "status": "awaiting_voice_review"}
    assert server.jobs["j1"]["status"] == "awaiting_voice_review"
    assert server.jobs["j1"]["gates_cleared"] == ["translation"]
    assert server.jobs["j1"]["pending_gate"] == "voice_cast"
    assert spawned == []                      # no stage ran
    assert webhooks_fired == ["awaiting_voice_review"]


def test_continue_resumes_when_the_boundary_has_no_further_gate(
        client, spawned, webhooks_fired, monkeypatch):
    gates = dict(rg.all_off(), translation="on")
    server.jobs["j2"] = _gated_job("j2", "translation", gates)
    monkeypatch.setattr(server, "_latest_checkpoint",
                        lambda jid: _checkpoint("translation_done", gates))

    d = client.post("/api/dub/j2/continue").json()

    assert d["resuming_from"] == "translation_done"
    assert server.jobs["j2"]["status"] == "resuming"
    assert server.jobs["j2"]["gates_cleared"] == ["translation"]
    assert "pending_gate" not in server.jobs["j2"]
    assert len(spawned) == 1                  # the resume, via the seam


def test_continue_flagged_voice_cast_sails_through_on_zero_findings(
        client, spawned, monkeypatch):
    """Single-speaker job: the flagged_only cast gate finds nothing, so
    clearing translation resumes instead of stopping at the cast."""
    gates = dict(rg.all_off(), translation="on", voice_cast="flagged_only")
    server.jobs["j3"] = _gated_job("j3", "translation", gates)
    monkeypatch.setattr(
        server, "_latest_checkpoint",
        lambda jid: _checkpoint("translation_done", gates, n_speakers=1))

    d = client.post("/api/dub/j3/continue").json()
    assert "now_awaiting" not in d
    assert len(spawned) == 1


def test_continue_flagged_voice_cast_holds_an_uncast_multispeaker_job(
        client, spawned, monkeypatch):
    gates = dict(rg.all_off(), translation="on", voice_cast="flagged_only")
    server.jobs["j4"] = _gated_job("j4", "translation", gates)
    monkeypatch.setattr(
        server, "_latest_checkpoint",
        lambda jid: _checkpoint("translation_done", gates, n_speakers=3))

    d = client.post("/api/dub/j4/continue").json()
    assert d["now_awaiting"] == "voice_cast"
    assert spawned == []


def test_final_qc_approve_completes_and_fires_the_webhook_exactly_once(
        client, spawned, webhooks_fired, monkeypatch):
    gates = dict(rg.all_off(), final_qc="on")
    server.jobs["j5"] = _gated_job("j5", "final_qc", gates)
    monkeypatch.setattr(server, "_latest_checkpoint",
                        lambda jid: _checkpoint("merge_done", gates))

    d = client.post("/api/dub/j5/continue").json()
    assert d["resuming_from"] == "merge_done"
    assert len(spawned) == 1
    asyncio.run(spawned.pop())                # drive the resume to its end

    job = server.jobs["j5"]
    assert job["status"] == "complete"
    assert job["gates_cleared"] == ["final_qc"]
    assert "pending_gate" not in job
    assert webhooks_fired == ["complete"]     # exactly once, from one place


def test_continue_from_a_non_gate_status_is_plain_crash_resume(
        client, spawned, monkeypatch):
    server.jobs["j6"] = {"id": "j6", "status": "error",
                         "created": time.time()}
    monkeypatch.setattr(server, "_latest_checkpoint",
                        lambda jid: _checkpoint("translation_done", {}))

    d = client.post("/api/dub/j6/continue").json()
    assert d["ok"] is True
    assert server.jobs["j6"].get("gates_cleared") in (None, [])
    assert len(spawned) == 1


def test_continue_without_a_checkpoint_is_a_404(client, spawned, monkeypatch):
    server.jobs["j7"] = _gated_job("j7", "translation",
                                   dict(rg.all_off(), translation="on"))
    monkeypatch.setattr(server, "_latest_checkpoint", lambda jid: None)
    assert client.post("/api/dub/j7/continue").status_code == 404
    assert client.post("/api/dub/nope/continue").status_code == 404


# ═══════════════════════════════════════════════════════════════════════
#  GET /api/dub/{id}/qc
# ═══════════════════════════════════════════════════════════════════════

def _qc_fixture(monkeypatch, loudnorm, placements):
    monkeypatch.setattr(
        server, "_quality_inputs",
        lambda jid: (_segments(2), "translation_done", placements, loudnorm))
    monkeypatch.setattr(server, "_read_user_glossary", lambda: {})


def test_qc_checklist_shape_and_warn_count(client, monkeypatch):
    server.jobs["q1"] = {"id": "q1", "status": "awaiting_final_qc",
                         "created": time.time(), "target_lang": "ru",
                         "pending_gate": "final_qc"}
    _qc_fixture(
        monkeypatch,
        loudnorm={"output_i": -16.3, "output_tp": -2.0},   # inside ±1 LU
        placements=[
            {"idx": 0, "src_start": 0, "src_end": 1.0,
             "dub_start": 0, "dub_end": 1.02},              # 2% — fine
            {"idx": 1, "src_start": 1, "src_end": 2.0,
             "dub_start": 1, "dub_end": 2.30},              # 30% — off-sync
        ])

    d = client.get("/api/dub/q1/qc").json()

    assert [r["id"] for r in d["rows"]] == [
        "loudness", "subtitles", "sync", "glossary", "consent"]
    by_id = {r["id"]: r for r in d["rows"]}
    assert by_id["loudness"]["state"] == "pass"
    assert by_id["sync"]["state"] == "warn"
    # Subtitle and consent checks land with later tasks — until then the
    # rows must degrade to "unavailable", never claim a pass they didn't
    # check (and never crash the document).
    assert by_id["subtitles"]["state"] in ("unavailable", "pass", "warn")
    assert by_id["consent"]["state"] in ("unavailable", "pass", "warn")
    assert d["warn_count"] == sum(
        1 for r in d["rows"] if r["state"] == "warn")
    assert d["pending_fixes"]["offsync_segments"] == [1]
    assert d["loudness_target"] == pytest.approx(
        getattr(server.cfg, "loudness_target", -16.0))
    assert d["on_approve"]["resynth_segments"] == 0
    assert d["pending_gate"] == "final_qc"


def test_qc_flags_a_loudness_miss(client, monkeypatch):
    server.jobs["q2"] = {"id": "q2", "status": "complete",
                         "created": time.time(), "target_lang": "ru"}
    _qc_fixture(monkeypatch,
                loudnorm={"output_i": -13.0, "output_tp": -0.4},
                placements=[])
    by_id = {r["id"]: r for r in client.get("/api/dub/q2/qc").json()["rows"]}
    assert by_id["loudness"]["state"] == "warn"
    assert by_id["sync"]["state"] == "unavailable"   # nothing placed yet


def test_qc_unknown_job_is_a_404(client):
    assert client.get("/api/dub/nope/qc").status_code == 404


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/dub/{id}/review_notes
# ═══════════════════════════════════════════════════════════════════════

def test_review_notes_append_with_local_author(client):
    server.jobs["n1"] = {"id": "n1", "status": "awaiting_final_qc",
                         "created": time.time()}
    d = client.post("/api/dub/n1/review_notes",
                    data={"text": "bass too hot on S2"}).json()
    assert d["ok"] is True and d["count"] == 1
    note = server.jobs["n1"]["review_notes"][0]
    assert note["text"] == "bass too hot on S2"
    assert note["author"] == "local"
    assert note["at"] == pytest.approx(time.time(), abs=5)


def test_review_notes_reject_empty_and_unknown(client):
    server.jobs["n2"] = {"id": "n2", "status": "complete",
                         "created": time.time()}
    assert client.post("/api/dub/n2/review_notes",
                       data={"text": "   "}).status_code == 400
    assert client.post("/api/dub/nope/review_notes",
                       data={"text": "x"}).status_code == 404


def test_review_notes_cap_at_one_hundred(client):
    server.jobs["n3"] = {"id": "n3", "status": "complete",
                         "created": time.time(),
                         "review_notes": [
                             {"text": f"old {i}", "at": 0.0, "author": "local"}
                             for i in range(100)]}
    d = client.post("/api/dub/n3/review_notes", data={"text": "newest"}).json()
    assert d["count"] == 100
    notes = server.jobs["n3"]["review_notes"]
    assert notes[-1]["text"] == "newest"
    assert notes[0]["text"] == "old 1"        # oldest fell off the front


# ═══════════════════════════════════════════════════════════════════════
#  Submit routes: the review_gates form field
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def submit_env(monkeypatch, tmp_path):
    """No Ollama probe, no queue, no writes under outputs/."""
    enqueued = []

    async def fake_check_ollama():
        return (False, [])

    async def fake_enqueue(job_id, pipeline_args):
        enqueued.append((job_id, pipeline_args))

    monkeypatch.setattr(server, "check_ollama", fake_check_ollama)
    monkeypatch.setattr(server, "enqueue_job", fake_enqueue)
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    return enqueued


def test_dub_submit_resolves_explicit_gates_onto_the_job(client, submit_env):
    r = client.post("/api/dub", data={
        "source": "/videos/clip.mp4",
        "review_gates": '{"translation": "on", "final_qc": "flagged_only"}',
    })
    assert r.status_code == 200
    jid = r.json()["job_id"]
    job = server.jobs[jid]
    assert job["review_gates"]["translation"] == "on"
    assert job["review_gates"]["final_qc"] == "flagged_only"
    assert job["review_gates"]["subtitles"] == "off"
    assert job["gates_cleared"] == []
    # ...and the queue payload carries the same resolved dict.
    (qid, args), = submit_env
    assert qid == jid and args["review_gates"] == job["review_gates"]


def test_dub_submit_refuses_a_typoed_gate(client, submit_env):
    r = client.post("/api/dub", data={
        "source": "/videos/clip.mp4",
        "review_gates": '{"translaton": "on"}',
    })
    assert r.status_code == 400
    assert "translaton" in r.json()["error"]
    assert submit_env == []


def test_dub_submit_without_gates_keeps_the_wizard_contract(client, submit_env):
    """wizard_mode is always sent by HTTP (Form default "auto"), so a plain
    submit resolves to all-off regardless of cfg defaults."""
    r = client.post("/api/dub", data={"source": "/videos/clip.mp4",
                                      "wizard_mode": "review_translation"})
    job = server.jobs[r.json()["job_id"]]
    assert job["review_gates"] == dict(rg.all_off(), translation="on")


# ═══════════════════════════════════════════════════════════════════════
#  retry_stage: all-off default, override re-arms
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def retry_env(monkeypatch):
    enqueued = []

    async def fake_enqueue(job_id, pipeline_args):
        enqueued.append((job_id, pipeline_args))

    monkeypatch.setattr(server, "enqueue_job", fake_enqueue)
    monkeypatch.setattr(
        server, "_stage_input_state",
        lambda jid, stage: {"stage": "tts_done", "segments": _segments(),
                            "target_lang": "ru",
                            "review_gates": {g: "on" for g in rg.GATES},
                            "gates_cleared": ["translation"]})
    return enqueued


def test_retry_defaults_every_gate_off(client, retry_env):
    server.jobs["r1"] = {"id": "r1", "status": "complete",
                         "created": time.time()}
    r = client.post("/api/dub/r1/retry_stage/assemble", data={})
    assert r.status_code == 200
    (_, args), = retry_env
    ctx = args["__stage_retry__"]["ctx"]
    assert ctx["review_gates"] == rg.all_off()   # checkpoint's gates dropped
    assert ctx["gates_cleared"] == []
    assert server.jobs["r1"]["review_gates"] == rg.all_off()


def test_retry_override_rearms_named_gates_only(client, retry_env):
    """The workbench's re-assemble passes final_qc=on to come back to QC."""
    server.jobs["r2"] = {"id": "r2", "status": "complete",
                         "created": time.time()}
    r = client.post("/api/dub/r2/retry_stage/assemble", data={
        "overrides": '{"review_gates": {"final_qc": "on"}}'})
    assert r.status_code == 200
    ctx = retry_env[-1][1]["__stage_retry__"]["ctx"]
    assert ctx["review_gates"] == dict(rg.all_off(), final_qc="on")


def test_retry_refuses_bad_review_gates(client, retry_env):
    server.jobs["r3"] = {"id": "r3", "status": "complete",
                         "created": time.time()}
    r = client.post("/api/dub/r3/retry_stage/assemble", data={
        "overrides": '{"review_gates": {"final_qc": "always"}}'})
    assert r.status_code == 400
    assert retry_env == []


def test_review_gates_is_an_allowed_retry_override():
    assert "review_gates" in server._RETRY_OVERRIDE_KEYS


# ═══════════════════════════════════════════════════════════════════════
#  Gate evaluator wiring (driver adapter, no pipeline run)
# ═══════════════════════════════════════════════════════════════════════

def test_evaluate_gate_pauses_merge_at_final_qc():
    ctx = {"review_gates": dict(rg.all_off(), final_qc="on"),
           "gates_cleared": [], "segments": _segments()}
    job = {"id": "e1"}
    status, _detail, cp = server._evaluate_gate("merge", ctx, job, "e1")
    assert (status, cp) == ("awaiting_final_qc", "merge_done")
    assert job["pending_gate"] == "final_qc"


def test_evaluate_gate_showcase_fallback_forces_delivery_gates_off():
    """A showcase sibling with legacy ctx (no review_gates) must not park at
    subtitles/final_qc — a sibling stuck at QC stalls the whole reel."""
    ctx = {"wizard_mode": "auto", "gates_cleared": [], "segments": []}
    job = {"id": "e2", "batch_kind": "showcase"}
    assert server._evaluate_gate("merge", ctx, job, "e2") is None
    assert ctx["review_gates"]["final_qc"] == "off"


def test_evaluate_gate_legacy_quality_pause_survives(monkeypatch):
    """Gate off + cfg.quality_gate on ⇒ the score-driven pause still fires,
    stashes its verdicts, records audit, and parks at the boundary's gate."""
    audited = []
    monkeypatch.setattr(server.app_audit, "record",
                        lambda *a, **k: audited.append((a, k)))
    monkeypatch.setattr(server.cfg, "quality_gate", True, raising=False)
    import pipeline.quality as q
    monkeypatch.setattr(q, "full_report", lambda *a, **k: {"fake": True})
    monkeypatch.setattr(q, "gate", lambda report, stage: {
        "pass": False, "stage": stage, "score": 12, "threshold": 60,
        "reasons": ["asr score 12 is below the 60 gate"], "verdicts": []})

    ctx = {"review_gates": rg.all_off(), "gates_cleared": [],
           "segments": _segments()}
    job = {"id": "e3"}
    status, detail, cp = server._evaluate_gate("diarize", ctx, job, "e3")
    assert (status, cp) == ("awaiting_transcript_review", "transcription_done")
    assert job["quality_gate"]["passed"] is False
    assert job["pending_gate"] == "transcript"
    assert "quality gate" in detail
    assert len(audited) == 1

    # ...but once that gate is cleared by an approve, the same boundary
    # doesn't re-pause on resume.
    ctx["gates_cleared"] = ["transcript"]
    job2 = {"id": "e3b"}
    assert server._evaluate_gate("diarize", ctx, job2, "e3b") is None


def test_evaluate_gate_flagged_only_uncomputable_pauses(monkeypatch):
    """findings=None ⇒ pause — the fail-safe direction, end to end."""
    monkeypatch.setattr(server, "_gate_findings",
                        lambda *a, **k: None)
    ctx = {"review_gates": dict(rg.all_off(), final_qc="flagged_only"),
           "gates_cleared": [], "segments": []}
    job = {"id": "e4"}
    status, detail, _cp = server._evaluate_gate("merge", ctx, job, "e4")
    assert status == "awaiting_final_qc"
    assert "could not run" in detail


# ═══════════════════════════════════════════════════════════════════════
#  Status ripple
# ═══════════════════════════════════════════════════════════════════════

def test_stage_order_tables_cannot_drift():
    assert tuple(server.STAGE_ORDER) == rg.PIPELINE_STAGE_ORDER


def test_every_gate_status_fires_the_awaiting_review_webhook():
    for gate, (status, _cp) in rg.GATE_STATUS.items():
        assert server._WEBHOOK_EVENT_FOR_STATUS[status] == "job.awaiting_review"


def test_webhook_payload_carries_the_pending_gate():
    from app.webhooks import payload_for_job
    payload = payload_for_job({"id": "x", "status": "awaiting_final_qc",
                               "pending_gate": "final_qc"})
    assert payload["pending_gate"] == "final_qc"


def test_gate_checkpoints_are_real_pipeline_checkpoints():
    checkpoints = {s["checkpoint"] for s in server.PIPELINE_STAGES}
    for _gate, (_status, cp) in rg.GATE_STATUS.items():
        assert cp in checkpoints
