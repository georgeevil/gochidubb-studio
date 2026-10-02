"""Transcribe-only mode: download → extract → transcribe → diarize, then done.

The transcript is the product (council minutes, interviews), so the mode has
to differ from a dub stopped early with `stop_after=diarize` in exactly the
ways that matter for a multi-hour recording: no Demucs run for a background
bed nobody will hear, no translation model required at submit, no review gate
or legacy quality-gate pause, every speaker kept (not folded into "main"), and
a terminal `complete` status instead of `paused`.

Route tests are hermetic like tests/test_partial_run.py (TestClient without
the context manager — see CLAUDE.md); runner tests replace the stage handlers
with fakes like tests/test_reuse_pipeline.py.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

import server


# ═══════════════════════════════════════════════════════════════════════
#  Mode plumbing
# ═══════════════════════════════════════════════════════════════════════

def test_transcribe_is_a_known_mode():
    assert "transcribe" in server.JOB_MODES
    assert server.normalize_job_mode(" Transcribe ") == "transcribe"


def test_transcribe_mode_walks_only_the_source_stages():
    assert server.stages_for_mode("transcribe") == [
        "download", "extract", "transcribe", "diarize"]
    for sid in server.stages_for_mode("transcribe"):
        assert sid in server.STAGE_ORDER


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/dub mode=transcribe
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def client(tmp_path, monkeypatch):
    saved = dict(server.jobs)
    server.jobs.clear()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    ollama_calls = []

    async def _ollama():
        ollama_calls.append(1)
        return True, []          # "running, nothing installed"

    enqueued = []

    async def _enqueue(job_id, args):
        enqueued.append((job_id, args))

    monkeypatch.setattr(server, "check_ollama", _ollama)
    monkeypatch.setattr(server, "enqueue_job", _enqueue)
    c = TestClient(server.app)
    c.enqueued = enqueued
    c.ollama_calls = ollama_calls
    try:
        yield c
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


def _submit(client, **extra):
    form = {"source": "/tmp/council.mp4", "mode": "transcribe",
            "source_lang": "es"}
    form.update(extra)
    return client.post("/api/dub", data=form)


def test_submit_needs_no_translation_model(client):
    r = _submit(client)
    assert r.status_code == 200, r.text
    # A machine with no LLM at all can still transcribe.
    assert client.ollama_calls == []
    assert client.enqueued[0][1]["mode"] == "transcribe"


def test_submit_forces_no_background_and_all_speakers(client):
    r = _submit(client, keep_bg="true", speaker_mode="main")
    job_id = r.json()["job_id"]
    args = client.enqueued[0][1]
    assert args["keep_bg"] is False
    assert args["speaker_mode"] == "all"
    assert server.jobs[job_id]["keep_bg"] is False


def test_speaker_hints_and_prompt_reach_the_queue(client):
    r = _submit(client, min_speakers="3", max_speakers="9",
                initial_prompt="Concejo Municipal de Talamanca, síndico")
    assert r.status_code == 200, r.text
    args = client.enqueued[0][1]
    assert args["min_speakers"] == 3
    assert args["max_speakers"] == 9
    assert args["initial_prompt"] == "Concejo Municipal de Talamanca, síndico"
    job = server.jobs[r.json()["job_id"]]
    assert job["min_speakers"] == 3 and job["max_speakers"] == 9


def test_unset_speaker_hints_mean_auto(client):
    _submit(client)
    args = client.enqueued[0][1]
    assert args["min_speakers"] is None
    assert args["max_speakers"] is None
    assert args["initial_prompt"] == ""


def test_inverted_speaker_range_is_refused(client):
    r = _submit(client, min_speakers="8", max_speakers="2")
    assert r.status_code == 400
    assert "min_speakers" in r.json()["error"]
    assert not client.enqueued


def test_dub_mode_still_validates_the_model(client):
    r = client.post("/api/dub", data={"source": "/tmp/x.mp4",
                                      "target_lang": "fr"})
    # Ollama "running with nothing installed" → refused, as before.
    assert r.status_code == 400
    assert client.ollama_calls


# ═══════════════════════════════════════════════════════════════════════
#  run_pipeline — the ctx a transcribe job runs with
# ═══════════════════════════════════════════════════════════════════════

def _run_pipeline_ctx(monkeypatch, **kw):
    seen = {}

    async def fake_stages(job_id, ctx, **skw):
        seen["ctx"] = dict(ctx)
        seen["kw"] = skw

    monkeypatch.setattr(server, "run_pipeline_stages", fake_stages)
    monkeypatch.setitem(server.jobs, "t1", {"id": "t1"})
    args = dict(source="x", source_lang="es", target_lang="ru", model="m",
                keep_bg=True, whisper_model="large-v3", mode="transcribe",
                speaker_mode="main", reference_audio="/tmp/ref.wav",
                review_gates={"transcript": "on", "translation": "on"})
    args.update(kw)
    asyncio.run(server.run_pipeline("t1", **args))
    return seen


def test_run_pipeline_transcribe_ctx(monkeypatch):
    seen = _run_pipeline_ctx(monkeypatch, min_speakers=2, max_speakers=0,
                             initial_prompt="  Talamanca  ")
    ctx = seen["ctx"]
    assert ctx["keep_bg"] is False
    assert ctx["speaker_mode"] == "all"
    assert ctx["reference_audio"] == ""
    assert set(ctx["review_gates"].values()) == {"off"}
    assert ctx["min_speakers"] == 2
    assert ctx["max_speakers"] is None          # 0 = auto
    assert ctx["initial_prompt"] == "Talamanca"
    assert seen["kw"] == {"start_stage": "download", "stop_after": "diarize"}


def test_run_pipeline_dub_keeps_its_settings(monkeypatch):
    seen = _run_pipeline_ctx(monkeypatch, mode="dub")
    ctx = seen["ctx"]
    assert ctx["keep_bg"] is True
    assert ctx["speaker_mode"] == "main"
    assert ctx["review_gates"]["transcript"] == "on"
    assert seen["kw"]["stop_after"] == ""


# ═══════════════════════════════════════════════════════════════════════
#  run_pipeline_stages — terminal status, no gates
# ═══════════════════════════════════════════════════════════════════════

class _Cfg:
    reuse_enabled = False
    quality_gate = True        # the legacy backstop is ON
    whisper_model = "large-v3"

    def __init__(self, **over):
        for k, v in over.items():
            setattr(self, k, v)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "cfg", _Cfg())
    monkeypatch.setattr(server, "save_job", lambda job: None)
    monkeypatch.setattr(server, "_save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(server.app_audit, "AUDIT_FILE", tmp_path / "audit.jsonl")
    ran = []

    def fake(stage):
        async def handler(job, work, ctx, update, perf):
            ran.append(stage)
            if stage == "transcribe":
                # Garbage ASR that the legacy quality gate would refuse.
                ctx["segments"] = [{"idx": 0, "start": 0.0, "end": 1.0,
                                    "text": "", "no_speech_prob": 0.99,
                                    "avg_logprob": -3.0}]
        return handler

    monkeypatch.setattr(server, "STAGE_HANDLERS",
                        {sid: fake(sid) for sid in server.STAGE_ORDER})
    return ran


def _drive(job, ctx, stop_after=""):
    server.jobs[job["id"]] = job
    try:
        asyncio.run(server.run_pipeline_stages(
            job["id"], ctx, start_stage="download", stop_after=stop_after))
    finally:
        server.jobs.pop(job["id"], None)
    return job


def test_transcribe_job_completes_after_diarize(rig):
    job = _drive({"id": "tj1", "mode": "transcribe"},
                 {"review_gates": {"transcript": "on"}}, stop_after="")
    assert rig == ["download", "extract", "transcribe", "diarize"]
    assert job["status"] == "complete"
    assert job["progress"] == 100
    assert job["checkpoint_stage"] == "transcription_done"
    assert job["transcript_url"] == "/api/dub/tj1/transcript"
    assert "pending_gate" not in job


def test_transcribe_ceiling_beats_a_later_stop_after(rig):
    # retry_stage / continue can hand the runner any stop_after; the mode's
    # ceiling must still hold.
    job = _drive({"id": "tj2", "mode": "transcribe"}, {}, stop_after="merge")
    assert "translate" not in rig
    assert job["status"] == "complete"


def test_dub_job_still_pauses_on_the_legacy_gate(rig):
    job = _drive({"id": "dj1", "mode": "dub"},
                 {"review_gates": {}}, stop_after="diarize")
    assert job["status"] == "awaiting_transcript_review"


def test_evaluate_gate_never_pauses_transcribe(monkeypatch):
    monkeypatch.setattr(server, "cfg", _Cfg())
    job = {"id": "x", "mode": "transcribe"}
    ctx = {"review_gates": {"transcript": "on"}, "segments": []}
    assert server._evaluate_gate("diarize", ctx, job, "x") is None


# ═══════════════════════════════════════════════════════════════════════
#  Stage handlers — the hints actually reach ASR and diarization
# ═══════════════════════════════════════════════════════════════════════

def test_transcribe_stage_passes_prompt_and_records_backend(tmp_path, monkeypatch):
    seen = {}

    def fake_transcribe(path, lang, model, **kw):
        seen.update(kw, path=path)
        kw["info"].update(backend="mlx", model="mlx-community/whisper-x")
        return [{"start": 0.0, "end": 2.0, "text": "Buenas tardes"}], "es"

    monkeypatch.setattr(server, "transcribe", fake_transcribe)
    ctx = {"audio_16k": str(tmp_path / "a.wav"), "duration": 2.0,
           "source_lang": "es", "whisper_model": "large-v3",
           "initial_prompt": "Talamanca"}
    asyncio.run(server._stage_transcribe(
        {"id": "x"}, tmp_path, ctx, lambda **k: None, {}))
    assert seen["initial_prompt"] == "Talamanca"
    assert ctx["asr"] == {"backend": "mlx", "model": "mlx-community/whisper-x"}


def test_diarize_stage_passes_speaker_hints_and_uses_timeline_audio(
        tmp_path, monkeypatch):
    timeline = tmp_path / "audio_16k_clean.wav"
    timeline.write_bytes(b"x")
    seen = {}

    def fake_diarize(path, **kw):
        seen.update(kw, path=path)
        return [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]

    def fake_refs(path, turns, out_dir, main_only=False):
        seen["refs_path"] = path
        seen["main_only"] = main_only
        return {"SPEAKER_00": "a.wav", "SPEAKER_01": "b.wav"}

    monkeypatch.setattr(server, "diarize_speakers", fake_diarize)
    monkeypatch.setattr(server, "extract_speaker_audio", fake_refs)
    monkeypatch.setattr(server, "effective_hf_token", lambda: "hf_x")
    ctx = {"segments": [{"idx": 0, "start": 0.0, "end": 2.0, "text": "Hola."},
                        {"idx": 1, "start": 2.0, "end": 4.0, "text": "Sí."}],
           "audio_16k": str(tmp_path / "audio_16k_vad.wav"),
           "audio_16k_timeline": str(timeline),
           "vad_intervals": [[0.0, 4.0]],
           "speaker_mode": "all", "min_speakers": 2, "max_speakers": 7}
    asyncio.run(server._stage_diarize(
        {"id": "x"}, tmp_path, ctx, lambda **k: None, {}))
    assert seen["min_speakers"] == 2 and seen["max_speakers"] == 7
    assert seen["path"] == str(timeline)
    assert seen["refs_path"] == str(timeline)
    assert seen["main_only"] is False
    assert ctx["diarization"]["model"] == server.DIARIZATION_MODELS[0]
    assert {s["speaker"] for s in ctx["segments"]} == {"SPEAKER_00", "SPEAKER_01"}


def test_timeline_audio_falls_back_without_vad(tmp_path):
    assert server._timeline_audio({"audio_16k": "a.wav"}) == "a.wav"
    # No vad_intervals → VAD passed the audio through; audio_16k is fine.
    assert server._timeline_audio({"audio_16k": "a.wav",
                                   "audio_16k_timeline": "b.wav"}) == "a.wav"
