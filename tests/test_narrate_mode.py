"""Narration jobs: POST /api/narrate and the runner behind it.

A narration is the one job that does not start from a video, so it does not
go through the stage walker. What it must still share with a dub is tested
here: it is queued (never run inline, because it synthesizes on the same
GPU), it honours Cancel, it lands on the same terminal statuses, and a
failure is recorded the same way.

Route tests are hermetic like tests/test_transcribe_mode.py (TestClient
without the context manager — see CLAUDE.md). Runner tests drive
run_narration directly with fake engines that write real WAV files, so the
timeline, the mix and the subtitle files are the real ones.
"""
import asyncio
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import server
from pipeline import narration as narr
from pipeline.synthesizer import BaseTTSEngine

MD = """# Briefing

## Cover
![](cover.png)
Seven years ago the court gave the agency twelve months.

## Timeline
![](timeline.png)
The ruling came in 2019. Nothing was delimited.
"""

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/narrate
# ═══════════════════════════════════════════════════════════════════════

class _Cfg:
    mode = "local"
    voice_consent_policy = "off"

    def __init__(self, **over):
        for k, v in over.items():
            setattr(self, k, v)


@pytest.fixture
def client(tmp_path, monkeypatch):
    saved = dict(server.jobs)
    server.jobs.clear()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "cfg", _Cfg())
    monkeypatch.setattr(server, "save_job", lambda job: None)
    monkeypatch.setattr(server, "scan_file_presets", lambda: {})
    enqueued = []

    async def _enqueue(job_id, args):
        enqueued.append((job_id, args))

    monkeypatch.setattr(server, "enqueue_job", _enqueue)
    c = TestClient(server.app)
    c.enqueued = enqueued
    c.out = tmp_path
    try:
        yield c
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


def _images(*names):
    return [("images", (name, io.BytesIO(PNG), "image/png")) for name in names]


def _submit(client, script=MD, images=("cover.png", "timeline.png"), **form):
    data = {"script": script}
    data.update(form)
    return client.post("/api/narrate", data=data,
                       files=_images(*images) or None)


def test_submit_queues_a_narration_job(client):
    r = _submit(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slides"] == 2 and body["lines"] == 3 and body["video"] is True
    assert body["estimated_duration"] > 0

    (job_id, args), = client.enqueued
    assert job_id == body["job_id"]
    # Routed to run_narration by the queue worker, never to run_pipeline.
    assert set(args) == {"__narrate__"}
    spec = args["__narrate__"]
    assert spec["target_lang"] == "en" and spec["render_video"] is True
    assert spec["script"]["title"] == "Briefing"

    job = server.jobs[job_id]
    assert job["mode"] == "narrate" and job["status"] == "queued"
    assert job["source_label"] == "Briefing" and job["target_lang"] == "en"
    assert job["narration"]["lines"] == 3 and job["narration"]["video"] is True


def test_submit_stores_the_script_and_the_named_pictures(client):
    job_id = _submit(client).json()["job_id"]
    work = client.out / job_id
    stored = json.loads((work / "narration_script.json").read_text("utf-8"))
    assert [s["id"] for s in stored["slides"]] == ["cover", "timeline"]
    assert sorted(p.name for p in (work / "slides").iterdir()) == [
        "cover.png", "timeline.png"]
    assert (work / "slides" / "cover.png").read_bytes() == PNG


def test_an_unreferenced_upload_is_reported_not_stored(client):
    r = _submit(client, images=("cover.png", "timeline.png", "spare.png"))
    assert r.json()["ignored_images"] == ["spare.png"]
    work = client.out / r.json()["job_id"]
    assert not (work / "slides" / "spare.png").exists()


def test_an_upload_name_cannot_escape_the_slides_folder(client):
    r = client.post("/api/narrate", data={"script": MD}, files=[
        ("images", ("../../cover.png", io.BytesIO(PNG), "image/png")),
        ("images", ("timeline.png", io.BytesIO(PNG), "image/png")),
    ])
    assert r.status_code == 200, r.text
    work = client.out / r.json()["job_id"]
    assert (work / "slides" / "cover.png").exists()
    assert not (client.out.parent / "cover.png").exists()


def test_missing_pictures_are_named(client):
    r = _submit(client, images=("cover.png",))
    assert r.status_code == 400
    assert "timeline.png" in r.json()["error"]
    assert "cover.png" not in r.json()["error"].split("uploaded:")[1].split(".")[0]
    assert client.enqueued == [] and server.jobs == {}


def test_no_video_means_no_pictures_are_needed(client):
    r = _submit(client, images=(), render_video="false")
    assert r.status_code == 200, r.text
    assert r.json()["video"] is False
    job_id = r.json()["job_id"]
    assert client.enqueued[0][1]["__narrate__"]["render_video"] is False
    assert not (client.out / job_id / "slides").exists()


def test_a_script_without_pictures_is_audio_only(client):
    r = _submit(client, script="## One\nJust words.", images=())
    assert r.status_code == 200, r.text
    assert r.json()["video"] is False


def test_the_first_slide_needs_a_picture_for_a_video(client):
    md = "## Cover\nNo picture here.\n\n## Two\n![](timeline.png)\nWords."
    r = _submit(client, script=md, images=("timeline.png",))
    assert r.status_code == 400
    assert "first slide" in r.json()["error"]


@pytest.mark.parametrize("form, fragment", [
    ({"script": "## Only a heading"}, "No slide has any narration"),
    ({"target_lang": "xx"}, "Unsupported language"),
    ({"tts_speed": "ludicrous"}, "tts_speed"),
    ({"voice_preset": "morgan_freeman"}, "Unknown voice"),
    ({"timing": "slide_gap=2"}, "timing must be a JSON object"),
    ({"timing": "[1, 2]"}, "timing must be a JSON object"),
])
def test_bad_submissions_are_refused_before_anything_is_queued(
        client, form, fragment):
    r = _submit(client, **form)
    assert r.status_code == 400, r.text
    assert fragment in r.json()["error"]
    assert client.enqueued == [] and server.jobs == {}


def test_a_non_image_upload_is_refused(client):
    r = client.post("/api/narrate", data={"script": MD}, files=[
        ("images", ("cover.exe", io.BytesIO(b"MZ"), "application/octet-stream"))])
    assert r.status_code == 400 and "cover.exe" in r.json()["error"]


def test_timing_and_voice_reach_the_queue(client):
    r = _submit(client, timing='{"slide_gap": 2.5}', voice_preset="female_narrator",
                tts_speed="quality", burn_subs="true", title="Members' briefing")
    assert r.status_code == 200, r.text
    spec = client.enqueued[0][1]["__narrate__"]
    assert spec["timing"] == {"slide_gap": 2.5}
    assert spec["voice_preset"] == "female_narrator"
    assert spec["tts_speed"] == "quality" and spec["burn_subs"] is True
    job = server.jobs[r.json()["job_id"]]
    assert job["title"] == "Members' briefing"
    assert job["voice_mode"] == "preset"


def test_a_free_text_style_is_a_custom_voice(client):
    r = _submit(client, voice_style="calm mature female narrator")
    assert server.jobs[r.json()["job_id"]]["voice_mode"] == "custom"


# ── consent: a library voice is a recording of a real person ────────────

def _library(monkeypatch, consent):
    presets = {"file:anna": {"name": "Anna", "reference_file": "/x/anna.wav",
                             **({"consent": {"by": "anna"}} if consent else {})}}
    monkeypatch.setattr(server, "scan_file_presets", lambda: presets)


def test_require_policy_refuses_a_library_voice_without_consent(
        client, monkeypatch):
    _library(monkeypatch, consent=False)
    monkeypatch.setattr(server, "cfg", _Cfg(voice_consent_policy="require"))
    r = _submit(client, voice_preset="file:anna")
    assert r.status_code == 400 and "consent" in r.json()["error"]
    assert client.enqueued == []


def test_warn_policy_narrates_and_marks_the_job(client, monkeypatch):
    _library(monkeypatch, consent=False)
    monkeypatch.setattr(server, "cfg", _Cfg(voice_consent_policy="warn"))
    r = _submit(client, voice_preset="file:anna")
    assert r.status_code == 200, r.text
    job = server.jobs[r.json()["job_id"]]
    assert job["narration"]["voice_consent"] == "missing"
    assert job["voice_mode"] == "upload"


def test_an_attested_library_voice_passes_the_require_policy(
        client, monkeypatch):
    _library(monkeypatch, consent=True)
    monkeypatch.setattr(server, "cfg", _Cfg(voice_consent_policy="require"))
    r = _submit(client, voice_preset="file:anna")
    assert r.status_code == 200, r.text
    assert "voice_consent" not in server.jobs[r.json()["job_id"]]["narration"]


def test_designed_voices_need_no_consent(client, monkeypatch):
    monkeypatch.setattr(server, "cfg", _Cfg(voice_consent_policy="require"))
    assert _submit(client, voice_preset="male_warm").status_code == 200


# ── hosted mode ─────────────────────────────────────────────────────────

def test_narrate_is_part_of_the_dub_write_surface():
    assert server._scope_for("POST", "/api/narrate") == "dub:write"


# ═══════════════════════════════════════════════════════════════════════
#  run_narration
# ═══════════════════════════════════════════════════════════════════════

SR = 24000


def _write_clip(path, text):
    seconds = max(0.4, len(text) / 30.0)
    t = np.arange(int(SR * seconds)) / SR
    sf.write(path, (0.4 * np.sin(2 * np.pi * 200 * t)).astype("float32"), SR,
             subtype="PCM_16")


class _AsyncEngine:
    """Stands in for edge-tts: natively async, fixed voice, no cloning."""
    sample_rate = SR

    def __init__(self, fail=(), fail_once=()):
        self.fail, self.fail_once = set(fail), set(fail_once)
        self.calls = []

    async def synthesize_segments_async(self, segments, output_dir, target_lang,
                                        voice="", progress_callback=None):
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.calls.append([s["idx"] for s in segments])
        for i, seg in enumerate(segments):
            idx = seg["idx"]
            if idx in self.fail or idx in self.fail_once:
                self.fail_once.discard(idx)
                seg["audio_path"] = None
            else:
                seg["audio_path"] = os.path.join(output_dir, f"seg_{i:04d}.wav")
                _write_clip(seg["audio_path"], seg["translated_text"])
            if progress_callback:
                progress_callback(i + 1, len(segments))
        return segments


class _CloningEngine(server.VoxCPMSynthesizer):
    """A VoxCPMSynthesizer with the model taken out: same call signatures,
    so the runner takes the designed-voice path it takes on a real install."""

    def __init__(self, design_ok=True):
        BaseTTSEngine.__init__(self)
        self._sample_rate = SR
        self.design_ok = design_ok
        self.designed, self.synth = [], []

    def design_reference(self, style, output_path, voice_seed=None,
                         target_lang="en", sample_text=""):
        self.designed.append((style, voice_seed, target_lang))
        if not self.design_ok:
            return None
        _write_clip(output_path, "x" * 60)
        return output_path

    def synthesize_segments(self, segments, output_dir, speaker_refs=None,
                            speaker_transcripts=None, progress_callback=None,
                            voice_seed=None, tts_speed="balanced",
                            is_cross_lingual=False, target_lang="ru",
                            cfg_override=None, steps_override=None):
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.synth.append({
            "refs": dict(speaker_refs or {}), "seed": voice_seed,
            "speed": tts_speed, "lang": target_lang,
            "cross": is_cross_lingual,
            "tts_texts": [s.get("tts_text") for s in segments],
            "speakers": {s["speaker"] for s in segments},
        })
        for i, seg in enumerate(segments):
            seg["audio_path"] = os.path.join(output_dir, f"seg_{i:04d}.wav")
            _write_clip(seg["audio_path"], seg["translated_text"])
        return segments

    def last_run_stats(self):
        return {}

    def last_failure_detail(self):
        return "CUDA out of memory"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    saved = dict(server.jobs)
    server.jobs.clear()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "DESIGNED_VOICES_DIR", tmp_path / "designed")
    monkeypatch.setattr(server, "save_job", lambda job: None)
    monkeypatch.setattr(server, "scan_file_presets", lambda: {})
    monkeypatch.setattr(server, "_say_map_for", lambda lang: {})
    # The suite runs without ffmpeg (requirements-dev.txt): loudness is the
    # assembler's, tested there; the slideshow encode is replaced below.
    import pipeline.assembler as assembler
    monkeypatch.setattr(assembler, "_normalize_loudness_inplace",
                        lambda *a, **k: None)
    rendered = []

    def _render(spans, audio_path, output_path, **kw):
        rendered.append({"spans": spans, "audio": audio_path,
                         "out": output_path, **kw})
        with open(output_path, "wb") as f:
            f.write(b"mp4")
        return output_path

    monkeypatch.setattr(narr, "render_slideshow", _render)

    class Rig:
        out = tmp_path

        def engine(self, eng):
            monkeypatch.setattr(server, "get_tts_engine", lambda: eng)
            return eng

        def run(self, job_id="nj1", script=MD, **args):
            spec = {"script": narr.parse_script(script), "target_lang": "en",
                    "voice_preset": "auto", "voice_style": "",
                    "tts_speed": "balanced", "timing": {},
                    "render_video": True, "burn_subs": False}
            spec.update(args)
            job = {"id": job_id, "status": "queued", "mode": "narrate",
                   "narration": {"slides": len(spec["script"]["slides"])}}
            server.jobs[job_id] = job
            asyncio.run(server.run_narration(job_id, spec))
            return job

    r = Rig()
    r.rendered = rendered
    try:
        yield r
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


def test_a_narration_completes_with_track_subtitles_and_video(rig):
    rig.engine(_AsyncEngine())
    job = rig.run()
    work = rig.out / "nj1"
    assert job["status"] == "complete" and job["progress"] == 100
    assert job["step_detail"] == "Narration ready"
    assert job["stage_id"] is None and "error" not in job

    data, sr = sf.read(str(work / "narration.wav"))
    assert sr == SR
    assert len(data) / sr == pytest.approx(job["duration"], abs=0.01)
    assert np.abs(data).max() > 0.2

    srt = (work / "subtitles.srt").read_text("utf-8")
    assert "Seven years ago" in srt and "Nothing was delimited." in srt
    assert (work / "subtitles.vtt").read_text("utf-8").startswith("WEBVTT")

    assert job["narration_url"].startswith("/outputs/nj1/narration.wav")
    assert job["srt_url"] == "/outputs/nj1/subtitles.srt"
    assert job["vtt_url"] == "/outputs/nj1/subtitles.vtt"
    assert job["timeline_url"] == "/outputs/nj1/narration_timeline.json"
    assert job["output_url"].startswith("/outputs/nj1/dubbed_video.mp4")
    assert job["narration"] == {"slides": 2, "lines": 3, "synthesized": 3,
                                "failed_lines": [], "cues": job["narration"]["cues"]}


def test_the_timeline_document_is_portable_and_consistent(rig):
    rig.engine(_AsyncEngine())
    job = rig.run()
    doc = json.loads(
        (rig.out / "nj1" / "narration_timeline.json").read_text("utf-8"))
    assert doc["duration"] == job["duration"] and doc["language"] == "en"
    assert [s["id"] for s in doc["slides"]] == ["cover", "timeline"]
    assert doc["slides"][0]["start"] == 0.0
    assert doc["slides"][1]["start"] == doc["slides"][0]["end"]
    assert doc["slides"][-1]["end"] == doc["duration"]
    assert doc["failed_lines"] == []
    # No machine paths in a file meant to travel with the WAV.
    assert "audio_path" not in json.dumps(doc)
    assert str(rig.out) not in json.dumps(doc)


def test_the_slideshow_gets_each_picture_for_its_slide(rig):
    rig.engine(_AsyncEngine())
    job = rig.run(burn_subs=True)
    (call,) = rig.rendered
    work = rig.out / "nj1"
    assert [p for p, _d in call["spans"]] == [
        str(work / "slides" / "cover.png"), str(work / "slides" / "timeline.png")]
    assert sum(d for _p, d in call["spans"]) == pytest.approx(
        job["duration"], abs=0.01)
    assert call["audio"] == str(work / "narration.wav")
    assert call["out"] == str(work / "dubbed_video.mp4")
    assert call["srt_path"] == str(work / "subtitles.srt")
    assert call["burn_subs"] is True
    assert call["duration"] == job["duration"]


def test_no_video_is_rendered_when_it_was_not_asked_for(rig):
    rig.engine(_AsyncEngine())
    job = rig.run(render_video=False)
    assert job["status"] == "complete" and rig.rendered == []
    assert "output_url" not in job
    assert (rig.out / "nj1" / "narration.wav").exists()


def test_a_script_without_pictures_renders_no_video(rig):
    rig.engine(_AsyncEngine())
    job = rig.run(script="## One\nJust words. And more words.")
    assert job["status"] == "complete" and rig.rendered == []
    assert "output_url" not in job


def test_a_failed_line_is_retried_once(rig):
    eng = rig.engine(_AsyncEngine(fail_once={1}))
    job = rig.run()
    assert eng.calls == [[0, 1, 2], [1]]
    assert job["narration"]["failed_lines"] == []
    assert job["narration"]["synthesized"] == 3
    assert job["step_detail"] == "Narration ready"


def test_a_line_that_keeps_failing_is_reported_not_hidden(rig):
    eng = rig.engine(_AsyncEngine(fail={1}))
    job = rig.run()
    assert eng.calls == [[0, 1, 2], [1]]
    assert job["status"] == "complete"
    assert job["narration"]["failed_lines"] == [1]
    assert "1 line(s) have no audio" in job["step_detail"]
    doc = json.loads(
        (rig.out / "nj1" / "narration_timeline.json").read_text("utf-8"))
    assert doc["failed_lines"] == [1]
    assert doc["segments"][1]["missing_audio"] is True
    # The subtitle is still there, so the slide can be read in the silence.
    assert "The ruling came in 2019." in (
        rig.out / "nj1" / "subtitles.srt").read_text("utf-8")


def test_total_failure_is_an_error_naming_the_cause(rig):
    eng = rig.engine(_CloningEngine())
    eng.synthesize_segments = lambda segments, *a, **k: [
        dict(s, audio_path=None) for s in segments]
    job = rig.run()
    assert job["status"] == "error"
    assert "All 3 narration lines failed" in job["error"]
    assert "CUDA out of memory" in job["error"]
    assert job["last_error"]["stage"] == "tts"
    assert not (rig.out / "nj1" / "narration.wav").exists()


def test_auto_voice_is_a_designed_narrator_drawn_once_and_cloned(rig):
    eng = rig.engine(_CloningEngine())
    job = rig.run()
    assert job["status"] == "complete"
    default = server.VOICE_PRESETS[server._NARRATION_DEFAULT_PRESET]
    assert eng.designed == [(default["style"], default["seed"], "en")]
    (call,) = eng.synth
    (ref,) = call["refs"].values()
    assert set(call["refs"]) == {narr.NARRATOR} == call["speakers"]
    assert ref.startswith(str(rig.out / "designed"))
    assert call["seed"] == default["seed"] and call["lang"] == "en"
    assert call["cross"] is False
    # Cloned from the reference: no style prefix on any line.
    assert call["tts_texts"] == [None, None, None]
    assert job["voice_preset"] == server._NARRATION_DEFAULT_PRESET
    assert job["voice_mode"] == "custom"


def test_a_free_text_style_designs_that_voice(rig):
    eng = rig.engine(_CloningEngine())
    rig.run(voice_style="calm mature female narrator")
    assert eng.designed[0][0] == "calm mature female narrator"


def test_design_failure_falls_back_to_a_style_prefix_per_line(rig):
    eng = rig.engine(_CloningEngine(design_ok=False))
    job = rig.run(voice_preset="female_narrator")
    assert job["status"] == "complete"
    (call,) = eng.synth
    assert call["refs"] == {}
    style = server.VOICE_PRESETS["female_narrator"]["style"]
    assert all(t and t.startswith(f"({style})") for t in call["tts_texts"])
    # …and the prefix never reaches a subtitle.
    assert style not in (rig.out / "nj1" / "subtitles.srt").read_text("utf-8")


def test_a_library_voice_is_cloned_from_its_file(rig, monkeypatch, tmp_path):
    ref = tmp_path / "anna.wav"
    _write_clip(str(ref), "x" * 60)
    monkeypatch.setattr(server, "scan_file_presets", lambda: {
        "file:anna": {"name": "Anna", "reference_file": str(ref)}})
    eng = rig.engine(_CloningEngine())
    job = rig.run(voice_preset="file:anna")
    assert eng.designed == []
    assert eng.synth[0]["refs"] == {narr.NARRATOR: str(ref)}
    assert job["voice_mode"] == "upload"


def test_glossary_respellings_apply_to_a_script(rig, monkeypatch):
    monkeypatch.setattr(server, "_say_map_for",
                        lambda lang: {"delimited": "dee-LIM-it-ed"})
    eng = rig.engine(_CloningEngine())
    rig.run()
    texts = eng.synth[0]["tts_texts"]
    assert texts[2] and "dee-LIM-it-ed" in texts[2]
    assert "dee-LIM-it-ed" not in (
        rig.out / "nj1" / "subtitles.srt").read_text("utf-8")


def test_timing_overrides_stretch_the_timeline(rig):
    rig.engine(_AsyncEngine())
    plain = rig.run(job_id="a")["duration"]
    slow = rig.run(job_id="b", timing={"slide_gap": 5.0})["duration"]
    assert slow == pytest.approx(plain + 5.0 - narr.DEFAULT_TIMING["slide_gap"],
                                 abs=0.01)


def test_cancel_is_honoured_and_reaches_the_queue_worker(rig):
    rig.engine(_AsyncEngine())
    job = {"id": "nc1", "status": "queued", "mode": "narrate",
           "cancel_requested": True}
    server.jobs["nc1"] = job
    with pytest.raises(server.JobCancelled):
        asyncio.run(server.run_narration(
            "nc1", {"script": narr.parse_script(MD)}))
    # The worker, not the runner, writes 'cancelled' — same as a dub.
    assert job["status"] == "queued"
    assert not (rig.out / "nc1" / "narration.wav").exists()


def test_an_encode_failure_is_an_error_but_keeps_the_track(rig, monkeypatch):
    rig.engine(_AsyncEngine())

    def _boom(*a, **k):
        raise RuntimeError("ffmpeg exited 1: no such encoder libx264")

    monkeypatch.setattr(narr, "render_slideshow", _boom)
    job = rig.run()
    assert job["status"] == "error" and "libx264" in job["error"]
    assert job["last_error"]["stage"] == "merge"
    assert (rig.out / "nj1" / "narration.wav").exists()
    assert (rig.out / "nj1" / "subtitles.srt").exists()


def test_the_queue_worker_routes_narrate_args_to_the_runner(monkeypatch):
    """The dispatch in _job_queue_worker: a dict carrying __narrate__ goes to
    run_narration and nothing else."""
    seen = []

    async def _narration(job_id, spec):
        seen.append(("narrate", job_id, spec))

    async def _pipeline(job_id, **kw):
        seen.append(("pipeline", job_id, kw))

    monkeypatch.setattr(server, "run_narration", _narration)
    monkeypatch.setattr(server, "run_pipeline", _pipeline)
    monkeypatch.setattr(server, "save_job", lambda job: None)
    monkeypatch.setattr(server, "_intake_gate", None)
    monkeypatch.setattr(server, "_intake_changed", None)
    saved = dict(server.jobs)
    server.jobs.clear()
    server.jobs["q1"] = {"id": "q1", "status": "queued"}

    async def _drive():
        monkeypatch.setattr(server, "_job_queue", asyncio.Queue())
        await server._job_queue.put(("q1", {"__narrate__": {"script": "x"}}))
        worker = asyncio.ensure_future(server._job_queue_worker())
        await server._job_queue.join()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    try:
        asyncio.run(_drive())
    finally:
        server.jobs.clear()
        server.jobs.update(saved)
    assert seen == [("narrate", "q1", {"script": "x"})]
