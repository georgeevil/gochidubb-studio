"""GET /api/dub/{id}/transcript — speaker-attributed transcript export.

The JSON shape is a contract with an external consumer (a public minutes
archive), so it is pinned exactly here. The rendered formats must name a
speaker by the human-confirmed name first, then the editor's label, then the
diarization id — and must never print a voice-profile *match*, which is only
a suggestion until a human confirms it.
"""
import json

import pytest
from fastapi.testclient import TestClient

import server
from pipeline.transcript import build_transcript, render, speaker_display_names


CP = {
    "stage": "transcription_done",
    "source": "https://youtu.be/council",
    "duration": 12.0,
    "effective_src": "es",
    "asr": {"backend": "mlx", "model": "mlx-community/whisper-large-v3-mlx"},
    "diarization": {"model": "pyannote/speaker-diarization-3.1"},
    "segments": [
        {"idx": 0, "start": 0.0, "end": 4.0, "text": " Buenas tardes.",
         "speaker": "SPEAKER_01", "avg_logprob": -0.2, "no_speech_prob": 0.01,
         "translated_text": "ignored", "word_conf_mean": 0.9},
        {"idx": 1, "start": 4.0, "end": 5.0, "text": "Gracias.",
         "speaker": "SPEAKER_00"},
        {"idx": 2, "start": 5.0, "end": 6.0, "text": "[música]",
         "speaker": "SPEAKER_00", "non_speech": True},
        {"idx": 3, "start": 6.0, "end": 9.5, "text": "Se aprueba el acta.",
         "speaker": "SPEAKER_01"},
    ],
}


def _doc(**kw):
    return build_transcript("j1", CP, source="https://youtu.be/council", **kw)


# ═══════════════════════════════════════════════════════════════════════
#  build_transcript — exact shape
# ═══════════════════════════════════════════════════════════════════════

def test_top_level_shape_is_exact():
    doc = _doc()
    assert list(doc) == ["job_id", "source", "duration", "language", "asr",
                         "diarization", "speakers", "segments"]
    assert doc["job_id"] == "j1"
    assert doc["duration"] == 12.0
    assert doc["language"] == "es"
    assert doc["asr"] == {"backend": "mlx",
                          "model": "mlx-community/whisper-large-v3-mlx"}
    assert doc["diarization"] == {"model": "pyannote/speaker-diarization-3.1"}


def test_segment_shape_is_exact_and_non_speech_is_dropped():
    segs = _doc()["segments"]
    assert [s["idx"] for s in segs] == [0, 1, 3]
    for s in segs:
        assert list(s) == ["idx", "start", "end", "text", "speaker",
                           "avg_logprob", "no_speech_prob"]
    assert segs[0]["text"] == "Buenas tardes."
    assert segs[0]["avg_logprob"] == -0.2
    assert segs[1]["avg_logprob"] is None and segs[1]["no_speech_prob"] is None


def test_speakers_sorted_by_talk_with_labels_matches_confirmations():
    match = {"profile_id": "p1", "name": "Ana Mora", "role": "regidora",
             "score": 0.81234}
    doc = _doc(labels={"SPEAKER_00": "Secretaria"},
               matches={"SPEAKER_01": match},
               confirmed={"SPEAKER_00": {"profile_id": None,
                                         "name": "Persona del público",
                                         "role": "público"}})
    sp = doc["speakers"]
    assert [s["id"] for s in sp] == ["SPEAKER_01", "SPEAKER_00"]
    assert list(sp[0]) == ["id", "label", "talk_secs", "segments", "match",
                           "confirmed"]
    assert sp[0] == {"id": "SPEAKER_01", "label": None, "talk_secs": 7.5,
                     "segments": 2,
                     "match": {"profile_id": "p1", "name": "Ana Mora",
                               "role": "regidora", "score": 0.8123},
                     "confirmed": None}
    assert sp[1]["label"] == "Secretaria"
    assert sp[1]["confirmed"] == {"profile_id": None,
                                  "name": "Persona del público",
                                  "role": "público"}
    assert sp[1]["segments"] == 1          # the non-speech line is not counted


def test_old_checkpoint_without_asr_or_diarization_is_null():
    cp = {"segments": [{"start": 0, "end": 1, "text": "hola"}]}
    doc = build_transcript("j", cp)
    assert doc["asr"] is None and doc["diarization"] is None
    assert doc["duration"] is None and doc["language"] is None
    assert doc["segments"][0]["speaker"] == "SPEAKER_00"


# ═══════════════════════════════════════════════════════════════════════
#  Rendering — confirmed > label > id, never the match
# ═══════════════════════════════════════════════════════════════════════

def test_display_name_precedence():
    doc = _doc(labels={"SPEAKER_00": "Secretaria", "SPEAKER_01": "Presidente"},
               matches={"SPEAKER_01": {"profile_id": "p", "name": "Guess",
                                       "score": 0.9}})
    names = speaker_display_names(doc)
    # The match is NOT used — only the label.
    assert names == {"SPEAKER_00": "Secretaria", "SPEAKER_01": "Presidente"}
    doc = _doc(labels={"SPEAKER_01": "Presidente"},
               confirmed={"SPEAKER_01": {"name": "Ana Mora", "role": "x"}})
    assert speaker_display_names(doc)["SPEAKER_01"] == "Ana Mora"
    assert speaker_display_names(_doc())["SPEAKER_00"] == "SPEAKER_00"


def test_srt():
    out = render(_doc(confirmed={"SPEAKER_01": {"name": "Ana Mora"}}), "srt")
    assert out.startswith("1\n00:00:00,000 --> 00:00:04,000\n"
                          "Ana Mora: Buenas tardes.\n\n2\n")
    assert "SPEAKER_00: Gracias." in out
    assert "[música]" not in out


def test_vtt():
    out = render(_doc(), "vtt")
    assert out.startswith("WEBVTT\n\n1\n00:00:00.000 --> 00:00:04.000\n"
                          "SPEAKER_01: Buenas tardes.")


def test_txt():
    out = render(_doc(labels={"SPEAKER_01": "Presidencia"}), "txt")
    assert "Source: https://youtu.be/council" in out
    assert "ASR: mlx mlx-community/whisper-large-v3-mlx" in out
    assert "[00:00:06] Presidencia: Se aprueba el acta." in out


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        render(_doc(), "docx")


# ═══════════════════════════════════════════════════════════════════════
#  Route
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def client(tmp_path, monkeypatch):
    saved = dict(server.jobs)
    server.jobs.clear()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    try:
        yield c
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


def _job(tmp_path, job_id="t1", cp_stage="transcription_done", **job):
    work = tmp_path / job_id
    work.mkdir()
    (work / f"checkpoint_{cp_stage}.json").write_text(
        json.dumps(CP), encoding="utf-8")
    server.jobs[job_id] = {"id": job_id, "status": "complete",
                           "mode": "transcribe",
                           "source": "https://youtu.be/council",
                           "duration": 12.0, **job}


def test_route_json(client, tmp_path):
    _job(tmp_path, speaker_labels={"SPEAKER_00": "Secretaria"})
    r = client.get("/api/dub/t1/transcript")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == "t1"
    assert body["source"] == "https://youtu.be/council"
    assert {s["id"]: s["label"] for s in body["speakers"]} == {
        "SPEAKER_01": None, "SPEAKER_00": "Secretaria"}


@pytest.mark.parametrize("fmt,needle", [
    ("srt", "00:00:00,000 --> 00:00:04,000"),
    ("vtt", "WEBVTT"),
    ("txt", "[00:00:00] SPEAKER_01: Buenas tardes."),
])
def test_route_text_formats(client, tmp_path, fmt, needle):
    _job(tmp_path)
    r = client.get(f"/api/dub/t1/transcript?format={fmt}")
    assert r.status_code == 200
    assert needle in r.text
    assert f'transcript_t1.{fmt}"' in r.headers["content-disposition"]


def test_route_reads_later_checkpoints_too(client, tmp_path):
    _job(tmp_path, cp_stage="translation_done")
    r = client.get("/api/dub/t1/transcript?format=txt")
    assert r.status_code == 200
    assert "ignored" not in r.text      # translations are not exported


def test_route_errors(client, tmp_path):
    assert client.get("/api/dub/nope/transcript").status_code == 404
    server.jobs["empty"] = {"id": "empty", "status": "transcribing"}
    r = client.get("/api/dub/empty/transcript")
    assert r.status_code == 404
    _job(tmp_path)
    r = client.get("/api/dub/t1/transcript?format=docx")
    assert r.status_code == 400
    assert "json" in r.json()["error"]


def test_finished_transcribe_job_allows_speaker_edits(client, tmp_path):
    _job(tmp_path)
    r = client.post("/api/dub/t1/speakers/edit", json={
        "ops": [{"op": "rename", "speaker": "SPEAKER_01", "label": "Presidencia"}]})
    assert r.status_code == 200, r.text
    assert server.jobs["t1"]["speaker_labels"]["SPEAKER_01"] == "Presidencia"
    # A finished *dub* still refuses: its audio is already synthesized.
    server.jobs["t1"]["mode"] = "dub"
    r = client.post("/api/dub/t1/speakers/edit", json={
        "ops": [{"op": "rename", "speaker": "SPEAKER_01", "label": "X"}]})
    assert r.status_code == 409


# ═══════════════════════════════════════════════════════════════════════
#  CLI wiring
# ═══════════════════════════════════════════════════════════════════════

def test_cli_parses_transcribe_and_transcript():
    from tools.gochidubb_cli import build_parser, cmd_transcribe, cmd_transcript
    a = build_parser().parse_args([
        "transcribe", "https://youtu.be/x", "--source-lang", "es",
        "--min-speakers", "3", "--max-speakers", "9", "--prompt", "Talamanca"])
    assert a.handler is cmd_transcribe
    assert (a.min_speakers, a.max_speakers, a.prompt) == (3, 9, "Talamanca")
    a = build_parser().parse_args(["transcript", "t1", "--format", "srt",
                                   "-o", "out.srt"])
    assert a.handler is cmd_transcript
    assert (a.format, a.output) == ("srt", "out.srt")
