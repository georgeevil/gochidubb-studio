"""Route tests for the CLD-267/CLD-272/CLD-266 workbench endpoints.

Same hermetic contract as tests/test_creator_routes.py: TestClient without
its context manager (no lifespan, so the real gochidubb.db never opens),
checkpoints monkeypatched, every write aimed at tmp_path. See that module's
docstring for why this pattern is load-bearing.
"""
import json
import time

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client():
    saved = dict(server.jobs)
    server.jobs.clear()
    try:
        yield TestClient(server.app)
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


def _job(jid, **kw):
    job = {"id": jid, "status": "awaiting_translation_review",
           "created": time.time()}
    job.update(kw)
    return job


def _checkpoints(monkeypatch, cps):
    """Patch checkpoint IO onto an in-memory dict; return the saved-state
    view so assertions can see exactly what write-through wrote."""
    store = {k: json.loads(json.dumps(v)) for k, v in cps.items()}
    saved = {}

    def load(job_id, stage):
        return store.get(stage)

    def save(job_id, work, stage=None, data=None, **kw):
        store[stage] = data
        saved[stage] = data

    monkeypatch.setattr(server, "_load_checkpoint", load)
    monkeypatch.setattr(server, "_save_checkpoint", save)
    return saved


_SEGS = [
    {"idx": 0, "start": 0.0, "end": 2.0, "speaker": "S1", "text": "a",
     "translated_text": "α"},
    {"idx": 1, "start": 2.0, "end": 4.0, "speaker": "S3", "text": "b",
     "translated_text": "β"},
    {"idx": 2, "start": 4.0, "end": 5.0, "speaker": "S3", "text": "c",
     "translated_text": "γ"},
]


def _cp():
    return {
        "segments": json.loads(json.dumps(_SEGS)),
        "speaker_refs": {"S1": "r1.wav", "S3": "r3.wav"},
        "speaker_transcripts": {"S1": "a", "S3": "b"},
        "speaker_voice_map": {"S1": "source", "S3": "source"},
    }


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/dub/{id}/speakers/edit
# ═══════════════════════════════════════════════════════════════════════

def test_speaker_edit_unknown_job_is_404(client):
    r = client.post("/api/dub/nope/speakers/edit", json={"ops": []})
    assert r.status_code == 404


def test_speaker_edit_refuses_while_synthesizing(client, monkeypatch):
    server.jobs["j"] = _job("j", status="synthesizing")
    r = client.post("/api/dub/j/speakers/edit",
                    json={"ops": [{"op": "rename", "speaker": "S1",
                                   "label": "Host"}]})
    assert r.status_code == 409
    assert "synthesizing" in r.json()["error"]


def test_speaker_edit_refuses_after_tts(client, monkeypatch):
    """Post-TTS relabels are the ghost-voice window; the guard closes it."""
    server.jobs["j"] = _job("j", status="complete")
    r = client.post("/api/dub/j/speakers/edit",
                    json={"ops": [{"op": "merge", "from": "S3",
                                   "into": "S1"}]})
    assert r.status_code == 409


def test_speaker_edit_requires_a_transcription_checkpoint(client, monkeypatch):
    server.jobs["j"] = _job("j")
    monkeypatch.setattr(server, "_load_checkpoint", lambda *a: None)
    r = client.post("/api/dub/j/speakers/edit",
                    json={"ops": [{"op": "rename", "speaker": "S1",
                                   "label": "x"}]})
    assert r.status_code == 409
    assert "checkpoint" in r.json()["error"].lower()


@pytest.mark.parametrize("ops,fragment", [
    ([], "non-empty"),
    ([{"op": "explode"}], "op"),
    ([{"op": "merge", "from": "S3"}], "merge"),
    ([{"op": "reassign", "segment_idx": "x", "to": "S1"}], "reassign"),
])
def test_speaker_edit_validates_ops(client, monkeypatch, ops, fragment):
    server.jobs["j"] = _job("j")
    _checkpoints(monkeypatch, {"transcription_done": _cp()})
    r = client.post("/api/dub/j/speakers/edit", json={"ops": ops})
    assert r.status_code == 400
    assert fragment.lower() in r.json()["error"].lower()


def test_merge_rewrites_both_checkpoints_and_the_cast(client, monkeypatch):
    job = _job("j", speaker_voice_map={"S1": "source", "S3": "preset:x"})
    server.jobs["j"] = job
    saved = _checkpoints(monkeypatch, {"transcription_done": _cp(),
                                       "translation_done": _cp()})
    r = client.post("/api/dub/j/speakers/edit",
                    json={"ops": [{"op": "merge", "from": "S3",
                                   "into": "S1"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["applied"]["merged_segments"] == 2
    assert body["applied"]["checkpoints_updated"] == 2
    for stage in ("transcription_done", "translation_done"):
        cp = saved[stage]
        assert {s["speaker"] for s in cp["segments"]} == {"S1"}
        # The ghost's artifacts go with them.
        assert "S3" not in cp["speaker_refs"]
        assert "S3" not in cp["speaker_voice_map"]
    # Job-level cast loses the ghost too; indices never shifted.
    assert job["speaker_voice_map"] == {"S1": "source"}
    assert [s["idx"] for s in saved["translation_done"]["segments"]] == [0, 1, 2]


def test_reassign_moves_one_segment_only(client, monkeypatch):
    server.jobs["j"] = _job("j")
    saved = _checkpoints(monkeypatch, {"transcription_done": _cp()})
    r = client.post("/api/dub/j/speakers/edit",
                    json={"ops": [{"op": "reassign", "segment_idx": 1,
                                   "to": "S1"}]})
    assert r.status_code == 200
    segs = saved["transcription_done"]["segments"]
    assert [s["speaker"] for s in segs] == ["S1", "S1", "S3"]


def test_rename_is_display_only(client, monkeypatch):
    job = _job("j")
    server.jobs["j"] = job
    saved = _checkpoints(monkeypatch, {"transcription_done": _cp()})
    r = client.post("/api/dub/j/speakers/edit",
                    json={"ops": [{"op": "rename", "speaker": "S1",
                                   "label": "Host"}]})
    assert r.status_code == 200
    assert job["speaker_labels"] == {"S1": "Host"}
    # The stable S-id survives underneath.
    assert {s["speaker"] for s in saved["transcription_done"]["segments"]} \
        == {"S1", "S3"}


def test_mark_non_speech_flags_without_removing(client, monkeypatch):
    server.jobs["j"] = _job("j")
    saved = _checkpoints(monkeypatch, {"transcription_done": _cp()})
    r = client.post("/api/dub/j/speakers/edit",
                    json={"ops": [{"op": "mark_non_speech",
                                   "segment_idxs": [1, 2]}]})
    assert r.status_code == 200
    segs = saved["transcription_done"]["segments"]
    assert len(segs) == 3, "indices must never shift"
    assert [bool(s.get("non_speech")) for s in segs] == [False, True, True]


def test_sibling_fanout_is_explicit_and_reports_refusals(client, monkeypatch):
    server.jobs["a"] = _job("a", batch_id="b1")
    server.jobs["b"] = _job("b", batch_id="b1")
    server.jobs["c"] = _job("c", batch_id="b1", status="synthesizing")
    _checkpoints(monkeypatch, {"transcription_done": _cp()})

    ops = [{"op": "rename", "speaker": "S1", "label": "Host"}]
    r = client.post("/api/dub/a/speakers/edit", json={"ops": ops})
    assert r.status_code == 200
    assert server.jobs["b"].get("speaker_labels") is None, \
        "fan-out must be opt-in"

    r = client.post("/api/dub/a/speakers/edit",
                    json={"ops": ops, "apply_to_siblings": True})
    sibs = {s["job_id"]: s for s in r.json()["siblings"]}
    assert sibs["b"]["ok"] is True
    assert sibs["c"]["ok"] is False
    assert server.jobs["b"]["speaker_labels"] == {"S1": "Host"}


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/dub/{id}/consent
# ═══════════════════════════════════════════════════════════════════════

def test_consent_unknown_job_is_404(client):
    assert client.post("/api/dub/nope/consent",
                       data={"speaker": "S1", "attested": "true"}
                       ).status_code == 404


def test_consent_requires_a_speaker(client):
    server.jobs["j"] = _job("j")
    r = client.post("/api/dub/j/consent", data={"attested": "true"})
    assert r.status_code == 400


def test_consent_attests_and_withdraws(client):
    job = _job("j")
    server.jobs["j"] = job
    r = client.post("/api/dub/j/consent",
                    data={"speaker": "S1", "attested": "true"})
    assert r.status_code == 200
    rec = job["voice_consent"]["S1"]
    assert rec["attested_by"] == "local" and rec["attested_at"] > 0

    r = client.post("/api/dub/j/consent",
                    data={"speaker": "S1", "attested": "false"})
    assert r.json()["voice_consent"] == {}
    assert job["voice_consent"] == {}


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/glossary/term — the `say` field
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def glossary(tmp_path, monkeypatch):
    path = tmp_path / "user_glossary.json"
    monkeypatch.setattr(server, "USER_GLOSSARY_FILE", path)
    return path


def _read(path):
    return json.loads(path.read_text())["domains"][0]["terms"]


def test_say_without_translation_is_a_valid_term(client, glossary):
    r = client.post("/api/glossary/term",
                    data={"term": "GoChi", "say": "GOH-chee",
                          "target_lang": "es"})
    assert r.status_code == 200
    assert _read(glossary)["GoChi"] == {"say": "GOH-chee"}


def test_adding_say_keeps_the_stored_translation(client, glossary):
    client.post("/api/glossary/term",
                data={"term": "studio", "translation": "el estudio",
                      "target_lang": "es"})
    client.post("/api/glossary/term",
                data={"term": "studio", "say": "ess-TOO-dyo",
                      "target_lang": "es"})
    assert _read(glossary)["studio"] == {"dst": "el estudio",
                                         "say": "ess-TOO-dyo"}


def test_adding_translation_keeps_the_stored_say(client, glossary):
    client.post("/api/glossary/term",
                data={"term": "studio", "say": "ess-TOO-dyo",
                      "target_lang": "es"})
    client.post("/api/glossary/term",
                data={"term": "studio", "translation": "el estudio",
                      "target_lang": "es"})
    assert _read(glossary)["studio"] == {"dst": "el estudio",
                                         "say": "ess-TOO-dyo"}


def test_plain_terms_stay_plain_strings(client, glossary):
    """Backward compatibility: no `say` means the legacy string shape."""
    client.post("/api/glossary/term",
                data={"term": "studio", "translation": "el estudio",
                      "target_lang": "es"})
    assert _read(glossary)["studio"] == "el estudio"


def test_overlong_say_is_refused(client, glossary):
    r = client.post("/api/glossary/term",
                    data={"term": "x", "say": "y" * 500, "target_lang": "es"})
    assert r.status_code == 400
    assert "Respelling" in r.json()["error"]


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/dub/{id}/voice_preview — the `text` param (validation only;
#  synthesis paths need an engine and belong to the e2e run)
# ═══════════════════════════════════════════════════════════════════════

def test_voice_preview_unknown_job_is_404(client):
    r = client.post("/api/dub/nope/voice_preview", json={"text": "hola"})
    assert r.status_code == 404


def test_voice_preview_refuses_mid_synthesis(client):
    server.jobs["j"] = _job("j", status="synthesizing")
    r = client.post("/api/dub/j/voice_preview", json={"text": "hola"})
    assert r.status_code == 409
