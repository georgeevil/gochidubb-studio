"""Speaker identification with enrolled voice profiles.

No torch / pyannote here: the embedding model is replaced by a fake that
maps each speaker's audio to a fixed vector, so what is under test is the
selection, the maths, the one-to-one assignment, the persistence and the
routes' contract — in particular that a match is only ever a suggestion and
that a missing pyannote is a 503, not a crash.

Route tests follow tests/test_creator_routes.py: TestClient without the
context manager, the profile store pointed at a temp SQLite file.
"""
import json
import sys

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import server
from app import voice_profiles as store
from pipeline import voiceprint as vp


# ═══════════════════════════════════════════════════════════════════════
#  Pure helpers
# ═══════════════════════════════════════════════════════════════════════

SEGS = [
    {"start": 0.0, "end": 20.0, "speaker": "A"},
    {"start": 20.0, "end": 21.5, "speaker": "A"},
    {"start": 21.5, "end": 21.9, "speaker": "A"},            # too short
    {"start": 22.0, "end": 30.0, "speaker": "A", "no_speech_prob": 0.9},
    {"start": 30.0, "end": 40.0, "speaker": "A", "non_speech": True},
    {"start": 40.0, "end": 43.0, "speaker": "B"},
]


def test_speaker_speech_secs_skips_non_speech():
    secs = vp.speaker_speech_secs(SEGS)
    assert secs == {"A": pytest.approx(29.9), "B": 3.0}


def test_select_clips_longest_clean_capped_chronological():
    clips = vp.select_clips(SEGS, "A")
    # 20 s segment capped to MAX_CLIP_SECS; the noisy, short and non-speech
    # segments are out.
    assert clips == [(0.0, vp.MAX_CLIP_SECS), (20.0, 21.5)]
    assert vp.select_clips(SEGS, "A", max_total=5.0) == [(0.0, 5.0)]
    assert vp.select_clips(SEGS, "nobody") == []


def test_cosine_and_centroid():
    assert vp.cosine([1, 0], [2, 0]) == pytest.approx(1.0)
    assert vp.cosine([1, 0], [0, 3]) == pytest.approx(0.0)
    assert vp.cosine([1, 0], [1, 0, 0]) == 0.0           # shape mismatch
    c = vp.centroid([[1, 0], [0, 1]])
    assert np.allclose(c, [2 ** -0.5, 2 ** -0.5])
    with pytest.raises(ValueError):
        vp.centroid([])


P1 = {"id": "p1", "name": "Ana", "role": "regidora", "embeddings": [[1, 0, 0]]}
P2 = {"id": "p2", "name": "Luis", "role": None, "embeddings": [[0, 1, 0]]}


def test_assign_matches_is_one_to_one_by_score():
    clusters = {"S0": [0.9, 0.1, 0], "S1": [1, 0.05, 0], "S2": [0.1, 1, 0]}
    m = vp.assign_matches(clusters, [P1, P2], threshold=0.6)
    # S1 is closer to Ana than S0 is, so Ana goes to S1 only.
    assert m["S1"]["profile_id"] == "p1"
    assert m["S2"]["profile_id"] == "p2"
    assert "S0" not in m


def test_assign_matches_respects_threshold():
    m = vp.assign_matches({"S0": [1, 1, 0]}, [P1], threshold=0.8)
    assert m == {}          # cos = 0.707


def test_missing_pyannote_is_voiceprint_unavailable(monkeypatch):
    monkeypatch.setattr(vp, "_inference", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    with pytest.raises(vp.VoiceprintUnavailable, match="pyannote"):
        vp._load_inference("")


def test_embed_clips_reads_only_the_clips(tmp_path, monkeypatch):
    sr = 16000
    wav = np.zeros(sr * 10, dtype=np.float32)
    wav[sr * 2: sr * 4] = 0.5            # the clip we ask for
    path = tmp_path / "a.wav"
    sf.write(path, wav, sr)
    seen = []
    monkeypatch.setattr(vp, "_load_inference", lambda token: "inf")

    def fake_embed(inference, w, rate):
        seen.append((len(w), float(w.max())))
        return np.array([1.0, 0.0])

    monkeypatch.setattr(vp, "_embed_waveform", fake_embed)
    vec = vp.embed_clips(str(path), [(2.0, 4.0), (9.5, 12.0)])
    # The second clip runs off the end and is too short once clipped.
    assert seen == [(2 * sr, 0.5)]
    assert np.allclose(vec, [1.0, 0.0])
    assert vp.embed_clips(str(path), [(20.0, 30.0)]) is None


# ═══════════════════════════════════════════════════════════════════════
#  Store
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_DB_PATH", None)
    store.init_store(tmp_path / "vp.db")
    yield
    monkeypatch.setattr(store, "_DB_PATH", None)


def test_store_crud(db):
    p = store.create("Ana Mora", " Talamanca ", [1.0, 0.0], role="regidora",
                     source={"job_id": "j", "speaker": "S1"})
    assert p["group"] == "talamanca" and p["embeddings_count"] == 1
    assert "embeddings" not in p
    full = store.get(p["id"], with_embeddings=True)
    assert full["embeddings"] == [[1.0, 0.0]]
    assert [x["id"] for x in store.list_profiles("TALAMANCA")] == [p["id"]]
    assert store.list_profiles("limon") == []
    assert store.update(p["id"], role="presidenta")["role"] == "presidenta"
    assert store.update("nope", name="x") is None
    assert store.delete(p["id"]) is True
    assert store.delete(p["id"]) is False


def test_add_embedding_dedupes_source_and_caps(db, monkeypatch):
    p = store.create("Ana", "g", [1.0, 0.0])
    src = {"job_id": "j1", "speaker": "S0"}
    store.add_embedding(p["id"], [0.9, 0.1], src)
    store.add_embedding(p["id"], [0.9, 0.1], src)       # same source again
    assert store.get(p["id"])["embeddings_count"] == 2
    monkeypatch.setattr(store, "MAX_EMBEDDINGS", 3)
    for i in range(5):
        store.add_embedding(p["id"], [1.0, i], {"job_id": f"k{i}", "speaker": "S"})
    assert store.get(p["id"])["embeddings_count"] == 3


def test_store_not_initialised_raises(monkeypatch):
    monkeypatch.setattr(store, "_DB_PATH", None)
    with pytest.raises(store.StoreUnavailable):
        store.list_profiles()


# ═══════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════

# Fake "voices": the speaker each clip belongs to is recovered from its start
# time, and each speaker has a fixed direction in embedding space.
VOICES = {"SPEAKER_00": [1.0, 0.0, 0.0], "SPEAKER_01": [0.0, 1.0, 0.0],
          "SPEAKER_02": [0.0, 0.0, 1.0]}
CP_SEGS = [
    {"idx": 0, "start": 0.0, "end": 12.0, "text": "Se abre la sesión.",
     "speaker": "SPEAKER_00"},
    {"idx": 1, "start": 100.0, "end": 108.0, "text": "Gracias, presidenta.",
     "speaker": "SPEAKER_01"},
    {"idx": 2, "start": 200.0, "end": 202.0, "text": "Sí.",
     "speaker": "SPEAKER_02"},                     # under 5 s — skipped
]


def _speaker_at(t):
    return {0: "SPEAKER_00", 1: "SPEAKER_01", 2: "SPEAKER_02"}[int(t // 100)]


@pytest.fixture
def client(tmp_path, monkeypatch, db):
    saved = dict(server.jobs)
    server.jobs.clear()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "save_job", lambda job: None)
    monkeypatch.setattr(server, "effective_hf_token", lambda: "")
    monkeypatch.setattr(server.app_audit, "AUDIT_FILE", tmp_path / "audit.jsonl")
    calls = []

    def fake_embed_clips(audio, clips, token=""):
        calls.append(clips)
        return np.array(VOICES[_speaker_at(clips[0][0])])

    monkeypatch.setattr(vp, "embed_clips", fake_embed_clips)
    c = TestClient(server.app)
    c.embed_calls = calls
    try:
        yield c
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


def _job(tmp_path, job_id="j1", **job):
    work = tmp_path / job_id
    work.mkdir()
    (work / "audio_16k.wav").write_bytes(b"x")
    cp = {"segments": CP_SEGS, "audio_16k": str(work / "audio_16k.wav")}
    (work / "checkpoint_transcription_done.json").write_text(json.dumps(cp))
    server.jobs[job_id] = {"id": job_id, "status": "complete",
                           "mode": "transcribe", **job}


def test_enroll_from_job_speaker(client, tmp_path):
    _job(tmp_path)
    r = client.post("/api/voiceprints", json={
        "name": "Ana Mora", "role": "presidenta", "group": "Talamanca",
        "job_id": "j1", "speaker": "SPEAKER_00"})
    assert r.status_code == 200, r.text
    p = r.json()["profile"]
    assert p["name"] == "Ana Mora" and p["group"] == "talamanca"
    assert p["sources"] == [{"job_id": "j1", "speaker": "SPEAKER_00"}]
    assert store.get(p["id"], True)["embeddings"] == [[1.0, 0.0, 0.0]]


def test_enroll_validation(client, tmp_path):
    _job(tmp_path)
    assert client.post("/api/voiceprints", json={"group": "g"}).status_code == 400
    r = client.post("/api/voiceprints", json={"name": "x", "group": "g"})
    assert r.status_code == 400
    r = client.post("/api/voiceprints", json={
        "name": "x", "group": "g", "job_id": "nope", "speaker": "S"})
    assert r.status_code == 404
    r = client.post("/api/voiceprints", json={
        "name": "x", "group": "g", "job_id": "j1", "speaker": "SPEAKER_02"})
    assert r.status_code == 400
    assert "5s" in r.json()["error"]


def test_enroll_from_upload(client, monkeypatch):
    seen = {}

    def fake_extract(src, dst):
        seen["src"] = src
        open(dst, "wb").write(b"wav")
        return dst

    monkeypatch.setattr(server, "extract_audio", fake_extract)
    monkeypatch.setattr(vp, "embed_file",
                        lambda path, token="": np.array([0.0, 1.0, 0.0]))
    r = client.post("/api/voiceprints",
                    data={"name": "Luis", "group": "limon"},
                    files={"file": ("luis.m4a", b"audio", "audio/mp4")})
    assert r.status_code == 200, r.text
    assert r.json()["profile"]["sources"] == []
    assert seen["src"].endswith(".m4a")


def test_identify_suggests_one_to_one_and_skips_short(client, tmp_path):
    _job(tmp_path)
    ana = store.create("Ana", "talamanca", [1.0, 0.1, 0.0], role="presidenta")
    store.create("Luis", "talamanca", [0.1, 1.0, 0.0])
    store.create("Elsewhere", "limon", [1.0, 0.0, 0.0])
    r = client.post("/api/dub/j1/speakers/identify", json={"group": "talamanca"})
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {row["speaker"]: row for row in body["speakers"]}
    assert body["threshold"] == server.cfg.voice_match_threshold
    assert rows["SPEAKER_00"]["match"]["profile_id"] == ana["id"]
    assert rows["SPEAKER_00"]["match"]["role"] == "presidenta"
    assert rows["SPEAKER_01"]["match"]["name"] == "Luis"
    assert rows["SPEAKER_02"]["skipped"] == "too_little_speech"
    assert rows["SPEAKER_02"]["match"] is None
    job = server.jobs["j1"]
    assert set(job["speaker_matches"]) == {"SPEAKER_00", "SPEAKER_01"}
    # Suggestions only — nothing confirmed.
    assert not job.get("speaker_confirmed")
    # Cached: identifying again does not re-embed.
    n = len(client.embed_calls)
    client.post("/api/dub/j1/speakers/identify", json={"group": "talamanca"})
    assert len(client.embed_calls) == n


def test_identify_threshold_and_validation(client, tmp_path):
    _job(tmp_path)
    store.create("Ana", "g", [1.0, 1.0, 0.0])
    r = client.post("/api/dub/j1/speakers/identify",
                    json={"group": "g", "threshold": 0.9})
    assert all(row["match"] is None for row in r.json()["speakers"])
    assert server.jobs["j1"]["speaker_matches"] == {}
    assert client.post("/api/dub/j1/speakers/identify",
                       json={}).status_code == 400
    assert client.post("/api/dub/j1/speakers/identify",
                       json={"group": "g", "threshold": 2}).status_code == 400
    assert client.post("/api/dub/zz/speakers/identify",
                       json={"group": "g"}).status_code == 404


def test_identify_without_pyannote_is_503(client, tmp_path, monkeypatch):
    _job(tmp_path)
    store.create("Ana", "g", [1.0, 0.0, 0.0])

    def unavailable(*a, **k):
        raise vp.VoiceprintUnavailable("Speaker identification needs pyannote.audio")

    monkeypatch.setattr(vp, "embed_clips", unavailable)
    r = client.post("/api/dub/j1/speakers/identify", json={"group": "g"})
    assert r.status_code == 503
    assert "pyannote" in r.json()["error"]


def test_routes_503_when_store_missing(client, monkeypatch):
    monkeypatch.setattr(store, "_DB_PATH", None)
    assert client.get("/api/voiceprints").status_code == 503
    r = client.post("/api/voiceprints", json={"name": "x", "group": "g"})
    assert r.status_code == 503


def test_confirm_with_profile_learns_and_exports(client, tmp_path):
    _job(tmp_path)
    ana = store.create("Ana Mora", "talamanca", [1.0, 0.0, 0.0], role="presidenta")
    r = client.post("/api/dub/j1/speakers/confirm", json={
        "speaker": "SPEAKER_00", "profile_id": ana["id"]})
    assert r.status_code == 200, r.text
    assert r.json()["confirmed"] == {"profile_id": ana["id"],
                                     "name": "Ana Mora", "role": "presidenta"}
    assert r.json()["profile_updated"] is True
    assert store.get(ana["id"])["embeddings_count"] == 2
    assert {"job_id": "j1", "speaker": "SPEAKER_00"} in store.get(ana["id"])["sources"]
    txt = client.get("/api/dub/j1/transcript?format=txt").text
    assert "Ana Mora: Se abre la sesión." in txt


def test_confirm_public_name_and_clear(client, tmp_path):
    _job(tmp_path)
    r = client.post("/api/dub/j1/speakers/confirm",
                    json={"speaker": "SPEAKER_01", "public": True})
    assert r.json()["confirmed"] == {"profile_id": None,
                                     "name": "Persona del público",
                                     "role": "público"}
    r = client.post("/api/dub/j1/speakers/confirm",
                    json={"speaker": "SPEAKER_02", "name": "Síndico Pérez"})
    assert r.json()["confirmed"]["name"] == "Síndico Pérez"
    assert r.json()["confirmed"]["profile_id"] is None
    doc = client.get("/api/dub/j1/transcript").json()
    conf = {s["id"]: s["confirmed"] for s in doc["speakers"]}
    assert conf["SPEAKER_01"]["name"] == "Persona del público"
    r = client.post("/api/dub/j1/speakers/confirm",
                    json={"speaker": "SPEAKER_01", "clear": True})
    assert r.json()["confirmed"] is None
    assert "SPEAKER_01" not in server.jobs["j1"]["speaker_confirmed"]


def test_confirm_validation(client, tmp_path):
    _job(tmp_path)
    post = lambda body: client.post("/api/dub/j1/speakers/confirm", json=body)  # noqa: E731
    assert post({}).status_code == 400
    assert post({"speaker": "SPEAKER_09", "public": True}).status_code == 400
    assert post({"speaker": "SPEAKER_00"}).status_code == 400
    assert post({"speaker": "SPEAKER_00", "profile_id": "nope"}).status_code == 404


def test_confirm_survives_an_embedding_failure(client, tmp_path, monkeypatch):
    _job(tmp_path)
    ana = store.create("Ana", "g", [1.0, 0.0, 0.0])

    def unavailable(*a, **k):
        raise vp.VoiceprintUnavailable("no pyannote")

    monkeypatch.setattr(vp, "embed_clips", unavailable)
    r = client.post("/api/dub/j1/speakers/confirm",
                    json={"speaker": "SPEAKER_00", "profile_id": ana["id"]})
    assert r.status_code == 200
    assert r.json()["profile_updated"] is False
    assert "no pyannote" in r.json()["warning"]
    assert server.jobs["j1"]["speaker_confirmed"]["SPEAKER_00"]["name"] == "Ana"


def test_list_patch_delete(client):
    p = store.create("Ana", "talamanca", [1.0, 0.0])
    store.create("Luis", "limon", [0.0, 1.0])
    assert len(client.get("/api/voiceprints").json()["profiles"]) == 2
    only = client.get("/api/voiceprints?group=limon").json()["profiles"]
    assert [x["name"] for x in only] == ["Luis"]
    assert "embeddings" not in only[0]
    withv = client.get("/api/voiceprints?group=limon&include_embeddings=true")
    assert withv.json()["profiles"][0]["embeddings"] == [[0.0, 1.0]]
    r = client.patch(f"/api/voiceprints/{p['id']}", json={"role": "regidora"})
    assert r.json()["profile"]["role"] == "regidora"
    assert client.patch(f"/api/voiceprints/{p['id']}",
                        json={"name": " "}).status_code == 400
    assert client.patch("/api/voiceprints/nope", json={"name": "x"}).status_code == 404
    assert client.delete(f"/api/voiceprints/{p['id']}").status_code == 200
    assert client.delete(f"/api/voiceprints/{p['id']}").status_code == 404


def test_merge_drops_identity_of_both_sides(client, tmp_path):
    _job(tmp_path, speaker_matches={"SPEAKER_00": {"profile_id": "a"},
                                    "SPEAKER_01": {"profile_id": "b"}},
         speaker_confirmed={"SPEAKER_01": {"name": "Luis"}})
    r = client.post("/api/dub/j1/speakers/edit", json={
        "ops": [{"op": "merge", "from": "SPEAKER_01", "into": "SPEAKER_00"}]})
    assert r.status_code == 200, r.text
    job = server.jobs["j1"]
    assert job["speaker_matches"] == {}
    assert job["speaker_confirmed"] == {}


def test_hosted_scope_for_voiceprints():
    assert server._scope_for("GET", "/api/voiceprints") == "jobs:read"
    assert server._scope_for("POST", "/api/voiceprints") == "voices:write"
    assert server._scope_for("DELETE", "/api/voiceprints/x") == "voices:write"


def test_cli_parses_voiceprints_and_speakers():
    from tools.gochidubb_cli import build_parser
    a = build_parser().parse_args(["voiceprints", "enroll", "Ana Mora",
                                   "--group", "talamanca", "--job", "j1",
                                   "--speaker", "SPEAKER_00"])
    assert (a.vp_cmd, a.name, a.group, a.job) == ("enroll", "Ana Mora",
                                                  "talamanca", "j1")
    a = build_parser().parse_args(["speakers", "confirm", "j1", "SPEAKER_01",
                                   "--public"])
    assert a.sp_cmd == "confirm" and a.public is True
    with pytest.raises(SystemExit):
        build_parser().parse_args(["speakers", "confirm", "j1", "SPEAKER_01"])
