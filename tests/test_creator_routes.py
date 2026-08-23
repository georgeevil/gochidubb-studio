"""Route-level tests for the five endpoints Creator mode added.

The rest of the suite tests pure functions. These are the first tests in the
repo that drive `server.app` itself, and they stay hermetic the same way the
pure tests do:

* `TestClient(app)` is used **without** its context manager on purpose. The
  lifespan (server.py) is what calls `init_db()` and `load_all_jobs()`, so
  skipping it means `app.db._DB_PATH` stays None, the real `gochidubb.db` is
  never opened, and `save_job_sync` is a no-op. A test run cannot touch a
  real job.
* Nothing here passes an `http` source to `/api/estimate`. `_probe_meta_cached`
  returns `{}` immediately for anything that is not a URL, so no yt-dlp probe
  and no network call ever happens.
* Every test that writes touches a tmp_path, never `presets/user_glossary.json`
  or `user_prefs.json`.

Some tests are marked `xfail(strict=True)`. Those are defects found by testing
the running server; the assertion states the behaviour that *should* hold. When
one is fixed the strict marker turns the pass into a failure, which is the
signal to delete the marker rather than to leave a stale one behind.
"""
import asyncio
import json
import pathlib
import time

import pytest
from fastapi.testclient import TestClient

import server
from app import billing as app_billing


@pytest.fixture
def client():
    """A client over the real app with an empty, isolated job table."""
    saved = dict(server.jobs)
    server.jobs.clear()
    try:
        yield TestClient(server.app)
    finally:
        server.jobs.clear()
        server.jobs.update(saved)


def _job(jid, **kw):
    # `created` has to be recent: billing.summarize filters on it, and the
    # estimate route only counts the last `window_days` of history.
    job = {"id": jid, "status": "complete", "progress": 100,
           "created": time.time()}
    job.update(kw)
    return job


# ═══════════════════════════════════════════════════════════════════════
#  GET /api/estimate
# ═══════════════════════════════════════════════════════════════════════

def test_estimate_with_no_params_is_a_zero_quote(client):
    """Step 1 of the wizard renders before a language is picked."""
    d = client.get("/api/estimate").json()
    assert d["duration_sec"] == 0.0
    assert d["langs"] == []
    assert d["cost"] == 0.0
    assert d["eta_sec"] == 0
    assert d["estimate"] is True


@pytest.mark.parametrize("langs", ["", ",,,", "  ,  "])
def test_estimate_treats_blank_language_lists_as_none(client, langs):
    assert client.get("/api/estimate", params={"langs": langs}).json()["langs"] == []


def test_estimate_rejects_an_unknown_language(client):
    r = client.get("/api/estimate", params={"langs": "zz"})
    assert r.status_code == 400
    assert "zz" in r.json()["error"]


@pytest.mark.parametrize("langs", ["es,es", "ES,es"])
def test_estimate_rejects_duplicates_case_insensitively(client, langs):
    r = client.get("/api/estimate", params={"langs": langs})
    assert r.status_code == 400
    assert "Duplicate" in r.json()["error"]


def test_estimate_rejects_more_than_the_batch_ceiling(client):
    codes = list(server._QUICK_TEST_KNOWN_LANGS)[:server.MAX_TARGET_LANGS + 1]
    r = client.get("/api/estimate", params={"langs": ",".join(codes)})
    assert r.status_code == 400
    assert str(server.MAX_TARGET_LANGS) in r.json()["error"]


def test_estimate_accepts_exactly_the_batch_ceiling(client):
    codes = list(server._QUICK_TEST_KNOWN_LANGS)[:server.MAX_TARGET_LANGS]
    r = client.get("/api/estimate",
                   params={"langs": ",".join(codes), "duration_sec": 600})
    assert r.status_code == 200
    assert len(r.json()["langs"]) == server.MAX_TARGET_LANGS


def test_estimate_rejects_a_non_numeric_duration(client):
    assert client.get("/api/estimate",
                      params={"duration_sec": "banana"}).status_code == 422


@pytest.mark.parametrize("bad", [-500, -0.001])
def test_estimate_floors_a_negative_duration_at_zero(client, bad):
    d = client.get("/api/estimate",
                   params={"duration_sec": bad, "langs": "es"}).json()
    assert d["duration_sec"] == 0.0
    assert d["cost"] == 0.0


def test_estimate_bills_source_length_times_language_count(client):
    d = client.get("/api/estimate",
                   params={"duration_sec": 600, "langs": "es,fr,ja"}).json()
    assert d["billable_minutes"] == pytest.approx(30.0)  # 10 min x 3


def test_estimate_cost_is_the_marginal_price_not_a_standalone_one(client):
    """The quote must be what the meter's total moves by.

    Pricing the new minutes from zero would quote the first band for all of
    them on a workspace already past it, and the meter would then disagree.
    """
    # 500 minutes of history puts the next minute in the second band.
    server.jobs.update({
        f"j{i}": _job(f"j{i}", duration=6000, target_lang="es",
                      started_at=1.0, completed_at=2.0)
        for i in range(5)
    })
    d = client.get("/api/estimate",
                   params={"duration_sec": 600, "langs": "es"}).json()
    expected = app_billing.marginal_cost(d["used_minutes"], d["billable_minutes"])
    assert d["cost"] == expected["cost"]
    assert d["used_minutes"] > 0


def test_estimate_eta_multiplies_by_language_count(client):
    """The queue is single-consumer, so N languages run one after another."""
    one = client.get("/api/estimate",
                     params={"duration_sec": 600, "langs": "es"}).json()
    three = client.get("/api/estimate",
                       params={"duration_sec": 600, "langs": "es,fr,ja"}).json()
    assert three["eta_sec"] == pytest.approx(one["eta_sec"] * 3, rel=0.01)
    assert three["eta_per_lang_sec"] == pytest.approx(one["eta_sec"], rel=0.01)


def test_estimate_says_when_the_eta_is_a_default_rather_than_measured(client):
    d = client.get("/api/estimate",
                   params={"duration_sec": 600, "langs": "es"}).json()
    assert d["eta_basis"] == "default"
    assert d["eta_realtime_factor"] == pytest.approx(server.cfg.eta_realtime_factor)


def test_estimate_uses_a_measured_factor_once_there_are_enough_samples(client):
    # Three finished jobs at a 2x realtime factor.
    server.jobs.update({
        f"m{i}": _job(f"m{i}", duration=100.0, target_lang="es",
                      started_at=1000.0, completed_at=1200.0)
        for i in range(3)
    })
    d = client.get("/api/estimate",
                   params={"duration_sec": 600, "langs": "es"}).json()
    assert d["eta_basis"] == "measured"
    assert d["eta_realtime_factor"] == pytest.approx(2.0)


def test_estimate_refuses_an_over_long_video_before_start(client, monkeypatch):
    """The wizard should refuse a too-long source without downloading it."""
    monkeypatch.setattr(server.cfg, "max_source_duration_sec", 600)
    d = client.get("/api/estimate",
                   params={"duration_sec": 1800, "langs": "es"}).json()
    assert d["duration_gate_error"]
    # Rendered verbatim on a consumer screen: no config identifiers in it.
    assert "max_source_duration_sec" not in d["duration_gate_error"]


def test_estimate_gate_is_silent_under_the_limit(client, monkeypatch):
    monkeypatch.setattr(server.cfg, "max_source_duration_sec", 3600)
    d = client.get("/api/estimate",
                   params={"duration_sec": 600, "langs": "es"}).json()
    assert d["duration_gate_error"] is None


def test_estimate_does_not_probe_a_non_url_source(client):
    """Anything that is not http never reaches yt-dlp — keeps this hermetic."""
    d = client.get("/api/estimate",
                   params={"source": "not-a-url", "duration_sec": 90,
                           "langs": "es"}).json()
    assert d["duration_sec"] == 90.0
    assert d["title"] == ""
    assert d["source_type"] == "upload"


def test_estimate_survives_a_non_finite_duration(client):
    """Browsers report Infinity for a <video> duration before metadata loads,
    and duration_sec is read client-side, so this value is not ours to trust.
    It used to reach round()/json.dumps(allow_nan=False) and answer 500."""
    r = client.get("/api/estimate", params={"duration_sec": "inf", "langs": "es"})
    assert r.status_code == 200
    assert r.json()["duration_sec"] == 0


def test_estimate_survives_nan_and_negative_infinity(client):
    """Both already floor to zero — pin it so a clamp fix does not regress."""
    for bad in ("nan", "-inf"):
        d = client.get("/api/estimate",
                       params={"duration_sec": bad, "langs": "es"}).json()
        assert d["duration_sec"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  GET /api/dub/{job_id}/flags
# ═══════════════════════════════════════════════════════════════════════

CHECKPOINT = {
    "target_lang": "ru",
    "source_lang": "en",
    "segments": [
        {"idx": 0, "start": 0.0, "end": 2.0,
         "text": "We flew to Kyoto last spring.",
         "translated_text": "Мы летали в Киото прошлой весной."},
        {"idx": 1, "start": 2.0, "end": 4.0,
         "text": "Kyoto was full of tourists then.",
         "translated_text": "Кёто был полон туристов тогда."},
        {"idx": 2, "start": 4.0, "end": 6.0,
         "text": "I would go back to Kyoto tomorrow.",
         "translated_text": "Я бы вернулся в Киото хоть завтра."},
    ],
}


@pytest.fixture
def flagged(client, monkeypatch):
    """A job whose translation checkpoint yields at least one flag."""
    server.jobs["cp1"] = _job("cp1", status="awaiting_translation_review",
                              target_lang="ru", source_lang="en")
    monkeypatch.setattr(server, "_load_checkpoint",
                        lambda jid, stage: dict(CHECKPOINT)
                        if (jid == "cp1" and stage == "translation_done") else None)
    return client


def test_flags_404s_on_an_unknown_job(client):
    r = client.get("/api/dub/nope/flags")
    assert r.status_code == 404
    assert r.json()["error"] == "Job not found"


def test_flags_404s_when_there_is_no_translation_checkpoint(client, monkeypatch):
    server.jobs["bare"] = _job("bare", status="error")
    monkeypatch.setattr(server, "_load_checkpoint", lambda jid, stage: None)
    r = client.get("/api/dub/bare/flags")
    assert r.status_code == 404
    assert "checkpoint" in r.json()["error"]


def test_a_handler_404_carries_error_not_fastapis_detail(client, monkeypatch):
    """creator.html tells a missing route from a handler's own 404 by this.

    `routeMissing()` falls back to mock data only for FastAPI's unmatched-route
    body, `{"detail": "Not Found"}`. If this route ever answered in that shape
    the review screen would quietly show invented flagged terms for a real job.
    """
    server.jobs["bare"] = _job("bare")
    monkeypatch.setattr(server, "_load_checkpoint", lambda jid, stage: None)
    body = client.get("/api/dub/bare/flags").json()
    assert "error" in body
    assert body.get("detail") != "Not Found"


def test_flags_returns_the_documented_shape(flagged):
    d = flagged.get("/api/dub/cp1/flags").json()
    assert d["job_id"] == "cp1"
    assert d["target_lang"] == "ru"
    assert d["source_lang"] == "en"
    assert d["count"] == len(d["flags"])
    assert d["audio_url"] == "/outputs/cp1/audio_16k.wav"
    for f in d["flags"]:
        assert {"idx", "kind", "start", "end", "source_text",
                "translated_text", "source_span", "target_span",
                "reason", "score", "variants"} <= set(f)


def test_flags_finds_the_inconsistent_name(flagged):
    """Киото / Кёто is the detector the whole review loop leads with."""
    d = flagged.get("/api/dub/cp1/flags").json()
    assert any(f["reason"] == "name_inconsistent" for f in d["flags"])


def test_flags_source_lang_auto_is_not_passed_through(client, monkeypatch):
    """"auto" is a pipeline token, not a language — it must not reach flags."""
    server.jobs["cp2"] = _job("cp2", target_lang="ru", source_lang="auto")
    cp = dict(CHECKPOINT)
    cp.pop("source_lang")
    monkeypatch.setattr(server, "_load_checkpoint", lambda jid, stage: dict(cp))
    assert client.get("/api/dub/cp2/flags").json()["source_lang"] == ""


def test_flags_negative_max_returns_nothing(flagged):
    d = flagged.get("/api/dub/cp1/flags", params={"max_flags": -1}).json()
    assert d["count"] == 0
    assert d["flags"] == []


def test_flags_max_is_clamped_to_twenty(flagged):
    d = flagged.get("/api/dub/cp1/flags", params={"max_flags": 99999}).json()
    assert d["count"] <= 20


def test_flags_rejects_a_non_numeric_max(flagged):
    assert flagged.get("/api/dub/cp1/flags",
                       params={"max_flags": "abc"}).status_code == 422


def test_flags_honours_a_small_cap(flagged):
    assert flagged.get("/api/dub/cp1/flags",
                       params={"max_flags": 1}).json()["count"] <= 1


def test_flags_zero_max_returns_nothing(flagged):
    """Zero is a real request for none. `max_flags or 5` answered it with five."""
    assert flagged.get("/api/dub/cp1/flags",
                       params={"max_flags": 0}).json()["count"] == 0


def test_a_clean_transcript_produces_no_flags(client, monkeypatch):
    """Zero flags is the expected result, not an error state."""
    server.jobs["clean"] = _job("clean", target_lang="ru")
    monkeypatch.setattr(server, "_load_checkpoint", lambda jid, stage: {
        "target_lang": "ru", "source_lang": "en",
        "segments": [{"idx": 0, "start": 0.0, "end": 2.0,
                      "text": "The weather was good.",
                      "translated_text": "Погода была хорошая."}],
    })
    d = client.get("/api/dub/clean/flags").json()
    assert d["count"] == 0
    assert d["flags"] == []


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/glossary/term
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def glossary(client, monkeypatch, tmp_path):
    """Redirect the glossary at a tmp file — never the real presets/ one."""
    path = tmp_path / "user_glossary.json"
    monkeypatch.setattr(server, "USER_GLOSSARY_FILE", path)
    cleared = []
    monkeypatch.setattr(server, "clear_glossary_cache",
                        lambda lang=None: cleared.append(lang))
    client.path = path
    client.cleared = cleared
    return client


def _post(c, **kw):
    return c.post("/api/glossary/term", data=kw)


def test_glossary_term_requires_its_fields(glossary):
    """A missing field gets the readable 400, not a pydantic 422 blob.

    This asserted 422 when it was written — Form(...) made FastAPI reject
    the request before any of the handler's friendlier messages could run,
    so those branches were unreachable and a consumer UI would have had to
    render a validation-error structure it knows nothing about.
    """
    r = _post(glossary)
    assert r.status_code == 400
    assert r.json()["error"] == "term and translation are required"


@pytest.mark.parametrize("field", ["term", "translation"])
def test_glossary_term_caps_length(glossary, field):
    """Every term is rendered into the prompt of every later translation for
    that language, so an unbounded one corrupts translations rather than
    merely filling a disk."""
    payload = {"term": "Kyoto", "translation": "Киото", "target_lang": "ru"}
    payload[field] = "x" * 20_000
    r = _post(glossary, **payload)
    assert r.status_code == 400
    assert "too long" in r.json()["error"]
    assert not glossary.path.exists()


@pytest.mark.parametrize("domain", ["../../etc/passwd", "a" * 65, "x\ty",
                                   "a/b", "a\\b", 'a"b'])
def test_glossary_term_validates_the_domain(glossary, domain):
    r = _post(glossary, term="Kyoto", translation="Киото", target_lang="ru",
              domain=domain)
    assert r.status_code == 400
    assert "Domain must be" in r.json()["error"]


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_domain_means_the_default_one(glossary, blank):
    """Whitespace-only is "not supplied", not an invalid name."""
    r = _post(glossary, term="Kyoto", translation="Киото", target_lang="ru",
              domain=blank)
    assert r.status_code == 200
    assert r.json()["domain"] == server._GLOSSARY_REVIEW_DOMAIN


def test_glossary_term_accepts_an_ordinary_domain_name(glossary):
    r = _post(glossary, term="Kyoto", translation="Киото", target_lang="ru",
              domain="My Channel v2.1")
    assert r.status_code == 200
    assert r.json()["domain"] == "My Channel v2.1"


@pytest.mark.parametrize("field", ["term", "translation", "target_lang"])
def test_glossary_term_rejects_a_blank_field(glossary, field):
    payload = {"term": "Kyoto", "translation": "Киото", "target_lang": "ru"}
    payload[field] = ""
    assert _post(glossary, **payload).status_code in (400, 422)


def test_glossary_term_rejects_an_unknown_language(glossary):
    r = _post(glossary, term="Kyoto", translation="Киото", target_lang="banana")
    assert r.status_code == 400
    assert "banana" in r.json()["error"]


def test_glossary_term_normalises_the_language_code(glossary):
    d = _post(glossary, term="Kyoto", translation="Kioto", target_lang="ES").json()
    assert d["target_lang"] == "es"


def test_glossary_term_writes_the_expected_file_shape(glossary):
    _post(glossary, term="Kyoto", translation="Киото", target_lang="ru")
    data = json.loads(glossary.path.read_text(encoding="utf-8"))
    domain = data["domains"][0]
    assert domain["name"] == server._GLOSSARY_REVIEW_DOMAIN
    assert domain["target_lang"] == "ru"
    assert domain["terms"] == {"Kyoto": "Киото"}
    assert domain["triggers"] == []


def test_glossary_term_merges_rather_than_replacing(glossary):
    """The review screen saves one term at a time; the second must not win alone.

    POST /api/glossary is a whole-file replace, which is why this route exists.
    """
    _post(glossary, term="Kyoto", translation="Киото", target_lang="ru")
    d = _post(glossary, term="Osaka", translation="Осака", target_lang="ru").json()
    assert d["total_terms"] == 2
    terms = json.loads(glossary.path.read_text(encoding="utf-8"))["domains"][0]["terms"]
    assert terms == {"Kyoto": "Киото", "Osaka": "Осака"}


def test_glossary_term_reports_what_it_replaced(glossary):
    _post(glossary, term="Kyoto", translation="Кёто", target_lang="ru")
    d = _post(glossary, term="Kyoto", translation="Киото", target_lang="ru").json()
    assert d["replaced"] == "Кёто"
    assert d["total_terms"] == 1


def test_glossary_term_keeps_languages_in_their_own_domains(glossary):
    _post(glossary, term="Kyoto", translation="Киото", target_lang="ru")
    _post(glossary, term="Kyoto", translation="Kioto", target_lang="es")
    domains = json.loads(glossary.path.read_text(encoding="utf-8"))["domains"]
    assert {d["target_lang"] for d in domains} == {"ru", "es"}


def test_glossary_term_survives_unicode_and_markup(glossary):
    r = _post(glossary, term="Кёто 🍣",
              translation='京都 <script>alert(1)</script>', target_lang="ja")
    assert r.status_code == 200
    terms = json.loads(glossary.path.read_text(encoding="utf-8"))["domains"][0]["terms"]
    assert terms["Кёто 🍣"] == '京都 <script>alert(1)</script>'


def test_glossary_term_clears_the_translator_cache(glossary):
    """Without this the decision does not reach the next language until restart.

    The review screen's "we'll remember this for future videos" is exactly
    this call; it was inert before the route existed.
    """
    _post(glossary, term="Kyoto", translation="Киото", target_lang="ru")
    assert glossary.cleared == ["ru"]


def test_glossary_term_accepts_a_named_domain(glossary):
    d = _post(glossary, term="guard", translation="гард",
              target_lang="ru", domain="bjj").json()
    assert d["domain"] == "bjj"


# ═══════════════════════════════════════════════════════════════════════
#  GET /api/jobs?compact=1  and  GET /api/languages
# ═══════════════════════════════════════════════════════════════════════

def test_compact_jobs_projects_to_the_home_screen_keys(client):
    server.jobs["a"] = _job(
        "a", target_lang="es", source_label="clip.mp4", duration=60,
        error="", batch_id="qt_1", output_url="/outputs/a/dubbed_video.mp4",
        transcript="..." * 5000, context_hint="a very long hint",
        meta={"title": "T", "thumbnail": "u", "description": "x" * 5000},
    )
    row = client.get("/api/jobs", params={"compact": 1}).json()["jobs"][0]
    assert set(row) <= set(server._COMPACT_JOB_KEYS) | {"meta"}
    assert "transcript" not in row
    assert "context_hint" not in row
    # meta["description"] is the thousands-of-chars field this exists to drop.
    assert set(row["meta"]) <= set(server._COMPACT_META_KEYS)
    assert "description" not in row["meta"]


def test_compact_keeps_what_a_failed_card_needs(client):
    server.jobs["e"] = _job("e", status="error", error="Download failed",
                            target_lang="ru")
    row = client.get("/api/jobs", params={"compact": 1}).json()["jobs"][0]
    assert row["status"] == "error"
    assert row["error"] == "Download failed"
    assert row["target_lang"] == "ru"


def test_uncompacted_jobs_still_carry_the_full_payload(client):
    """Pro mode reads fields compact drops — the default must not change."""
    server.jobs["a"] = _job("a", target_lang="es", context_hint="hint",
                            whisper_model="medium", voice_preset="auto")
    row = client.get("/api/jobs").json()["jobs"][0]
    assert row["context_hint"] == "hint"
    assert row["whisper_model"] == "medium"
    assert row["voice_preset"] == "auto"


def test_languages_stays_a_bare_code_list(client):
    """tools/gochidubb_client.py returns this verbatim and the CLI joins it.

    Turning `languages` into [{code, name}] would break the CLI and MCP; the
    display names are additive instead.
    """
    d = client.get("/api/languages").json()
    assert isinstance(d["languages"], list)
    assert all(isinstance(c, str) for c in d["languages"])
    assert ",".join(d["languages"])  # what the CLI does


def test_language_names_cover_every_code(client):
    d = client.get("/api/languages").json()
    assert set(d["names"]) == set(d["languages"])
    assert [c["code"] for c in d["catalog"]] == d["languages"]
    assert all(c["name"] for c in d["catalog"])


def test_estimate_validates_against_the_same_language_set(client):
    """A code the picker offers must be a code the quote accepts."""
    codes = client.get("/api/languages").json()["languages"]
    r = client.get("/api/estimate",
                   params={"langs": ",".join(codes[:5]), "duration_sec": 60})
    assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
#  Which front door: GET / , /pro , /creator
# ═══════════════════════════════════════════════════════════════════════
#
# The highest-consequence change in Creator mode: every failure mode of
# reading the preference must resolve to Pro, which is what / has always
# served. An existing user must never be stranded on a page they did not
# ask for by a missing file, an unreadable one, or a value nobody wrote.

def _is_creator(resp):
    return b"creator.css" in resp.content


@pytest.fixture
def prefs(client, monkeypatch, tmp_path):
    """Point PREFS_FILE at a tmp file — never the real user_prefs.json."""
    path = tmp_path / "user_prefs.json"
    monkeypatch.setattr(server, "PREFS_FILE", path)
    client.path = path
    return client


PREFS_THAT_MUST_SERVE_PRO = [
    pytest.param(None, id="file-missing"),
    pytest.param("", id="empty-file"),
    pytest.param("   ", id="whitespace-only"),
    pytest.param("{}", id="empty-object"),
    pytest.param('{"ui_mode":"pro"}', id="explicitly-pro"),
    pytest.param('{"ui_mode":"CREATOR"}', id="wrong-case"),
    pytest.param('{"ui_mode":" creator "}', id="padded"),
    pytest.param('{"ui_mode":"banana"}', id="unknown-value"),
    pytest.param('{"ui_mode":null}', id="null"),
    pytest.param('{"ui_mode":true}', id="bool"),
    pytest.param('{"ui_mode":["creator"]}', id="list"),
    pytest.param('[{"ui_mode":"creator"}]', id="top-level-array"),
    pytest.param('"creator"', id="top-level-string"),
    pytest.param("{ui_mode: creator", id="unparseable"),
    pytest.param("creator", id="bare-text"),
    pytest.param("{" * 5000, id="huge-junk"),
]


@pytest.mark.parametrize("content", PREFS_THAT_MUST_SERVE_PRO)
def test_root_serves_pro_for_anything_but_an_explicit_creator(prefs, content):
    if content is not None:
        prefs.path.write_text(content, encoding="utf-8")
    r = prefs.get("/")
    assert r.status_code == 200
    assert not _is_creator(r)


def test_root_serves_creator_only_on_the_exact_preference(prefs):
    prefs.path.write_text('{"ui_mode":"creator"}', encoding="utf-8")
    assert _is_creator(prefs.get("/"))


def test_root_falls_back_to_pro_when_the_creator_page_is_missing(
        prefs, monkeypatch, tmp_path):
    """A preference pointing at a page not on disk would be a blank screen."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>pro</html>", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC_DIR", static)
    prefs.path.write_text('{"ui_mode":"creator"}', encoding="utf-8")
    r = prefs.get("/")
    assert r.status_code == 200
    assert b"pro" in r.content


@pytest.mark.parametrize("content", PREFS_THAT_MUST_SERVE_PRO
                         + [pytest.param('{"ui_mode":"creator"}', id="creator")])
def test_pro_is_unconditional(prefs, content):
    """The escape hatch: /pro never reads the preference."""
    if content is not None:
        prefs.path.write_text(content, encoding="utf-8")
    r = prefs.get("/pro")
    assert r.status_code == 200
    assert not _is_creator(r)


@pytest.mark.parametrize("content", PREFS_THAT_MUST_SERVE_PRO)
def test_creator_is_unconditional(prefs, content):
    """Each mode always has a direct address that works."""
    if content is not None:
        prefs.path.write_text(content, encoding="utf-8")
    r = prefs.get("/creator")
    assert r.status_code == 200
    assert _is_creator(r)


def test_the_creator_page_does_not_pull_a_cdn(client):
    """The pitch is that everything runs on your machine.

    index.html loads React and Babel from unpkg, so Pro mode does not work
    offline. The consumer front door must not inherit that.
    """
    body = client.get("/creator").content.decode("utf-8", "replace")
    assert "unpkg.com" not in body
    assert "cdn.jsdelivr" not in body


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/quick_test?background=1 — the preparing status must terminate
# ═══════════════════════════════════════════════════════════════════════
# A job left in "preparing" never reaches a pipeline, never times out, and is
# skipped by bulk delete, so it sits on the home screen claiming to be working
# until the server restarts. Every path out of the background task therefore
# has to reach a terminal status, including the ones nobody planned for.
#
# The task is captured rather than raced. `TestClient` used without its
# context manager tears the event-loop portal down when the request ends,
# which cancels anything still running — measured, a one-millisecond delay
# inside the task is enough to lose. So `_spawn_background` is patched to
# hand the coroutine back, and the test runs it to completion itself. No
# sleeps, no polling, no race.

@pytest.fixture
def background(client, monkeypatch, tmp_path):
    """A quick_test client whose download, queue and task spawn are stubs."""
    monkeypatch.setattr(server, "save_job", lambda j: None)

    async def _no_probe(url):
        return {}
    monkeypatch.setattr(server, "_probe_meta_cached", _no_probe)

    def _download(url, out):
        dest = pathlib.Path(out) / "v.mp4"
        dest.write_bytes(b"\0" * 64)
        return str(dest)
    monkeypatch.setattr(server, "download_video", _download)

    async def _enqueue(jid, args):
        server.jobs[jid]["status"] = "queued"
    monkeypatch.setattr(server, "enqueue_job", _enqueue)
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)

    spawned = []
    monkeypatch.setattr(server, "_spawn_background", spawned.append)
    client.spawned = spawned
    client.uploads = tmp_path
    return client


def _submit(c, langs="es,fr"):
    r = c.post("/api/quick_test", data={
        "source": "https://example.com/v", "target_langs": langs,
        "model": "aya-expanse:8b", "background": "1"})
    assert r.status_code == 200, r.text
    return r.json()["job_ids"]


def _finish(c):
    """Run the captured background task to completion, deterministically."""
    assert len(c.spawned) == 1, f"expected one background task, got {len(c.spawned)}"
    asyncio.run(c.spawned.pop())


def test_background_submit_returns_before_the_download_starts(background):
    ids = _submit(background)
    # The point of background=1: jobs exist and the caller is already gone.
    assert [server.jobs[j]["status"] for j in ids] == ["preparing", "preparing"]
    assert server.jobs[ids[0]]["step_detail"] == "Fetching your video…"


def test_background_submit_reaches_the_queue(background):
    ids = _submit(background)
    _finish(background)
    assert [server.jobs[j]["status"] for j in ids] == ["queued", "queued"]


def test_a_failed_download_does_not_strand_the_batch(background, monkeypatch):
    def _boom(url, out):
        raise RuntimeError("404 not found")
    monkeypatch.setattr(server, "download_video", _boom)
    ids = _submit(background)
    _finish(background)
    assert [server.jobs[j]["status"] for j in ids] == ["error", "error"]
    assert "404 not found" in server.jobs[ids[0]]["error"]


def test_a_crash_after_the_download_does_not_strand_the_batch(
        background, monkeypatch):
    """The defect this section exists to pin: `_prepare` runs detached, so an
    exception in it goes to the event loop's exception handler and nowhere
    near a user. Without an outer guard the jobs sit in `preparing` forever."""
    async def _boom(jid, args):
        raise RuntimeError("queue exploded")
    monkeypatch.setattr(server, "enqueue_job", _boom)
    ids = _submit(background)
    _finish(background)
    assert [server.jobs[j]["status"] for j in ids] == ["error", "error"]
    assert "queue exploded" in server.jobs[ids[0]]["error"]


def test_a_persist_failure_still_frees_every_sibling(background, monkeypatch):
    """A failing save_job must not abort the loop that frees the others —
    the in-memory store is the source of truth during a live run, so the
    status change has already done the important half of the work."""
    async def _boom(jid, args):
        raise RuntimeError("queue exploded")
    monkeypatch.setattr(server, "enqueue_job", _boom)
    ids = _submit(background, langs="es,fr,de")

    def _bad_save(job):
        raise RuntimeError("db exploded")
    monkeypatch.setattr(server, "save_job", _bad_save)
    _finish(background)
    assert [server.jobs[j]["status"] for j in ids] == ["error", "error", "error"]


def test_a_cancelled_task_does_not_strand_the_batch(background, monkeypatch):
    """Server shutdown cancels the task mid-download."""
    async def _hang(*a, **kw):
        raise asyncio.CancelledError()
    monkeypatch.setattr(asyncio, "to_thread", _hang)
    ids = _submit(background)
    with pytest.raises(asyncio.CancelledError):
        _finish(background)
    assert [server.jobs[j]["status"] for j in ids] == ["error", "error"]
    assert "shut down" in server.jobs[ids[0]]["error"]


def test_a_failed_download_leaves_no_empty_directory(background, monkeypatch):
    def _boom(url, out):
        raise RuntimeError("404")
    monkeypatch.setattr(server, "download_video", _boom)
    _submit(background)
    _finish(background)
    assert [p.name for p in background.uploads.glob("qt_*") if p.is_dir()] == []


def test_a_fully_cancelled_batch_discards_its_download(background):
    """Cancelled while the download ran: nothing will read the file, and
    uploads/ has no other sweeper."""
    ids = _submit(background)
    for jid in ids:
        assert background.post(f"/api/dub/{jid}/cancel").json()["ok"] is True
    _finish(background)
    assert [p.name for p in background.uploads.glob("qt_*") if p.is_dir()] == []
    assert [server.jobs[j]["status"] for j in ids] == ["cancelled", "cancelled"]


def test_a_partly_cancelled_batch_still_starts_the_rest(background):
    ids = _submit(background)
    background.post(f"/api/dub/{ids[0]}/cancel")
    _finish(background)
    assert [server.jobs[j]["status"] for j in ids] == ["cancelled", "queued"]


def test_a_half_created_batch_is_not_left_in_the_job_table(
        background, monkeypatch):
    """save_job failing during creation used to leave the siblings it had
    already registered stuck in `preparing`, with only a 500 to show for it."""
    def _bad_save(job):
        raise RuntimeError("db exploded")
    monkeypatch.setattr(server, "save_job", _bad_save)
    before = set(server.jobs)
    r = background.post("/api/quick_test", data={
        "source": "https://example.com/v", "target_langs": "es,fr",
        "model": "aya-expanse:8b", "background": "1"})
    assert r.status_code == 500
    assert "error" in r.json()
    assert set(server.jobs) == before
    assert background.spawned == []        # nothing was spawned to leak


def test_cancelling_a_preparing_job_takes_effect_at_once(client):
    """Cancel used to sit until the download finished — leaving the card on
    "preparing", a status nothing can delete, for the length of a download.

    Driven against the route directly rather than a live background task:
    TestClient without its context manager tears the event-loop portal down
    when the request ends, which cancels anything create_task spawned, so a
    genuinely slow download cannot be held open across two requests here.
    """
    server.jobs["prep1"] = _job("prep1", status="preparing",
                                step_detail="Fetching your video…")
    r = client.post("/api/dub/prep1/cancel")
    assert r.json() == {"ok": True, "cancelled_from": "preparing"}
    assert server.jobs["prep1"]["status"] == "cancelled"
    # Terminal, so the card stops saying "preparing" and can be deleted.
    assert "cancelled" not in server._UNDELETABLE_STATUSES
    # And the background task will skip it rather than enqueue it.
    assert server.jobs["prep1"]["cancel_requested"] is True


def test_bulk_delete_skips_a_preparing_job_but_single_delete_does_not(client):
    """Worth pinning precisely, because it is easy to overstate.

    `_UNDELETABLE_STATUSES` gates only POST /api/jobs/bulk_delete —
    DELETE /api/job/{id} removes a job whatever its status. So a job stuck
    in "preparing" is skipped by the bulk path and keeps claiming to be
    working, but it is not unremovable. That is what makes it a defect
    rather than an emergency.
    """
    assert "preparing" in server._UNDELETABLE_STATUSES
    server.jobs["prep2"] = _job("prep2", status="preparing")
    r = client.post("/api/jobs/bulk_delete", data={"job_ids": "prep2"})
    assert r.json()["skipped"][0]["reason"].startswith("job is preparing")
    assert "prep2" in server.jobs

    assert client.delete("/api/job/prep2").status_code == 200
    assert "prep2" not in server.jobs


def test_discard_download_removes_the_file_and_its_directory(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    d = tmp_path / "qt_deadbeef"
    d.mkdir()
    f = d / "v.mp4"
    f.write_bytes(b"\0" * 8)
    server._discard_download(f)
    assert not f.exists()
    assert not d.exists()


def test_discard_download_never_touches_a_user_upload(tmp_path, monkeypatch):
    """It must only remove the per-download directory this module creates —
    an uploaded file sits directly in uploads/ and is not ours to delete."""
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    other = tmp_path / "someone_elses.mp4"
    other.write_bytes(b"\0" * 8)
    server._discard_download(other)
    assert tmp_path.exists()          # uploads/ itself survives
    keeper = tmp_path / "qt_keepme"
    keeper.mkdir()
    (keeper / "v.mp4").write_bytes(b"\0")
    (keeper / "other.mp4").write_bytes(b"\0")
    server._discard_download(keeper / "v.mp4")
    # A directory with anything else still in it is left alone.
    assert keeper.exists() and (keeper / "other.mp4").exists()
