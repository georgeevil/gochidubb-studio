"""Route-level tests for the vendor admin console's endpoints.

Same hermetic rules as `tests/test_creator_routes.py`, and for the same
reasons — see that module's docstring:

* `TestClient(app)` **without** its context manager, so the lifespan never
  runs, `app.db._DB_PATH` stays None and `save_job_sync` is a no-op. A test
  run cannot touch the real `gochidubb.db`.
* No network. Nothing here submits a job or probes a URL.

Two things specific to this suite:

* `app.apikeys` reads the real `apikeys.json` at the project root. Every test
  that cares about keys patches `app_apikeys.list_keys`, so a developer's own
  keys neither leak into an assertion nor get revoked by one.
* `/api/admin/overview`, `/revenue` and `/revenue.csv` call `storage_stats()`,
  which walks `outputs/`. With `server.jobs` emptied by the fixture that walk
  visits nothing, so it stays fast and touches no real output directory.
"""
import csv
import io
import time

import pytest
from fastapi.testclient import TestClient

import server
from app import admin as app_admin
from app import apikeys as app_apikeys


@pytest.fixture
def client():
    """A client over the real app with an empty, isolated job table."""
    saved = dict(server.jobs)
    server.jobs.clear()
    server._admin_stage_cache.clear()
    try:
        yield TestClient(server.app)
    finally:
        server.jobs.clear()
        server.jobs.update(saved)
        server._admin_stage_cache.clear()


@pytest.fixture
def no_keys(monkeypatch):
    """Never read (or write) the developer's real apikeys.json."""
    monkeypatch.setattr(app_apikeys, "list_keys", lambda: [])
    return []


def _job(jid, **kw):
    job = {"id": jid, "status": "complete", "progress": 100,
           "duration": 600.0, "target_lang": "es", "created": time.time()}
    job.update(kw)
    return job


# ═══════════════════════════════════════════════════════════════════════
#  GET /admin — the page
# ═══════════════════════════════════════════════════════════════════════

def test_admin_page_is_served_unconditionally(client):
    """Like /pro and /creator: a direct address that always works."""
    r = client.get("/admin")
    assert r.status_code == 200
    assert "admin.css" in r.text


def test_admin_page_loads_nothing_from_a_cdn(client):
    """Same rule as creator.html — the page must paint offline.

    A product whose pitch is "everything runs on your machine" cannot have
    its operations console blank out when unpkg is unreachable.
    """
    body = client.get("/admin").text
    for host in ("unpkg.com", "cdn.jsdelivr", "cdnjs.", "fonts.googleapis",
                 "fonts.gstatic"):
        assert host not in body, host


# ═══════════════════════════════════════════════════════════════════════
#  GET /api/admin/overview
# ═══════════════════════════════════════════════════════════════════════

def test_overview_renders_on_an_empty_install(client, no_keys):
    d = client.get("/api/admin/overview").json()
    assert d["window_days"] == 30
    assert len(d["tiles"]) == 5
    assert d["attention"] == []
    assert d["top_sources"] == []
    assert d["estimate"] is True
    assert "bills nobody" in d["disclaimer"]


def test_overview_counts_real_jobs(client, no_keys):
    server.jobs["a"] = _job("a", duration=600.0)
    server.jobs["b"] = _job("b", duration=1200.0, status="error")
    d = client.get("/api/admin/overview").json()
    tiles = {t["key"]: t for t in d["tiles"]}
    assert tiles["minutes"]["value"] == 10.0      # the failed job meters nothing
    assert tiles["jobs"]["value"] == 2
    assert tiles["success"]["value"] == 50.0


def test_overview_window_is_honoured_and_clamped(client, no_keys):
    assert client.get("/api/admin/overview",
                      params={"window_days": 90}).json()["window_days"] == 90
    # 0 falls back to the 30-day default, the same `int(x or 30)` idiom
    # /api/billing/usage uses; a negative would make `since` later than `now`
    # and hide everything, so it floors to one day instead.
    assert client.get("/api/admin/overview",
                      params={"window_days": 0}).json()["window_days"] == 30
    assert client.get("/api/admin/overview",
                      params={"window_days": -5}).json()["window_days"] == 1


def test_overview_reports_the_deployment_mode(client, no_keys):
    assert client.get("/api/admin/overview").json()["mode"] in ("local", "hosted")


def test_overview_survives_a_job_with_an_unreadable_duration(client, no_keys):
    """A probe that could not read a stream must not 500 the whole console."""
    server.jobs["bad"] = _job("bad", duration=float("nan"))
    server.jobs["worse"] = _job("worse", duration="N/A")
    r = client.get("/api/admin/overview")
    assert r.status_code == 200
    assert r.json()["current"]["minutes"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  GET /api/admin/accounts and /account/{id}
# ═══════════════════════════════════════════════════════════════════════

def test_accounts_returns_one_workspace_and_the_keys(client, monkeypatch):
    monkeypatch.setattr(app_apikeys, "list_keys", lambda: [
        {"id": "k1", "name": "ci", "environment": "live", "scopes": ["jobs:read"],
         "created": time.time() - 86400, "expires": None, "last_used": None,
         "revoked": None, "masked": "gcd_live_••••abcd"},
    ])
    server.jobs["a"] = _job("a")
    d = client.get("/api/admin/accounts").json()
    assert d["workspace"]["minutes"] == 10.0
    assert [k["id"] for k in d["keys"]] == ["k1"]
    assert d["counts"]["total"] == 1
    assert d["spend_attributable"] is False


def test_account_detail_404s_for_an_unknown_key(client, no_keys):
    r = client.get("/api/admin/account/nope")
    assert r.status_code == 404
    assert r.json()["error"]


def test_account_detail_carries_the_scope_vocabulary_and_audit_trail(
        client, monkeypatch):
    monkeypatch.setattr(app_apikeys, "list_keys", lambda: [
        {"id": "k1", "name": "ci", "environment": "live",
         "scopes": ["jobs:read"], "created": 0, "expires": None,
         "last_used": None, "revoked": None, "masked": "gcd_live_••••abcd"},
    ])
    monkeypatch.setattr(server.app_audit, "recent", lambda limit=200: [
        {"ts": 1.0, "action": "apikey.create", "target": "k1", "actor": "local"},
        {"ts": 2.0, "action": "apikey.create", "target": "other", "actor": "local"},
    ])
    d = client.get("/api/admin/account/k1").json()
    assert d["key"]["name"] == "ci"
    assert d["state"] == "idle"
    assert d["scopes"] == app_apikeys.SCOPES
    assert [e["target"] for e in d["audit"]] == ["k1"]


def test_account_detail_never_returns_a_usable_secret(client, monkeypatch):
    """`list_keys` strips hash and salt; the console must not put them back."""
    monkeypatch.setattr(app_apikeys, "list_keys", lambda: [
        {"id": "k1", "name": "ci", "environment": "live", "scopes": [],
         "created": 0, "expires": None, "last_used": None, "revoked": None,
         "masked": "gcd_live_••••abcd"},
    ])
    body = client.get("/api/admin/account/k1").text
    assert "salt" not in body and "hash" not in body


# ═══════════════════════════════════════════════════════════════════════
#  GET /api/admin/revenue and revenue.csv
# ═══════════════════════════════════════════════════════════════════════

def test_revenue_publishes_the_full_rate_card(client):
    d = client.get("/api/admin/revenue").json()
    rates = [b["rate"] for b in d["rate_card"]["bands"]]
    assert rates == [0.080, 0.065, 0.050]
    assert d["adjustments"]["available"] is False
    assert "bills nobody" in d["disclaimer"]


def test_revenue_csv_is_a_download_with_a_header_row(client):
    server.jobs["a"] = _job("a")
    r = client.get("/api/admin/revenue.csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    assert ".csv" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0][0] == "job_id"
    assert rows[1][0] == "a"


def test_revenue_csv_is_empty_but_valid_with_no_jobs(client):
    rows = list(csv.reader(io.StringIO(client.get("/api/admin/revenue.csv").text)))
    assert len(rows) == 1 and rows[0][0] == "job_id"


# ═══════════════════════════════════════════════════════════════════════
#  GET /api/admin/fleet
# ═══════════════════════════════════════════════════════════════════════

def test_fleet_reports_the_real_queue_depth(client, no_keys, monkeypatch):
    class _Q:
        def qsize(self): return 3
    monkeypatch.setattr(server, "_job_queue", _Q())
    tiles = {t["key"]: t for t in client.get("/api/admin/fleet").json()["tiles"]}
    assert tiles["queue"]["value"] == 3


def test_fleet_reports_zero_depth_when_the_queue_does_not_exist_yet(
        client, no_keys, monkeypatch):
    """The lifespan never runs under TestClient, so `_job_queue` is None."""
    monkeypatch.setattr(server, "_job_queue", None)
    tiles = {t["key"]: t for t in client.get("/api/admin/fleet").json()["tiles"]}
    assert tiles["queue"]["value"] == 0


def test_fleet_includes_the_review_queue(client, no_keys):
    server.jobs["v"] = _job("v", voice_mode="upload")
    d = client.get("/api/admin/fleet").json()
    assert any(i["kind"] == "voice_clone" for i in d["review_queue"])


# ═══════════════════════════════════════════════════════════════════════
#  POST /api/admin/intake — the design's "Pause new jobs"
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def gate(monkeypatch):
    """A real asyncio.Event standing in for the one the lifespan creates."""
    import asyncio
    ev = asyncio.Event()
    ev.set()
    monkeypatch.setattr(server, "_intake_gate", ev)
    monkeypatch.setattr(server, "_intake_changed", asyncio.Event())
    monkeypatch.setattr(server, "_intake_paused_since", 0.0)
    return ev


def test_intake_pause_and_resume_move_the_gate(client, gate, no_keys):
    d = client.post("/api/admin/intake", data={"paused": "true"}).json()
    assert d["paused"] is True and d["since"] > 0
    assert not gate.is_set()
    assert server._intake_is_paused() is True

    d = client.post("/api/admin/intake", data={"paused": "false"}).json()
    assert d["paused"] is False and d["since"] == 0.0
    assert gate.is_set()
    assert server._intake_is_paused() is False


def test_pausing_twice_keeps_the_original_timestamp(client, gate, no_keys):
    """A second click is not a new pause; the "since" clock must not restart."""
    first = client.post("/api/admin/intake", data={"paused": "true"}).json()["since"]
    second = client.post("/api/admin/intake", data={"paused": "true"}).json()["since"]
    assert first == second


def test_fleet_reflects_a_pause(client, gate, no_keys):
    client.post("/api/admin/intake", data={"paused": "true"})
    d = client.get("/api/admin/fleet").json()
    assert d["intake_paused"] is True
    assert d["status"] == "warn"
    assert "intake paused" in d["status_text"]


def test_intake_is_recorded_in_the_audit_log(client, gate, no_keys, monkeypatch):
    """`audit.record` appends to a real file; capture it instead."""
    seen = []
    monkeypatch.setattr(server.app_audit, "record",
                        lambda action, **kw: seen.append((action, kw)))
    client.post("/api/admin/intake", data={"paused": "true"})
    assert seen and seen[0][0] == "admin.intake"
    assert seen[0][1]["detail"] == "paused"


def test_intake_503s_when_the_queue_is_not_running(client, no_keys, monkeypatch):
    monkeypatch.setattr(server, "_intake_gate", None)
    r = client.post("/api/admin/intake", data={"paused": "true"})
    assert r.status_code == 503


def test_pause_does_not_survive_a_restart(client, gate, no_keys):
    """A stuck pause after a crash would silently strand every queued job.

    The lifespan re-creates the gate open every time; this asserts the
    default rather than trusting the comment that says so.
    """
    client.post("/api/admin/intake", data={"paused": "true"})
    import asyncio
    fresh = asyncio.Event()
    fresh.set()                       # what the lifespan does on startup
    assert fresh.is_set()


# ═══════════════════════════════════════════════════════════════════════
#  The queue worker's admission gate
# ═══════════════════════════════════════════════════════════════════════

def test_worker_waits_before_dequeuing_while_paused(monkeypatch):
    """A paused worker must not hold a job it has already taken.

    Checking the gate *after* the get() would leave a job out of `qsize()`
    while it was plainly still waiting, so the console's queue depth would
    under-report for as long as the pause lasted. This drives the real
    worker loop and asserts the queue is untouched while the gate is clear.
    """
    import asyncio

    async def scenario():
        queue = asyncio.Queue()
        gate = asyncio.Event()        # clear == paused
        monkeypatch.setattr(server, "_job_queue", queue)
        monkeypatch.setattr(server, "_intake_gate", gate)

        ran = []

        async def fake_pipeline(job_id, **kw):
            ran.append(job_id)

        monkeypatch.setattr(server, "run_pipeline", fake_pipeline)
        await queue.put(("j1", {}))

        worker = asyncio.create_task(server._job_queue_worker())
        # Give the worker every chance to misbehave.
        for _ in range(20):
            await asyncio.sleep(0)
        assert queue.qsize() == 1, "a paused worker dequeued anyway"
        assert ran == []

        gate.set()
        await asyncio.wait_for(queue.join(), timeout=5)
        assert ran == ["j1"]

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())


def test_pausing_an_idle_worker_still_stops_the_next_job(monkeypatch):
    """The hole a plain `await gate.wait()` before `get()` leaves open.

    An idle worker is already parked *inside* `_job_queue.get()`, which
    returns the instant something is enqueued — so a job submitted after the
    pause would start anyway while the console went on saying intake was
    paused. This is the case that made the extra `_intake_changed` race
    worth its lines.
    """
    import asyncio

    async def scenario():
        queue = asyncio.Queue()
        gate = asyncio.Event(); gate.set()
        changed = asyncio.Event()
        monkeypatch.setattr(server, "_job_queue", queue)
        monkeypatch.setattr(server, "_intake_gate", gate)
        monkeypatch.setattr(server, "_intake_changed", changed)

        ran = []

        async def fake_pipeline(job_id, **kw):
            ran.append(job_id)

        monkeypatch.setattr(server, "run_pipeline", fake_pipeline)

        worker = asyncio.create_task(server._job_queue_worker())
        for _ in range(10):
            await asyncio.sleep(0)        # let it park inside get()

        gate.clear(); changed.set()       # what POST /api/admin/intake does
        for _ in range(10):
            await asyncio.sleep(0)

        await queue.put(("j1", {}))       # a job arrives during the pause
        for _ in range(30):
            await asyncio.sleep(0)
        assert ran == [], "a paused worker started a job anyway"
        assert queue.qsize() == 1, "the job was taken off the queue while paused"

        gate.set(); changed.set()
        await asyncio.wait_for(queue.join(), timeout=5)
        assert ran == ["j1"]

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())


def test_worker_gate_is_cancellable_while_paused(monkeypatch):
    """Shutdown must not hang behind a pause somebody forgot to lift."""
    import asyncio

    async def scenario():
        monkeypatch.setattr(server, "_job_queue", asyncio.Queue())
        monkeypatch.setattr(server, "_intake_gate", asyncio.Event())  # paused
        worker = asyncio.create_task(server._job_queue_worker())
        for _ in range(10):
            await asyncio.sleep(0)
        worker.cancel()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True),
                               timeout=5)

    asyncio.run(scenario())


# ═══════════════════════════════════════════════════════════════════════
#  Stage sampling
# ═══════════════════════════════════════════════════════════════════════

def test_stage_samples_split_recent_from_baseline(monkeypatch, tmp_path):
    now = time.time()
    server.jobs.clear()
    server._admin_stage_cache.clear()
    server.jobs["new"] = _job("new", created=now - 3600)
    server.jobs["old"] = _job("old", created=now - 4 * 86400)

    for jid in ("new", "old"):
        (tmp_path / jid).mkdir()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "load_metrics", lambda work: {
        "stages": {"tts": {"status": "ok", "duration_sec": 12.0}}})

    # 12s of stage over a 600s source = 0.02x realtime, in both windows.
    recent, baseline = server._admin_stage_samples(now)
    assert recent == {"tts": [0.02]}
    assert baseline == {"tts": [0.02]}
    server.jobs.clear()
    server._admin_stage_cache.clear()


def test_stage_samples_are_normalised_by_source_length(monkeypatch, tmp_path):
    """A long video and a short one at the same speed must sample the same.

    Raw stage seconds mostly measure how long the video was, so a raw p95
    comparison flags every stage on any day whose videos ran longer than last
    week's. Measured on a real install that was a wall of eight false "3-34x
    degraded" alarms, which is worse than no signal at all.
    """
    now = time.time()
    server.jobs.clear()
    server._admin_stage_cache.clear()
    server.jobs["long"] = _job("long", duration=3600.0, created=now - 60)
    server.jobs["short"] = _job("short", duration=60.0, created=now - 120)
    for jid in ("long", "short"):
        (tmp_path / jid).mkdir()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    # Both ran the stage at 2x realtime; only the absolute seconds differ.
    monkeypatch.setattr(server, "load_metrics", lambda work: {"stages": {
        "tts": {"status": "ok",
                "duration_sec": server.jobs[work.name]["duration"] * 2}}})

    recent, _base = server._admin_stage_samples(now)
    assert recent == {"tts": [2.0, 2.0]}
    server.jobs.clear()
    server._admin_stage_cache.clear()


def test_stage_samples_skip_a_job_with_no_measured_duration(monkeypatch, tmp_path):
    """Nothing to normalise by — contribute nothing rather than divide by zero."""
    now = time.time()
    server.jobs.clear()
    server._admin_stage_cache.clear()
    server.jobs["a"] = _job("a", duration=0.0, created=now - 60)
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "load_metrics", lambda work: {
        "stages": {"tts": {"status": "ok", "duration_sec": 12.0}}})

    assert server._admin_stage_samples(now) == ({}, {})
    server.jobs.clear()
    server._admin_stage_cache.clear()


def test_stage_samples_ignore_failed_stage_runs(monkeypatch, tmp_path):
    """A stage that raised after four seconds did not get faster."""
    now = time.time()
    server.jobs.clear()
    server._admin_stage_cache.clear()
    server.jobs["a"] = _job("a", created=now - 60)
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "load_metrics", lambda work: {"stages": {
        "tts": {"status": "error", "duration_sec": 4.0},
        "translate": {"status": "ok", "duration_sec": 30.0},
    }})
    recent, _base = server._admin_stage_samples(now)
    assert recent == {"translate": [0.05]}       # 30s over a 600s source
    server.jobs.clear()
    server._admin_stage_cache.clear()


def test_stage_samples_are_cached(monkeypatch, tmp_path):
    """The fleet screen polls; the walk must not run per request."""
    now = time.time()
    server.jobs.clear()
    server._admin_stage_cache.clear()
    server.jobs["a"] = _job("a", created=now - 60)
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)

    calls = []
    def counting(work):
        calls.append(work)
        return {"stages": {"tts": {"status": "ok", "duration_sec": 5.0}}}
    monkeypatch.setattr(server, "load_metrics", counting)

    server._admin_stage_samples(now)
    server._admin_stage_samples(now + 1)
    assert len(calls) == 1
    server._admin_stage_samples(now + server._ADMIN_STAGE_TTL + 1)
    assert len(calls) == 2
    server.jobs.clear()
    server._admin_stage_cache.clear()


# ═══════════════════════════════════════════════════════════════════════
#  Cross-checks against the aggregation module
# ═══════════════════════════════════════════════════════════════════════

def test_routes_and_module_agree_on_the_disclaimer(client, no_keys):
    wanted = app_admin.billing_disclaimer()
    for path in ("/api/admin/overview", "/api/admin/revenue"):
        assert client.get(path).json()["disclaimer"] == wanted


def test_no_admin_route_invents_a_figure_without_labelling_it(client, no_keys):
    """Anything carrying money must also carry `estimate: true`."""
    for path in ("/api/admin/overview", "/api/admin/accounts", "/api/admin/revenue"):
        assert client.get(path).json()["estimate"] is True
