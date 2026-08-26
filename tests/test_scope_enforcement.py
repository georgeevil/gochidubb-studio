"""Route-level tests for hosted-mode API-key scope enforcement (CLD-249).

app/apikeys.py's verify() and scope checks were unit-tested from the day the
keys screen shipped, but no route ever rejected a bad key — enforcement is the
`_enforce_api_scopes` middleware in server.py, and these tests exercise it the
only way that counts: through real requests.

The contract under test:
  * local mode (the default): every route open, exactly as it always was.
  * hosted mode: routes the scope table claims require `Authorization:
    Bearer <key>`; a missing/unknown/expired/revoked key is 401, a valid key
    without the needed scope is 403, and a valid key with it passes through
    to the route.
  * loopback callers are exempt even in hosted mode — the operator on the
    box (and the session-less browser UI) must never be locked out.

Uses TestClient(app) WITHOUT the context manager on purpose — see
tests/test_creator_routes.py for why (no lifespan → no real DB writes).
TestClient requests arrive from host "testclient", which is not loopback, so
enforcement actually fires here.
"""
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import server
from app import apikeys as app_apikeys


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
def keystore(tmp_path, monkeypatch):
    """An isolated apikeys.json so tests never touch the real one."""
    monkeypatch.setattr(app_apikeys, "KEYS_FILE", tmp_path / "apikeys.json")
    yield


@pytest.fixture
def hosted(monkeypatch, keystore):
    monkeypatch.setattr(server.cfg, "mode", "hosted")
    yield


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


# ── the scope table ──────────────────────────────────────────────────

def test_scope_table_maps_the_agent_surface():
    f = server._scope_for
    assert f("POST", "/api/dub") == "dub:write"
    assert f("POST", "/api/quick_test") == "dub:write"
    assert f("POST", "/api/showcase") == "dub:write"
    assert f("POST", "/api/job/j1/redub") == "dub:write"
    assert f("POST", "/api/dub/j1/cancel") == "dub:write"
    assert f("DELETE", "/api/job/j1") == "dub:write"
    assert f("GET", "/api/jobs") == "jobs:read"
    assert f("GET", "/api/job/j1") == "jobs:read"
    assert f("GET", "/api/dub/j1/flags") == "jobs:read"
    assert f("GET", "/outputs/j1/dubbed.mp4") == "outputs:read"
    assert f("POST", "/api/webhooks") == "webhooks:manage"
    assert f("GET", "/api/webhooks") == "webhooks:manage"
    assert f("POST", "/api/voice_presets") == "voices:write"
    # Reading voices is not a write, and has no scope of its own.
    assert f("GET", "/api/voices") is None
    # Routes outside the table stay open — the browser UI's surface.
    assert f("GET", "/api/system") is None
    assert f("GET", "/api/config") is None
    assert f("GET", "/pro") is None


# ── local mode: nothing enforced ─────────────────────────────────────

def test_local_mode_stays_unauthenticated(client, keystore, monkeypatch):
    monkeypatch.setattr(server.cfg, "mode", "local")
    r = client.get("/api/jobs")
    assert r.status_code == 200
    # Even a nonsense key is ignored rather than rejected.
    r = client.get("/api/jobs", headers=_bearer("gcd_live_nonsense"))
    assert r.status_code == 200


# ── hosted mode ──────────────────────────────────────────────────────

def test_hosted_rejects_a_missing_key(client, hosted):
    r = client.get("/api/jobs")
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"
    assert "API key" in r.json()["error"]


def test_hosted_rejects_an_unknown_key(client, hosted):
    r = client.get("/api/jobs", headers=_bearer("gcd_live_nonsense"))
    assert r.status_code == 401


def test_hosted_accepts_a_valid_key_with_the_scope(client, hosted):
    _, token = app_apikeys.create("reader", ["jobs:read"])
    r = client.get("/api/jobs", headers=_bearer(token))
    assert r.status_code == 200


def test_hosted_rejects_a_valid_key_without_the_scope(client, hosted):
    rec, token = app_apikeys.create("writer-only", ["dub:write"])
    r = client.get("/api/jobs", headers=_bearer(token))
    assert r.status_code == 403
    assert "jobs:read" in r.json()["error"]
    # And the mirror image: a read key cannot start work. 403 comes from the
    # middleware before the route ever parses the (empty) form.
    _, rt = app_apikeys.create("reader", ["jobs:read"])
    r = client.post("/api/dub", headers=_bearer(rt))
    assert r.status_code == 403


def test_hosted_write_key_reaches_the_route(client, hosted):
    _, token = app_apikeys.create("writer", ["dub:write"])
    # No source in the form, so the route itself complains — the point is
    # that the middleware let the request through (not 401/403).
    r = client.post("/api/dub", headers=_bearer(token))
    assert r.status_code not in (401, 403)


def test_hosted_rejects_a_revoked_key(client, hosted):
    rec, token = app_apikeys.create("doomed", ["jobs:read"])
    assert client.get("/api/jobs", headers=_bearer(token)).status_code == 200
    app_apikeys.revoke(rec["id"])
    assert client.get("/api/jobs", headers=_bearer(token)).status_code == 401


def test_hosted_rejects_an_expired_key(client, hosted):
    rec, token = app_apikeys.create("expiring", ["jobs:read"],
                                    expires_days=1)
    # Age the stored record past its expiry rather than sleeping.
    with app_apikeys._lock:
        recs = app_apikeys._read()
        for r_ in recs:
            if r_["id"] == rec["id"]:
                r_["expires"] = time.time() - 1
        app_apikeys._write(recs)
    assert client.get("/api/jobs", headers=_bearer(token)).status_code == 401


def test_hosted_leaves_unmapped_routes_open(client, hosted):
    # The browser UI's surface (system status, config) needs no key even in
    # hosted mode — accounts/sessions are §9 of the plan, out of scope.
    assert client.get("/api/system").status_code == 200


def test_hosted_exempts_loopback_callers(hosted):
    """The operator's own box can never be locked out.

    TestClient can't spoof its peer address, so this goes through httpx's
    ASGI transport, which can.
    """
    import asyncio

    async def _call(host):
        transport = httpx.ASGITransport(app=server.app, client=(host, 1234))
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t") as c:
            return (await c.get("/api/jobs")).status_code

    assert asyncio.run(_call("127.0.0.1")) == 200
    assert asyncio.run(_call("::1")) == 200
    assert asyncio.run(_call("203.0.113.9")) == 401
