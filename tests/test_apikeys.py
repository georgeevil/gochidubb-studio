"""Tests for the API key store (app/apikeys.py).

The security-relevant promises are the ones worth pinning down: the plaintext
token is never recoverable from the file, revoked and expired keys stop
verifying, and scope validation rejects nonsense before a key is minted.
"""
import time

import pytest

from app import apikeys


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the store at a temp file so tests never touch the real one."""
    monkeypatch.setattr(apikeys, "KEYS_FILE", tmp_path / "apikeys.json")
    yield


def test_create_returns_token_once_and_stores_only_a_hash():
    rec, token = apikeys.create("ci", ["dub:write"])
    assert token.startswith(apikeys.PREFIX_LIVE)

    # The public record never carries the secret material.
    assert "hash" not in rec and "salt" not in rec
    assert token not in str(rec)

    # Nor does the file on disk.
    raw = apikeys.KEYS_FILE.read_text(encoding="utf-8")
    assert token not in raw

    # And the token is not recoverable from any listing.
    assert token not in str(apikeys.list_keys())


def test_verify_accepts_the_token_and_rejects_others():
    _, token = apikeys.create("ci", ["dub:write"])
    assert apikeys.verify(token)["name"] == "ci"
    assert apikeys.verify(token + "x") is None
    assert apikeys.verify("") is None
    assert apikeys.verify("gcd_live_nonsense") is None


def test_masked_shows_only_the_tail():
    rec, token = apikeys.create("ci", ["dub:write"])
    assert rec["masked"].endswith(token[-4:])
    # Four characters is not enough to reconstruct anything.
    assert len(rec["masked"]) < len(token)


def test_revoked_keys_stop_verifying():
    rec, token = apikeys.create("ci", ["dub:write"])
    assert apikeys.verify(token)
    assert apikeys.revoke(rec["id"]) is True
    assert apikeys.verify(token) is None
    # Revoking twice is not an error the caller should act on.
    assert apikeys.revoke(rec["id"]) is False
    # The record survives, so the list stays honest about what existed.
    assert any(k["id"] == rec["id"] for k in apikeys.list_keys())


def test_expired_keys_stop_verifying():
    rec, token = apikeys.create("ci", ["dub:write"], expires_days=1)
    assert apikeys.verify(token)
    assert apikeys.is_active(dict(rec, expires=time.time() - 1)) is False
    # Verified through the store, not just the helper.
    apikeys.KEYS_FILE.write_text(
        apikeys.KEYS_FILE.read_text(encoding="utf-8").replace(
            str(rec["expires"]), str(time.time() - 10)),
        encoding="utf-8")
    assert apikeys.verify(token) is None


def test_scope_validation():
    with pytest.raises(ValueError):
        apikeys.create("bad", ["not:a:scope"])
    with pytest.raises(ValueError):
        apikeys.create("bad", [])
    with pytest.raises(ValueError):
        apikeys.create("   ", ["dub:write"])
    with pytest.raises(ValueError):
        apikeys.create("bad", ["dub:write"], environment="staging")


def test_scopes_are_deduped_and_sorted():
    rec, _ = apikeys.create("ci", ["jobs:read", "dub:write", "jobs:read"])
    assert rec["scopes"] == ["dub:write", "jobs:read"]


def test_has_scope():
    rec, _ = apikeys.create("ci", ["dub:write"])
    assert apikeys.has_scope(rec, "dub:write")
    assert not apikeys.has_scope(rec, "webhooks:manage")
    assert not apikeys.has_scope(None, "dub:write")


def test_test_environment_uses_its_own_prefix():
    _, token = apikeys.create("staging", ["jobs:read"], environment="test")
    assert token.startswith(apikeys.PREFIX_TEST)


def test_legacy_records_without_environment_list_as_live():
    """Keys minted before the environment split must render as live."""
    import json

    rec, _ = apikeys.create("old-timer", ["dub:write"])
    raw = json.loads(apikeys.KEYS_FILE.read_text(encoding="utf-8"))
    for r in raw:
        r.pop("environment", None)
    apikeys.KEYS_FILE.write_text(json.dumps(raw), encoding="utf-8")

    (listed,) = apikeys.list_keys()
    assert listed["environment"] == "live"
    # The stored record itself stays untouched — only the projection defaults.
    on_disk = json.loads(apikeys.KEYS_FILE.read_text(encoding="utf-8"))
    assert "environment" not in on_disk[0]


def test_last_used_is_recorded_on_verify():
    rec, token = apikeys.create("ci", ["dub:write"])
    assert rec["last_used"] is None
    apikeys.verify(token)
    assert apikeys.list_keys()[0]["last_used"] is not None


def test_two_keys_are_independent():
    _, a = apikeys.create("a", ["dub:write"])
    rec_b, b = apikeys.create("b", ["jobs:read"])
    apikeys.revoke(rec_b["id"])
    assert apikeys.verify(a) is not None
    assert apikeys.verify(b) is None
