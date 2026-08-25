"""Tests for app/secrets.py — file-backed secret store with env override."""
import json
import os
import stat

import pytest

import app.secrets as secrets_mod
from app.secrets import KNOWN_SECRETS, SecretsStore


@pytest.fixture
def store(tmp_path):
    return SecretsStore(tmp_path / "secrets.json")


@pytest.fixture
def no_env(monkeypatch):
    """Ensure env overrides are absent so file values are visible."""
    for env_k in KNOWN_SECRETS.values():
        monkeypatch.delenv(env_k, raising=False)


# ── set / get / status roundtrip ─────────────────────────────────────

def test_roundtrip(store, no_env):
    assert store.get("vk_access_token") == ""
    store.set("vk_access_token", "tok123")
    assert store.get("vk_access_token") == "tok123"
    status = store.status()
    assert set(status) == set(KNOWN_SECRETS)
    assert status["vk_access_token"] is True
    assert not any(v for k, v in status.items() if k != "vk_access_token")


def test_persists_across_instances(tmp_path, no_env):
    path = tmp_path / "secrets.json"
    SecretsStore(path).set("vk_group_id", "987654")
    reloaded = SecretsStore(path)
    assert reloaded.get("vk_group_id") == "987654"
    assert reloaded.get("vk_access_token") == ""


def test_set_empty_clears(store, no_env):
    store.set("vk_access_token", "tok")
    store.set("vk_access_token", "")
    assert store.get("vk_access_token") == ""
    assert store.status()["vk_access_token"] is False


def test_values_stripped(store, no_env):
    store.set("vk_access_token", "  tok  ")
    assert store.get("vk_access_token") == "tok"


def test_file_only_contains_set_keys(store, no_env, tmp_path):
    store.set("vk_access_token", "tok")
    data = json.loads((tmp_path / "secrets.json").read_text())
    assert data == {"vk_access_token": "tok"}


# ── permissions ──────────────────────────────────────────────────────

@pytest.mark.skipif(os.name == "nt", reason="chmod is a no-op on Windows")
def test_file_mode_0600(store, tmp_path):
    store.set("vk_access_token", "tok")
    mode = stat.S_IMODE((tmp_path / "secrets.json").stat().st_mode)
    assert mode == 0o600


# ── env override precedence ──────────────────────────────────────────

def test_env_overrides_file(store, monkeypatch):
    store.set("vk_access_token", "filetok")
    monkeypatch.setenv("VK_ACCESS_TOKEN", "envtok")
    assert store.get("vk_access_token") == "envtok"
    assert store.status()["vk_access_token"] is True


def test_env_present_without_file_value(store, monkeypatch):
    monkeypatch.setenv("VK_GROUP_ID", "123")
    assert store.get("vk_group_id") == "123"
    assert store.status()["vk_group_id"] is True


def test_blank_env_falls_through_to_file(store, monkeypatch):
    store.set("vk_access_token", "filetok")
    monkeypatch.setenv("VK_ACCESS_TOKEN", "   ")
    assert store.get("vk_access_token") == "filetok"


# ── unknown keys ─────────────────────────────────────────────────────

def test_unknown_key_get(store):
    with pytest.raises(KeyError):
        store.get("aws_secret")


def test_unknown_key_set(store):
    with pytest.raises(KeyError):
        store.set("aws_secret", "x")


# ── corrupt file is non-fatal ────────────────────────────────────────

def test_corrupt_file_treated_as_empty(tmp_path, no_env):
    path = tmp_path / "secrets.json"
    path.write_text("{not json", encoding="utf-8")
    s = SecretsStore(path)
    assert s.status() == {k: False for k in KNOWN_SECRETS}


# ── module-level singleton API ───────────────────────────────────────

def test_module_functions_delegate(tmp_path, no_env, monkeypatch):
    monkeypatch.setattr(secrets_mod, "_store", SecretsStore(tmp_path / "secrets.json"))
    secrets_mod.set_secret("vk_access_token", "tok")
    assert secrets_mod.get_secret("vk_access_token") == "tok"
    status = secrets_mod.secret_status()
    assert set(status) == set(KNOWN_SECRETS)
    assert status["vk_access_token"] is True
    assert not any(v for k, v in status.items() if k != "vk_access_token")
    with pytest.raises(KeyError):
        secrets_mod.set_secret("nope", "x")
