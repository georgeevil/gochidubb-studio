"""API keys — scoped, revocable credentials for driving GoChiDUBB.

The design's API Keys screen creates named keys with scope checkboxes, shows
the token exactly once, and lists the rest masked with a revoke button. This
is the store behind it.

Three rules shape the implementation:

1. **Keys never live in ``config-user.json``.** ``GET /api/config`` is
   unauthenticated and CORS is permissive, so anything on ``UserConfig`` is
   readable by any local caller or any webpage the user has open. The same
   reasoning that put VK tokens in ``secrets.json`` (see ``app/secrets.py``)
   applies here, so keys get their own file with the same atomic, 0600 write.

2. **Only a hash is stored.** The plaintext token is returned once, at
   creation, and never again — not by any route, not in any log. A stolen
   ``apikeys.json`` therefore yields no usable credential. The hash is a
   salted SHA-256: these are 256 bits of `secrets.token_urlsafe` entropy, not
   user-chosen passwords, so there is nothing to brute-force and a slow KDF
   would buy nothing while costing a hash on every request.

3. **Enforcement is gated on hosted mode.** In ``local`` mode — the default,
   and the project's charter — every route stays open exactly as it always
   has. The scope machinery is written and tested from the start so that
   turning it on is a config change rather than a rewrite, but a bug in it
   must never be able to lock a self-hoster out of their own server.

Usage:
    from app import apikeys
    rec, token = apikeys.create("ci-pipeline", ["dub:write", "jobs:read"])
    apikeys.verify(token)          # -> record dict, or None
    apikeys.revoke(rec["id"])
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets as _secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("gochidubb.apikeys")

BASE = Path(__file__).parent.parent.resolve()
KEYS_FILE = BASE / "apikeys.json"

# Scope vocabulary, matching the design's checkboxes exactly.
SCOPES: Dict[str, str] = {
    "dub:write": "Start, retry and cancel dubs",
    "jobs:read": "List jobs and read their status",
    "outputs:read": "Download finished media and subtitles",
    "mcp:invoke": "Drive the MCP tools",
    "webhooks:manage": "Create and remove webhooks",
    "voices:write": "Add and edit voice presets",
}

# Live keys carry this prefix, test keys the other. Purely a human signal —
# both are enforced identically; the design shows the distinction so a key
# pasted into the wrong environment is obvious at a glance.
PREFIX_LIVE = "gcd_live_"
PREFIX_TEST = "gcd_test_"

_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _hash(token: str, salt: str) -> str:
    return hashlib.sha256((salt + token).encode("utf-8")).hexdigest()


def _read() -> List[Dict[str, Any]]:
    if not KEYS_FILE.exists():
        return []
    try:
        data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        # Never log file contents — only why it could not be read.
        log.warning(f"[apikeys] Could not read {KEYS_FILE.name}: {e}")
        return []
    return data if isinstance(data, list) else []


def _write(records: List[Dict[str, Any]]) -> None:
    """Atomic, 0600 — the same shape as app/secrets.py's writer."""
    try:
        fd, tmp = tempfile.mkstemp(dir=str(KEYS_FILE.parent),
                                   prefix=".apikeys-", suffix=".tmp")
        try:
            try:
                os.fchmod(fd, 0o600)
            except (AttributeError, OSError):
                pass  # Windows / exotic filesystems
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(tmp, KEYS_FILE)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        try:
            os.chmod(KEYS_FILE, 0o600)
        except OSError:
            pass
    except Exception as e:
        log.warning(f"[apikeys] Could not save {KEYS_FILE.name}: {e}")


def public(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The safe projection of a key: everything except hash and salt."""
    out = {k: v for k, v in rec.items() if k not in ("hash", "salt")}
    # Keys minted before the environment split were all live-prefixed.
    out.setdefault("environment", "live")
    return out


def list_keys() -> List[Dict[str, Any]]:
    """All keys, newest first, without their hashes."""
    with _lock:
        recs = _read()
    recs.sort(key=lambda r: r.get("created", 0), reverse=True)
    return [public(r) for r in recs]


def create(name: str, scopes: List[str], *, environment: str = "live",
           expires_days: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
    """Mint a key. Returns (public record, plaintext token).

    The token is the only time the caller ever sees the secret; it is not
    recoverable afterwards.
    """
    bad = [s for s in scopes if s not in SCOPES]
    if bad:
        raise ValueError(f"unknown scope(s): {', '.join(sorted(bad))}")
    if not scopes:
        raise ValueError("a key needs at least one scope")
    name = (name or "").strip()
    if not name:
        raise ValueError("a key needs a name")
    if environment not in ("live", "test"):
        raise ValueError("environment must be 'live' or 'test'")

    prefix = PREFIX_LIVE if environment == "live" else PREFIX_TEST
    token = prefix + _secrets.token_urlsafe(32)
    salt = _secrets.token_hex(16)
    rec: Dict[str, Any] = {
        "id": _secrets.token_hex(8),
        "name": name,
        "environment": environment,
        "scopes": sorted(set(scopes)),
        "created": _now(),
        "expires": _now() + expires_days * 86400 if expires_days else None,
        "last_used": None,
        "revoked": None,
        # Enough of the token to recognise it in a list, never enough to use.
        "masked": f"{prefix}••••{token[-4:]}",
        "salt": salt,
        "hash": _hash(token, salt),
    }
    with _lock:
        recs = _read()
        recs.append(rec)
        _write(recs)
    log.info(f"[apikeys] created key {rec['id']} ({name}) "
             f"scopes={','.join(rec['scopes'])}")
    return public(rec), token


def revoke(key_id: str) -> bool:
    """Mark a key revoked. Revoked keys are kept so the list stays honest."""
    with _lock:
        recs = _read()
        for r in recs:
            if r.get("id") == key_id and not r.get("revoked"):
                r["revoked"] = _now()
                _write(recs)
                log.info(f"[apikeys] revoked key {key_id}")
                return True
    return False


def delete(key_id: str) -> bool:
    with _lock:
        recs = _read()
        kept = [r for r in recs if r.get("id") != key_id]
        if len(kept) == len(recs):
            return False
        _write(kept)
    return True


def is_active(rec: Dict[str, Any], now: Optional[float] = None) -> bool:
    now = now if now is not None else _now()
    if rec.get("revoked"):
        return False
    exp = rec.get("expires")
    return not (exp and now >= exp)


def verify(token: str, *, touch: bool = True) -> Optional[Dict[str, Any]]:
    """Return the public record for a valid token, else None.

    Compared with `compare_digest` so a wrong key cannot be discovered a byte
    at a time by timing the response.
    """
    if not token:
        return None
    with _lock:
        recs = _read()
        now = _now()
        for r in recs:
            if not _secrets.compare_digest(r.get("hash", ""),
                                           _hash(token, r.get("salt", ""))):
                continue
            if not is_active(r, now):
                return None
            if touch:
                r["last_used"] = now
                _write(recs)
            return public(r)
    return None


def has_scope(rec: Optional[Dict[str, Any]], scope: str) -> bool:
    return bool(rec) and scope in (rec.get("scopes") or [])
