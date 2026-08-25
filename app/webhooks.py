"""Webhooks — push job lifecycle events instead of making callers poll.

The design fires on three events: ``job.completed``, ``job.failed`` and
``job.awaiting_review``. This module owns the subscription list, the delivery
attempt, and a bounded log of what happened, so the UI can show response codes
and offer a manual re-send.

Deliberate choices:

* **Fire-and-forget, never blocking the pipeline.** A dub must not slow down
  or fail because someone's endpoint is down, so deliveries run as detached
  asyncio tasks with a short timeout and a single retry. `httpx` is already a
  dependency; nothing new is added.
* **Config lives beside the API keys, not in ``config-user.json``.** A webhook
  URL can itself be a credential (many services put a token in the path), and
  ``GET /api/config`` is unauthenticated — the same reasoning as
  ``app/secrets.py`` and ``app/apikeys.py``.
* **Payloads carry no media and no transcript** — an id, a status, and the
  handful of fields a listener needs to decide whether to come and fetch
  something. Less to leak, and it keeps the request small.
* **Signed when a secret is set.** An HMAC-SHA256 of the body goes in
  ``X-GoChiDUBB-Signature``, so a receiver can tell a real delivery from
  anything else that finds the URL.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

log = logging.getLogger("gochidubb.webhooks")

BASE = Path(__file__).parent.parent.resolve()
HOOKS_FILE = BASE / "webhooks.json"

EVENTS = ("job.completed", "job.failed", "job.awaiting_review")

# Deliveries kept for the UI's log. Bounded, like every other in-memory buffer
# here — this is a recent history, not an archive.
DELIVERY_CAPACITY = 200

_lock = threading.Lock()
_deliveries: Deque[Dict[str, Any]] = deque(maxlen=DELIVERY_CAPACITY)
_delivery_seq = 0


# ── Subscriptions ────────────────────────────────────────────────────
def _read() -> List[Dict[str, Any]]:
    if not HOOKS_FILE.exists():
        return []
    try:
        data = json.loads(HOOKS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"[webhooks] Could not read {HOOKS_FILE.name}: {e}")
        return []
    return data if isinstance(data, list) else []


def _write(records: List[Dict[str, Any]]) -> None:
    try:
        fd, tmp = tempfile.mkstemp(dir=str(HOOKS_FILE.parent),
                                   prefix=".webhooks-", suffix=".tmp")
        try:
            try:
                os.fchmod(fd, 0o600)
            except (AttributeError, OSError):
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(tmp, HOOKS_FILE)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        try:
            os.chmod(HOOKS_FILE, 0o600)
        except OSError:
            pass
    except Exception as e:
        log.warning(f"[webhooks] Could not save {HOOKS_FILE.name}: {e}")


def public(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Everything except the signing secret."""
    out = {k: v for k, v in rec.items() if k != "secret"}
    out["signed"] = bool(rec.get("secret"))
    return out


def list_hooks() -> List[Dict[str, Any]]:
    with _lock:
        return [public(r) for r in _read()]


def add(url: str, events: List[str], *, secret: str = "") -> Dict[str, Any]:
    url = (url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("url must start with http:// or https://")
    bad = [e for e in events if e not in EVENTS]
    if bad:
        raise ValueError(f"unknown event(s): {', '.join(sorted(bad))}")
    if not events:
        raise ValueError("subscribe to at least one event")
    import secrets as _s
    rec = {
        "id": _s.token_hex(8),
        "url": url,
        "events": sorted(set(events)),
        "secret": (secret or "").strip(),
        "created": time.time(),
        "enabled": True,
    }
    with _lock:
        recs = _read()
        recs.append(rec)
        _write(recs)
    log.info(f"[webhooks] added {rec['id']} -> {url} ({','.join(rec['events'])})")
    return public(rec)


def remove(hook_id: str) -> bool:
    with _lock:
        recs = _read()
        kept = [r for r in recs if r.get("id") != hook_id]
        if len(kept) == len(recs):
            return False
        _write(kept)
    log.info(f"[webhooks] removed {hook_id}")
    return True


def set_enabled(hook_id: str, enabled: bool) -> bool:
    with _lock:
        recs = _read()
        for r in recs:
            if r.get("id") == hook_id:
                r["enabled"] = bool(enabled)
                _write(recs)
                return True
    return False


# ── Deliveries ───────────────────────────────────────────────────────
def _record_delivery(hook_id: str, event: str, url: str, *,
                     status: Optional[int] = None, error: Optional[str] = None,
                     ms: Optional[float] = None,
                     payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global _delivery_seq
    with _lock:
        _delivery_seq += 1
        d = {
            "id": _delivery_seq,
            "hook_id": hook_id,
            "event": event,
            "url": url,
            "ts": time.time(),
            "status": status,
            "error": error,
            "ms": round(ms, 1) if ms is not None else None,
            "ok": bool(status and 200 <= status < 300),
            "payload": payload,
        }
        _deliveries.append(d)
    return d


def deliveries(limit: int = 50) -> List[Dict[str, Any]]:
    with _lock:
        items = list(_deliveries)
    items.reverse()
    return items[:max(0, limit)]


def sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


async def deliver_one(hook: Dict[str, Any], event: str,
                      payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST one payload. Never raises — failures become delivery records."""
    import httpx
    body = json.dumps({"event": event, "sent_at": time.time(), "data": payload},
                      ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json",
               "User-Agent": "GoChiDUBB-Webhook/1"}
    if hook.get("secret"):
        headers["X-GoChiDUBB-Signature"] = sign(body, hook["secret"])
    started = time.time()
    last_err = None
    # One retry: a listener restarting should not lose an event, but a
    # genuinely dead endpoint must not hold a task open for long.
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                r = await client.post(hook["url"], content=body, headers=headers)
            return _record_delivery(hook.get("id", "?"), event, hook["url"],
                                    status=r.status_code,
                                    ms=(time.time() - started) * 1000.0,
                                    payload=payload)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 2:
                break
    log.warning(f"[webhooks] delivery to {hook.get('id')} failed: {last_err}")
    return _record_delivery(hook.get("id", "?"), event, hook["url"],
                            error=last_err,
                            ms=(time.time() - started) * 1000.0,
                            payload=payload)


def hooks_for(event: str) -> List[Dict[str, Any]]:
    with _lock:
        recs = _read()
    return [r for r in recs
            if r.get("enabled", True) and event in (r.get("events") or [])]


def payload_for_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of a job worth pushing — no media, no transcript."""
    return {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "title": job.get("title") or job.get("source_label"),
        "source": job.get("source"),
        "target_lang": job.get("target_lang"),
        "duration_sec": job.get("duration"),
        "error": job.get("error"),
        "batch_id": job.get("batch_id"),
        # Why a job is awaiting review, when it is there because a gate said
        # so rather than because the user asked to review every job. Carries
        # the same verdicts the UI shows, so an agent can act on the payload
        # without a second round trip.
        "quality_gate": job.get("quality_gate"),
        # Which review gate is holding the job when status is awaiting_* —
        # the receiver of a job.awaiting_review event needs to know *what*
        # to review without fetching the job.
        "pending_gate": job.get("pending_gate"),
    }
