"""Activity stream — what the agent (or a human, or the server) just did.

Concept 1a of the SaaS redesign makes an activity feed the home screen: agent
runs appear as cards showing the tool calls they made, with job progress and
cost underneath, and one-line system events below that. Nothing in GoChiDUBB
recorded any of it, because the pieces live in different processes:

  * The MCP server (`tools/gochidubb_mcp.py`) and the CLI are separate
    processes. They reach this server over plain HTTP, so from the server's
    side an agent's `gochidubb.dub(...)` call is indistinguishable from the
    browser's `POST /api/dub` — same route, same shape.
  * Job progress *is* recorded, but only as the current state of each job.
    "Transcribe finished 40s ago" is not something the jobs dict can answer.

So two things are needed, and this module is the second half of both:

  1. Callers identify themselves. `GoChiDUBBClient` sends an
     `X-GoChiDUBB-Client` header, which the MCP server sets to
     `mcp/<agent>`; a middleware in server.py turns header-carrying requests
     into `tool_call` events here. Requests without the header — i.e. the UI —
     are not recorded as tool calls, because they are not.
  2. The job runner calls `record_job(...)` as jobs change state.

Reads are served from a bounded in-memory deque, exactly like `app/logbuf.py`.
The feed used to be *only* in-memory, but once it became the home screen
(CLD-274) an empty feed after every restart meant the home screen greeted the
user with nothing — so events are now also persisted to an `activity` table in
gochidubb.db. `attach_store()` (called from the server lifespan, right after
`init_db()`) opens its own SQLite connection and a daemon writer thread that
batches inserts off a bounded queue; on startup it backfills the newest rows
into the deque and re-seeds the id counter from MAX(id), so `since_id`
paging survives a restart unchanged. The queue drops (and counts) events
rather than ever blocking the job runner on SQLite — which is also why this
is still not an audit trail: a record that can be silently dropped under
pressure is worse than none. The append-only audit log (`app/audit.py`) is a
separate, persisted concern — do not conflate them.

Everything is redacted on the way in via `pipeline.notices.redact` — in
`_append`, before the event reaches either the deque or the write queue, so
nothing unredacted can persist. Same reason logbuf does it: the buffer is
served over HTTP and GOCHIDUBB_HOST can put that on the network.
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Union

from pipeline.notices import redact

log = logging.getLogger("gochidubb.activity")

# Roughly a day of ordinary use. The feed only ever renders the newest page of
# this, so the cap is about bounding memory, not about the UI.
DEFAULT_CAPACITY = 600

# Event kinds, matching the filter tabs in the design (All · Runs · Tool calls
# · System). "run" groups the tool calls an agent made; "tool_call" is one
# call; "job" is a pipeline state change; "system" is everything else.
KINDS = ("run", "tool_call", "job", "system")

# Store tuning. The queue bound is what keeps `_append` non-blocking: when the
# writer cannot keep up, the oldest queued event is dropped (and counted)
# instead of stalling the job runner on SQLite.
_QUEUE_MAX = 2000
_BATCH_MAX = 50          # write a batch at this many events…
_BATCH_WINDOW = 1.0      # …or after this many seconds, whichever comes first
_PRUNE_EVERY = 500       # run retention after this many inserted rows
RETENTION_DAYS = 30
RETENTION_ROWS = 10_000

_lock = threading.Lock()
_events: Deque[Dict[str, Any]] = deque(maxlen=DEFAULT_CAPACITY)
_seq = 0

# Persistence state — all None/zero until attach_store() is called.
_store_conn: Optional[sqlite3.Connection] = None
_store_queue: Optional["queue.Queue[Union[Dict[str, Any], threading.Event, object]]"] = None
_writer: Optional[threading.Thread] = None
_dropped = 0

# Sentinel telling the writer thread to flush what it has and exit.
_STOP = object()


def _append(kind: str, **fields: Any) -> Dict[str, Any]:
    """Add one event and return it (already redacted)."""
    global _seq, _dropped
    ev: Dict[str, Any] = {"kind": kind, "ts": time.time()}
    for k, v in fields.items():
        if v is not None:
            ev[k] = v
    # redact() walks strings; anything that is not a string passes through.
    # This runs before the event reaches the deque OR the write queue, so
    # nothing unredacted can persist.
    for k, v in list(ev.items()):
        if isinstance(v, str):
            ev[k] = redact(v)
    with _lock:
        _seq += 1
        ev["id"] = _seq
        _events.append(ev)
        q = _store_queue
        if q is not None:
            try:
                q.put_nowait(ev)
            except queue.Full:
                # Drop the oldest queued event, not the newest — the feed
                # favours recency — and never block the caller.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                _dropped += 1
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    _dropped += 1
    return ev


def record_tool_call(tool: str, actor: str, *, job_id: Optional[str] = None,
                     status: Optional[int] = None, detail: Optional[str] = None,
                     ms: Optional[float] = None,
                     prompt: Optional[str] = None) -> Dict[str, Any]:
    """One agent/CLI call, named as the tool it corresponds to.

    `actor` is the raw client identity from the header (e.g. "mcp/claude-code"
    or "cli"), so the feed can say who did it. `prompt` is the natural-language
    request that led to the call, when the caller chose to send one — the
    sentence the design's run cards quote above the tool calls. Optional and
    purely informational, like the actor header it travels next to.
    """
    return _append("tool_call", tool=tool, actor=actor, job_id=job_id,
                   status=status, detail=detail,
                   ms=round(ms, 1) if ms is not None else None,
                   prompt=prompt)


def record_job(job_id: str, status: str, *, title: Optional[str] = None,
               stage: Optional[str] = None, actor: Optional[str] = None,
               detail: Optional[str] = None) -> Dict[str, Any]:
    """A job changed state. Called from the job runner on transitions."""
    return _append("job", job_id=job_id, status=status, title=title,
                   stage=stage, actor=actor, detail=detail)


def record_system(title: str, *, severity: str = "info",
                  detail: Optional[str] = None) -> Dict[str, Any]:
    """Server-level event — startup, a webhook delivery, a budget warning."""
    return _append("system", title=title, severity=severity, detail=detail)


def recent(limit: int = 100, kinds: Optional[List[str]] = None,
           since_id: int = 0) -> List[Dict[str, Any]]:
    """Newest-first events, optionally filtered by kind.

    `since_id` lets the UI poll for only what it has not seen; it is compared
    against the monotonic per-event id, not a timestamp, so two events in the
    same millisecond cannot hide each other.
    """
    with _lock:
        items = list(_events)
    if kinds:
        want = set(kinds)
        items = [e for e in items if e.get("kind") in want]
    if since_id:
        items = [e for e in items if e["id"] > since_id]
    items.reverse()
    return items[:max(0, limit)]


def last_id() -> int:
    with _lock:
        return _seq


def dropped_count() -> int:
    """Events lost to a full write queue since the store was attached."""
    with _lock:
        return _dropped


def clear() -> None:
    """Drop the in-memory feed and reset ids. For tests.

    Only safe with no store attached (call `detach_store()` first) — resetting
    `_seq` while a store holds higher persisted ids would mint colliding ids.
    """
    global _seq, _dropped
    with _lock:
        _events.clear()
        _seq = 0
        _dropped = 0


# ---------------------------------------------------------------------------
# Persistence (CLD-274)

def _write_batch(conn: sqlite3.Connection, batch: List[Dict[str, Any]]) -> int:
    if not batch:
        return 0
    rows = [(e["id"], e["ts"], e["kind"], json.dumps(e, ensure_ascii=False))
            for e in batch]
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO activity (id, ts, kind, data) "
            "VALUES (?, ?, ?, ?)", rows)
        conn.commit()
        return len(rows)
    except Exception as e:
        log.warning(f"[activity] store write failed ({len(rows)} events): {e}")
        return 0


def _prune(conn: sqlite3.Connection) -> None:
    """Retention: drop rows past the age or count cap."""
    try:
        cutoff = time.time() - RETENTION_DAYS * 86400
        conn.execute(
            "DELETE FROM activity WHERE ts < ? OR id <= "
            "(SELECT COALESCE(MAX(id), 0) FROM activity) - ?",
            (cutoff, RETENTION_ROWS))
        conn.commit()
    except Exception as e:
        log.warning(f"[activity] retention prune failed: {e}")


def _writer_loop(conn: sqlite3.Connection, q: "queue.Queue") -> None:
    """Daemon thread: batch queued events into SQLite (~1 s or 50 events).

    Owns the connection exclusively once started. Flush waiters (bare
    `threading.Event`s in the queue) are set only after the batch — and any
    due prune — has committed, so `flush()` is a real barrier.
    """
    since_prune = 0
    stopping = False
    while not stopping:
        try:
            item = q.get(timeout=_BATCH_WINDOW)
        except queue.Empty:
            continue
        batch: List[Dict[str, Any]] = []
        waiters: List[threading.Event] = []
        deadline = time.monotonic() + _BATCH_WINDOW
        while True:
            if item is _STOP:
                stopping = True
            elif isinstance(item, threading.Event):
                waiters.append(item)
            else:
                batch.append(item)
            # A stop or flush request ends the collection window early.
            if stopping or waiters or len(batch) >= _BATCH_MAX:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = q.get(timeout=remaining)
            except queue.Empty:
                break
        if stopping:
            # Drain whatever is still queued so nothing is lost on shutdown.
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, threading.Event):
                    waiters.append(item)
                elif item is not _STOP:
                    batch.append(item)
        since_prune += _write_batch(conn, batch)
        if since_prune >= _PRUNE_EVERY:
            _prune(conn)
            since_prune = 0
        for w in waiters:
            w.set()


def attach_store(db_path: Any) -> None:
    """Start persisting the feed to the `activity` table in `db_path`.

    Called from the server lifespan right after `init_db()`. Opens its own
    connection (WAL; app/db.py's jobs connection is untouched), backfills the
    newest DEFAULT_CAPACITY rows into the deque, re-seeds `_seq` from MAX(id)
    so `since_id` semantics survive the restart, then hands the connection to
    the writer thread.
    """
    global _store_conn, _store_queue, _writer, _seq
    if _store_conn is not None:
        detach_store()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS activity ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts REAL, kind TEXT, data TEXT)")
    conn.commit()
    rows = conn.execute(
        "SELECT id, data FROM activity ORDER BY id DESC LIMIT ?",
        (DEFAULT_CAPACITY,)).fetchall()
    max_id = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM activity").fetchone()[0]
    backfill: List[Dict[str, Any]] = []
    for rid, data in reversed(rows):  # oldest first
        try:
            ev = json.loads(data)
        except Exception:
            continue
        ev["id"] = rid
        backfill.append(ev)
    q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
    with _lock:
        # Persisted history goes in front of anything already in memory
        # (normally nothing — attach happens at startup).
        have = {e["id"] for e in _events}
        merged = [e for e in backfill if e["id"] not in have] + list(_events)
        merged.sort(key=lambda e: e["id"])
        _events.clear()
        _events.extend(merged[-DEFAULT_CAPACITY:])
        _seq = max(_seq, max_id)
        _store_conn = conn
        _store_queue = q
    t = threading.Thread(target=_writer_loop, args=(conn, q),
                         name="activity-store", daemon=True)
    _writer = t
    t.start()
    log.info(f"[activity] store attached ({len(backfill)} events backfilled, "
             f"next id {_seq + 1})")


def detach_store() -> None:
    """Flush pending events, stop the writer, close the connection. For tests
    (and a clean shutdown); safe to call when no store is attached."""
    global _store_conn, _store_queue, _writer
    with _lock:
        conn, q, t = _store_conn, _store_queue, _writer
        _store_queue = None  # stop feeding the queue first
    if q is not None:
        try:
            q.put(_STOP, timeout=5)
        except queue.Full:
            pass
    if t is not None:
        t.join(timeout=10)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    with _lock:
        _store_conn = None
        _writer = None


def flush(timeout: float = 5.0) -> bool:
    """Block until everything appended so far is committed. For tests."""
    with _lock:
        q = _store_queue
    if q is None:
        return True
    done = threading.Event()
    try:
        q.put(done, timeout=timeout)
    except queue.Full:
        return False
    return done.wait(timeout)
