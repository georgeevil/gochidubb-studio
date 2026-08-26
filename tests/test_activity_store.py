"""Tests for the activity feed's SQLite store (app/activity.py, CLD-274).

The persistence contract: events reach rows already-redacted, `since_id`
paging survives a restart (re-attach backfills the deque and re-seeds the id
counter from MAX(id)), a full write queue drops-and-counts instead of
blocking, and retention pruning bounds the table.

Every test attaches to a throwaway db under tmp_path — the live gochidubb.db
is never touched.
"""
import json
import queue
import sqlite3
import time

import pytest

from app import activity


@pytest.fixture
def store_db(tmp_path):
    """A clean feed attached to a temp db; detached and cleared afterwards."""
    activity.detach_store()
    activity.clear()
    db = tmp_path / "activity-test.db"
    activity.attach_store(db)
    yield db
    activity.detach_store()
    activity.clear()


def _rows(db):
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        return conn.execute(
            "SELECT id, ts, kind, data FROM activity ORDER BY id").fetchall()
    finally:
        conn.close()


def test_events_persist_as_rows(store_db):
    a = activity.record_tool_call("dub", "mcp/claude-code", job_id="job_1")
    b = activity.record_job("job_1", "downloading")
    c = activity.record_system("server started")
    assert activity.flush()

    rows = _rows(store_db)
    assert [(r[0], r[2]) for r in rows] == [
        (a["id"], "tool_call"), (b["id"], "job"), (c["id"], "system")]
    # The data blob round-trips the whole event, id included.
    ev = json.loads(rows[0][3])
    assert ev["tool"] == "dub" and ev["actor"] == "mcp/claude-code"
    assert ev["id"] == a["id"] and rows[0][1] == pytest.approx(a["ts"])


def test_rows_are_redacted(store_db):
    secret = "hf_abcdefghijklmnopqrstuvwxyz012345"
    activity.record_tool_call("dub", "cli", detail=f"using token {secret}")
    assert activity.flush()
    (row,) = _rows(store_db)
    assert secret not in row[3]


def test_since_id_survives_restart(store_db):
    activity.record_tool_call("dub", "cli")
    activity.record_job("job_1", "queued")
    last = activity.record_job("job_1", "downloading")
    assert activity.flush()
    before = activity.recent()
    assert len(before) == 3

    # Simulated restart: detach, wipe the in-memory feed, re-attach.
    activity.detach_store()
    activity.clear()
    assert activity.recent() == [] and activity.last_id() == 0
    activity.attach_store(store_db)

    # The deque is backfilled and the id counter continues where it left off.
    assert activity.recent() == before
    assert activity.last_id() == last["id"]
    new = activity.record_system("back up")
    assert new["id"] == last["id"] + 1
    assert [e["id"] for e in activity.recent(since_id=last["id"])] == [new["id"]]

    # And the new event lands in the same table without id collisions.
    assert activity.flush()
    assert [r[0] for r in _rows(store_db)] == [1, 2, 3, 4]


def test_backfill_is_capped_at_default_capacity(store_db):
    total = activity.DEFAULT_CAPACITY + 25
    for i in range(total):
        activity.record_job("job_1", f"s{i}")
    assert activity.flush()

    activity.detach_store()
    activity.clear()
    activity.attach_store(store_db)

    events = activity.recent(limit=100_000)
    assert len(events) == activity.DEFAULT_CAPACITY
    # Newest survive; ids keep counting from the true maximum.
    assert events[0]["status"] == f"s{total - 1}"
    assert activity.last_id() == total


def test_full_queue_drops_oldest_and_counts(monkeypatch):
    """The enqueue path never blocks: a full queue sheds its oldest event."""
    activity.detach_store()
    activity.clear()
    # A tiny queue with no writer thread draining it.
    q = queue.Queue(maxsize=3)
    monkeypatch.setattr(activity, "_store_queue", q)
    try:
        for i in range(5):
            activity.record_system(f"e{i}")
        assert activity.dropped_count() == 2
        # The newest three are what remain queued.
        assert [item["title"] for item in list(q.queue)] == ["e2", "e3", "e4"]
        # The in-memory feed itself lost nothing.
        assert len(activity.recent()) == 5
    finally:
        monkeypatch.setattr(activity, "_store_queue", None)
        activity.clear()


def test_retention_prunes_by_row_count(store_db, monkeypatch):
    monkeypatch.setattr(activity, "_PRUNE_EVERY", 1)
    monkeypatch.setattr(activity, "RETENTION_ROWS", 10)
    for i in range(30):
        activity.record_job("job_1", f"s{i}")
    assert activity.flush()
    # flush() returns only after the due prune committed, so this is stable:
    # everything at or below max_id - 10 is gone.
    assert [r[0] for r in _rows(store_db)] == list(range(21, 31))


def test_retention_prunes_by_age(store_db, monkeypatch):
    monkeypatch.setattr(activity, "_PRUNE_EVERY", 1)
    activity.record_system("ancient")
    activity.record_system("recent")
    assert activity.flush()

    # Age one row past the retention window, directly in the table.
    conn = sqlite3.connect(str(store_db))
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("UPDATE activity SET ts = ts - ? WHERE id = 1",
                     (activity.RETENTION_DAYS * 86400 + 3600,))
        conn.commit()
    finally:
        conn.close()

    activity.record_system("trigger prune")
    assert activity.flush()
    ids = [r[0] for r in _rows(store_db)]
    assert 1 not in ids and ids == [2, 3]


def test_unpersisted_feed_still_works():
    """No store attached — the module behaves exactly as before."""
    activity.detach_store()
    activity.clear()
    try:
        ev = activity.record_system("hello")
        assert activity.recent()[0]["id"] == ev["id"]
        assert activity.flush()  # a no-op, not an error
    finally:
        activity.clear()


def test_detach_flushes_pending_events(store_db):
    for i in range(7):
        activity.record_job("job_1", f"s{i}")
    # No explicit flush — detach itself must not lose the queued tail.
    activity.detach_store()
    assert len(_rows(store_db)) == 7
    # Re-attach so the fixture's teardown detach stays a no-op.
    activity.attach_store(store_db)


def test_time_based_prune_uses_wall_clock(store_db, monkeypatch):
    """A fresh event is never inside the retention window's delete range."""
    monkeypatch.setattr(activity, "_PRUNE_EVERY", 1)
    now = time.time()
    activity.record_system("fresh")
    assert activity.flush()
    (row,) = _rows(store_db)
    assert row[1] >= now - 5
