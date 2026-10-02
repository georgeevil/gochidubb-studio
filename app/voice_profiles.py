"""Enrolled voice profiles for speaker identification (pipeline/voiceprint.py).

One `voice_profiles` table in the same SQLite file as `jobs`, following
app/artifact_store.py: initialised once at startup next to init_db(), every
call opens its own short-lived connection (routes run embedding work in a
thread pool, and sqlite3 connections are not shareable across threads).

A profile is {id, name, role, group, embeddings: [[float…]…], created_at,
updated_at, sources: [{job_id, speaker}]}. `group` scopes matching — the
councillors of one municipality are only ever compared with each other.
Embeddings accumulate as a human confirms the profile on new recordings, so
matching improves with use; the list is capped (oldest dropped) so a profile
confirmed on every weekly session for years stays small.

A voice embedding is biometric data about a real person. It never leaves
this machine through this module, and listings omit the vectors unless asked.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("gochidubb.voice_profiles")

_DB_PATH: Optional[Path] = None

# Newest embeddings kept per profile.
MAX_EMBEDDINGS = 32
MAX_NAME_LEN = 120


class StoreUnavailable(RuntimeError):
    """init_store() has not run (e.g. a route test without the lifespan)."""


def init_store(db_path: Path) -> None:
    """Create the table. Call once at startup, next to init_db()."""
    global _DB_PATH
    _DB_PATH = db_path
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id         TEXT PRIMARY KEY,
                grp        TEXT NOT NULL,
                name       TEXT NOT NULL,
                role       TEXT,
                data       TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_voice_profiles_grp "
                     "ON voice_profiles(grp)")
        conn.commit()
    finally:
        conn.close()


def available() -> bool:
    return _DB_PATH is not None


def normalize_group(group) -> str:
    return str(group or "").strip().lower()


def _connect() -> sqlite3.Connection:
    if _DB_PATH is None:
        raise StoreUnavailable("voice profile store is not initialised")
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _row(r: sqlite3.Row, with_embeddings: bool) -> Dict[str, Any]:
    data = json.loads(r["data"] or "{}")
    embs = data.get("embeddings") or []
    out = {
        "id": r["id"], "name": r["name"], "role": r["role"],
        "group": r["grp"],
        "embeddings_count": len(embs),
        "created_at": r["created_at"], "updated_at": r["updated_at"],
        "sources": data.get("sources") or [],
    }
    if with_embeddings:
        out["embeddings"] = embs
    return out


def _vec(v) -> List[float]:
    return [round(float(x), 6) for x in v]


def create(name: str, group: str, embedding, *, role: Optional[str] = None,
           source: Optional[dict] = None) -> Dict[str, Any]:
    now = time.time()
    pid = uuid.uuid4().hex[:12]
    data = {"embeddings": [_vec(embedding)],
            "sources": [source] if source else []}
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO voice_profiles(id, grp, name, role, data, created_at,"
            " updated_at) VALUES(?,?,?,?,?,?,?)",
            (pid, normalize_group(group), name.strip()[:MAX_NAME_LEN],
             (role or "").strip()[:MAX_NAME_LEN] or None,
             json.dumps(data), now, now))
        conn.commit()
    finally:
        conn.close()
    log.info(f"[voices] enrolled {pid} in group {normalize_group(group)!r}")
    return get(pid)


def get(profile_id: str, with_embeddings: bool = False) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        r = conn.execute("SELECT * FROM voice_profiles WHERE id=?",
                         (profile_id,)).fetchone()
    finally:
        conn.close()
    return _row(r, with_embeddings) if r else None


def list_profiles(group: Optional[str] = None,
                  with_embeddings: bool = False) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        if group:
            rows = conn.execute(
                "SELECT * FROM voice_profiles WHERE grp=? ORDER BY name",
                (normalize_group(group),)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM voice_profiles ORDER BY grp, name").fetchall()
    finally:
        conn.close()
    return [_row(r, with_embeddings) for r in rows]


def update(profile_id: str, *, name: Optional[str] = None,
           role: Optional[str] = None,
           group: Optional[str] = None) -> Optional[Dict[str, Any]]:
    sets, args = [], []
    if name is not None:
        sets.append("name=?")
        args.append(name.strip()[:MAX_NAME_LEN])
    if role is not None:
        sets.append("role=?")
        args.append(role.strip()[:MAX_NAME_LEN] or None)
    if group is not None:
        sets.append("grp=?")
        args.append(normalize_group(group))
    if not sets:
        return get(profile_id)
    sets.append("updated_at=?")
    args.append(time.time())
    conn = _connect()
    try:
        cur = conn.execute(
            f"UPDATE voice_profiles SET {', '.join(sets)} WHERE id=?",
            (*args, profile_id))
        conn.commit()
        if not cur.rowcount:
            return None
    finally:
        conn.close()
    return get(profile_id)


def delete(profile_id: str) -> bool:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM voice_profiles WHERE id=?", (profile_id,))
        conn.commit()
        return bool(cur.rowcount)
    finally:
        conn.close()


def add_embedding(profile_id: str, embedding,
                  source: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """Append one embedding (and where it came from) to a profile.

    A source already on the profile is not added twice — confirming the same
    speaker of the same job again must not double its weight.
    """
    conn = _connect()
    try:
        r = conn.execute("SELECT data FROM voice_profiles WHERE id=?",
                         (profile_id,)).fetchone()
        if not r:
            return None
        data = json.loads(r["data"] or "{}")
        sources = data.get("sources") or []
        if source and source in sources:
            return get(profile_id)
        embs = (data.get("embeddings") or []) + [_vec(embedding)]
        data["embeddings"] = embs[-MAX_EMBEDDINGS:]
        if source:
            sources.append(source)
        data["sources"] = sources
        conn.execute("UPDATE voice_profiles SET data=?, updated_at=? WHERE id=?",
                     (json.dumps(data), time.time(), profile_id))
        conn.commit()
    finally:
        conn.close()
    return get(profile_id)
