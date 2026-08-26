"""Route-level tests for the two feed data sources CLD-249 added.

1. The natural-language prompt behind an agent run: GoChiDUBBClient sends it
   percent-encoded in X-GoChiDUBB-Prompt, the `_record_agent_calls`
   middleware decodes, truncates and records it, and the feed's run card
   quotes it. The whole path is informational — a request without it records
   exactly what it always did.

2. The `spend` block on /api/activity: billing.summarize over the calendar
   month, riding the poll the feed already makes (the heavier
   /api/billing/usage walks output directories and must not be polled).

TestClient(app) without the context manager, as everywhere else.
"""
import time
import urllib.parse

import pytest
from fastapi.testclient import TestClient

import server
from app import activity
from tools.gochidubb_client import GoChiDUBBClient


@pytest.fixture
def client():
    saved = dict(server.jobs)
    server.jobs.clear()
    activity.clear()
    try:
        yield TestClient(server.app)
    finally:
        server.jobs.clear()
        server.jobs.update(saved)
        activity.clear()


# ── the prompt header, end to end ────────────────────────────────────

def test_prompt_header_is_decoded_and_recorded(client):
    prompt = "Дублируй это видео на французский и сделай showcase"
    client.get("/api/jobs", headers={
        "X-GoChiDUBB-Client": "mcp/claude-code",
        "X-GoChiDUBB-Prompt": urllib.parse.quote(prompt, safe=""),
    })
    ev = activity.recent(kinds=["tool_call"])[0]
    assert ev["tool"] == "list_jobs"
    assert ev["prompt"] == prompt


def test_prompt_is_truncated_and_optional(client):
    client.get("/api/jobs", headers={
        "X-GoChiDUBB-Client": "cli",
        "X-GoChiDUBB-Prompt": urllib.parse.quote("x" * 1000, safe=""),
    })
    assert len(activity.recent(kinds=["tool_call"])[0]["prompt"]) == 300

    client.get("/api/jobs", headers={"X-GoChiDUBB-Client": "cli"})
    assert "prompt" not in activity.recent(kinds=["tool_call"])[0]


def test_blank_prompt_records_no_prompt_field(client):
    client.get("/api/jobs", headers={
        "X-GoChiDUBB-Client": "cli",
        "X-GoChiDUBB-Prompt": "%20%20",
    })
    assert "prompt" not in activity.recent(kinds=["tool_call"])[0]


def test_client_prompt_headers_encode_and_skip_empty():
    h = GoChiDUBBClient._prompt_headers("Dub this into French")
    assert h == {"X-GoChiDUBB-Prompt": "Dub%20this%20into%20French"}
    # Round-trips through the header alphabet even for non-ASCII.
    h = GoChiDUBBClient._prompt_headers("Кёко says hi")
    assert h["X-GoChiDUBB-Prompt"].isascii()
    assert urllib.parse.unquote(h["X-GoChiDUBB-Prompt"]) == "Кёко says hi"
    assert GoChiDUBBClient._prompt_headers("") is None
    assert GoChiDUBBClient._prompt_headers("   ") is None
    assert GoChiDUBBClient._prompt_headers(None) is None


# ── the spend block ──────────────────────────────────────────────────

def test_activity_carries_a_month_to_date_spend_block(client):
    server.jobs["sp1"] = {
        "id": "sp1", "status": "complete", "duration": 600.0,
        "target_lang": "fr", "created": time.time(),
    }
    d = client.get("/api/activity").json()
    spend = d["spend"]
    # 10 minutes at the first tier's $0.080.
    assert spend["minutes"] == 10.0
    assert spend["cost"] == 0.80
    assert spend["rate"] == 0.080
    assert spend["estimate"] is True
    assert spend["mode"] == server.cfg.mode
    assert spend["next_tier"]["next_rate"] == 0.065


def test_spend_ignores_last_months_jobs_and_failures(client):
    server.jobs["old"] = {
        "id": "old", "status": "complete", "duration": 600.0,
        "target_lang": "fr", "created": time.time() - 40 * 86400,
    }
    server.jobs["dead"] = {
        "id": "dead", "status": "error", "duration": 600.0,
        "target_lang": "fr", "created": time.time(),
    }
    spend = client.get("/api/activity").json()["spend"]
    assert spend["minutes"] == 0.0
    assert spend["cost"] == 0.0
