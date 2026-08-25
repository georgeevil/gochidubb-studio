"""Tests for app/bugreport.py — signatures, assembly, and both sinks.

The error signature is a *contract with the outside world*: it is embedded
in Linear issue bodies as ``gcd-sig:<hash>`` and searched for on the next
occurrence. Changing the normalization rules or the hash orphans every
existing issue, so a golden-value test locks the algorithm.
"""
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import logbuf  # noqa: E402
from app.bugreport import (  # noqa: E402
    SIG_PREFIX, SUPPORT_EMAIL, EmailSink, LinearSink, build_bug_report,
    deliver_report, error_signature, get_email_sink, normalize_message,
    render_report_text, report_title, select_log_window,
)


# ═════════════════════════════════════════════════════════════════════
# Normalization + signature
# ═════════════════════════════════════════════════════════════════════
class TestNormalizeMessage:
    def test_posix_paths_collapse(self):
        a = normalize_message("cannot open /Users/alice/videos/clip.mp4")
        b = normalize_message("cannot open /home/bob/other/file.mp4")
        assert a == b
        assert "<path>" in a

    def test_windows_paths_collapse(self):
        a = normalize_message(r"cannot open C:\Users\alice\clip.mp4")
        b = normalize_message(r"cannot open D:\videos\other.mp4")
        assert a == b
        assert "<path>" in a

    def test_short_url_ish_fragment_survives(self):
        # One-component absolute fragments like "/v1" are meaningful
        # (endpoint paths) and must not be swallowed by the POSIX rule.
        assert "/v1" in normalize_message("POST /v1 failed")

    def test_uuids_collapse(self):
        a = normalize_message("job 0f3aab12-1234-4abc-9def-001122334455 died")
        b = normalize_message("job ffffffff-0000-4a4a-8b8b-aabbccddeeff died")
        assert a == b
        assert "<id>" in a

    def test_hex_runs_collapse(self):
        a = normalize_message("chunk deadbeefcafe0123 failed at 0x7fff")
        b = normalize_message("chunk 0123456789abcdef failed at 0xdead")
        assert a == b
        assert "<hex>" in a

    def test_numbers_collapse(self):
        a = normalize_message("failed on segment 17 after 120 seconds")
        b = normalize_message("failed on segment 3 after 95 seconds")
        assert a == b
        assert "<n>" in a

    def test_unit_suffixed_numbers_collapse(self):
        # Durations glue the unit onto the digits ("120s", "12.5s"); the
        # number rule must still swallow them or every timeout gets its own
        # Linear issue.
        a = normalize_message("request failed after 120s (retry 3, 12.5s)")
        b = normalize_message("request failed after 95s (retry 1, 3.2s)")
        assert a == b
        assert "<n>s" in a

    def test_case_and_whitespace_insensitive(self):
        assert normalize_message("  CUDA   Error ") == normalize_message("cuda error")

    def test_truncated_to_300(self):
        assert len(normalize_message("x" * 1000)) <= 300


class TestErrorSignature:
    def test_same_bug_different_run_hashes_identically(self):
        a = error_signature("tts", "no audio at /tmp/job-1/seg 17.wav after 42 ms")
        b = error_signature("tts", "no audio at /tmp/job-9/seg 3.wav after 95 ms")
        assert a == b

    def test_stage_distinguishes(self):
        assert (error_signature("translate", "backend timed out")
                != error_signature("tts", "backend timed out"))

    def test_shape(self):
        sig = error_signature("tts", "boom")
        assert len(sig) == 12
        assert sig == sig.lower()
        int(sig, 16)  # valid hex

    def test_golden_value(self):
        """Locks the algorithm. If this fails you changed normalization or
        hashing — which orphans every Linear issue already carrying a
        gcd-sig line. Do that only on purpose."""
        assert error_signature(
            "tts", "CUDA out of memory. Tried to allocate 512.00 MiB on device 0"
        ) == "3bce138aacbb"


# ═════════════════════════════════════════════════════════════════════
# Report assembly
# ═════════════════════════════════════════════════════════════════════
def _fake_job():
    return {
        "id": "job-1",
        "status": "error",
        "title": "Some lecture",
        "target_lang": "ru",
        "model": "qwen3",
        "created": 1700000000.0,
        "error": "synth exploded with token=abcd1234secret attached",
        "last_error": {
            "stage": "tts",
            "stage_label": "Voice synthesis",
            "type": "RuntimeError",
            "message": "synth exploded with token=abcd1234secret attached",
            "traceback_tail": "Traceback...\ntoken=abcd1234secret\nRuntimeError",
            "ts": 1700000100.0,
            "log_from": 5,
            "log_to": 9,
        },
        "error_history": [
            {"stage": "tts", "message": "earlier fail 1", "ts": 1.0},
        ],
        # Must never leak:
        "transcript": [{"text": "private words"}],
        "transcript_raw": [{"text": "private words"}],
        "_pending_args": {"hf_token": "hf_abcdefghijkl"},
    }


class TestBuildBugReport:
    def test_shape(self):
        rep = build_bug_report(_fake_job(), system={"platform": "TestOS"},
                               logs=[{"seq": 1, "message": "m"}])
        assert rep["report_version"] == 1
        assert rep["generated_at"] > 0
        assert rep["job"]["id"] == "job-1"
        assert rep["job"]["target_lang"] == "ru"
        assert rep["last_error"]["stage"] == "tts"
        assert rep["error_history"][0]["message"] == "earlier fail 1"
        assert rep["system"]["platform"] == "TestOS"
        assert rep["logs"] == [{"seq": 1, "message": "m"}]
        assert rep["signature"] == error_signature(
            "tts", _fake_job()["last_error"]["message"])

    def test_absent_job_fields_omitted(self):
        rep = build_bug_report({"id": "j", "last_error": {"stage": "tts",
                                                          "message": "x"}},
                               system={}, logs=[])
        assert "target_lang" not in rep["job"]

    def test_secrets_redacted_everywhere(self):
        rep = build_bug_report(_fake_job(), system={}, logs=[])
        blob = json.dumps(rep)
        assert "abcd1234secret" not in blob
        assert "[redacted]" in rep["last_error"]["traceback_tail"]
        assert "[redacted]" in rep["last_error"]["message"]

    def test_large_and_private_fields_never_leak(self):
        rep = build_bug_report(_fake_job(), system={}, logs=[])
        blob = json.dumps(rep)
        assert "private words" not in blob
        assert "_pending_args" not in blob
        assert "hf_abcdefghijkl" not in blob

    def test_legacy_top_level_error_reaches_the_report(self):
        """An interrupted job writes only job["error"]; it must still describe itself.

        The signature was always computed from the resolved message, so a
        report that showed "unknown error" was inconsistent with its own
        dedupe key — and useless to whoever received it.
        """
        job = {"id": "j9", "status": "error", "failed_stage": "tts",
               "error": "Interrupted by server restart"}
        rep = build_bug_report(job, system={}, logs=[])
        assert rep["last_error"]["message"] == "Interrupted by server restart"
        assert rep["last_error"]["stage"] == "tts"
        assert rep["signature"] == error_signature("tts", job["error"])
        assert "unknown error" not in report_title(rep)

    def test_structured_error_wins_over_the_legacy_field(self):
        job = {"id": "j9", "status": "error", "error": "stale summary",
               "last_error": {"stage": "translate", "message": "real cause"}}
        rep = build_bug_report(job, system={}, logs=[])
        assert rep["last_error"]["message"] == "real cause"
        assert rep["last_error"]["stage"] == "translate"

    def test_no_error_means_empty_signature(self):
        rep = build_bug_report({"id": "j", "status": "complete"},
                               system={}, logs=[])
        assert rep["signature"] == ""


# ═════════════════════════════════════════════════════════════════════
# Log window selection
# ═════════════════════════════════════════════════════════════════════
class TestSelectLogWindow:
    def _ring(self, n=120):
        ring = logbuf.LogRing(capacity=500)
        for i in range(1, n + 1):
            ring.add("INFO", "test", f"line {i}")
        return ring

    def test_error_window_honored(self):
        ring = self._ring()
        le = {"log_from": 30, "log_to": 40}
        out = select_log_window(ring.entries, le, limit=80)
        seqs = [e["seq"] for e in out]
        assert seqs[0] == 30
        # window is 11 lines; padded forward past log_to up to the limit
        assert 41 in seqs
        assert len(out) <= 80

    def test_long_window_keeps_tail(self):
        ring = self._ring()
        le = {"log_from": 1, "log_to": 110}
        out = select_log_window(ring.entries, le, limit=80)
        assert len(out) == 80
        assert out[-1]["seq"] == 110          # tail of the window, not the ring
        assert out[0]["seq"] == 31

    def test_no_error_falls_back_to_newest(self):
        ring = self._ring()
        out = select_log_window(ring.entries, None, limit=80)
        assert len(out) == 80
        assert out[-1]["seq"] == 120

    def test_limit_respected(self):
        ring = self._ring()
        out = select_log_window(ring.entries, {"log_from": 1, "log_to": 120},
                                limit=10)
        assert len(out) == 10


# ═════════════════════════════════════════════════════════════════════
# Linear sink (httpx.MockTransport — no network)
# ═════════════════════════════════════════════════════════════════════
httpx = pytest.importorskip("httpx")


def _report():
    job = _fake_job()
    return build_bug_report(job, system={"platform": "TestOS", "python": "3.11"},
                            logs=[{"seq": i, "level": "INFO", "logger": "t",
                                   "message": f"line {i}"} for i in range(5)])


def _transport(search_nodes, seen):
    def handler(request):
        payload = json.loads(request.content.decode("utf-8"))
        seen.append((request, payload))
        q = payload["query"]
        if "issueSearch" in q:
            return httpx.Response(200, json={
                "data": {"issueSearch": {"nodes": search_nodes}}})
        if "commentCreate" in q:
            return httpx.Response(200, json={
                "data": {"commentCreate": {"success": True,
                                           "comment": {"url": "https://linear.app/c/1"}}}})
        if "issueCreate" in q:
            return httpx.Response(200, json={
                "data": {"issueCreate": {"success": True,
                                         "issue": {"id": "iid", "identifier": "GCD-7",
                                                   "url": "https://linear.app/i/GCD-7"}}}})
        return httpx.Response(400, json={"errors": [{"message": "bad request"}]})
    return httpx.MockTransport(handler)


class TestLinearSink:
    def test_creates_when_no_match(self):
        seen = []
        sink = LinearSink("lin_api_KEY", "team-1",
                          transport=_transport([], seen))
        rep = _report()
        result = asyncio.run(sink.deliver(rep, note="I clicked dub"))
        assert result["ok"] is True
        assert result["action"] == "created"
        assert result["issue"] == "GCD-7"
        assert result["url"] == "https://linear.app/i/GCD-7"
        assert result["signature"] == rep["signature"]
        # Two calls: search then create.
        assert len(seen) == 2
        # Personal keys go raw — no Bearer prefix.
        for request, _ in seen:
            assert request.headers["Authorization"] == "lin_api_KEY"
        create_payload = seen[1][1]
        body = create_payload["variables"]["input"]["description"]
        assert f"{SIG_PREFIX}{rep['signature']}" in body
        assert "I clicked dub" in body
        assert create_payload["variables"]["input"]["teamId"] == "team-1"
        assert "projectId" not in create_payload["variables"]["input"]

    def test_project_id_forwarded_when_set(self):
        seen = []
        sink = LinearSink("k", "team-1", project_id="proj-9",
                          transport=_transport([], seen))
        asyncio.run(sink.deliver(_report()))
        assert seen[1][1]["variables"]["input"]["projectId"] == "proj-9"

    def test_comments_on_existing_issue(self):
        seen = []
        nodes = [{"id": "iid-0", "identifier": "GCD-3",
                  "url": "https://linear.app/i/GCD-3", "title": "t"}]
        sink = LinearSink("k", "team-1", transport=_transport(nodes, seen))
        result = asyncio.run(sink.deliver(_report()))
        assert result["ok"] is True
        assert result["action"] == "commented"
        assert result["issue"] == "GCD-3"
        assert result["url"] == "https://linear.app/c/1"
        assert seen[1][1]["variables"]["issueId"] == "iid-0"

    def test_search_query_uses_sig_prefix(self):
        seen = []
        rep = _report()
        sink = LinearSink("k", "team-1", transport=_transport([], seen))
        asyncio.run(sink.deliver(rep))
        assert seen[0][1]["variables"]["q"] == SIG_PREFIX + rep["signature"]

    def test_connect_error_never_raises(self):
        def boom(request):
            raise httpx.ConnectError("no route to host")
        sink = LinearSink("k", "team-1", transport=httpx.MockTransport(boom))
        result = asyncio.run(sink.deliver(_report()))
        assert result["ok"] is False
        assert result["action"] == "failed"
        assert result["error"]

    def test_graphql_error_becomes_failed_result(self):
        def denied(request):
            return httpx.Response(200, json={
                "errors": [{"message": "authentication failed"}]})
        sink = LinearSink("k", "team-1", transport=httpx.MockTransport(denied))
        result = asyncio.run(sink.deliver(_report()))
        assert result["ok"] is False
        assert result["action"] == "failed"
        assert "authentication failed" in result["error"]


# ═════════════════════════════════════════════════════════════════════
# Email sink — the fallback for when the tracker will not take the write
# ═════════════════════════════════════════════════════════════════════
class _FakeSMTP:
    """Records the conversation instead of holding one."""

    def __init__(self, fail_on_send=None):
        self.calls = []
        self.sent = []
        self._fail_on_send = fail_on_send

    def starttls(self):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        if self._fail_on_send is not None:
            raise self._fail_on_send
        self.calls.append("send_message")
        self.sent.append(msg)

    def quit(self):
        self.calls.append("quit")


def _email_sink(smtp, **kw):
    kw.setdefault("username", "bot@gochidubb.com")
    kw.setdefault("password", "hunter2")
    kw.setdefault("sender", "bot@gochidubb.com")
    return EmailSink("mail.example.com", smtp_factory=lambda: smtp, **kw)


class TestEmailSinkMessage:
    def test_subject_carries_signature_so_recurrences_thread(self):
        rep = _report()
        msg = _email_sink(_FakeSMTP()).build_message(rep)
        assert msg["Subject"].startswith("[gochidubb] ")
        assert f"{SIG_PREFIX}{rep['signature']}" in msg["Subject"]
        # And again as a header, for a filter that should not parse subjects.
        assert msg["X-GoChiDUBB-Signature"] == rep["signature"]

    def test_addressed_to_support_by_default(self):
        msg = _email_sink(_FakeSMTP()).build_message(_report())
        assert msg["To"] == SUPPORT_EMAIL
        assert msg["From"] == "bot@gochidubb.com"

    def test_body_is_plain_text_with_the_same_facts(self):
        rep = _report()
        msg = _email_sink(_FakeSMTP()).build_message(rep, note="I clicked dub")
        body = msg.get_body(preferencelist=("plain",)).get_content()
        assert "I clicked dub" in body
        assert rep["signature"] in body
        assert "job-1" in body
        # Markdown table pipes would be noise in a mail client.
        assert "| --- |" not in body

    def test_full_report_attached_as_json(self):
        rep = _report()
        msg = _email_sink(_FakeSMTP()).build_message(rep)
        parts = list(msg.iter_attachments())
        assert len(parts) == 1
        assert parts[0].get_filename().endswith(".json")
        payload = json.loads(parts[0].get_content())
        assert payload["signature"] == rep["signature"]
        assert payload["job"]["id"] == "job-1"

    def test_text_renderer_survives_an_empty_report(self):
        """A report is assembled from a failure; it must not need one itself."""
        text = render_report_text({})
        assert "unknown error" in text
        assert SIG_PREFIX not in text          # no signature, no dedupe claim

    def test_context_reaches_the_reader(self):
        """Why the tracker refused it is the point of the fallback mail."""
        msg = _email_sink(_FakeSMTP()).build_message(
            _report(), context="Filed here because the tracker said HTTP 400")
        body = msg.get_body(preferencelist=("plain",)).get_content()
        assert "HTTP 400" in body


class TestEmailSinkDelivery:
    def test_sends_and_reports_the_recipient(self):
        smtp = _FakeSMTP()
        result = asyncio.run(_email_sink(smtp).deliver(_report()))
        assert result["ok"] is True
        assert result["action"] == "emailed"
        assert result["recipient"] == SUPPORT_EMAIL
        assert smtp.calls == ["starttls", ("login", "bot@gochidubb.com", "hunter2"),
                              "send_message", "quit"]
        assert len(smtp.sent) == 1

    def test_ssl_mode_skips_starttls(self):
        smtp = _FakeSMTP()
        asyncio.run(_email_sink(smtp, security="ssl").deliver(_report()))
        assert "starttls" not in smtp.calls

    def test_anonymous_relay_never_logs_in(self):
        smtp = _FakeSMTP()
        asyncio.run(_email_sink(smtp, username="", password="",
                                sender="gochidubb@localhost",
                                security="none").deliver(_report()))
        assert smtp.calls == ["send_message", "quit"]

    def test_failure_never_raises_and_still_closes(self):
        smtp = _FakeSMTP(fail_on_send=OSError("connection reset"))
        result = asyncio.run(_email_sink(smtp).deliver(_report()))
        assert result["ok"] is False
        assert result["action"] == "failed"
        assert "connection reset" in result["error"]
        assert smtp.calls[-1] == "quit"      # the finally block ran

    def test_password_never_survives_into_the_result(self):
        """An SMTP server that echoes the password must not leak it onwards."""
        smtp = _FakeSMTP(
            fail_on_send=RuntimeError("535 auth rejected for hunter2"))
        result = asyncio.run(_email_sink(smtp).deliver(_report()))
        assert "hunter2" not in result["error"]
        assert "***" in result["error"]

    def test_send_does_not_run_on_the_event_loop(self):
        """A hung SMTP host must not freeze every other request."""
        import threading
        seen = {}

        class _ThreadSpy(_FakeSMTP):
            def send_message(self, msg):
                seen["thread"] = threading.current_thread().name
                super().send_message(msg)

        async def go():
            seen["loop"] = threading.current_thread().name
            await _email_sink(_ThreadSpy()).deliver(_report())

        asyncio.run(go())
        assert seen["thread"] != seen["loop"]


class TestEmailSinkConfig:
    def _secrets(self, monkeypatch, **values):
        import app.secrets as secrets_mod
        monkeypatch.setattr(secrets_mod, "get_secret",
                            lambda k: values.get(k, ""))

    def test_no_host_means_no_sink(self, monkeypatch):
        self._secrets(monkeypatch)
        assert get_email_sink() is None

    def test_host_alone_is_enough(self, monkeypatch):
        self._secrets(monkeypatch, smtp_host="localhost")
        sink = get_email_sink()
        assert sink is not None
        assert sink._port == 587 and sink._security == "starttls"

    def test_port_465_implies_implicit_tls(self, monkeypatch):
        self._secrets(monkeypatch, smtp_host="mail.example.com", smtp_port="465")
        assert get_email_sink()._security == "ssl"

    def test_explicit_security_wins(self, monkeypatch):
        self._secrets(monkeypatch, smtp_host="mail.example.com",
                      smtp_port="465", smtp_security="none")
        assert get_email_sink()._security == "none"

    def test_recipient_is_support_unless_overridden(self, monkeypatch):
        self._secrets(monkeypatch, smtp_host="h")
        assert get_email_sink()._recipient == SUPPORT_EMAIL
        self._secrets(monkeypatch, smtp_host="h", bugreport_email="me@example.com")
        assert get_email_sink()._recipient == "me@example.com"


# ═════════════════════════════════════════════════════════════════════
# The sink chain
# ═════════════════════════════════════════════════════════════════════
class _ChainSink:
    def __init__(self, name, ok, error=""):
        self.name = name
        self._ok = ok
        self._error = error
        self.calls = []

    async def deliver(self, report, note="", *, context=""):
        self.calls.append(context)
        return {"ok": self._ok, "sink": self.name,
                "action": "created" if self._ok else "failed",
                "url": "", "issue": "", "error": self._error,
                "signature": report.get("signature", "")}


class TestDeliverReport:
    def test_tracker_success_never_reaches_the_mailbox(self):
        linear, mail = _ChainSink("linear", True), _ChainSink("email", True)
        r = asyncio.run(deliver_report(_report(), sinks=[linear, mail]))
        assert r["ok"] is True and r["sink"] == "linear"
        assert r["fell_back"] is False
        assert mail.calls == []          # not tried at all

    def test_tracker_refusal_falls_through_with_its_reason(self):
        linear = _ChainSink("linear", False, "RuntimeError: Linear API returned HTTP 400")
        mail = _ChainSink("email", True)
        r = asyncio.run(deliver_report(_report(), sinks=[linear, mail]))
        assert r["ok"] is True and r["sink"] == "email"
        assert r["fell_back"] is True
        assert "HTTP 400" in mail.calls[0]
        assert [(a["sink"], a["ok"]) for a in r["attempts"]] == [
            ("linear", False), ("email", True)]

    def test_a_200_with_a_graphql_error_falls_back_too(self):
        """Linear reports some refusals as 200 + errors[], so status is no test."""
        linear = _ChainSink("linear", False, "Linear GraphQL error: over quota")
        mail = _ChainSink("email", True)
        r = asyncio.run(deliver_report(_report(), sinks=[linear, mail]))
        assert r["ok"] is True and r["sink"] == "email"

    def test_everything_failing_keeps_every_reason(self):
        a = _ChainSink("linear", False, "HTTP 400")
        b = _ChainSink("email", False, "SMTPConnectError")
        r = asyncio.run(deliver_report(_report(), sinks=[a, b]))
        assert r["ok"] is False
        assert r["fell_back"] is False
        assert [x["error"] for x in r["attempts"]] == ["HTTP 400", "SMTPConnectError"]

    def test_no_sinks_is_not_a_failure_to_deliver(self):
        """'Nowhere to send it' and 'sending failed' need different answers."""
        r = asyncio.run(deliver_report(_report(), sinks=[]))
        assert r["ok"] is False
        assert r["action"] == "unconfigured"
        assert r["attempts"] == []


# ═════════════════════════════════════════════════════════════════════
# Routes (handlers awaited directly — no HTTP client, no server start)
# ═════════════════════════════════════════════════════════════════════
# server.py transitively imports the heavy optional ML stack. Stub the
# modules that may not be installed in a test environment so the import
# works on a bare checkout.
for _name in ("voxcpm", "whisperx", "faster_whisper", "pyannote", "edge_tts",
              "demucs"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

server = pytest.importorskip("server")
from app import bugreport as bugreport_mod  # noqa: E402


class _Req:
    """Just enough of a Request for send_bug_report: .json()."""

    def __init__(self, body=None):
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _StubSink:
    name = "stub"

    def __init__(self, result):
        self.result = result
        self.calls = []

    async def deliver(self, report, note="", *, context=""):
        self.calls.append((report, note, context))
        return dict(self.result, signature=report.get("signature", ""))


class TestBugReportRoutes:
    @pytest.fixture(autouse=True)
    def _no_sinks(self, monkeypatch):
        """Both sinks off unless a test says otherwise.

        Without this the route would consult the developer's real
        secrets.json — a machine with smtp_host set would take the fallback
        and quietly pass a test that meant to exercise the tracker.
        """
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: None)
        monkeypatch.setattr(bugreport_mod, "get_email_sink", lambda: None)

    def test_get_unknown_job_404(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {})
        resp = asyncio.run(server.get_bug_report("nope"))
        assert resp.status_code == 404

    def test_post_unknown_job_404(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {})
        resp = asyncio.run(server.send_bug_report("nope", _Req()))
        assert resp.status_code == 404

    def test_post_job_without_error_400(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": {"id": "j1",
                                                    "status": "complete"}})
        resp = asyncio.run(server.send_bug_report("j1", _Req()))
        assert resp.status_code == 400
        assert b"no recorded error" in resp.body

    def test_post_unconfigured_400(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        resp = asyncio.run(server.send_bug_report("j1", _Req()))
        assert resp.status_code == 400
        # Names both ways out, not just the tracker.
        assert b"linear_api_key" in resp.body
        assert b"smtp_host" in resp.body

    def test_get_returns_report_and_config_flag(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        d = asyncio.run(server.get_bug_report("j1"))
        assert d["report"]["job"]["id"] == "job-1"
        assert d["signature"] == d["report"]["signature"]
        assert d["linear_configured"] is False
        assert d["email_configured"] is False
        # The UI addresses its mailto link from this, so it is never blank.
        assert d["support_email"]

    def test_post_success_delivers_and_echoes_result(self, monkeypatch):
        job = _fake_job()
        monkeypatch.setattr(server, "jobs", {"j1": job})
        sink = _StubSink({"ok": True, "sink": "stub", "action": "created",
                          "url": "https://linear.app/i/GCD-9",
                          "issue": "GCD-9", "error": ""})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: sink)
        d = asyncio.run(server.send_bug_report(
            "j1", _Req({"note": "was dubbing token=abcd1234secret"})))
        assert d == {"ok": True, "action": "created",
                     "url": "https://linear.app/i/GCD-9", "issue": "GCD-9",
                     "signature": sink.calls[0][0]["signature"],
                     "recipient": "", "fell_back": False,
                     "attempts": [{"sink": "stub", "ok": True,
                                   "action": "created", "error": ""}]}
        report, note, _ctx = sink.calls[0]
        assert report["job"]["id"] == "job-1"
        # The note is redacted before it reaches the sink.
        assert "abcd1234secret" not in note
        assert "[redacted]" in note

    def test_post_delivery_failure_502(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        sink = _StubSink({"ok": False, "sink": "stub", "action": "failed",
                          "url": "", "issue": "", "error": "HTTP 500"})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: sink)
        resp = asyncio.run(server.send_bug_report("j1", _Req({})))
        assert resp.status_code == 502
        assert b"HTTP 500" in resp.body

    def test_post_tolerates_missing_body(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        sink = _StubSink({"ok": True, "sink": "stub", "action": "commented",
                          "url": "u", "issue": "GCD-1", "error": ""})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: sink)
        d = asyncio.run(server.send_bug_report("j1", _Req()))   # body raises
        assert d["ok"] is True
        assert sink.calls[0][1] == ""

    def test_post_falls_back_to_email_and_says_so(self, monkeypatch):
        """The route must not report an emailed fallback as a filed issue."""
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        linear = _StubSink({"ok": False, "sink": "linear", "action": "failed",
                            "url": "", "issue": "",
                            "error": "RuntimeError: Linear API returned HTTP 400"})
        mail = _StubSink({"ok": True, "sink": "email", "action": "emailed",
                          "url": "", "issue": "", "error": "",
                          "recipient": SUPPORT_EMAIL})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: linear)
        monkeypatch.setattr(bugreport_mod, "get_email_sink", lambda: mail)
        d = asyncio.run(server.send_bug_report("j1", _Req({})))
        assert d["ok"] is True
        assert d["action"] == "emailed"
        assert d["fell_back"] is True
        assert d["recipient"] == SUPPORT_EMAIL
        assert [a["sink"] for a in d["attempts"]] == ["linear", "email"]
        # The mail carries why the tracker refused it.
        assert "HTTP 400" in mail.calls[0][2]

    def test_post_502_lists_every_attempt(self, monkeypatch):
        monkeypatch.setattr(server, "jobs", {"j1": _fake_job()})
        dead = _StubSink({"ok": False, "sink": "linear", "action": "failed",
                          "url": "", "issue": "", "error": "HTTP 400"})
        also_dead = _StubSink({"ok": False, "sink": "email", "action": "failed",
                               "url": "", "issue": "", "error": "SMTPConnectError"})
        monkeypatch.setattr(bugreport_mod, "get_sink", lambda: dead)
        monkeypatch.setattr(bugreport_mod, "get_email_sink", lambda: also_dead)
        resp = asyncio.run(server.send_bug_report("j1", _Req({})))
        assert resp.status_code == 502
        assert b"HTTP 400" in resp.body and b"SMTPConnectError" in resp.body
