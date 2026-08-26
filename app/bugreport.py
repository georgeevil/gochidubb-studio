"""Bug reports — package a failed job into something a maintainer can act on.

A GoChiDUBB failure happens on someone else's machine, with no telemetry and
no shared logs. This module turns the structured error a job already carries
(``last_error`` from ``server._set_job_error``) plus the log window around it
into a single report dict, and can deliver that report to an issue tracker.

Deliberate choices:

* **Deduplication by signature, not by message.** The same bug produces a
  slightly different message every time (paths, ids, durations). The
  signature hashes a *normalized* message together with the stage id, and the
  resulting ``gcd-sig:<hash>`` line is embedded in the issue body so a later
  occurrence finds the existing issue via search and lands as a comment
  instead of a duplicate.
* **Everything user-visible goes through ``redact``** (the same scrubber
  ``app/logbuf.py`` applies on log ingest — see pipeline/notices.redact).
  Reports leave the machine, so no config values, no ``_pending_args``, no
  transcript, and no credential-shaped strings may survive into one.
* **Delivery never raises and never logs the API key.** A bug report is a
  courtesy on top of a failure; it must not become a second failure. Sinks
  return a result dict either way.
* **Sinks are a protocol, and they chain.** Linear is tried first because
  it dedupes; when it refuses the write — an over-quota workspace, a
  revoked key, an outage — :func:`deliver_report` sends the same report to
  the support mailbox instead, carrying the tracker's own reason with it.
  A failure that cannot be filed must still reach a person. A future Slack
  sink is another class here plus a link in that chain.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from pipeline.notices import redact

log = logging.getLogger("gochidubb.bugreport")

LINEAR_API_URL = "https://api.linear.app/graphql"

# Where reports go when the issue tracker will not take them. Not a user
# preference — this is GoChiDUBB's own support mailbox, and the whole point
# of the fallback is that it needs no setup to be the right address. The
# ``bugreport_email`` secret overrides it for anyone self-hosting.
SUPPORT_EMAIL = "support@gochidubb.com"

# The dedupe marker embedded verbatim in issue bodies and searched for on the
# next occurrence. Changing this prefix (or the hash) orphans existing issues.
SIG_PREFIX = "gcd-sig:"


# ── Signature ────────────────────────────────────────────────────────
# Order matters: Windows paths first (a drive letter + hex-ish tail would be
# half-eaten by the generic rules), then POSIX paths (two-plus components so
# short URL-ish fragments like "/v1" survive), then ids, then bare numbers.
_NORMALIZE_RULES = (
    (re.compile(r"[a-z]:\\[^\s'\"]+"), "<path>"),
    (re.compile(r"(?:/[\w.@-]+){2,}"), "<path>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<id>"),
    (re.compile(r"\b[0-9a-f]{8,}\b"), "<hex>"),
    (re.compile(r"0x[0-9a-f]+"), "<hex>"),
    # No trailing \b: durations glue their unit to the digits ("120s",
    # "12.5s") and must still collapse, or every timeout files a new issue.
    (re.compile(r"\b\d+(?:\.\d+)?"), "<n>"),
)


def normalize_message(msg: str) -> str:
    """Collapse the run-specific parts of an error message.

    Two occurrences of the same bug must normalize identically even though
    their paths, ids and numbers differ — the normalized form is what gets
    hashed into the dedupe signature.
    """
    s = str(msg or "").lower().strip()
    for pattern, replacement in _NORMALIZE_RULES:
        s = pattern.sub(replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:300]


def error_signature(stage: str, message: str) -> str:
    """12-hex-char dedupe key for one (stage, normalized message) pair.

    The stage id is part of the hash so "translate timed out" and "tts timed
    out" stay distinct issues.
    """
    basis = f"{stage or ''}|{normalize_message(message)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


# ── Report assembly (pure) ───────────────────────────────────────────
# The whitelist of job fields a report may carry. Explicitly NOT here:
# transcript / transcript_raw (huge, and someone else's content),
# _pending_args (raw request payloads), and anything from config.
_JOB_FIELDS = (
    "id", "status", "title", "source_label", "source", "source_type",
    "target_lang", "source_lang", "model", "speaker_mode", "voice_preset",
    "voice_mode", "tts_speed", "wizard_mode", "mode", "created", "batch_id",
)


def _redacted_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow copy with every string value scrubbed."""
    return {k: (redact(v) if isinstance(v, str) else v) for k, v in d.items()}


def build_bug_report(job: Dict[str, Any], *, system: Dict[str, Any],
                     logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the full report for one failed job (pure — no I/O).

    Takes the in-memory job dict (the DB copy has large fields stripped and
    may lag the live run). ``logs`` should come from
    :func:`select_log_window`; entries are already redacted on ingest.
    """
    le = job.get("last_error") or {}
    stage = le.get("stage") or job.get("failed_stage") or ""
    message = le.get("message") or job.get("error") or ""
    # A job that failed before structured errors existed — or that was
    # interrupted, which writes only the legacy top-level `error` — has an
    # empty last_error. The signature was always computed from the resolved
    # values; carry them into the report too, or the title and body read
    # "unknown error" for a failure whose text is right here.
    resolved = _redacted_copy(le)
    if message and not resolved.get("message"):
        resolved["message"] = redact(message)
    if stage and not resolved.get("stage"):
        resolved["stage"] = stage
    return {
        "report_version": 1,
        "generated_at": time.time(),
        "job": _redacted_copy({k: job[k] for k in _JOB_FIELDS if k in job}),
        "last_error": resolved,
        "error_history": [_redacted_copy(h) for h in job.get("error_history") or []
                          if isinstance(h, dict)],
        "signature": error_signature(stage, message) if message else "",
        "logs": logs,
        "system": _redacted_copy(system),
    }


def select_log_window(snapshot_fn: Callable[..., Dict[str, Any]],
                      last_error: Optional[Dict[str, Any]],
                      limit: int = 80) -> List[Dict[str, Any]]:
    """The log lines worth attaching to a report.

    When the error carries a [log_from, log_to] ring-seq window (see
    ``server._set_job_error``), take that window's tail, padded forward with
    lines logged after the failure if the window is shorter than ``limit``.
    Without a window, just the newest ``limit`` lines. Entries are redacted
    at ingest time (see app/logbuf.LogRing.add), so they pass through as-is.
    """
    le = last_error or {}
    log_from = int(le.get("log_from") or 0)
    log_to = int(le.get("log_to") or 0)
    if not log_to:
        return list(snapshot_fn(limit=limit)["entries"])
    entries = snapshot_fn(limit=0, since_seq=max(0, log_from - 1))["entries"]
    window = [e for e in entries if e["seq"] <= log_to]
    window = window[-limit:]
    if len(window) < limit:
        after = [e for e in entries if e["seq"] > log_to]
        window = window + after[:limit - len(window)]
    return window


# ── Sinks ────────────────────────────────────────────────────────────
class Sink(Protocol):
    """Somewhere a report can be delivered. ``deliver`` never raises."""

    name: str

    async def deliver(self, report: Dict[str, Any], note: str = "", *,
                      context: str = "") -> Dict[str, Any]:
        ...


def summary_rows(report: Dict[str, Any]) -> tuple:
    """The (label, value) pairs every sink puts at the top of a report.

    One definition so an emailed report and a filed issue describe the same
    failure in the same terms — a maintainer reading both should not have to
    work out whether they are looking at one incident or two.
    """
    le = report.get("last_error") or {}
    job = report.get("job") or {}
    system = report.get("system") or {}
    return (
        ("Job id", job.get("id", "")),
        ("Stage", le.get("stage", "")),
        ("Error type", le.get("type", "")),
        ("Target language", job.get("target_lang", "")),
        ("Signature", report.get("signature") or ""),
        ("Platform", system.get("platform", "")),
        ("Python", system.get("python", "")),
        ("GPU", f"{system.get('gpu_backend', '')} {system.get('gpu') or ''}".strip()),
    )


def report_title(report: Dict[str, Any]) -> str:
    """One-line headline — an issue title, and an email subject."""
    le = report.get("last_error") or {}
    stage_label = le.get("stage_label") or le.get("stage") or "unknown stage"
    message = le.get("message") or "unknown error"
    return f"[gochidubb] {stage_label}: {message[:90]}"


def _fmt_log_lines(logs: List[Dict[str, Any]]) -> str:
    lines = []
    for e in logs or []:
        lines.append(f"[{e.get('level', '')}] {e.get('logger', '')} — "
                     f"{e.get('message', '')}")
    return "\n".join(lines)


class LinearSink:
    """Deliver reports to Linear: one issue per signature, comments after.

    Personal API keys are sent raw in the ``Authorization`` header (no
    ``Bearer`` prefix) — that is what Linear's GraphQL API expects for them.
    ``transport`` is a test seam forwarded to ``httpx.AsyncClient`` so tests
    can use ``httpx.MockTransport``.
    """

    name = "linear"

    _FIND_QUERY = (
        "query FindBySig($q: String!) { issueSearch(query: $q, first: 5) "
        "{ nodes { id identifier url title } } }"
    )
    _COMMENT_MUTATION = (
        "mutation AddComment($issueId: String!, $body: String!) "
        "{ commentCreate(input: {issueId: $issueId, body: $body}) "
        "{ success comment { url } } }"
    )
    _CREATE_MUTATION = (
        "mutation CreateIssue($input: IssueCreateInput!) "
        "{ issueCreate(input: $input) { success issue { id identifier url } } }"
    )

    def __init__(self, api_key: str, team_id: str, project_id: str = "",
                 transport=None):
        self._api_key = api_key
        self._team_id = team_id
        self._project_id = project_id
        self._transport = transport

    async def _gql(self, client, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        r = await client.post(
            LINEAR_API_URL,
            json={"query": query, "variables": variables},
            headers={"Authorization": self._api_key,
                     "Content-Type": "application/json"},
        )
        if r.status_code != 200:
            raise RuntimeError(f"Linear API returned HTTP {r.status_code}")
        payload = r.json()
        errors = payload.get("errors")
        if errors:
            msg = (errors[0] or {}).get("message") or "unknown GraphQL error"
            raise RuntimeError(f"Linear GraphQL error: {msg}")
        return payload.get("data") or {}

    # ── Body rendering ───────────────────────────────────────────────
    @staticmethod
    def _issue_content(report: Dict[str, Any], note: str, context: str = "") -> tuple:
        le = report.get("last_error") or {}
        sig = report.get("signature") or ""
        title = report_title(report)
        parts = []
        if context:
            parts += [f"> {context}", ""]
        parts += ["| field | value |", "| --- | --- |"]
        parts += [f"| {k} | {v} |" for k, v in summary_rows(report) if v != ""]
        parts += ["",
                  f"`{SIG_PREFIX}{sig}` — dedupe key, do not remove: later "
                  "occurrences of this error are matched to this issue by "
                  "searching for this line."]
        if note:
            parts += ["", f"**User note:** {note}"]
        if le.get("traceback_tail"):
            parts += ["", "```text", le["traceback_tail"], "```"]
        logs = report.get("logs") or []
        if logs:
            parts += ["", f"<details><summary>Last {len(logs)} log lines</summary>",
                      "", "```text", _fmt_log_lines(logs), "```", "", "</details>"]
        return title, "\n".join(parts)

    @staticmethod
    def _comment_body(report: Dict[str, Any], note: str, context: str = "") -> str:
        le = report.get("last_error") or {}
        job = report.get("job") or {}
        system = report.get("system") or {}
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(le.get("ts") or time.time()))
        parts = ([f"> {context}", ""] if context else [])
        parts += [f"New occurrence — job `{job.get('id', '?')}`, "
                 f"target `{job.get('target_lang', '?')}`, {ts} UTC",
                 f"{system.get('platform', '')} · python {system.get('python', '')} · "
                 f"{system.get('gpu_backend', '')}"]
        if note:
            parts += ["", f"**User note:** {note}"]
        if le.get("traceback_tail"):
            parts += ["", "```text", le["traceback_tail"], "```"]
        logs = report.get("logs") or []
        if logs:
            parts += ["", f"<details><summary>Last {len(logs)} log lines</summary>",
                      "", "```text", _fmt_log_lines(logs), "```", "", "</details>"]
        return "\n".join(parts)

    # ── Delivery ─────────────────────────────────────────────────────
    async def deliver(self, report: Dict[str, Any], note: str = "", *,
                      context: str = "") -> Dict[str, Any]:
        """Create or comment. Never raises; the key never reaches a log."""
        import httpx
        sig = report.get("signature") or ""
        result = {"ok": False, "sink": self.name, "action": "failed",
                  "url": "", "issue": "", "signature": sig, "error": ""}
        try:
            async with httpx.AsyncClient(timeout=10.0,
                                         transport=self._transport) as client:
                found = None
                if sig:
                    data = await self._gql(client, self._FIND_QUERY,
                                           {"q": SIG_PREFIX + sig})
                    nodes = (data.get("issueSearch") or {}).get("nodes") or []
                    found = nodes[0] if nodes else None
                if found:
                    data = await self._gql(
                        client, self._COMMENT_MUTATION,
                        {"issueId": found.get("id"),
                         "body": self._comment_body(report, note, context)})
                    cc = data.get("commentCreate") or {}
                    if not cc.get("success"):
                        raise RuntimeError("commentCreate reported success=false")
                    url = ((cc.get("comment") or {}).get("url")
                           or found.get("url") or "")
                    result.update(ok=True, action="commented", url=url,
                                  issue=found.get("identifier") or "")
                else:
                    title, body = self._issue_content(report, note, context)
                    inp = {"teamId": self._team_id, "title": title,
                           "description": body}
                    if self._project_id:
                        inp["projectId"] = self._project_id
                    data = await self._gql(client, self._CREATE_MUTATION,
                                           {"input": inp})
                    ic = data.get("issueCreate") or {}
                    if not ic.get("success"):
                        raise RuntimeError("issueCreate reported success=false")
                    issue = ic.get("issue") or {}
                    result.update(ok=True, action="created",
                                  url=issue.get("url") or "",
                                  issue=issue.get("identifier") or "")
        except Exception as e:
            # redact() as belt-and-braces; the exception text never contains
            # the key by construction, but a proxy error might echo headers.
            err = redact(f"{type(e).__name__}: {e}")
            result["error"] = err
            log.warning(f"[bugreport] Linear delivery failed: {err}")
        return result


def render_report_text(report: Dict[str, Any], note: str = "",
                       context: str = "") -> str:
    """The report as plain text, for a mailbox rather than an issue tracker.

    Same facts as the Linear body, without the markdown — a mail client
    shows a table as pipes and a ``<details>`` as literal tags.
    """
    le = report.get("last_error") or {}
    sig = report.get("signature") or ""
    parts = []
    if context:
        parts += [context, ""]
    parts += [le.get("message") or "unknown error", ""]
    parts += [f"{k}: {v}" for k, v in summary_rows(report) if v != ""]
    if sig:
        parts += ["", f"{SIG_PREFIX}{sig} — dedupe key. Recurrences of this "
                  "bug arrive under an identical subject line."]
    if note:
        parts += ["", f"User note: {note}"]
    if le.get("traceback_tail"):
        parts += ["", "--- traceback ---", le["traceback_tail"]]
    logs = report.get("logs") or []
    if logs:
        parts += ["", f"--- last {len(logs)} log lines ---",
                  _fmt_log_lines(logs)]
    parts += ["", "The full machine-readable report is attached as JSON."]
    return "\n".join(parts)


class EmailSink:
    """Deliver reports to a mailbox over SMTP.

    This is the fallback for when the issue tracker will not take the write.
    It exists because the tracker is not always available to be written to:
    an over-quota workspace answers 400, a revoked key answers 401, an
    outage answers nothing at all — and in every one of those cases the
    failure the user is reporting is still real and still unreported.

    There is no dedupe here; a mailbox is not an issue tracker. The
    signature goes in the subject line and in an ``X-GoChiDUBB-Signature``
    header instead, so recurrences of one bug thread together and can be
    filtered without being silently merged.

    ``smtp_factory`` is a test seam: a zero-argument callable returning
    something with ``starttls`` / ``login`` / ``send_message`` / ``quit``.
    """

    name = "email"

    def __init__(self, host: str, port: int = 587, username: str = "",
                 password: str = "", sender: str = "",
                 recipient: str = SUPPORT_EMAIL, security: str = "starttls",
                 timeout: float = 20.0, smtp_factory=None):
        self._host = host
        self._security = security if security in ("starttls", "ssl", "none") else "starttls"
        self._port = int(port or (465 if self._security == "ssl" else 587))
        self._username = username
        self._password = password
        self._sender = sender or username or f"gochidubb@{host}"
        self._recipient = recipient or SUPPORT_EMAIL
        self._timeout = timeout
        self._smtp_factory = smtp_factory

    # ── Message ──────────────────────────────────────────────────────
    def build_message(self, report: Dict[str, Any], note: str = "",
                      context: str = ""):
        """The full report as an :class:`email.message.EmailMessage`.

        The subject carries the signature so a mail client threads
        recurrences; the body is readable on its own, and the JSON
        attachment is what a maintainer actually pastes into a debugger.
        """
        import json as _json
        from email.message import EmailMessage

        sig = report.get("signature") or ""
        msg = EmailMessage()
        subject = report_title(report)
        if sig:
            subject = f"{subject} [{SIG_PREFIX}{sig}]"
        msg["Subject"] = subject
        msg["From"] = self._sender
        msg["To"] = self._recipient
        if sig:
            msg["X-GoChiDUBB-Signature"] = sig
        msg.set_content(render_report_text(report, note, context))
        msg.add_attachment(
            _json.dumps(report, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            maintype="application", subtype="json",
            filename=f"gochidubb-report-{sig or 'unknown'}.json")
        return msg

    # ── Delivery ─────────────────────────────────────────────────────
    def _scrub(self, text: str) -> str:
        """redact(), plus the SMTP password by exact match.

        redact() catches credential-*shaped* strings; an SMTP password is
        whatever the user chose, so remove the one value we actually hold
        before anything reaches a result dict or a log line.
        """
        s = str(text)
        if self._password:
            s = s.replace(self._password, "***")
        return redact(s)

    def _send(self, msg) -> None:
        """Blocking SMTP conversation. Runs on a worker thread, never the loop."""
        import smtplib
        if self._smtp_factory is not None:
            client = self._smtp_factory()
        elif self._security == "ssl":
            client = smtplib.SMTP_SSL(self._host, self._port, timeout=self._timeout)
        else:
            client = smtplib.SMTP(self._host, self._port, timeout=self._timeout)
        try:
            if self._security == "starttls":
                client.starttls()
            if self._username:
                client.login(self._username, self._password)
            client.send_message(msg)
        finally:
            try:
                client.quit()
            except Exception:
                pass

    async def deliver(self, report: Dict[str, Any], note: str = "", *,
                      context: str = "") -> Dict[str, Any]:
        """Send the report. Never raises; the password never reaches a log."""
        import asyncio
        sig = report.get("signature") or ""
        result = {"ok": False, "sink": self.name, "action": "failed",
                  "url": "", "issue": "", "signature": sig, "error": "",
                  "recipient": self._recipient}
        try:
            msg = self.build_message(report, note, context)
            # smtplib is blocking, and a hung SMTP host would freeze every
            # other request for the connect timeout — the same defect that
            # took the server out for the whole synthesis stage.
            await asyncio.to_thread(self._send, msg)
            result.update(ok=True, action="emailed")
        except Exception as e:
            err = self._scrub(f"{type(e).__name__}: {e}")
            result["error"] = err
            log.warning(f"[bugreport] email delivery failed: {err}")
        return result


# ── Sink selection ───────────────────────────────────────────────────
def get_sink() -> Optional[Sink]:
    """The configured issue-tracker sink, or None.

    Linear is the only tracker today. A future Slack sink is another class
    in this module plus a branch here on its own secrets.
    """
    from app.secrets import get_secret
    api_key = get_secret("linear_api_key")
    team_id = get_secret("linear_team_id")
    if not (api_key and team_id):
        return None
    return LinearSink(api_key, team_id,
                      project_id=get_secret("linear_project_id"))


def support_email() -> str:
    """Where the fallback mail goes. Configurable, but never unset."""
    from app.secrets import get_secret
    return get_secret("bugreport_email") or SUPPORT_EMAIL


def get_email_sink() -> Optional[Sink]:
    """The SMTP fallback sink, or None if no mail host is configured.

    Only ``smtp_host`` is required: a relay on localhost needs no
    credentials, and demanding them would turn the fallback off for the
    setup least likely to fail.
    """
    from app.secrets import get_secret
    host = get_secret("smtp_host")
    if not host:
        return None
    raw_port = (get_secret("smtp_port") or "").strip()
    security = (get_secret("smtp_security") or "").strip().lower()
    if security not in ("starttls", "ssl", "none"):
        # Port 465 is implicit TLS by convention; everything else negotiates.
        security = "ssl" if raw_port == "465" else "starttls"
    username = get_secret("smtp_username")
    port = int(raw_port) if raw_port.isdigit() else (465 if security == "ssl" else 587)
    return EmailSink(host, port=port, username=username,
                     password=get_secret("smtp_password"),
                     sender=get_secret("smtp_from") or username,
                     recipient=support_email(), security=security)


def sink_configured() -> bool:
    """Presence check on the issue tracker only — never touches the network."""
    return get_sink() is not None


def email_configured() -> bool:
    """Presence check on the SMTP fallback — never touches the network."""
    return get_email_sink() is not None


async def deliver_report(report: Dict[str, Any], note: str = "", *,
                         sinks: Optional[List[Sink]] = None) -> Dict[str, Any]:
    """Deliver to the issue tracker, falling back to the support mailbox.

    Linear goes first because it dedupes: the signature finds the existing
    issue and a recurrence lands as a comment. When it refuses the write the
    same report is emailed instead, carrying the tracker's own reason in the
    body so the failure to file is visible rather than inferred.

    **Falling back on any tracker failure, not only on HTTP 400,** is
    deliberate. Linear reports some refusals as a 200 with a GraphQL error
    body and others as a status code, so the code alone is not a reliable
    test for "it did not land" — and every refusal has the same consequence
    for the person who clicked report this.

    Returns the first successful result, annotated with ``attempts`` (one
    entry per sink tried) and ``fell_back``. When every sink fails, returns
    the last result with the same annotations; when none is configured,
    ``action`` is ``"unconfigured"``.
    """
    chain = sinks if sinks is not None else [
        s for s in (get_sink(), get_email_sink()) if s is not None]
    attempts: List[Dict[str, Any]] = []
    if not chain:
        return {"ok": False, "action": "unconfigured", "attempts": attempts,
                "fell_back": False, "signature": report.get("signature") or "",
                "error": "No delivery sink is configured."}
    context = ""
    result: Dict[str, Any] = {}
    for sink in chain:
        result = await sink.deliver(report, note, context=context)
        attempts.append({"sink": result.get("sink") or getattr(sink, "name", "?"),
                         "ok": bool(result.get("ok")),
                         "action": result.get("action") or "",
                         "error": result.get("error") or ""})
        if result.get("ok"):
            break
        context = ("Filed here because the issue tracker refused this report: "
                   f"{result.get('error') or 'delivery failed'}")
    result["attempts"] = attempts
    result["fell_back"] = bool(result.get("ok")) and len(attempts) > 1
    return result
