"""Vendor admin console — the aggregation behind ``static/admin.html``.

The design (``SaaS Admin Console.dc.html``, screens labelled ``3a``) draws the
pages *you* use to run GoChiDUBB as a business: business overview, customer
accounts, revenue ops, and GPU fleet + abuse queue. It assumes a hosted,
multi-tenant deployment. This repo is a single-tenant server that runs on one
machine and bills nobody.

Rather than hard-code the design's mock figures, every panel here is computed
from something this server actually has, and the panels whose subject does not
exist say so instead of inventing it. The mapping is written down in
``docs/design/admin-console-spec.md``; the short version:

* **Real.** Minutes dubbed, job counts and success rate, per-day and per-month
  revenue estimates, the rate-card bands, per-stage p95 latency against a
  7-day baseline, queue depth, GPU utilisation, storage, API keys and their
  scopes, and the audit trail.
* **Estimated, and labelled as such everywhere.** Every dollar figure. It is
  ``app/billing.py`` arithmetic — the design's published hosted rates applied
  to real usage — and that module's honesty boundary applies verbatim here:
  *the minutes are real, the money is an estimate, this server bills nobody.*
* **Absent.** Workspaces, invoices, payments, dunning, credits, DMCA claims,
  staff SSO and multi-node GPU pools. Nothing in this module fabricates them.
  Where the design has a panel for one, the console renders the nearest real
  thing under an honest title, or an explicit note that the subject needs a
  billing/tenancy system this build does not have.

The one place the mapping is a genuine stretch, and is called out on the page:
the design's **Accounts** list becomes **API keys**, because a key is the only
caller identity this server has. Keys are what a vendor would actually
administer here — scopes, age, expiry, revocation — and every action on them
already lands in ``app/audit.py``, which is exactly what the design's footnote
("every action here lands in the customer-visible audit log") promises.

Pure module: no I/O, no server imports, no clock reads that a caller cannot
override. Everything it needs — jobs, key records, a storage figure, a GPU
snapshot, per-stage durations — is passed in, so the whole console is testable
without a running pipeline.
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app import billing, estimate

# ── Status vocabulary ────────────────────────────────────────────────
# Mirrors server.py. Kept as its own copy on purpose: this module must not
# import server.py (that would drag the whole pipeline into a pure helper and
# into every test that touches it). The sets are small and the names are
# stable, and `tests/test_admin.py` asserts they still line up with the
# status strings the job runner writes.

# Finished, and the dub exists.
DONE_STATUSES = frozenset({"complete"})
# Finished, and it did not. `cancelled` is deliberately NOT here: a user
# stopping their own job is not the service failing, and counting it against
# the success rate would make "I changed my mind" look like an outage.
FAILED_STATUSES = frozenset({"error", "interrupted"})
CANCELLED_STATUSES = frozenset({"cancelled"})
# Stopped, waiting on a human. Minutes have already been spent on these.
ATTENTION_STATUSES = frozenset({"awaiting_translation_review", "paused"})
# Somewhere in the pipeline, or waiting for the single-consumer queue.
ACTIVE_STATUSES = frozenset({
    "preparing", "queued", "scheduled", "uploaded", "running", "downloading",
    "extracting", "transcribing", "diarizing", "translating", "synthesizing",
    "assembling", "merging",
})

SEVERITIES = ("ok", "warn", "bad")

# Below this many samples a p95 is one unlucky job, so the comparison is
# suppressed rather than reported as a degradation.
MIN_STAGE_SAMPLES = 4
MIN_BASELINE_SAMPLES = 8

# p95 / baseline p95 thresholds for the stage table's severity column.
STAGE_WARN_RATIO = 1.5
STAGE_BAD_RATIO = 3.0

# A day's minutes this far above the trailing median is worth a human look —
# on a hosted deployment it is the shape of a leaked key, and locally it is
# still the shape of a runaway script.
ANOMALY_RATIO = 3.0
ANOMALY_MIN_MINUTES = 30.0

# A live key with no expiry that has been around this long is worth rotating.
KEY_STALE_DAYS = 180.0

# Windows longer than this are bucketed by month rather than by day, so the
# chart never tries to draw 365 bars three pixels wide.
MONTHLY_BUCKET_DAYS = 120

DAY = 86400.0


# ── Small helpers ────────────────────────────────────────────────────

def _now(now: Optional[float]) -> float:
    return time.time() if now is None else float(now)


def _num(value: Any) -> float:
    try:
        out = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out and abs(out) != float("inf") else 0.0


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _month_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def job_state(job: Dict[str, Any]) -> str:
    """One of done / failed / cancelled / attention / active / other."""
    status = job.get("status") or ""
    if status in DONE_STATUSES:
        return "done"
    if status in FAILED_STATUSES:
        return "failed"
    if status in CANCELLED_STATUSES:
        return "cancelled"
    if status in ATTENTION_STATUSES:
        return "attention"
    if status in ACTIVE_STATUSES:
        return "active"
    return "other"


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. 0.0 for an empty sample.

    Nearest-rank rather than an interpolating estimator because these are
    small samples of wall-clock durations: reporting a p95 that no run
    actually took would make the fleet table harder to check against the
    job it came from, not easier.
    """
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return 0.0
    pct = min(100.0, max(0.0, float(pct)))
    rank = max(1, int(-(-pct / 100.0 * len(vals) // 1)))  # ceil
    return vals[min(rank, len(vals)) - 1]


def window(days: int, now: Optional[float] = None) -> Tuple[float, float, float]:
    """``(since, prev_since, now)`` for a trailing window of `days` days.

    `prev_since` bounds the equally-long window immediately before it, which
    is what every "vs last period" delta on the overview is measured against.
    """
    now_ts = _now(now)
    span = max(1, int(days or 30)) * DAY
    return now_ts - span, now_ts - 2 * span, now_ts


def _in_window(job: Dict[str, Any], since: float, until: float) -> bool:
    created = _num(job.get("created"))
    return since <= created < until


def _delta(current: float, previous: float) -> Optional[float]:
    """Percent change, or None when the previous period had nothing to compare.

    A jump from zero is not "+infinity% growth", it is a first period; the
    tile renders "no prior period" rather than a number that means nothing.
    """
    if previous <= 0:
        return None
    return round((current - previous) / previous * 100.0, 1)


def _source_label(job: Dict[str, Any]) -> str:
    """A human name for the source, without the fan-out suffix.

    ``/api/quick_test`` writes ``source_label`` as ``"<title> -> ES"`` so a
    per-language job is identifiable on its own. Grouping siblings back
    together means stripping that again.
    """
    label = (job.get("batch_label") or "").strip()
    if label:
        return label
    label = (job.get("title") or "").strip()
    if label:
        return label
    label = (job.get("source_label") or "").strip()
    if " -> " in label:
        label = label.rsplit(" -> ", 1)[0]
    return label or (job.get("id") or "unknown")


def _group_mode(jobs: Sequence[Dict[str, Any]]) -> str:
    """How this group of sibling jobs was submitted."""
    kinds = {(j.get("batch_kind") or "") for j in jobs}
    if "showcase" in kinds:
        return "showcase"
    if len(jobs) > 1:
        return "multi-language"
    return "single"


def _worst(states: Iterable[str]) -> str:
    """Severity of a group: bad beats warn beats ok."""
    seen = set(states)
    if "bad" in seen:
        return "bad"
    if "warn" in seen:
        return "warn"
    return "ok"


def _job_severity(job: Dict[str, Any]) -> str:
    state = job_state(job)
    if state == "failed":
        return "bad"
    if state in ("attention", "cancelled"):
        return "warn"
    return "ok"


# ── Usage roll-ups ───────────────────────────────────────────────────

def _billable(job: Dict[str, Any]) -> float:
    """Metered minutes for one job — delegated, never re-derived here.

    ``billing.job_minutes`` is the single definition of what a minute of
    usage is (source length x target languages, nothing for a job that
    failed). A second copy in this module would be a second answer.
    """
    return billing.job_minutes(job)


def usage_series(jobs: Iterable[Dict[str, Any]], *, since: float, until: float,
                 bucket: str = "day") -> List[Dict[str, Any]]:
    """Minutes, jobs and estimated cost per day (or per month) in a window.

    **The per-bucket cost is marginal, not standalone.** Tiers are cumulative
    (``billing.price_minutes``), so pricing each day's minutes from zero would
    charge every day the first-band rate and produce a series that sums to far
    more than the window total. Each bucket is therefore priced as the
    difference between the running total before and after it — the same
    reasoning ``billing.marginal_cost`` documents for the wizard quote — which
    makes the bars sum to exactly the headline figure.
    """
    keyfn = _month_key if bucket == "month" else _day_key
    buckets: Dict[str, Dict[str, Any]] = {}

    # Pre-seed every bucket in range so a quiet day is a zero-height bar
    # rather than a missing one; a gap in the chart reads as "no data", and
    # "nobody dubbed anything on Sunday" is data.
    cursor = since
    while cursor < until:
        buckets.setdefault(keyfn(cursor), {"minutes": 0.0, "jobs": 0})
        cursor += DAY
    buckets.setdefault(keyfn(max(since, until - 1)), {"minutes": 0.0, "jobs": 0})

    for job in jobs or ():
        if not _in_window(job, since, until):
            continue
        mins = _billable(job)
        slot = buckets.setdefault(keyfn(_num(job.get("created"))),
                                  {"minutes": 0.0, "jobs": 0})
        slot["minutes"] += mins
        slot["jobs"] += 1

    out: List[Dict[str, Any]] = []
    cumulative = 0.0
    priced_before = 0.0
    for key in sorted(buckets):
        slot = buckets[key]
        cumulative += slot["minutes"]
        priced_after = billing.price_minutes(cumulative)["cost"]
        out.append({
            "bucket": key,
            "minutes": round(slot["minutes"], 2),
            "jobs": slot["jobs"],
            "cost": round(priced_after - priced_before, 2),
        })
        priced_before = priced_after
    return out


def top_sources(jobs: Iterable[Dict[str, Any]], *, since: float, until: float,
                limit: int = 10) -> List[Dict[str, Any]]:
    """Biggest spenders in the window, grouped the way the UI groups them.

    Siblings of one fan-out share a ``batch_id``, so the unit here is the
    batch — one submitted source — not the per-language job.

    **Cost is allocated pro-rata by minutes, not priced per group.** Pricing
    each group from zero would give every one of them the first, most
    expensive band and the column would sum to more than the workspace
    actually owes; charging them in some arbitrary order would make the
    numbers depend on the sort. Sharing out the window's real total in
    proportion to minutes is the only split that adds up, and the effective
    rate column says what each group worked out at.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for job in jobs or ():
        if not _in_window(job, since, until):
            continue
        key = job.get("batch_id") or job.get("id") or "unknown"
        grp = groups.setdefault(key, {
            "id": key, "label": _source_label(job), "jobs": [],
            "langs": [], "minutes": 0.0,
        })
        grp["jobs"].append(job)
        grp["minutes"] += _billable(job)
        lang = (job.get("target_lang") or "").lower()
        if lang and lang not in grp["langs"]:
            grp["langs"].append(lang)

    total_minutes = sum(g["minutes"] for g in groups.values())
    total_cost = billing.price_minutes(total_minutes)["cost"]

    rows: List[Dict[str, Any]] = []
    for grp in groups.values():
        share = (grp["minutes"] / total_minutes) if total_minutes > 0 else 0.0
        cost = round(total_cost * share, 2)
        rows.append({
            "id": grp["id"],
            "label": grp["label"],
            "langs": sorted(grp["langs"]),
            "job_ids": [j.get("id") for j in grp["jobs"]],
            "jobs": len(grp["jobs"]),
            "minutes": round(grp["minutes"], 2),
            "cost": cost,
            "effective_rate": round(cost / grp["minutes"], 4) if grp["minutes"] else 0.0,
            "mode": _group_mode(grp["jobs"]),
            "health": _worst(_job_severity(j) for j in grp["jobs"]),
        })
    rows.sort(key=lambda r: (-r["minutes"], r["label"]))
    return rows[:max(0, int(limit or 10))]


def _counts(jobs: Iterable[Dict[str, Any]], since: float, until: float) -> Dict[str, Any]:
    """Minutes and per-state job counts for one window."""
    states = Counter()
    minutes = 0.0
    for job in jobs or ():
        if not _in_window(job, since, until):
            continue
        states[job_state(job)] += 1
        minutes += _billable(job)
    total = sum(states.values())
    graded = states["done"] + states["failed"]
    return {
        "minutes": round(minutes, 2),
        "cost": billing.price_minutes(minutes)["cost"],
        "jobs": total,
        "done": states["done"],
        "failed": states["failed"],
        "cancelled": states["cancelled"],
        "attention": states["attention"],
        "active": states["active"],
        # None, not 100%, when nothing has finished either way — a fresh
        # install has no success rate, and printing a perfect one is a lie
        # that only gets found out later.
        "success_pct": round(states["done"] / graded * 100.0, 1) if graded else None,
    }


# ── Screen 1: business overview ──────────────────────────────────────

def attention(jobs: Iterable[Dict[str, Any]],
              keys: Iterable[Dict[str, Any]] = (), *,
              now: Optional[float] = None,
              storage_gb: float = 0.0,
              stage_alerts: Sequence[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """"Needs attention" — real signals only, worst first.

    Every entry names something a human can go and do. A detector that
    cannot point at a job, a key or a setting does not belong here.
    """
    now_ts = _now(now)
    jobs = list(jobs or ())
    items: List[Dict[str, Any]] = []

    recent = [j for j in jobs if _num(j.get("created")) >= now_ts - 7 * DAY]

    failed = [j for j in recent if job_state(j) == "failed"]
    if failed:
        reasons = Counter((j.get("error") or "unknown").split("\n")[0][:80]
                          for j in failed)
        top_reason, top_n = reasons.most_common(1)[0]
        items.append({
            "severity": "bad",
            "title": f"{len(failed)} job{'s' if len(failed) != 1 else ''} failed in the last 7 days",
            "detail": f"most common: {top_reason} ({top_n})",
            "link": "jobs", "target": failed[0].get("id", ""),
        })

    waiting = [j for j in jobs if job_state(j) == "attention"]
    if waiting:
        items.append({
            "severity": "warn",
            "title": f"{len(waiting)} job{'s' if len(waiting) != 1 else ''} paused for a human",
            "detail": "translation review has not been approved — no GPU time "
                      "is being spent, and none will be until someone answers",
            "link": "jobs", "target": waiting[0].get("id", ""),
        })

    stale = [j for j in jobs
             if job_state(j) == "active" and _num(j.get("created")) < now_ts - DAY]
    if stale:
        items.append({
            "severity": "warn",
            "title": f"{len(stale)} job{'s' if len(stale) != 1 else ''} active for over a day",
            "detail": "still in a running status — either genuinely long, or "
                      "left behind by a restart the runner did not mark",
            "link": "jobs", "target": stale[0].get("id", ""),
        })

    for alert in stage_alerts or ():
        items.append({
            "severity": alert.get("severity", "warn"),
            "title": f"{alert.get('stage', 'a stage')} p95 is "
                     f"{alert.get('ratio', 0):.1f}x its 7-day baseline",
            "detail": f"p95 {alert.get('p95', 0):.2f}x realtime vs "
                      f"{alert.get('baseline_p95', 0):.2f}x baseline "
                      f"over {alert.get('runs', 0)} recent runs",
            "link": "fleet", "target": alert.get("stage", ""),
        })

    storage_gb = _num(storage_gb)
    if storage_gb > billing.STORAGE_INCLUDED_GB:
        over = storage_gb - billing.STORAGE_INCLUDED_GB
        items.append({
            "severity": "warn",
            "title": f"Outputs are {storage_gb:.1f} GB — past the "
                     f"{billing.STORAGE_INCLUDED_GB:.0f} GB included allowance",
            "detail": f"{over:.1f} GB billable at "
                      f"${billing.STORAGE_PER_GB:.2f}/GB/month on the published rates",
            "link": "revenue", "target": "",
        })

    for issue in key_issues(keys, now=now_ts):
        items.append({
            "severity": issue["severity"],
            "title": issue["title"],
            "detail": issue["detail"],
            "link": "accounts", "target": issue["key_id"],
        })

    order = {"bad": 0, "warn": 1, "ok": 2}
    items.sort(key=lambda i: order.get(i["severity"], 3))
    return items


def overview(jobs: Iterable[Dict[str, Any]],
             keys: Iterable[Dict[str, Any]] = (), *,
             days: int = 30,
             now: Optional[float] = None,
             storage_gb: float = 0.0,
             stage_alerts: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """Screen ``3a Business overview``.

    Five tiles, a revenue-by-bucket series, the attention list, and the top
    sources table — the design's layout, over this install's real jobs.
    """
    jobs = list(jobs or ())
    since, prev_since, now_ts = window(days, now)

    current = _counts(jobs, since, now_ts)
    previous = _counts(jobs, prev_since, since)
    bucket = "month" if int(days or 30) > MONTHLY_BUCKET_DAYS else "day"

    # The design's fourth tile is gross margin, which needs a GPU bill this
    # deployment does not receive. The nearest real efficiency number is the
    # one the wizard already quotes ETAs from: measured wall-clock per second
    # of source. None when too few jobs have finished to take a median —
    # `app/estimate.py` is explicit that a median of two is a coin flip.
    factor_now = estimate.realtime_factor(
        j for j in jobs if _in_window(j, since, now_ts))
    factor_prev = estimate.realtime_factor(
        j for j in jobs if _in_window(j, prev_since, since))

    tiles = [
        {
            "key": "revenue",
            # No "(est.)" here — the tile carries `estimate` and the UI
            # appends the marker itself, so putting it in the label too
            # renders "Revenue (est.) (est.)".
            "label": "Revenue",
            "value": current["cost"],
            "format": "money",
            "delta_pct": _delta(current["cost"], previous["cost"]),
            "note": f"est. at published rates · {days}d",
            "estimate": True,
        },
        {
            "key": "minutes",
            "label": "Minutes dubbed",
            "value": current["minutes"],
            "format": "number",
            "delta_pct": _delta(current["minutes"], previous["minutes"]),
            "note": "source length x target languages",
            "estimate": False,
        },
        {
            "key": "jobs",
            "label": "Jobs run",
            "value": current["jobs"],
            "format": "int",
            "delta_pct": _delta(current["jobs"], previous["jobs"]),
            "note": f"{current['done']} done · {current['failed']} failed",
            "estimate": False,
        },
        {
            "key": "throughput",
            "label": "Realtime factor",
            "value": round(factor_now, 2) if factor_now is not None else None,
            "format": "factor",
            # Lower is faster, so the sign of the delta is inverted before
            # it reaches the tile: a factor that fell is an improvement, and
            # rendering that as a red "▼" would read as a regression.
            "delta_pct": (None if (factor_now is None or factor_prev is None)
                          else _delta(factor_prev, factor_now)),
            "note": ("median wall-clock per second of source"
                     if factor_now is not None
                     else f"needs {estimate.MIN_SAMPLES} finished jobs to measure"),
            "estimate": False,
        },
        {
            "key": "success",
            "label": "Job success",
            "value": current["success_pct"],
            "format": "percent",
            "delta_pct": _delta(current["success_pct"] or 0.0,
                                previous["success_pct"] or 0.0),
            "note": ("cancelled jobs excluded" if current["cancelled"]
                     else "complete vs failed"),
            "estimate": False,
        },
    ]

    return {
        "window_days": int(days or 30),
        "since": since,
        "now": now_ts,
        "bucket": bucket,
        "tiles": tiles,
        "current": current,
        "previous": previous,
        "series": usage_series(jobs, since=since, until=now_ts, bucket=bucket),
        "attention": attention(jobs, keys, now=now_ts, storage_gb=storage_gb,
                               stage_alerts=stage_alerts),
        "top_sources": top_sources(jobs, since=since, until=now_ts),
        "storage_gb": round(_num(storage_gb), 2),
        "estimate": True,
        "disclaimer": billing_disclaimer(),
    }


def billing_disclaimer() -> str:
    """The one sentence every money figure on this console is qualified by."""
    return ("Minutes are measured from real job durations. Every dollar figure "
            "is an estimate: it applies the design's published hosted rates to "
            "that usage. This server has no accounts, no invoices and no "
            "payments — it bills nobody.")


# ── Screen 2: accounts (= API keys, the only caller identity here) ───

def key_issues(keys: Iterable[Dict[str, Any]], *,
               now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Credential hygiene problems worth surfacing on the overview."""
    now_ts = _now(now)
    out: List[Dict[str, Any]] = []
    for rec in keys or ():
        if rec.get("revoked"):
            continue
        name = rec.get("name") or rec.get("id") or "a key"
        expires = _num(rec.get("expires"))
        age_days = (now_ts - _num(rec.get("created"))) / DAY
        if expires and expires < now_ts:
            out.append({
                "severity": "bad", "key_id": rec.get("id", ""),
                "title": f"API key '{name}' has expired but is not revoked",
                "detail": "an expired key is refused, but leaving the record "
                          "live hides that a credential is out there",
            })
        elif expires and expires < now_ts + 7 * DAY:
            out.append({
                "severity": "warn", "key_id": rec.get("id", ""),
                "title": f"API key '{name}' expires in under a week",
                "detail": "rotate it before whatever uses it starts failing",
            })
        elif not expires and age_days > KEY_STALE_DAYS:
            out.append({
                "severity": "warn", "key_id": rec.get("id", ""),
                "title": f"API key '{name}' is {age_days / 30:.0f} months old "
                         f"with no expiry",
                "detail": "long-lived credentials with no end date are the "
                          "ones that leak quietly",
            })
    return out


def key_state(rec: Dict[str, Any], *, now: Optional[float] = None) -> str:
    """revoked / expired / expiring / idle / active."""
    now_ts = _now(now)
    if rec.get("revoked"):
        return "revoked"
    expires = _num(rec.get("expires"))
    if expires and expires < now_ts:
        return "expired"
    if expires and expires < now_ts + 7 * DAY:
        return "expiring"
    if not rec.get("last_used"):
        return "idle"
    return "active"


def accounts(jobs: Iterable[Dict[str, Any]],
             keys: Iterable[Dict[str, Any]] = (), *,
             now: Optional[float] = None,
             mode: str = "local",
             days: int = 30,
             enforced: bool = False) -> Dict[str, Any]:
    """Screen ``3a Customer accounts``, mapped onto what exists.

    There is one workspace — this machine — so it gets one row of its own,
    with the window's real usage on it. Everything below it is an API key:
    the only other caller identity the server can tell apart.

    ``spend_attributable`` is False and stays False until jobs record which
    credential started them. Showing a per-key spend column now would mean
    splitting the workspace's usage between keys on no evidence, so the
    column is absent rather than invented.
    """
    jobs = list(jobs or ())
    keys = list(keys or ())
    since, _prev, now_ts = window(days, now)
    current = _counts(jobs, since, now_ts)

    last_activity = max((_num(j.get("created")) for j in jobs), default=0.0)

    workspace = {
        "id": "local",
        "kind": "workspace",
        "name": "This machine",
        "detail": f"deployment mode: {mode}",
        "state": "active",
        "minutes": current["minutes"],
        "cost": current["cost"],
        "jobs": current["jobs"],
        "last_active": last_activity,
        "scopes": [],
    }

    rows: List[Dict[str, Any]] = []
    for rec in keys:
        rows.append({
            "id": rec.get("id", ""),
            "kind": "key",
            "name": rec.get("name") or rec.get("id", ""),
            "detail": rec.get("masked") or "",
            "environment": rec.get("environment") or "live",
            "state": key_state(rec, now=now_ts),
            "scopes": list(rec.get("scopes") or []),
            "created": _num(rec.get("created")),
            "expires": _num(rec.get("expires")) or None,
            "last_used": _num(rec.get("last_used")) or None,
            "revoked": _num(rec.get("revoked")) or None,
        })
    order = {"expired": 0, "expiring": 1, "active": 2, "idle": 3, "revoked": 4}
    rows.sort(key=lambda r: (order.get(r["state"], 9), -r["created"]))

    return {
        "mode": mode,
        "window_days": int(days or 30),
        "enforced": bool(enforced),
        "workspace": workspace,
        "keys": rows,
        "counts": {
            "total": len(rows),
            "active": sum(1 for r in rows if r["state"] in ("active", "idle")),
            "revoked": sum(1 for r in rows if r["state"] == "revoked"),
            "problem": sum(1 for r in rows if r["state"] in ("expired", "expiring")),
        },
        "issues": key_issues(keys, now=now_ts),
        "spend_attributable": False,
        # Rendered verbatim on the page, under the list. The design's
        # accounts screen is the one this build maps least directly, and the
        # console should say that where it is being looked at.
        "note": ("GoChiDUBB is single-tenant: there is one workspace, this "
                 "machine. The design's per-customer account list maps onto "
                 "API keys, which are the only callers this server can tell "
                 "apart. Per-key spend is not shown because jobs do not "
                 "record which credential started them — splitting the "
                 "workspace's usage between keys would be a guess."),
        "estimate": True,
        "disclaimer": billing_disclaimer(),
    }


# ── Screen 3: revenue ops ────────────────────────────────────────────

def rate_card(jobs: Iterable[Dict[str, Any]], *,
              since: float, until: float) -> Dict[str, Any]:
    """The published rate card, with this window's real position in it."""
    minutes = sum(_billable(j) for j in jobs or () if _in_window(j, since, until))
    priced = billing.price_minutes(minutes)
    bands = []
    for band in priced["bands"]:
        upper = band["to"]
        bands.append({
            "label": (f"first {upper:.0f} min" if band["from"] == 0 and upper
                      else f"{band['from']:.0f}+ min" if upper is None
                      else f"{band['from']:.0f}–{upper:.0f} min"),
            "from": band["from"],
            "to": upper,
            "rate": band["rate"],
            "minutes": band["minutes"],
            "cost": round(band["cost"], 2),
            "reached": band["minutes"] > 0,
        })
    return {
        "bands": bands,
        "minutes": priced["minutes"],
        "cost": priced["cost"],
        "marginal_rate": priced["rate"],
        "next_tier": billing.minutes_to_next_tier(minutes),
        "showcase_surcharge": billing.SHOWCASE_SURCHARGE,
        "storage_included_gb": billing.STORAGE_INCLUDED_GB,
        "storage_per_gb": billing.STORAGE_PER_GB,
    }


def unbilled(jobs: Iterable[Dict[str, Any]], *,
             since: float, until: float) -> Dict[str, Any]:
    """Work that ran but is not chargeable — the design's dunning panel slot.

    There is no dunning here because there are no invoices. What this server
    does have in the same shape is three buckets of minutes that will not
    turn into money: still running, paused waiting for a human, and failed.

    Failed jobs are priced at what they *would* have cost had they finished,
    which is the only interesting thing about them — ``billing.job_minutes``
    correctly returns zero for a failed job, so the exposure figure has to be
    computed from the source duration directly.
    """
    running: List[Dict[str, Any]] = []
    waiting: List[Dict[str, Any]] = []
    lost: List[Dict[str, Any]] = []

    for job in jobs or ():
        if not _in_window(job, since, until):
            continue
        state = job_state(job)
        row = {
            "id": job.get("id", ""),
            "label": _source_label(job),
            "lang": (job.get("target_lang") or "").lower(),
            "status": job.get("status") or "",
            "minutes": round(_num(job.get("duration")) / 60.0, 2),
            "error": (job.get("error") or "")[:120],
        }
        if state == "active":
            running.append(row)
        elif state == "attention":
            waiting.append(row)
        elif state == "failed":
            lost.append(row)

    def _bucket(rows: List[Dict[str, Any]], label: str, severity: str) -> Dict[str, Any]:
        mins = sum(r["minutes"] for r in rows)
        return {
            "label": label,
            "severity": severity,
            "count": len(rows),
            "minutes": round(mins, 2),
            "cost": billing.price_minutes(mins)["cost"],
            "rows": rows[:8],
        }

    return {
        "buckets": [
            _bucket(running, "in flight", "ok"),
            _bucket(waiting, "paused for review", "warn"),
            _bucket(lost, "failed — never billed", "bad"),
        ],
        "policy": ("Failed and cancelled jobs are never billed "
                   "(app/billing.py::UNBILLED_STATUSES). Minutes are metered "
                   "as source length x target languages; a job that never "
                   "measured a duration meters nothing rather than a guess."),
    }


def periods(jobs: Iterable[Dict[str, Any]], *, months: int = 6,
            now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Per-month roll-up, newest first — the design's invoice list slot.

    Not invoices: this server issues none. It is what each month's usage
    would have been billed at, which is the honest version of the same row.
    """
    now_ts = _now(now)
    since = now_ts - max(1, int(months or 6)) * 31 * DAY
    series = usage_series(jobs, since=since, until=now_ts, bucket="month")
    out = []
    for row in reversed(series):
        out.append({
            "period": row["bucket"],
            "minutes": row["minutes"],
            "jobs": row["jobs"],
            "cost": row["cost"],
        })
    return out[:max(1, int(months or 6))]


def revenue(jobs: Iterable[Dict[str, Any]], *,
            days: int = 30,
            now: Optional[float] = None,
            storage_gb: float = 0.0) -> Dict[str, Any]:
    """Screen ``3a Revenue ops``."""
    jobs = list(jobs or ())
    since, _prev, now_ts = window(days, now)
    summary = billing.summarize(jobs, since=since, storage_gb=_num(storage_gb))
    return {
        "window_days": int(days or 30),
        "since": since,
        "now": now_ts,
        "summary": summary,
        "rate_card": rate_card(jobs, since=since, until=now_ts),
        "unbilled": unbilled(jobs, since=since, until=now_ts),
        "periods": periods(jobs, now=now_ts),
        # The design's fourth card is "Credits & adjustments". There is no
        # ledger to credit against, so the console says that rather than
        # drawing an input that would write nowhere.
        "adjustments": {
            "available": False,
            "reason": ("Credits and adjustments need an invoice to adjust. "
                       "This server has no billing system: no accounts, no "
                       "invoices, no payments. The rate card above is the "
                       "design's published schedule applied to real usage so "
                       "the number means something, not a bill."),
        },
        "estimate": True,
        "disclaimer": billing_disclaimer(),
    }


def revenue_csv_rows(jobs: Iterable[Dict[str, Any]], *,
                     since: float, until: float) -> List[List[Any]]:
    """Rows for the design's ``Export CSV ↓``, header first.

    One row per job, because that is the grain someone exporting this wants
    to pivot on. The cost column is the job's pro-rata share of the window
    total, allocated the same way ``top_sources`` allocates it.
    """
    rows = [j for j in jobs or () if _in_window(j, since, until)]
    total_minutes = sum(_billable(j) for j in rows)
    total_cost = billing.price_minutes(total_minutes)["cost"]

    out: List[List[Any]] = [[
        "job_id", "created_utc", "label", "target_lang", "status",
        "source_seconds", "billable_minutes", "estimated_cost_usd",
        "batch_id", "batch_kind",
    ]]
    for job in sorted(rows, key=lambda j: _num(j.get("created"))):
        mins = _billable(job)
        share = (mins / total_minutes) if total_minutes > 0 else 0.0
        out.append([
            job.get("id", ""),
            datetime.fromtimestamp(_num(job.get("created")),
                                   timezone.utc).isoformat(timespec="seconds"),
            _source_label(job),
            (job.get("target_lang") or "").lower(),
            job.get("status") or "",
            round(_num(job.get("duration")), 2),
            round(mins, 3),
            round(total_cost * share, 4),
            job.get("batch_id") or "",
            job.get("batch_kind") or "",
        ])
    return out


# ── Screen 4: fleet & review queue ───────────────────────────────────

def stage_health(recent: Dict[str, Sequence[float]],
                 baseline: Dict[str, Sequence[float]]) -> List[Dict[str, Any]]:
    """Per-stage p95 against a 7-day baseline — the design's pool table.

    `recent` and `baseline` map a stage id to that stage's **rates**: seconds
    of stage time per second of source. The caller normalises (see
    ``server._admin_stage_samples``) because a raw duration mostly measures
    how long the video was, so a raw comparison flags every stage on any day
    whose videos ran longer than last week's.

    A stage with too few samples on either side reports its p95 with
    ``ratio: None`` rather than a comparison built on two runs.
    """
    out: List[Dict[str, Any]] = []
    for stage in sorted(set(recent) | set(baseline)):
        now_samples = list(recent.get(stage) or ())
        base_samples = list(baseline.get(stage) or ())
        p95 = percentile(now_samples, 95)
        base_p95 = percentile(base_samples, 95)
        ratio: Optional[float] = None
        severity = "ok"
        if (len(now_samples) >= MIN_STAGE_SAMPLES
                and len(base_samples) >= MIN_BASELINE_SAMPLES and base_p95 > 0):
            ratio = round(p95 / base_p95, 2)
            if ratio >= STAGE_BAD_RATIO:
                severity = "bad"
            elif ratio >= STAGE_WARN_RATIO:
                severity = "warn"
        out.append({
            "stage": stage,
            "runs": len(now_samples),
            # Three decimals: these are rates, and a fast stage like merge
            # runs at hundredths of realtime — one decimal would print 0.0
            # for most of the table.
            "p50": round(percentile(now_samples, 50), 3),
            "p95": round(p95, 3),
            "baseline_p95": round(base_p95, 3),
            "baseline_runs": len(base_samples),
            "ratio": ratio,
            "severity": severity,
        })
    return out


def stage_alerts(health: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The degraded rows of `stage_health`, worst first."""
    bad = [h for h in health if h.get("severity") in ("warn", "bad")]
    bad.sort(key=lambda h: -(h.get("ratio") or 0))
    return bad


def review_queue(jobs: Iterable[Dict[str, Any]],
                 keys: Iterable[Dict[str, Any]] = (), *,
                 now: Optional[float] = None,
                 presets: Iterable[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """The design's abuse queue, over signals this server can actually see.

    Three detector families, all real:

    * **Usage anomaly.** Yesterday's metered minutes against the trailing
      median. On a hosted deployment this is the shape of a leaked key; on a
      local one it is still the shape of a script that got away.
    * **Unattested voice references.** Jobs that cloned from an uploaded
      sample without a consent attestation (``job["voice_consent"]``,
      written by POST /api/dub/{id}/consent). Clones that carry one are not
      listed — they were checked, and passed. `presets`, when given
      (scan_file_presets().values()), adds the library-preset attestation
      count to the same item.
    * **Credential hygiene.** The same checks the accounts screen runs.

    Deliberately absent: DMCA and copyright. A claim queue needs somewhere
    for a claim to arrive, and this server has no such intake.
    """
    now_ts = _now(now)
    jobs = list(jobs or ())
    items: List[Dict[str, Any]] = []

    series = usage_series(jobs, since=now_ts - 14 * DAY, until=now_ts, bucket="day")
    if len(series) >= 8:
        today = series[-1]["minutes"]
        prior = sorted(row["minutes"] for row in series[:-1])
        median = prior[len(prior) // 2]
        if today >= ANOMALY_MIN_MINUTES and median > 0 and today / median >= ANOMALY_RATIO:
            items.append({
                "kind": "usage_anomaly",
                "severity": "warn",
                "title": "Usage anomaly — today is well above baseline",
                "detail": f"{today:.0f} metered minutes today vs a "
                          f"{median:.0f}-minute median over the prior 13 days "
                          f"({today / median:.1f}x)",
                "at": now_ts,
                "actions": ["Open jobs", "Review API keys"],
            })

    cloned = [j for j in jobs
              if (j.get("voice_mode") or "") == "upload"
              and _num(j.get("created")) >= now_ts - 30 * DAY]
    if cloned:
        items.append({
            "kind": "voice_clone",
            "severity": "warn",
            "title": f"{len(cloned)} voice clone{'s' if len(cloned) != 1 else ''} "
                     f"from an uploaded reference (30d)",
            "detail": "this build stores no consent attestation with an "
                      "uploaded voice sample, so there is nothing to check "
                      "these against — listed so they can be looked at, not "
                      "because anything is known to be wrong",
            "at": max(_num(j.get("created")) for j in cloned),
            "job_ids": [j.get("id") for j in cloned[:8]],
            "actions": ["Open job", "Listen to reference"],
        })

    for issue in key_issues(keys, now=now_ts):
        items.append({
            "kind": "credential",
            "severity": issue["severity"],
            "title": issue["title"],
            "detail": issue["detail"],
            "at": now_ts,
            "key_id": issue["key_id"],
            "actions": ["Revoke key"],
        })

    order = {"bad": 0, "warn": 1, "ok": 2}
    items.sort(key=lambda i: (order.get(i["severity"], 3), -i.get("at", 0)))
    return items


def fleet(jobs: Iterable[Dict[str, Any]],
          keys: Iterable[Dict[str, Any]] = (), *,
          now: Optional[float] = None,
          queue_depth: int = 0,
          gpu: Optional[Dict[str, Any]] = None,
          gpu_backend: str = "",
          gpu_name: str = "",
          health: Sequence[Dict[str, Any]] = (),
          intake_paused: bool = False) -> Dict[str, Any]:
    """Screen ``3a Fleet and abuse queue``.

    "Fleet" is one node — the machine this is running on — and the console
    says so rather than drawing three invented pools. What it does have that
    the design asks for, and that a hosted fleet view would want first, is
    the per-stage p95-against-baseline table.
    """
    now_ts = _now(now)
    jobs = list(jobs or ())

    running = [j for j in jobs if job_state(j) == "active"]
    day_jobs = [j for j in jobs if _num(j.get("created")) >= now_ts - DAY]
    day_done = sum(1 for j in day_jobs if job_state(j) == "done")
    day_failed = sum(1 for j in day_jobs if job_state(j) == "failed")
    graded = day_done + day_failed
    fail_pct = round(day_failed / graded * 100.0, 1) if graded else None

    reasons = Counter((j.get("error") or "unknown").split("\n")[0][:60]
                      for j in day_jobs if job_state(j) == "failed")
    top_failure = reasons.most_common(1)[0][0] if reasons else ""

    gpu = dict(gpu or {})
    util = gpu.get("util_pct")
    alerts = stage_alerts(health)

    # Each branch names the cause it actually fired on. An earlier version
    # said "see stages below" whenever anything was wrong, which sent the
    # reader to a table of eight healthy rows when the real problem was the
    # failure rate.
    if intake_paused:
        status, status_text = "warn", "intake paused — nothing new will start"
    elif any(a["severity"] == "bad" for a in alerts):
        status = "bad"
        status_text = f"{alerts[0]['stage']} is badly degraded"
    elif alerts:
        status = "warn"
        status_text = f"{alerts[0]['stage']} is slower than its baseline"
    elif fail_pct is not None and fail_pct > 10:
        status = "warn"
        status_text = f"{fail_pct:.0f}% of jobs failed in the last 24h"
    else:
        status, status_text = "ok", "all systems nominal"

    return {
        "now": now_ts,
        "status": status,
        "status_text": status_text,
        "intake_paused": bool(intake_paused),
        "tiles": [
            {"key": "queue", "label": "Queue depth", "value": queue_depth,
             "note": f"{len(running)} in a running status"},
            {"key": "running", "label": "Running now", "value": len(running),
             "note": "the queue is single-consumer by design"},
            {"key": "util", "label": "GPU util",
             "value": round(util, 1) if isinstance(util, (int, float)) else None,
             "note": gpu_backend or "no GPU telemetry"},
            {"key": "failures", "label": "Failures · 24h", "value": fail_pct,
             "note": f"top cause: {top_failure}" if top_failure else "none"},
        ],
        "node": {
            "name": gpu_name or "this machine",
            "backend": gpu_backend or "",
            "gpu": gpu,
            "running": len(running),
            "queued": queue_depth,
            # Said out loud on the page: the design's three-pool table has no
            # counterpart here, and pretending otherwise would make a
            # capacity screen that cannot be trusted.
            "note": ("One node. GoChiDUBB runs the pipeline in-process on the "
                     "machine serving this page — there is no scheduler, no "
                     "autoscaler and no second pool to route around."),
        },
        "stages": list(health),
        "alerts": alerts,
        "review_queue": review_queue(jobs, keys, now=now_ts),
    }
