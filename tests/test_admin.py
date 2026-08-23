"""Unit tests for `app/admin.py` — the vendor admin console's aggregation.

The module is pure and takes its clock as an argument, so every test here
pins `now` and asserts on exact numbers rather than on shapes. That matters
more than usual for this module: it is the one that turns real jobs into
dollar figures, and a rounding or allocation bug would be invisible on the
page and wrong on the export.
"""
import time

import pytest

from app import admin, billing


NOW = 1_780_000_000.0        # a fixed clock; nothing here reads time.time()
DAY = 86400.0


def job(jid, *, status="complete", duration=600.0, langs=1, created_ago=3600.0,
        **kw):
    """A job dict shaped like the ones server.py's runner writes."""
    out = {
        "id": jid,
        "status": status,
        "duration": duration,
        "target_lang": kw.pop("target_lang", "es"),
        "created": NOW - created_ago,
    }
    if langs > 1:
        out["target_langs"] = ["es", "fr", "de", "ja", "pt"][:langs]
    out.update(kw)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Status vocabulary
# ═══════════════════════════════════════════════════════════════════════

def test_cancelled_is_not_counted_as_a_failure():
    """A user stopping their own job is not the service failing.

    If `cancelled` ever slid into FAILED_STATUSES the success rate would
    drop every time somebody changed their mind, and the overview would
    report an outage that never happened.
    """
    assert "cancelled" not in admin.FAILED_STATUSES
    assert admin.job_state({"status": "cancelled"}) == "cancelled"

    jobs = [job("a"), job("b", status="cancelled")]
    counts = admin._counts(jobs, NOW - 30 * DAY, NOW)
    assert counts["success_pct"] == 100.0
    assert counts["cancelled"] == 1


def test_success_rate_is_none_before_anything_has_finished():
    counts = admin._counts([job("a", status="transcribing")], NOW - DAY, NOW)
    assert counts["success_pct"] is None


def test_every_status_the_runner_writes_is_classified():
    """No status may fall through to "other" — that bucket counts nowhere.

    The list is server.py's job vocabulary. A new status added there without
    a home here would silently vanish from the tiles.
    """
    runner_statuses = [
        "preparing", "queued", "scheduled", "uploaded", "downloading",
        "extracting", "transcribing", "diarizing", "translating",
        "synthesizing", "assembling", "merging", "awaiting_translation_review",
        "paused", "complete", "error", "cancelled", "interrupted",
    ]
    for status in runner_statuses:
        assert admin.job_state({"status": status}) != "other", status


# ═══════════════════════════════════════════════════════════════════════
#  usage_series — the revenue chart
# ═══════════════════════════════════════════════════════════════════════

def test_series_buckets_sum_to_the_window_total():
    """The bars must add up to the headline figure, exactly.

    This is the whole reason per-bucket cost is computed marginally. Pricing
    each day's minutes from zero would charge every day the first-band rate;
    with enough volume to cross a tier the bars would visibly out-total the
    number printed above them.
    """
    # 4,000 minutes of source spread over four days — well past the 500 and
    # 2,000 minute tier boundaries, so a from-zero-per-day bug shows up.
    jobs = [job(f"j{i}", duration=60_000.0, created_ago=(i + 1) * DAY)
            for i in range(4)]
    series = admin.usage_series(jobs, since=NOW - 10 * DAY, until=NOW)
    total = billing.price_minutes(sum(admin._billable(j) for j in jobs))["cost"]
    assert round(sum(s["cost"] for s in series), 2) == pytest.approx(total, abs=0.02)


def test_series_seeds_quiet_days_as_zero_bars():
    """A day nobody dubbed is a zero bar, not a missing one."""
    series = admin.usage_series([job("a", created_ago=DAY)],
                                since=NOW - 7 * DAY, until=NOW)
    assert len(series) >= 7
    assert sum(1 for s in series if s["minutes"] == 0) >= 6
    assert [s["bucket"] for s in series] == sorted(s["bucket"] for s in series)


def test_series_can_bucket_by_month():
    jobs = [job("a", created_ago=5 * DAY), job("b", created_ago=70 * DAY)]
    series = admin.usage_series(jobs, since=NOW - 120 * DAY, until=NOW,
                                bucket="month")
    assert all(len(s["bucket"]) == 7 for s in series)     # YYYY-MM
    assert sum(s["jobs"] for s in series) == 2


# ═══════════════════════════════════════════════════════════════════════
#  top_sources — pro-rata allocation
# ═══════════════════════════════════════════════════════════════════════

def test_top_sources_allocation_sums_to_the_window_cost():
    jobs = [
        job("a", duration=60_000.0, batch_id="b1", batch_label="Reel one"),
        job("b", duration=30_000.0, batch_id="b1", batch_label="Reel one"),
        job("c", duration=90_000.0, batch_id="b2", batch_label="Reel two"),
    ]
    rows = admin.top_sources(jobs, since=NOW - 30 * DAY, until=NOW)
    total = billing.price_minutes(sum(admin._billable(j) for j in jobs))["cost"]
    assert round(sum(r["cost"] for r in rows), 2) == pytest.approx(total, abs=0.02)


def test_top_sources_groups_siblings_by_batch_and_keeps_the_worst_health():
    jobs = [
        job("a", batch_id="b1", target_lang="es"),
        job("b", batch_id="b1", target_lang="fr", status="error"),
    ]
    rows = admin.top_sources(jobs, since=NOW - 30 * DAY, until=NOW)
    assert len(rows) == 1
    assert rows[0]["jobs"] == 2
    assert rows[0]["langs"] == ["es", "fr"]
    assert rows[0]["health"] == "bad"
    assert rows[0]["mode"] == "multi-language"


def test_top_sources_strips_the_fan_out_suffix_from_a_label():
    """`/api/quick_test` writes "<title> -> ES"; siblings regroup under one name."""
    rows = admin.top_sources([job("a", source_label="Nordic hiking guide -> ES")],
                             since=NOW - 30 * DAY, until=NOW)
    assert rows[0]["label"] == "Nordic hiking guide"


def test_top_sources_survives_a_window_with_no_billable_minutes():
    """Every job failed: minutes are zero, and the division must not blow up."""
    rows = admin.top_sources([job("a", status="error")],
                             since=NOW - 30 * DAY, until=NOW)
    assert rows[0]["cost"] == 0.0
    assert rows[0]["effective_rate"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  overview
# ═══════════════════════════════════════════════════════════════════════

def test_overview_deltas_compare_against_the_preceding_window():
    jobs = [
        job("now1", duration=600.0, created_ago=2 * DAY),
        job("now2", duration=600.0, created_ago=3 * DAY),
        job("old", duration=600.0, created_ago=40 * DAY),
    ]
    d = admin.overview(jobs, days=30, now=NOW)
    minutes = next(t for t in d["tiles"] if t["key"] == "minutes")
    assert minutes["value"] == 20.0                 # 2 x 10 min
    assert d["previous"]["minutes"] == 10.0
    assert minutes["delta_pct"] == 100.0


def test_overview_reports_no_delta_rather_than_infinite_growth():
    d = admin.overview([job("a", created_ago=DAY)], days=30, now=NOW)
    assert next(t for t in d["tiles"] if t["key"] == "minutes")["delta_pct"] is None


def test_overview_realtime_tile_is_blank_until_there_are_enough_samples():
    from app import estimate
    few = [job(f"j{i}", created_ago=DAY, started_at=NOW - 900,
               completed_at=NOW - 300) for i in range(estimate.MIN_SAMPLES - 1)]
    tile = next(t for t in admin.overview(few, days=30, now=NOW)["tiles"]
                if t["key"] == "throughput")
    assert tile["value"] is None
    assert str(estimate.MIN_SAMPLES) in tile["note"]

    many = [job(f"k{i}", created_ago=DAY, duration=600.0,
                started_at=NOW - 4000, completed_at=NOW - 1000)
            for i in range(estimate.MIN_SAMPLES)]
    tile = next(t for t in admin.overview(many, days=30, now=NOW)["tiles"]
                if t["key"] == "throughput")
    assert tile["value"] == pytest.approx(5.0)      # 3000s wall / 600s source


def test_overview_window_of_a_year_switches_to_monthly_buckets():
    d = admin.overview([job("a", created_ago=DAY)], days=365, now=NOW)
    assert d["bucket"] == "month"
    assert len(d["series"]) <= 14


# ═══════════════════════════════════════════════════════════════════════
#  attention
# ═══════════════════════════════════════════════════════════════════════

def test_attention_surfaces_failures_paused_jobs_and_stale_runs():
    jobs = [
        job("f", status="error", error="tts timeout", created_ago=DAY),
        job("p", status="awaiting_translation_review", created_ago=DAY),
        job("s", status="transcribing", created_ago=3 * DAY),
    ]
    titles = " ".join(i["title"] for i in admin.attention(jobs, now=NOW))
    assert "failed" in titles
    assert "paused for a human" in titles
    assert "active for over a day" in titles


def test_attention_is_ordered_worst_first():
    jobs = [job("p", status="awaiting_translation_review", created_ago=DAY),
            job("f", status="error", created_ago=DAY)]
    items = admin.attention(jobs, now=NOW)
    assert items[0]["severity"] == "bad"


def test_attention_flags_storage_only_past_the_included_allowance():
    assert not any("allowance" in i["title"]
                   for i in admin.attention([], now=NOW, storage_gb=50.0))
    over = admin.attention([], now=NOW,
                           storage_gb=billing.STORAGE_INCLUDED_GB + 20.0)
    assert any("allowance" in i["title"] for i in over)


def test_attention_is_empty_on_a_clean_install():
    assert admin.attention([], [], now=NOW, storage_gb=0.0) == []


# ═══════════════════════════════════════════════════════════════════════
#  Keys
# ═══════════════════════════════════════════════════════════════════════

def key(kid="k1", **kw):
    out = {"id": kid, "name": kid, "environment": "live", "scopes": ["jobs:read"],
           "created": NOW - 10 * DAY, "expires": None, "last_used": None,
           "revoked": None, "masked": "gcd_live_••••abcd"}
    out.update(kw)
    return out


@pytest.mark.parametrize("rec,expected", [
    (key(revoked=NOW - DAY), "revoked"),
    (key(expires=NOW - DAY), "expired"),
    (key(expires=NOW + 2 * DAY), "expiring"),
    (key(), "idle"),
    (key(last_used=NOW - 60), "active"),
])
def test_key_state(rec, expected):
    assert admin.key_state(rec, now=NOW) == expected


def test_a_revoked_key_raises_no_hygiene_issue():
    """Revoking is the fix; keeping the record is not a second problem."""
    assert admin.key_issues([key(expires=NOW - DAY, revoked=NOW)], now=NOW) == []


def test_expired_but_live_key_is_the_worst_hygiene_issue():
    issues = admin.key_issues([key(expires=NOW - DAY)], now=NOW)
    assert issues[0]["severity"] == "bad"
    assert issues[0]["key_id"] == "k1"


def test_an_old_key_with_no_expiry_is_flagged_and_a_young_one_is_not():
    old = key(created=NOW - (admin.KEY_STALE_DAYS + 10) * DAY)
    assert admin.key_issues([old], now=NOW)[0]["severity"] == "warn"
    assert admin.key_issues([key()], now=NOW) == []


def test_accounts_never_invents_a_per_key_spend_column():
    """Jobs do not record which credential started them.

    Splitting the workspace's usage between keys would be a guess dressed up
    as a number, so the flag stays False and the note says why.
    """
    d = admin.accounts([job("a")], [key()], now=NOW)
    assert d["spend_attributable"] is False
    assert all("cost" not in k and "minutes" not in k for k in d["keys"])
    assert "single-tenant" in d["note"]


def test_accounts_sorts_problem_keys_to_the_top():
    keys = [key("ok", last_used=NOW), key("dead", expires=NOW - DAY)]
    assert [k["id"] for k in admin.accounts([], keys, now=NOW)["keys"]] == ["dead", "ok"]


# ═══════════════════════════════════════════════════════════════════════
#  Revenue ops
# ═══════════════════════════════════════════════════════════════════════

def test_rate_card_publishes_every_band_including_unreached_ones():
    """The screen shows the whole schedule and where you are in it."""
    card = admin.rate_card([job("a", duration=600.0)],
                           since=NOW - 30 * DAY, until=NOW)
    assert len(card["bands"]) == len(billing.TIERS)
    assert [b["reached"] for b in card["bands"]] == [True, False, False]
    assert card["bands"][0]["rate"] == billing.TIERS[0][1]


def test_unbilled_prices_failed_jobs_at_what_they_would_have_cost():
    """`billing.job_minutes` returns 0 for a failure — correctly, for billing.

    The exposure figure is the interesting thing about a failed job, so this
    bucket has to reach past that and read the source duration itself.
    """
    d = admin.unbilled([job("f", status="error", duration=600.0)],
                       since=NOW - 30 * DAY, until=NOW)
    lost = next(b for b in d["buckets"] if b["severity"] == "bad")
    assert lost["count"] == 1
    assert lost["minutes"] == 10.0
    assert lost["cost"] > 0
    assert billing.job_minutes(job("f", status="error")) == 0.0


def test_unbilled_separates_in_flight_from_paused_from_failed():
    jobs = [job("r", status="synthesizing"),
            job("p", status="awaiting_translation_review"),
            job("f", status="error")]
    counts = [b["count"] for b in
              admin.unbilled(jobs, since=NOW - 30 * DAY, until=NOW)["buckets"]]
    assert counts == [1, 1, 1]


def test_periods_are_newest_first():
    jobs = [job("a", created_ago=5 * DAY), job("b", created_ago=70 * DAY)]
    periods = admin.periods(jobs, months=6, now=NOW)
    assert [p["period"] for p in periods] == sorted(
        (p["period"] for p in periods), reverse=True)


def test_revenue_says_credits_are_unavailable_rather_than_drawing_a_form():
    d = admin.revenue([job("a")], days=30, now=NOW)
    assert d["adjustments"]["available"] is False
    assert "no billing system" in d["adjustments"]["reason"]


def test_csv_has_a_header_and_one_row_per_job_in_window():
    jobs = [job("a", created_ago=DAY), job("b", created_ago=2 * DAY),
            job("old", created_ago=90 * DAY)]
    rows = admin.revenue_csv_rows(jobs, since=NOW - 30 * DAY, until=NOW)
    assert rows[0][0] == "job_id"
    assert len(rows) == 3
    assert {r[0] for r in rows[1:]} == {"a", "b"}


def test_csv_cost_column_sums_to_the_window_total():
    jobs = [job(f"j{i}", duration=60_000.0, created_ago=(i + 1) * DAY)
            for i in range(4)]
    rows = admin.revenue_csv_rows(jobs, since=NOW - 30 * DAY, until=NOW)
    total = billing.price_minutes(sum(admin._billable(j) for j in jobs))["cost"]
    assert sum(r[7] for r in rows[1:]) == pytest.approx(total, abs=0.02)


# ═══════════════════════════════════════════════════════════════════════
#  Fleet
# ═══════════════════════════════════════════════════════════════════════

def test_percentile_is_nearest_rank():
    assert admin.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert admin.percentile([1, 2, 3, 4], 50) == 2
    assert admin.percentile([], 95) == 0.0


def test_stage_health_suppresses_a_comparison_built_on_too_few_runs():
    health = admin.stage_health({"tts": [10.0, 900.0]}, {"tts": [10.0, 11.0]})
    assert health[0]["ratio"] is None
    assert health[0]["severity"] == "ok"
    assert health[0]["p95"] == 900.0            # still reported, just uncompared


def test_stage_health_grades_a_real_degradation():
    recent = {"tts": [20.0, 22.0, 24.0, 40.0, 42.0]}
    baseline = {"tts": [10.0] * 10}
    row = admin.stage_health(recent, baseline)[0]
    assert row["ratio"] == pytest.approx(4.2, abs=0.05)
    assert row["severity"] == "bad"
    assert admin.stage_alerts([row]) == [row]


def test_fleet_status_prefers_the_pause_then_the_worst_stage():
    bad = admin.stage_health({"tts": [40.0] * 5}, {"tts": [10.0] * 10})
    degraded = admin.fleet([], now=NOW, health=bad)
    assert degraded["status"] == "bad"
    assert degraded["status_text"] == "tts is badly degraded"

    paused = admin.fleet([], now=NOW, health=bad, intake_paused=True)
    assert paused["status"] == "warn"
    assert "intake paused" in paused["status_text"]

    assert admin.fleet([], now=NOW)["status_text"] == "all systems nominal"


def test_fleet_status_names_the_cause_it_actually_fired_on():
    """"Degraded — see stages below" with eight healthy rows helps nobody."""
    jobs = [job("a", status="error", created_ago=600.0),
            job("b", created_ago=600.0)]
    d = admin.fleet(jobs, now=NOW)          # 50% failures, no stage data
    assert d["status"] == "warn"
    assert "failed in the last 24h" in d["status_text"]
    assert "stages" not in d["status_text"]


def test_fleet_failure_rate_covers_only_the_last_day():
    jobs = [job("a", status="error", created_ago=3600.0),
            job("b", created_ago=3600.0),
            job("old", status="error", created_ago=5 * DAY)]
    tile = next(t for t in admin.fleet(jobs, now=NOW)["tiles"]
                if t["key"] == "failures")
    assert tile["value"] == 50.0
    assert "tts" not in tile["note"]


def test_fleet_says_there_is_one_node():
    node = admin.fleet([], now=NOW)["node"]
    assert "One node" in node["note"]


# ═══════════════════════════════════════════════════════════════════════
#  Review queue
# ═══════════════════════════════════════════════════════════════════════

def test_review_queue_detects_a_usage_spike_against_the_trailing_median():
    jobs = [job(f"base{i}", duration=600.0, created_ago=(i + 1) * DAY)
            for i in range(13)]
    jobs += [job(f"spike{i}", duration=600.0, created_ago=600.0)
             for i in range(20)]
    kinds = [i["kind"] for i in admin.review_queue(jobs, now=NOW)]
    assert "usage_anomaly" in kinds


def test_review_queue_ignores_a_spike_that_is_only_a_few_minutes():
    """Two jobs on a quiet install is a ratio, not an incident."""
    jobs = [job(f"base{i}", duration=60.0, created_ago=(i + 1) * DAY)
            for i in range(13)]
    jobs += [job("s1", duration=60.0, created_ago=600.0),
             job("s2", duration=60.0, created_ago=600.0),
             job("s3", duration=60.0, created_ago=600.0),
             job("s4", duration=60.0, created_ago=600.0)]
    assert not any(i["kind"] == "usage_anomaly"
                   for i in admin.review_queue(jobs, now=NOW))


def test_review_queue_lists_uploaded_voice_references_without_accusing():
    jobs = [job("v", voice_mode="upload", created_ago=DAY)]
    item = next(i for i in admin.review_queue(jobs, now=NOW)
                if i["kind"] == "voice_clone")
    assert item["job_ids"] == ["v"]
    assert "consent attestation" in item["detail"]


def test_review_queue_does_not_invent_a_dmca_backlog():
    kinds = {i["kind"] for i in admin.review_queue([job("a")], [], now=NOW)}
    assert "dmca" not in kinds and "copyright" not in kinds


def test_review_queue_is_empty_on_a_clean_install():
    assert admin.review_queue([], [], now=NOW) == []


# ═══════════════════════════════════════════════════════════════════════
#  Robustness
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"),
                                 "twelve", None, -5.0])
def test_a_nonsense_duration_meters_nothing_rather_than_exploding(bad):
    """NaN and infinity survive arithmetic and only die at round()/json.dumps.

    `/api/estimate` learned this the hard way; the console reads `duration`
    off the same job dicts, so it has to be just as unimpressed by them.
    """
    jobs = [job("a", duration=bad)]
    d = admin.overview(jobs, days=30, now=NOW)
    assert d["current"]["minutes"] >= 0.0
    admin.revenue(jobs, days=30, now=NOW)
    admin.fleet(jobs, now=NOW)
    admin.revenue_csv_rows(jobs, since=NOW - DAY, until=NOW)


def test_every_screen_renders_from_an_empty_install():
    assert admin.overview([], [], days=30, now=NOW)["current"]["jobs"] == 0
    assert admin.accounts([], [], now=NOW)["counts"]["total"] == 0
    assert admin.revenue([], days=30, now=NOW)["summary"]["minutes"] == 0.0
    assert admin.fleet([], [], now=NOW)["status"] == "ok"


def test_window_defaults_to_the_real_clock_when_none_is_given():
    since, prev, now = admin.window(30)
    assert now == pytest.approx(time.time(), abs=5)
    assert now - since == pytest.approx(30 * DAY)
    assert since - prev == pytest.approx(30 * DAY)
