"""Tests for the usage meter (app/billing.py).

Tier arithmetic is exactly the kind of thing that looks right and is off by a
band, so the boundaries get pinned down explicitly. The other thing worth
guarding is that failed work is never billed.
"""
import pytest

from app import billing


def test_first_band_is_linear():
    r = billing.price_minutes(100)
    assert r["cost"] == pytest.approx(100 * 0.080)
    assert r["rate"] == 0.080


def test_bands_are_cumulative_not_replacing():
    # 600 min = 500 @ 0.080 + 100 @ 0.065, NOT 600 @ 0.065.
    r = billing.price_minutes(600)
    assert r["cost"] == pytest.approx(500 * 0.080 + 100 * 0.065)
    assert r["rate"] == 0.065


def test_third_band():
    # 2,500 min = 500@.08 + 1500@.065 + 500@.05
    r = billing.price_minutes(2500)
    assert r["cost"] == pytest.approx(500 * 0.080 + 1500 * 0.065 + 500 * 0.050)
    assert r["rate"] == 0.050


@pytest.mark.parametrize("minutes,expected", [
    (0, 0.080), (499.9, 0.080),
    (500, 0.065), (1999.9, 0.065),
    (2000, 0.050), (10_000, 0.050),
])
def test_tier_boundaries(minutes, expected):
    """The rate must change exactly at the boundary, not one minute either side."""
    assert billing.tier_for(minutes) == expected


def test_zero_and_negative_minutes_cost_nothing():
    assert billing.price_minutes(0)["cost"] == 0
    assert billing.price_minutes(-5)["cost"] == 0


def test_next_tier_countdown():
    nxt = billing.minutes_to_next_tier(400)
    assert nxt["minutes"] == 100 and nxt["next_rate"] == 0.065
    assert billing.minutes_to_next_tier(2500) is None


def test_job_minutes_multiplies_by_language_count():
    assert billing.job_minutes({"duration": 120, "status": "complete"}) == 2.0
    multi = {"duration": 120, "status": "complete",
             "target_langs": ["fr", "es", "ja"]}
    assert billing.job_minutes(multi) == 6.0


@pytest.mark.parametrize("status", ["error", "cancelled", "failed"])
def test_failed_jobs_are_never_billed(status):
    assert billing.job_minutes({"duration": 600, "status": status}) == 0.0


def test_job_without_duration_is_not_guessed():
    assert billing.job_minutes({"status": "complete"}) == 0.0
    assert billing.job_minutes({"duration": 0, "status": "complete"}) == 0.0


def test_summarize_counts_only_real_work():
    jobs = [
        {"id": "a", "duration": 600, "status": "complete", "target_lang": "fr"},
        {"id": "b", "duration": 600, "status": "error", "target_lang": "es"},
        {"id": "c", "duration": 300, "status": "complete", "target_lang": "fr"},
    ]
    s = billing.summarize(jobs)
    assert s["minutes"] == pytest.approx(15.0)   # 10 + 5, the error ignored
    assert s["jobs_counted"] == 2
    assert s["jobs_unbilled"] == 1
    assert s["by_lang"]["fr"] == pytest.approx(15.0)
    assert "es" not in s["by_lang"]
    assert s["estimate"] is True


def test_summarize_honours_the_since_window():
    jobs = [
        {"id": "old", "duration": 600, "status": "complete", "created": 100},
        {"id": "new", "duration": 600, "status": "complete", "created": 900},
    ]
    assert billing.summarize(jobs, since=500)["jobs_counted"] == 1


def test_showcase_surcharge_applies_on_top():
    jobs = [{"id": "s", "duration": 600, "status": "complete",
             "batch_kind": "showcase", "target_lang": "fr"}]
    s = billing.summarize(jobs)
    assert s["showcase_minutes"] == pytest.approx(10.0)
    assert s["showcase_surcharge"] == pytest.approx(10.0 * 0.020)
    assert s["cost"] == pytest.approx(10.0 * 0.080 + 10.0 * 0.020)


def test_storage_is_free_up_to_the_included_allowance():
    jobs = [{"id": "a", "duration": 60, "status": "complete"}]
    assert billing.summarize(jobs, storage_gb=50)["storage_cost"] == 0
    assert billing.summarize(jobs, storage_gb=100)["storage_cost"] == 0
    over = billing.summarize(jobs, storage_gb=150)
    assert over["storage_cost"] == pytest.approx(50 * 0.02)


def test_bands_sum_to_the_total_cost():
    r = billing.price_minutes(3000)
    assert sum(b["cost"] for b in r["bands"]) == pytest.approx(r["cost"])


def test_every_band_is_reported_even_when_unreached():
    """The screen shows the whole schedule with your position in it.

    Reporting only the band you have reached hides the cheaper rates ahead,
    which is exactly what the first cut of this did.
    """
    r = billing.price_minutes(100)
    assert len(r["bands"]) == len(billing.TIERS)
    assert r["bands"][0]["minutes"] == pytest.approx(100)
    assert [b["minutes"] for b in r["bands"][1:]] == [0, 0]
    # And the unreached bands still advertise their rate.
    assert [b["rate"] for b in r["bands"]] == [0.080, 0.065, 0.050]


# ── marginal_cost — what the wizard quotes before Start ───────────────

def test_marginal_cost_is_a_difference_not_a_standalone_price():
    """The bug this pins: pricing the new minutes from zero.

    A workspace 480 minutes into the month buying 40 more pays the first
    rate for 20 of them and the second for the other 20. Pricing 40 minutes
    standalone charges all 40 at the first rate — $3.20 in the wizard,
    $2.90 on the meter, for the same job.
    """
    r = billing.marginal_cost(480, 40)
    assert r["cost"] == pytest.approx(20 * 0.080 + 20 * 0.065)
    assert r["cost"] != pytest.approx(billing.price_minutes(40)["cost"])


def test_marginal_cost_from_zero_matches_a_standalone_price():
    r = billing.marginal_cost(0, 36)
    assert r["cost"] == pytest.approx(billing.price_minutes(36)["cost"])


def test_marginal_plus_used_equals_the_new_total():
    """Adding the quote to this month's bill must land on next month's
    reading of the same meter — no rounding drift between the two."""
    used, new = 1900.0, 250.0
    r = billing.marginal_cost(used, new)
    assert r["total_cost"] == pytest.approx(billing.price_minutes(used + new)["cost"])
    assert r["cost"] + billing.price_minutes(used)["cost"] == pytest.approx(
        r["total_cost"], abs=0.01)


def test_marginal_bands_describe_only_the_new_minutes():
    r = billing.marginal_cost(480, 40)
    assert [b["minutes"] for b in r["bands"]] == [20, 20, 0]
    assert sum(b["cost"] for b in r["bands"]) == pytest.approx(r["cost"])


def test_marginal_rate_is_where_the_purchase_starts():
    r = billing.marginal_cost(480, 40)
    assert r["rate"] == 0.080                       # first new minute
    assert r["effective_rate"] == pytest.approx(r["cost"] / 40)


def test_marginal_cost_of_nothing_is_free():
    r = billing.marginal_cost(120, 0)
    assert r["cost"] == 0
    assert r["rate"] == 0.080
    assert r["estimate"] is True


def test_marginal_cost_ignores_negative_input():
    assert billing.marginal_cost(-10, -10)["cost"] == 0


def test_marginal_cost_is_what_the_meter_will_move_by():
    """`cost` is the difference of two *displayed* totals, not of the raw
    arithmetic — so it is exactly what the meter's headline changes by.

    Pinned because it is tempting to "fix" the cent of disagreement with
    the band breakdown by recomputing `cost` unrounded, which would make
    the wizard quote a number the meter then contradicts.
    """
    used, new = 490.0, 20.0
    r = billing.marginal_cost(used, new)
    meter_before = billing.price_minutes(used)["cost"]
    meter_after = billing.price_minutes(used + new)["cost"]
    assert r["cost"] == pytest.approx(meter_after - meter_before, abs=1e-9)


def test_marginal_bands_are_a_decomposition_not_the_quote():
    """The bands may disagree with `cost` by a cent, by design.

    They are the exact unrounded split of the same minutes, rounded once
    each for display; `cost` is a difference of already-rounded totals. A
    caller must render `cost` as the total rather than summing the bands.
    """
    for used, new in [(490.0, 20.0), (0.0, 36.2), (1990.0, 25.0),
                      (123.45, 67.89)]:
        r = billing.marginal_cost(used, new)
        band_sum = sum(b["cost"] for b in r["bands"])
        # Same money, to within the rounding that separates them.
        assert abs(band_sum - r["cost"]) <= 0.01 + 1e-9, (used, new)
        # And the minutes always reconcile exactly — only money rounds.
        assert sum(b["minutes"] for b in r["bands"]) == pytest.approx(
            r["new_minutes"], abs=0.02)
