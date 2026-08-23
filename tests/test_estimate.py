"""Tests for the pre-flight estimate (app/estimate.py).

Two things are worth pinning here, and both are places where a plausible
implementation would quietly lie to the user:

* the ETA multiplies by the language count, because the job queue is
  single-consumer and N languages run one after another;
* the measured factor is only used once there is enough data to measure,
  and jobs that never finished contribute nothing.
"""
import pytest

from app import estimate


# started_at/completed_at are unix epochs on a real job; 0 means "never
# recorded", which is why the helper offsets from a plausible clock.
T0 = 1_700_000_000.0


def _job(duration, started, completed, status="complete"):
    return {"status": status, "duration": duration,
            "started_at": (T0 + started) if started else started,
            "completed_at": (T0 + completed) if completed else completed}


# ── realtime_factor ───────────────────────────────────────────────────

def test_too_few_samples_returns_none():
    jobs = [_job(60, 100, 460), _job(60, 500, 860)]
    assert estimate.realtime_factor(jobs) is None


def test_median_of_completed_jobs():
    # 6x, 4x, 8x -> median 6x
    jobs = [_job(60, 10, 370), _job(60, 10, 250), _job(60, 10, 490)]
    assert estimate.realtime_factor(jobs) == pytest.approx(6.0)


def test_unfinished_jobs_are_not_samples():
    # Three jobs, but only one of them finished — not enough to measure.
    jobs = [
        _job(60, 10, 370),
        _job(60, 10, 370, status="error"),
        _job(60, 10, 370, status="synthesizing"),
    ]
    assert estimate.realtime_factor(jobs) is None


def test_jobs_missing_timing_are_not_samples():
    jobs = [
        _job(60, 10, 370),
        {"status": "complete", "duration": 60},              # no timestamps
        {"status": "complete", "started_at": T0, "completed_at": T0 + 60},
        _job(0, 10, 370),                                    # zero duration
    ]
    assert estimate.realtime_factor(jobs) is None


def test_implausible_ratios_are_discarded():
    # A job whose clock spanned an overnight pause is not a measurement of
    # anything. Dropping it leaves two samples, which is below the floor.
    jobs = [_job(60, 10, 370), _job(60, 10, 310), _job(10, 10, 100_000)]
    assert estimate.realtime_factor(jobs) is None


def test_empty_and_none_are_safe():
    assert estimate.realtime_factor([]) is None
    assert estimate.realtime_factor(None) is None


# ── eta_seconds ───────────────────────────────────────────────────────

def test_eta_multiplies_by_language_count():
    """The queue is serial — three languages take three times as long.

    This is the number the design wanted flattered ("about 20 minutes" for a
    three-language 12-minute video). It is not achievable on one GPU worker
    and quoting it would make every other number on the screen suspect.
    """
    one = estimate.eta_seconds(720, 1, 6.0)
    three = estimate.eta_seconds(720, 3, 6.0)
    assert one == pytest.approx(4320)
    assert three == pytest.approx(3 * one)


def test_eta_zero_languages_is_zero():
    assert estimate.eta_seconds(720, 0, 6.0) == 0


def test_eta_tolerates_junk():
    assert estimate.eta_seconds(None, None, None) == 0
    assert estimate.eta_seconds(-5, 3, 6.0) == 0


# ── billable_minutes ──────────────────────────────────────────────────

def test_billable_minutes_matches_the_meter():
    """Same arithmetic as billing.job_minutes, so the quote and the meter
    cannot disagree about what was metered."""
    from app import billing
    quoted = estimate.billable_minutes(724, 3)
    metered = billing.job_minutes(
        {"status": "complete", "duration": 724, "target_langs": ["es", "fr", "ja"]})
    assert quoted == pytest.approx(metered)


def test_billable_minutes_zero_languages():
    assert estimate.billable_minutes(724, 0) == 0
