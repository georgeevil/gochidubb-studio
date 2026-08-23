"""Pre-flight estimates — how long a job will take, before it starts.

The wizard needs two numbers on the screen where Start is pressed: a price
and a wait. The price comes from `app/billing.py`; the wait comes from here.

Two things about the wait are worth stating plainly, because both are easy
to get flatteringly wrong:

* **The queue is serial.** `enqueue_job` feeds one `_job_queue_worker`
  (server.py), deliberately — concurrent dubs would OOM the GPU. Three
  languages therefore run one after another, so the ETA multiplies by the
  language count. Quoting single-language time for a three-language batch
  would be off by 3x on the very first thing a new user does.
* **The factor is measured, not asserted.** `realtime_factor` reads real
  finished jobs off this install and takes the median seconds-of-wall-clock
  per second-of-source. Only when there are too few samples does it fall
  back to `cfg.eta_realtime_factor`, and the caller is expected to say
  which of the two it used.

Pure module: no I/O, no model loads, no server imports.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional

# Below this many completed jobs a median is a coin-flip, so the configured
# default wins instead.
MIN_SAMPLES = 3

# Ratios outside this window are not measurements of anything — a job whose
# clock spanned an overnight pause, or one whose duration was mis-recorded.
# The median already resists outliers; this stops a pathological one from
# being the median when there are only three samples.
MIN_PLAUSIBLE_FACTOR = 0.05
MAX_PLAUSIBLE_FACTOR = 200.0

# Statuses whose timing says nothing about how long a dub takes.
_COMPLETED_STATUSES = {"complete"}


def _sample(job: Dict[str, Any]) -> Optional[float]:
    """Seconds of wall clock per second of source, or None if unmeasurable."""
    if (job.get("status") or "") not in _COMPLETED_STATUSES:
        return None
    try:
        started = float(job.get("started_at") or 0.0)
        completed = float(job.get("completed_at") or 0.0)
        duration = float(job.get("duration") or 0.0)
    except (TypeError, ValueError):
        return None
    if started <= 0 or completed <= started or duration <= 0:
        return None
    factor = (completed - started) / duration
    if not (MIN_PLAUSIBLE_FACTOR <= factor <= MAX_PLAUSIBLE_FACTOR):
        return None
    return factor


def realtime_factor(jobs: Iterable[Dict[str, Any]], *,
                    min_samples: int = MIN_SAMPLES) -> Optional[float]:
    """Median realtime factor over this install's finished jobs.

    Returns None when fewer than `min_samples` jobs carry all of
    `status="complete"`, `started_at`, `completed_at` and `duration` — the
    caller then uses the configured default and labels the ETA as such.
    """
    samples: List[float] = []
    for job in jobs or ():
        s = _sample(job)
        if s is not None:
            samples.append(s)
    if len(samples) < max(1, int(min_samples)):
        return None
    return float(statistics.median(samples))


def eta_seconds(duration_sec: float, n_langs: int, factor: float) -> float:
    """Wall-clock seconds to dub `duration_sec` of source into `n_langs`.

    Multiplied by the language count on purpose — see the module docstring.
    """
    duration = max(0.0, float(duration_sec or 0.0))
    langs = max(0, int(n_langs or 0))
    f = max(0.0, float(factor or 0.0))
    return duration * langs * f


def billable_minutes(duration_sec: float, n_langs: int) -> float:
    """Metered minutes for one source dubbed into N languages.

    Mirrors `billing.job_minutes` — source length x target languages — so
    the wizard's quote and the meter's later reading are the same
    arithmetic on the same inputs.
    """
    duration = max(0.0, float(duration_sec or 0.0))
    langs = max(0, int(n_langs or 0))
    return (duration / 60.0) * langs
