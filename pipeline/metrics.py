"""Per-stage timing and resource observability for the dubbing pipeline.

Every pipeline stage (download → extract → transcribe → diarize → translate →
tts → assemble → merge) is wrapped in a `stage_timer()` context manager which:

  1. Logs a `[perf]` line on entry and exit with wall-clock duration.
  2. Samples CPU / RAM / GPU utilization on a background thread for the
     duration of the stage, and reports avg + peak.
  3. Appends the result to `<work_dir>/metrics.json` so the UI (and any
     post-hoc analysis) can show where the time actually went.

Everything here degrades gracefully: psutil, pynvml, torch and nvidia-smi are
all optional. When a probe is unavailable the corresponding fields are simply
absent from the summary rather than raising — observability must never be the
reason a dub fails.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("gochidubb.metrics")

# ── Optional probes ──────────────────────────────────────────────────
try:
    import psutil  # type: ignore
    _PROC = psutil.Process(os.getpid())
    # Prime the CPU counters — the first call to cpu_percent() always
    # returns 0.0 because it has no previous sample to diff against.
    _PROC.cpu_percent(None)
    psutil.cpu_percent(None)
except Exception:  # pragma: no cover - psutil is in requirements but optional
    psutil = None  # type: ignore
    _PROC = None

_CPU_COUNT = (psutil.cpu_count(logical=True) if psutil else os.cpu_count()) or 1

# Shortest gap between two CPU readings that yields a meaningful rate. Under
# this, cpu_percent(None) is dividing by a wall-clock delta close to zero.
_MIN_CPU_WINDOW = 0.05

METRICS_FILENAME = "metrics.json"


# ═════════════════════════════════════════════════════════════════════
# GPU probing
# ═════════════════════════════════════════════════════════════════════
class _GpuProbe:
    """Reads GPU utilization + memory through whichever backend exists.

    Backend priority:
      1. pynvml       — cheapest and most accurate on NVIDIA (no subprocess)
      2. torch.cuda   — memory only; utilization via torch.cuda.utilization()
      3. nvidia-smi   — subprocess fallback when neither library is installed
      4. torch.mps    — Apple Silicon: allocated memory only (no util API)

    The backend is resolved once, lazily, and cached. `read()` returns None
    when no GPU is visible so callers can omit GPU fields entirely.
    """

    def __init__(self) -> None:
        self._backend: Optional[str] = None
        self._handle = None
        self._nvml = None
        self._torch = None
        self._resolved = False

    def _resolve(self) -> None:
        if self._resolved:
            return
        self._resolved = True

        # 1. pynvml
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self._nvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._backend = "pynvml"
                log.debug("[perf] GPU probe backend: pynvml")
                return
        except Exception:
            pass

        # 2/4. torch (cuda, then mps)
        try:
            import torch  # type: ignore
            self._torch = torch
            if torch.cuda.is_available():
                self._backend = "torch.cuda"
                log.debug("[perf] GPU probe backend: torch.cuda")
                return
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._backend = "torch.mps"
                log.debug("[perf] GPU probe backend: torch.mps (memory only)")
                return
        except Exception:
            pass

        # 3. nvidia-smi
        if shutil.which("nvidia-smi"):
            self._backend = "nvidia-smi"
            log.debug("[perf] GPU probe backend: nvidia-smi")
            return

        self._backend = None
        log.debug("[perf] No GPU probe available — GPU metrics disabled")

    @property
    def backend(self) -> Optional[str]:
        self._resolve()
        return self._backend

    def read(self) -> Optional[Dict[str, float]]:
        """Return {util_pct?, mem_used_mb, mem_total_mb?} or None."""
        self._resolve()
        b = self._backend
        try:
            if b == "pynvml":
                nv = self._nvml
                util = nv.nvmlDeviceGetUtilizationRates(self._handle)
                mem = nv.nvmlDeviceGetMemoryInfo(self._handle)
                return {
                    "util_pct": float(util.gpu),
                    "mem_used_mb": mem.used / 1024 / 1024,
                    "mem_total_mb": mem.total / 1024 / 1024,
                }
            if b == "torch.cuda":
                t = self._torch
                free_b, total_b = t.cuda.mem_get_info()
                out = {
                    "mem_used_mb": (total_b - free_b) / 1024 / 1024,
                    "mem_total_mb": total_b / 1024 / 1024,
                }
                try:
                    # Requires pynvml under the hood on some builds; guard it.
                    out["util_pct"] = float(t.cuda.utilization())
                except Exception:
                    pass
                return out
            if b == "torch.mps":
                t = self._torch
                return {
                    "mem_used_mb": t.mps.current_allocated_memory() / 1024 / 1024,
                }
            if b == "nvidia-smi":
                r = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                line = (r.stdout or "").strip().splitlines()[0]
                util, used, total = [p.strip() for p in line.split(",")]
                return {
                    "util_pct": float(util),
                    "mem_used_mb": float(used),
                    "mem_total_mb": float(total),
                }
        except Exception as e:  # pragma: no cover — probe failures are non-fatal
            log.debug(f"[perf] GPU read failed ({b}): {e}")
        return None


_gpu = _GpuProbe()


def gpu_backend() -> Optional[str]:
    """Which GPU probe (if any) is active — surfaced in /api/system."""
    return _gpu.backend


def gpu_snapshot() -> Optional[Dict[str, float]]:
    """One-shot GPU reading, for endpoints that want a live value."""
    return _gpu.read()


# ═════════════════════════════════════════════════════════════════════
# Resource sampling
# ═════════════════════════════════════════════════════════════════════
class ResourceSampler:
    """Samples CPU/RAM/GPU on a daemon thread until stopped.

    Sampling runs on its own thread rather than inside the asyncio loop
    because most stages (WhisperX, VoxCPM, ffmpeg) block the loop for long
    stretches — an async sampler would simply never get scheduled and we'd
    record zero samples for exactly the stages we care about most.
    """

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = max(0.25, interval)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_proc: list[float] = []
        self._cpu_sys: list[float] = []
        self._rss_mb: list[float] = []
        self._gpu_util: list[float] = []
        self._gpu_mem: list[float] = []
        self._gpu_total_mb: Optional[float] = None
        # Set properly in start(); a sampler that is never started still has
        # a number here rather than an attribute error.
        self._last_cpu_at = 0.0

    def _sample_once(self) -> None:
        if _PROC is not None:
            try:
                # RSS is a reading, not a rate — always valid, however soon
                # after the last one it is taken.
                self._rss_mb.append(_PROC.memory_info().rss / 1024 / 1024)

                # CPU is a rate: cpu_percent(None) divides the CPU-time delta
                # by the wall-clock delta since the previous call. start()
                # primes the counters and _run() samples immediately after, so
                # that delta can be microseconds and the quotient enormous —
                # CI measured a 988% "peak" on a normalized scale whose
                # ceiling is 100. Below a window this short the number is an
                # artefact of thread-start latency, not a measurement, so it
                # is skipped rather than recorded and averaged in.
                now = time.monotonic()
                if now - self._last_cpu_at >= _MIN_CPU_WINDOW:
                    self._last_cpu_at = now
                    # Divide by core count so 100% means "all cores saturated"
                    # instead of psutil's per-core-sum convention (800% on 8).
                    self._cpu_proc.append(_PROC.cpu_percent(None) / _CPU_COUNT)
                    self._cpu_sys.append(psutil.cpu_percent(None))
            except Exception:
                pass
        g = _gpu.read()
        if g:
            if "util_pct" in g:
                self._gpu_util.append(g["util_pct"])
            if "mem_used_mb" in g:
                self._gpu_mem.append(g["mem_used_mb"])
            if g.get("mem_total_mb"):
                self._gpu_total_mb = g["mem_total_mb"]

    def _run(self) -> None:
        # Take one sample immediately so short stages still get a data point.
        self._sample_once()
        while not self._stop.wait(self.interval):
            self._sample_once()

    def start(self) -> "ResourceSampler":
        # Reset psutil's CPU counters so the first sample measures the window
        # since *this stage* started, not since some unrelated earlier call.
        # psutil keeps that state per-process-object, so this assumes one
        # sampler at a time — which holds: the job queue runs pipelines
        # serially to avoid GPU contention.
        try:
            if _PROC is not None:
                _PROC.cpu_percent(None)
                psutil.cpu_percent(None)
        except Exception:
            pass
        self._last_cpu_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="gochidubb-perf-sampler", daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1.0)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        def stats(vals: list[float], prefix: str, out: Dict[str, Any]) -> None:
            if not vals:
                return
            out[f"{prefix}_avg"] = round(sum(vals) / len(vals), 1)
            out[f"{prefix}_peak"] = round(max(vals), 1)

        # Counted from RSS: it is recorded on every pass, while CPU is
        # skipped when the window since the last reading is too short to
        # divide by. A stage that sampled once still reports one sample.
        out: Dict[str, Any] = {
            "samples": (len(self._rss_mb) or len(self._cpu_proc)
                        or len(self._gpu_util)),
        }
        stats(self._cpu_proc, "cpu_proc_pct", out)
        stats(self._cpu_sys, "cpu_sys_pct", out)
        stats(self._rss_mb, "rss_mb", out)
        stats(self._gpu_util, "gpu_util_pct", out)
        stats(self._gpu_mem, "gpu_mem_mb", out)
        if self._gpu_total_mb:
            out["gpu_mem_total_mb"] = round(self._gpu_total_mb, 1)
        if _gpu.backend:
            out["gpu_backend"] = _gpu.backend
        out["cpu_cores"] = _CPU_COUNT
        return out


def format_resources(res: Dict[str, Any]) -> str:
    """Compact one-line rendering of a sampler summary for log output."""
    parts = []
    if "cpu_proc_pct_avg" in res:
        parts.append(
            f"cpu {res['cpu_proc_pct_avg']:.0f}%avg/{res['cpu_proc_pct_peak']:.0f}%peak"
        )
    if "rss_mb_peak" in res:
        parts.append(f"rss {res['rss_mb_peak']:.0f}MB")
    if "gpu_util_pct_avg" in res:
        parts.append(
            f"gpu {res['gpu_util_pct_avg']:.0f}%avg/{res['gpu_util_pct_peak']:.0f}%peak"
        )
    if "gpu_mem_mb_peak" in res:
        total = res.get("gpu_mem_total_mb")
        parts.append(
            f"vram {res['gpu_mem_mb_peak']:.0f}MB"
            + (f"/{total:.0f}MB" if total else "")
        )
    if not parts:
        return "no resource samples"
    return " · ".join(parts)


# ═════════════════════════════════════════════════════════════════════
# Metrics persistence
# ═════════════════════════════════════════════════════════════════════
# metrics.json layout:
# {
#   "job_id": "...",
#   "stages": {                 # latest run of each stage (what the UI shows)
#       "transcribe": {"status": "ok", "duration_sec": 41.2, "started_at": ...,
#                       "attempt": 2, "resources": {...}, "detail": {...}}
#   },
#   "history": [ {stage, ...}, ... ]   # every attempt, append-only, capped
# }
_HISTORY_CAP = 200
_metrics_lock = threading.Lock()


def _metrics_path(work_dir: Path) -> Path:
    return Path(work_dir) / METRICS_FILENAME


def load_metrics(work_dir: Path) -> Dict[str, Any]:
    p = _metrics_path(work_dir)
    if not p.exists():
        return {"stages": {}, "history": []}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("stages", {})
        data.setdefault("history", [])
        return data
    except Exception as e:
        log.warning(f"[perf] Could not read {p.name}: {e}")
        return {"stages": {}, "history": []}


def record_stage(
    work_dir: Path,
    job_id: str,
    stage: str,
    status: str,
    duration_sec: float,
    started_at: float,
    resources: Optional[Dict[str, Any]] = None,
    detail: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    """Persist one stage attempt to metrics.json. Returns the entry written."""
    entry: Dict[str, Any] = {
        "stage": stage,
        "status": status,
        "duration_sec": round(duration_sec, 3),
        "started_at": started_at,
        "finished_at": started_at + duration_sec,
        "resources": resources or {},
    }
    if detail:
        entry["detail"] = detail
    if error:
        entry["error"] = error[:2000]

    with _metrics_lock:
        data = load_metrics(work_dir)
        prev = data["stages"].get(stage) or {}
        entry["attempt"] = int(prev.get("attempt", 0)) + 1
        data["job_id"] = job_id
        data["stages"][stage] = entry
        data["history"].append(entry)
        if len(data["history"]) > _HISTORY_CAP:
            data["history"] = data["history"][-_HISTORY_CAP:]
        try:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            tmp = _metrics_path(work_dir).with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _metrics_path(work_dir))
        except Exception as e:
            log.warning(f"[perf] Could not write metrics.json: {e}")
    return entry


@contextmanager
def stage_timer(
    work_dir: Path,
    job_id: str,
    stage: str,
    interval: float = 2.0,
    detail: Optional[Dict[str, Any]] = None,
):
    """Time a pipeline stage and record CPU/GPU usage while it runs.

    Yields a mutable dict — anything a stage puts in it (segment counts,
    model names, chosen backend) is stored alongside the timing as `detail`,
    which is what makes the numbers interpretable after the fact:

        with stage_timer(work, job_id, "transcribe") as t:
            segments, lang = transcribe(...)
            t["segments"] = len(segments)

    On exception the stage is recorded with status="error" and the exception
    is re-raised — a failed stage still tells you how long it ran before dying.
    """
    ctx: Dict[str, Any] = dict(detail or {})
    started = time.time()
    sampler = ResourceSampler(interval=interval).start()
    log.info(f"[perf] ▶ stage='{stage}' job={job_id} started")
    status, err = "ok", ""
    try:
        yield ctx
    except BaseException as e:
        status = "cancelled" if e.__class__.__name__ == "JobCancelled" else "error"
        err = f"{e.__class__.__name__}: {e}"
        raise
    finally:
        dur = time.time() - started
        res = sampler.stop()
        record_stage(
            work_dir, job_id, stage, status, dur, started,
            resources=res, detail=ctx, error=err,
        )
        detail_str = " · ".join(f"{k}={v}" for k, v in ctx.items() if v is not None)
        log.info(
            f"[perf] ■ stage='{stage}' job={job_id} status={status} "
            f"took={dur:.2f}s · {format_resources(res)}"
            + (f" · {detail_str}" if detail_str else "")
            + (f" · error={err}" if err else "")
        )
