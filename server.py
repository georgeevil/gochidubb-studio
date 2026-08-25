"""
GoChiDUBB Studio - Plug-and-Play AI Video Dubbing
==================================================
Created by TachikomaRed and smolemaru
Run: python server.py
Open: http://localhost:8910
"""
# ═══════════════════════════════════════════════════════════════════
# SPEECHBRAIN / K2 WORKAROUND — MUST RUN BEFORE ANY OTHER IMPORT
# ═══════════════════════════════════════════════════════════════════
# speechbrain 1.x uses lazy modules for `integrations.k2_fsa` and a few
# deprecated-redirect paths. On Windows the `k2` wheel doesn't exist, so
# these lazy imports fail the moment anything walks speechbrain's
# namespace (e.g. inspect.getmembers during TTS). We pre-populate
# sys.modules with empty stubs so importlib.import_module returns the
# stub instead of trying to actually load the broken chain.
#
# IMPORTANT: this block runs BEFORE pipeline imports so that WhisperX
# and pyannote (which transitively import speechbrain) see the stubs
# from the very first load.
import sys as _sys
import types as _types


# `_sys` / `_types` are deleted at the end of this block (see the `del`
# below), so a linter reads them as unbound here. They are not: this helper
# is only ever called from the loop directly beneath it, long before the
# del runs. Annotated rather than restructured — CLAUDE.md marks this block
# load-bearing and warns against moving anything above it.
def _gochidubb_stub_module(_name: str) -> None:
    if _name in _sys.modules:          # noqa: F821
        return
    m = _types.ModuleType(_name)       # noqa: F821
    m.__file__ = f"<gochidubb-stub:{_name}>"
    m.__path__ = []
    _sys.modules[_name] = m            # noqa: F821


for _n in (
    "k2",
    "speechbrain.k2_integration",
    "speechbrain.integrations.k2_fsa",
    "speechbrain.integrations.k2_fsa.ctc_loss",
    "speechbrain.integrations.k2_fsa.graph_compiler",
    "speechbrain.integrations.k2_fsa.lattice_decoder",
    "speechbrain.integrations.k2_fsa.lexicon",
    "speechbrain.integrations.k2_fsa.losses",
    "speechbrain.integrations.k2_fsa.prepare_lang",
    "speechbrain.integrations.k2_fsa.utils",
    "speechbrain.wordemb",
    "speechbrain.lobes.models.huggingface_transformers",
):
    _gochidubb_stub_module(_n)

del _sys, _types, _gochidubb_stub_module, _n

# ═══════════════════════════════════════════════════════════════════

import asyncio
import csv
import io
import json
import math
import re
import logging
import os
import platform
import shutil
import signal
import sys
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Load .env file if python-dotenv is installed (HF_TOKEN, etc)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        print(f"[env] Loaded {_env_path}")
except ImportError:
    pass  # dotenv optional


# Fallback .env loader for installs without python-dotenv.
#
# This MUST run before the `from pipeline...` imports further down: those
# modules capture their configuration (LM_STUDIO_URL, LM_STUDIO_MODEL,
# request timeouts, HF_TOKEN) into module-level constants at import time.
# The loader used to sit below those imports, so on any machine without
# python-dotenv every one of those settings silently kept its built-in
# default and the user's .env was ignored.
def _load_dotenv_simple():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv_simple()


# ─────────────────────────────────────────────────────────────
# Windows: help torchcodec find FFmpeg DLLs
# ─────────────────────────────────────────────────────────────
# torchcodec ships libtorchcodec_coreN.dll which in turn loads avformat/
# avcodec/avutil DLLs from wherever they happen to live. On Windows those
# DLLs come from ffmpeg's bin/ folder (e.g. installed via winget or
# C:\ffmpeg\bin). If that folder isn't on PATH *for DLL search*, the load
# fails with the "Could not find module" cascade seen in prior logs.
#
# Python 3.8+ requires os.add_dll_directory() explicitly — just having it
# on PATH is no longer enough. We scan typical install locations and
# register any that contain avformat*.dll. No-op if torchcodec is absent.
if sys.platform == "win32":
    try:
        import os as _os
        _ffmpeg_candidates = []
        # 1. FFMPEG_DIR / FFMPEG_PATH env override
        for _env_var in ("FFMPEG_DIR", "FFMPEG_PATH"):
            _p = _os.environ.get(_env_var, "").strip()
            if _p and _os.path.isdir(_p):
                _ffmpeg_candidates.append(_p)
                _bin = _os.path.join(_p, "bin")
                if _os.path.isdir(_bin):
                    _ffmpeg_candidates.append(_bin)
        # 2. Locate via `where ffmpeg` PATH lookup
        _ffmpeg_on_path = shutil.which("ffmpeg")
        if _ffmpeg_on_path:
            _ffmpeg_candidates.append(_os.path.dirname(_ffmpeg_on_path))
        # 3. Common install roots
        for _root in (
            r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
            _os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
        ):
            if _os.path.isdir(_root):
                _ffmpeg_candidates.append(_root)
        _added = set()
        for _d in _ffmpeg_candidates:
            if _d in _added or not _os.path.isdir(_d):
                continue
            # Only add if it actually has avformat (real ffmpeg DLLs)
            try:
                _has_av = any(
                    f.lower().startswith("avformat") and f.lower().endswith(".dll")
                    for f in _os.listdir(_d)
                )
                if _has_av:
                    _os.add_dll_directory(_d)
                    _added.add(_d)
                    print(f"[ffmpeg] Registered DLL dir for torchcodec: {_d}")
            except Exception:
                continue
        if not _added:
            # Not fatal — torchcodec is optional; pyannote has a fallback.
            # Just note it in dev logs.
            pass
    except Exception as _e:
        print(f"[ffmpeg] DLL registration skipped: {_e}")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from pipeline.downloader import download_video, probe_metadata, curate_metadata
from pipeline.audio import (
    extract_audio, extract_audio_hq, separate_background, get_duration,
    SeparationAborted,
)
from pipeline.transcriber import transcribe
from pipeline.diarizer import (
    diarize_speakers, assign_speakers_to_segments,
    extract_speaker_audio, extract_fallback_reference, effective_hf_token,
)
from pipeline.translator import (
    translate_segments, check_ollama, ollama_pull_stream,
    unload_ollama_model, clear_glossary_cache, LANGUAGE_NAMES,
)
from pipeline.flags import flag_segments
from pipeline.synthesizer import VoxCPMSynthesizer, F5TTSEngine, EdgeTTSFallback
from pipeline.assembler import assemble_dubbed_audio, merge_audio_video, write_srt
from pipeline.models import (
    get_system_status, MODEL_CATALOG, USE_LM_STUDIO,
    LM_STUDIO_MODELS_ENDPOINT,
)
from pipeline.vad import apply_vad_filter, remap_segments as vad_remap_segments
from pipeline.metrics import (
    stage_timer, load_metrics, gpu_backend, gpu_snapshot,
)
from pipeline import diagnostics as diag
from pipeline.notices import (
    mask_secret, merge_notices, worst_severity, notice as pnotice,
)

from app.config import cfg, coerce_field, BASE, UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR
from app.db import init_db, save_job_sync, load_all_jobs, delete_job_db
from app import (logbuf, artifact_store, reuse_runtime, reuse as app_reuse,
                 activity, apikeys as app_apikeys, webhooks as app_webhooks,
                 billing as app_billing, audit as app_audit,
                 estimate as app_estimate, admin as app_admin)


# Force UTF-8 stdout for foreign-language transcripts on Windows cp1252 consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Paths come from app.config (already created at import time)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("gochidubb.server")

# Mirror everything into an in-memory ring so GET /api/logs can show it. This
# has to happen right after basicConfig and before the pipeline does any work:
# it rebinds the console handler to the real stderr before teeing stdio, which
# is what stops every log line being captured twice. See app/logbuf.py.
logbuf.install()


# Silence extremely repetitive polling endpoints that flood the console
# (UI polls /api/system and /api/job/<id> every few seconds, httpx logs
# every Ollama /api/tags healthcheck call).
class _QuietPolling(logging.Filter):
    _QUIET_SUBSTRINGS = (
        "/api/system",
        "/api/tags",
        "/api/job/",
        "/api/voices",
        "/outputs/",       # range-requests for video playback after completion
    )
    # Only the *successful* noise is worth hiding. Suppressing failures too
    # makes a broken /outputs range-request (the UI's video player silently
    # never loading) invisible in the logs with nothing to grep for.
    # uvicorn logs '%s - "%s %s HTTP/%s" %d' — the status code is the last
    # field, with no trailing space. The "Not Found" phrase you see in the
    # console is added afterwards by uvicorn's AccessFormatter, so it is NOT
    # part of record.getMessage(). Don't require anything after the digits.
    _STATUS_RE = re.compile(r'"\s+(\d{3})(?:\s|$)')

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if not any(q in msg for q in self._QUIET_SUBSTRINGS):
            return True
        m = self._STATUS_RE.search(msg)
        if m and int(m.group(1)) >= 400:
            return True     # keep errors
        return False


for _logger_name in ("uvicorn.access", "httpx", "httpcore"):
    logging.getLogger(_logger_name).addFilter(_QuietPolling())
# httpx logs Ollama calls at INFO, demote to WARNING so only errors show
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Windows ProactorEventLoop spams harmless WinError 10054 ("connection
# forcibly closed by remote host") every time a browser tab is closed or
# the video player seeks. These are not actionable errors — the tab went
# away, not a real problem. Filter them out so real errors stand out.
class _WindowsConnResetFilter(logging.Filter):
    _NOISE_SUBSTRINGS = (
        "WinError 10054",
        "ConnectionResetError",
        "_call_connection_lost",
    )
    def filter(self, record):
        try:
            msg = record.getMessage()
            if any(n in msg for n in self._NOISE_SUBSTRINGS):
                return False
            # Also check exception info (WinError 10054 often logged via exc_info)
            if record.exc_info:
                exc_str = str(record.exc_info[1])
                if any(n in exc_str for n in self._NOISE_SUBSTRINGS):
                    return False
        except Exception:
            pass
        return True


logging.getLogger("asyncio").addFilter(_WindowsConnResetFilter())

jobs: dict = {}

# Statuses that mean a job is mid-flight. Used to mark jobs stale after a
# restart, and to refuse settings changes that would swap the TTS model out
# from under a running synthesis.
ACTIVE_STATUSES = frozenset({
    # "preparing" is a batch whose source is still being fetched by the
    # background task /api/quick_test spawns — no pipeline owns it yet, but a
    # restart kills the task, so it must be marked stale like the rest.
    "preparing",
    "queued", "running", "downloading", "extracting",
    "transcribing", "translating", "synthesizing",
    "assembling", "merging",
})
_tts_engine = None


def _free_gpu_memory():
    """Best-effort GPU memory cleanup. Call before loading a heavy model
    when another one may have left VRAM cached. Safe to call even if
    torch isn't imported — fails silently.

    What this does:
      1. Force Python GC so any dead tensor references get collected
      2. torch.cuda.empty_cache() releases PyTorch's cached allocator back
         to the driver (Ollama / llama.cpp don't share this cache so their
         unloads are separate)
      3. torch.cuda.ipc_collect() releases handles from forked processes
    """
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            log.info(f"[gpu] Cleared torch cache; free VRAM ~{free_mb:.0f} MB")
    except Exception as e:
        log.debug(f"_free_gpu_memory: {e}")

# ═══════════════════════════════════════════════════════════════════════
#  Job queue — ensures only one GPU-heavy job runs at a time
# ═══════════════════════════════════════════════════════════════════════
# Multiple concurrent dubs would OOM a 12GB GPU (WhisperX + VoxCPM +
# pyannote ≈ 10GB peak). Instead we queue. User can submit 5 videos in
# a row — they all get job_ids immediately and appear in History with
# status="queued". The scheduler processes them serially.
# User-facing benefit: "fire and forget" — drop 3 videos on the app,
# walk away, come back to 3 finished dubs.
# ═══════════════════════════════════════════════════════════════════════
_job_queue: "asyncio.Queue[tuple]" = None  # set in lifespan
_queue_worker_task = None
_scheduler_task = None

# Admission gate for the queue worker — the admin console's "Pause new jobs".
# Set = running, clear = paused, which is the direction asyncio.Event reads
# naturally in `await _intake_gate.wait()`.
#
# The gate is checked BEFORE the dequeue, never after. Holding a job the
# worker has already taken would leave it out of `qsize()` while it is plainly
# still waiting, so the console's queue depth would under-report for as long
# as the pause lasted.
#
# That alone is not enough, though, and the gap is the interesting part: an
# idle worker is parked inside `_job_queue.get()`, which returns the instant
# something is enqueued — so a job submitted during a pause would start
# anyway, while the console went on claiming intake was paused. A safety
# control that lies is worse than not having one.
#
# So the worker races the dequeue against `_intake_changed`, which the pause
# route pulses on every toggle. When the toggle wins, the pending `get()` is
# cancelled and the worker loops back to park on the gate. Cancelling an
# `asyncio.Queue.get()` cannot lose a job: the item lives in the queue's own
# deque and a getter only pops it after being woken, so a cancelled getter
# leaves it exactly where it was.
_intake_gate: "asyncio.Event" = None      # set in lifespan; set == open
_intake_changed: "asyncio.Event" = None   # pulsed when intake is toggled
_intake_paused_since: float = 0.0


def _intake_is_paused() -> bool:
    return _intake_gate is not None and not _intake_gate.is_set()

# Separate queue for platform uploads (Phase 3C). Uploads are network-bound,
# not GPU-bound, so they must not wait behind (or block) the dub pipeline.
_upload_queue: "asyncio.Queue[str]" = None  # set in lifespan; holds job_ids
_upload_worker_task = None


async def _scheduler_loop():
    """Background loop that moves scheduled jobs into the live queue when
    their scheduled_at time arrives. Polls every 30 seconds — good enough
    resolution for "start at 2 AM" use cases, and doesn't thrash CPU.

    Survives server restarts because the job state (status='scheduled' +
    scheduled_at timestamp + _pending_args) is persisted to disk. If the
    server was down when the time passed, jobs whose scheduled_at is in
    the past get enqueued immediately on next poll.
    """
    log.info("[scheduler] Loop started (polls every 30s)")
    while True:
        try:
            await asyncio.sleep(30)
            now = time.time()
            ready = [
                j for j in jobs.values()
                if j.get("status") == "scheduled"
                and j.get("scheduled_at", 0) > 0
                and j.get("scheduled_at", 0) <= now
            ]
            for j in ready:
                args = j.get("_pending_args")
                if not args:
                    j["status"] = "error"
                    j["error"] = "Scheduled job missing pipeline args"
                    save_job(j)
                    continue
                log.info(f"[scheduler] Job {j['id']} reached scheduled time — enqueueing")
                j.pop("_pending_args", None)
                await enqueue_job(j["id"], args)
        except asyncio.CancelledError:
            log.info("[scheduler] Loop cancelled, exiting")
            return
        except Exception as e:
            log.warning(f"[scheduler] Iteration failed (continuing): {e}")


async def enqueue_job_stub_removed_marker():
    pass


# ═══════════════════════════════════════════════════════════════════════
#  Cancellation — make the Cancel button actually work
# ═══════════════════════════════════════════════════════════════════════
# cancel_job() sets `j["cancel_requested"] = True` on the job dict.
# The pipeline checks this flag at every stage boundary (via update())
# and at per-segment progress callbacks, raising JobCancelled when set.
# The pipeline wrapper catches it and marks status=cancelled cleanly.
# ═══════════════════════════════════════════════════════════════════════
class JobCancelled(Exception):
    """Raised inside pipeline stages when the user has requested cancel."""
    pass


# Set once uvicorn begins tearing down, so long in-process computations
# (demucs) can bail out instead of pinning an executor thread the event loop
# will later join with no timeout.
_shutting_down = False


def _cancel_requested(job_id: str) -> bool:
    return bool(job_id in jobs and jobs[job_id].get("cancel_requested"))


def _should_abort_long_work(job_id: str):
    """Predicate handed to pipeline stages that can poll for an early exit."""
    def _check() -> bool:
        return _shutting_down or _cancel_requested(job_id)
    return _check


def _mark_interrupted(job_id: str) -> None:
    """Park a job that a shutdown caught mid-run on a non-active status.

    Anything still holding an in-flight status ('merging', 'extracting', ...)
    reads as a running job to the UI and to `serverctl status` forever, since
    the process that was running it is gone.
    """
    j = jobs.get(job_id)
    if not j or j.get("status") in ("complete", "cancelled", "error"):
        return
    j["status"] = "interrupted"
    j["step_detail"] = "Interrupted by server shutdown"
    j.setdefault(
        "error",
        "Interrupted by server shutdown — retry from the last checkpoint",
    )
    try:
        save_job(j)
    except Exception as e:
        log.debug(f"[queue] could not persist interrupted status: {e}")


def _set_job_error(job: dict, exc, stage_id: str = "") -> dict:
    """Attach a structured error to the job (and keep the legacy string).

    The structured form is what lets the UI say *where* it failed and show
    the exact log lines: `log_from` is the ring seq recorded when the failed
    stage started (see run_pipeline_stages), `log_to` the seq at failure —
    by which point the caller must already have logged the traceback, or the
    window won't cover it. History survives retries so a fail→retry→fail
    sequence stays legible; `last_error` and `error` are cleared on restart
    by _clear_job_error.
    """
    if isinstance(exc, BaseException):
        message = str(exc) or type(exc).__name__
        etype = type(exc).__name__
        tb_tail = ""
        if exc.__traceback__ is not None:
            import traceback as _tb
            tb_tail = "".join(
                _tb.format_exception(type(exc), exc, exc.__traceback__))[-2500:]
    else:
        message, etype, tb_tail = str(exc), "", ""
    stage = stage_id or job.get("failed_stage") or job.get("stage_id") or ""
    err = {
        "stage": stage,
        "stage_label": (STAGE_BY_ID.get(stage) or {}).get("label", stage),
        "type": etype,
        "message": message,
        "traceback_tail": tb_tail,
        "ts": time.time(),
        "log_from": int(job.get("_stage_log_from") or 0),
        "log_to": logbuf.current_seq(),
    }
    job["error"] = message           # legacy consumers (CLI, MCP, old UI paths)
    job["last_error"] = err
    hist = job.setdefault("error_history", [])
    hist.append({k: v for k, v in err.items() if k != "traceback_tail"})
    del hist[:-8]                    # cap: an error loop must not bloat the job
    return err


def _clear_job_error(job: dict) -> None:
    """Reset error state when a job is (re)started.

    Every restart path must call this, or the previous failure keeps
    rendering until the new run finishes — the exact stale-error bug this
    replaces. `error_history` deliberately survives.
    """
    job.pop("error", None)
    job.pop("last_error", None)
    job.pop("failed_stage", None)
    job.pop("stale_from_restart", None)
    job.pop("download_hint", None)


def _maybe_terminate_tts_worker():
    """Best-effort kill of the persistent VoxCPM TTS subprocess.
    Called when we detect cancel mid-synthesis so the next iteration
    of stdout.readline() exits immediately instead of waiting for the
    current segment to finish rendering."""
    try:
        tts = _tts_engine
        if tts is None:
            return
        proc = getattr(tts, "_worker_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                log.info("[cancel] TTS worker terminated")
            except Exception as e:
                log.debug(f"[cancel] TTS worker terminate failed: {e}")
            # Reap it. Dropping the handle without waiting leaves a zombie,
            # and a worker that ignores SIGTERM (mid-CUDA-call, say) would
            # otherwise survive us entirely — it is spawned without its own
            # process group, so nothing else will clean it up.
            import subprocess
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception as e:
                    log.debug(f"[cancel] TTS worker kill failed: {e}")
            except Exception as e:
                log.debug(f"[cancel] TTS worker reap failed: {e}")
            try:
                tts._worker_proc = None
            except Exception:
                pass
    except Exception as e:
        log.debug(f"[cancel] _maybe_terminate_tts_worker error: {e}")


# The multiprocessing resource tracker unlinks POSIX semaphores/shared memory
# that a dying process failed to release. Killing it is what turns the benign
# "leaked semaphore objects to clean up at shutdown" warning into an actual
# kernel-namespace leak, so it is excluded from every sweep below.
_PROTECTED_CHILD_MARKERS = ("multiprocessing.resource_tracker",
                            "multiprocessing.semaphore_tracker",
                            "resource_tracker.main")


def _terminate_child_processes(timeout: float = 5.0) -> int:
    """SIGTERM then SIGKILL every live descendant (ffmpeg, yt-dlp, TTS daemon).

    Shutting down cancels the queue worker, but that only raises CancelledError
    at the `await` — the worker thread underneath is still blocked in
    run_ffmpeg's `proc.wait()` loop and cannot be interrupted. asyncio then
    calls loop.shutdown_default_executor(), which joins that thread with no
    timeout on 3.11, so the process hangs until ffmpeg finishes on its own
    (potentially tens of minutes) and has to be SIGKILLed from outside.

    Killing the children first lets those threads unwind in ~1s.

    Returns the number of processes terminated.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return 0  # psutil is in requirements but treated as optional
    try:
        me = psutil.Process(os.getpid())
        children = me.children(recursive=True)
    except Exception as e:
        log.debug(f"[shutdown] could not enumerate children: {e}")
        return 0

    victims = []
    for p in children:
        try:
            cmdline = " ".join(p.cmdline())
        except Exception:
            cmdline = ""
        if any(m in cmdline for m in _PROTECTED_CHILD_MARKERS):
            continue
        victims.append(p)

    if not victims:
        return 0

    names = []
    for p in victims:
        try:
            names.append(f"{p.name()}[{p.pid}]")
        except Exception:
            names.append(f"?[{p.pid}]")
        try:
            p.terminate()
        except Exception:
            pass

    _, alive = psutil.wait_procs(victims, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=2)

    log.info(f"[shutdown] terminated {len(victims)} child process(es): "
             f"{', '.join(names)}")
    return len(victims)


async def enqueue_job(job_id, pipeline_args):
    """Add a job to the queue; scheduler will pick it up. If queue empty
    and scheduler idle, starts processing immediately.

    pipeline_args: dict of keyword args for run_pipeline (preferred),
        or legacy tuple of positional args (old enqueue sites).
    """
    global _job_queue
    if _job_queue is None:
        # Queue not initialized yet (shouldn't happen after startup)
        _job_queue = asyncio.Queue()
    # Mark job as queued in the store so UI shows "queued" badge
    if job_id in jobs:
        jobs[job_id]["status"] = "queued"
        jobs[job_id]["queue_position"] = _job_queue.qsize() + 1
        save_job(jobs[job_id])
    await _job_queue.put((job_id, pipeline_args))
    log.info(f"[queue] Job {job_id} enqueued (position {_job_queue.qsize()})")
    # Auto-activate sleep prevention when the queue has jobs waiting.
    # This ensures long unattended runs don't halt because Windows slept.
    _apply_sleep_prevention(True)


# ═══════════════════════════════════════════════════════════════════════
#  Windows sleep prevention — keeps PC awake during long batch runs
# ═══════════════════════════════════════════════════════════════════════
# Night-mode workflow: user drops 5 courses in the queue, goes to bed.
# Without this, Windows would go to sleep ~20 min into the first video
# and pause everything. We call SetThreadExecutionState to tell Windows
# "keep the system awake AND keep CPU busy" while we have work.
# ES_CONTINUOUS=0x80000000, ES_SYSTEM_REQUIRED=0x00000001
# The flag is persistent until cleared (not a timer), so we must clear
# it when the queue empties. Called from both enqueue and worker-idle.
# ═══════════════════════════════════════════════════════════════════════
_sleep_lock_active = False


def _apply_sleep_prevention(keep_awake: bool):
    """Toggle Windows sleep prevention. Safe no-op on non-Windows OS."""
    global _sleep_lock_active
    try:
        import ctypes
        if not hasattr(ctypes, "windll"):
            return  # not Windows
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        # ES_AWAYMODE_REQUIRED could be added on desktop to keep CPU active
        # even with lid closed; we leave it off to allow screen sleep but
        # prevent system sleep.
        if keep_awake and not _sleep_lock_active:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
            _sleep_lock_active = True
            log.info("[power] Sleep prevention ON — PC stays awake during queue")
        elif not keep_awake and _sleep_lock_active:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            _sleep_lock_active = False
            log.info("[power] Sleep prevention OFF — PC can sleep again")
    except Exception as e:
        log.debug(f"[power] Sleep prevention toggle failed: {e}")


async def _job_queue_worker():
    """Background worker — runs pipelines serially from the queue."""
    global _job_queue
    log.info("[queue] Worker started")
    while True:
        try:
            if _intake_gate is not None and not _intake_gate.is_set():
                log.info("[queue] Intake paused — not starting anything new")
                await _intake_gate.wait()
                log.info("[queue] Intake resumed")
            if _intake_changed is None:
                job_id, pipeline_args = await _job_queue.get()
            else:
                # See the `_intake_gate` comment: race the dequeue against a
                # pause so an idle worker cannot admit a job the operator has
                # just closed the door on.
                getter = asyncio.ensure_future(_job_queue.get())
                toggled = asyncio.ensure_future(_intake_changed.wait())
                try:
                    done, _pending = await asyncio.wait(
                        {getter, toggled}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    toggled.cancel()
                if getter in done:
                    # A job and a pause can land in the same tick. The job is
                    # already out of the queue by then, so it runs — the
                    # pause takes effect from the next one.
                    _intake_changed.clear()
                    job_id, pipeline_args = getter.result()
                else:
                    getter.cancel()
                    _intake_changed.clear()
                    continue
        except asyncio.CancelledError:
            log.info("[queue] Worker cancelled, exiting")
            return
        try:
            # Update queue positions for waiting jobs so UI reflects movement
            for j in jobs.values():
                if j.get("status") == "queued":
                    # Decrement: this one just left the queue,
                    # others move up
                    pos = j.get("queue_position", 1) - 1
                    j["queue_position"] = max(pos, 1)
            log.info(f"[queue] Processing job {job_id}")
            # Mark actual start time so elapsed/ETA measurements are
            # accurate (created = enqueue time, which can be much earlier
            # in a big batch). See "#3 started_at is never recorded" audit.
            if job_id in jobs:
                jobs[job_id]["started_at"] = time.time()
                # A re-run must not display the previous run's failure.
                _clear_job_error(jobs[job_id])
                save_job(jobs[job_id])
            # Queue stores args as a dict keyword, not positional tuple — so we
            # can add new pipeline params without breaking old enqueue sites
            if isinstance(pipeline_args, dict):
                # Stage retries ride the same queue as full runs so they
                # can't collide with another job on the GPU, and so they
                # inherit the cancel / post-success handling below.
                retry = pipeline_args.get("__stage_retry__")
                if retry:
                    await run_pipeline_stages(
                        job_id, retry["ctx"],
                        start_stage=retry["start_stage"],
                        stop_after=retry.get("stop_after", ""),
                    )
                else:
                    await run_pipeline(job_id, **pipeline_args)
            else:
                # Legacy positional tuple (kept for back-compat with older queued jobs)
                await run_pipeline(job_id, *pipeline_args[:12],
                                   wizard_mode=pipeline_args[12])
        except JobCancelled:
            # Cancel is a user action, not an error. Status was already
            # set by the exception raiser; just log cleanly.
            log.info(f"[queue] Job {job_id} cancelled by user")
            if job_id in jobs:
                jobs[job_id]["status"] = "cancelled"
                jobs[job_id].pop("cancel_requested", None)
                save_job(jobs[job_id])
        except asyncio.CancelledError:
            # Shutdown. run_pipeline_stages already marked the job
            # 'interrupted'; re-raise so this worker task actually stops.
            log.warning(f"[queue] Job {job_id} interrupted (server shutting down)")
            _mark_interrupted(job_id)
            raise
        except SeparationAborted:
            # Same cause, but delivered as a normal exception from inside a
            # worker thread. Do not re-raise — the loop is still healthy and
            # will exit at its own cancellation point.
            log.warning(f"[queue] Job {job_id} interrupted during separation")
            _mark_interrupted(job_id)
        except Exception as e:
            log.error(f"[queue] Job {job_id} crashed: {e}", exc_info=True)
            if job_id in jobs:
                jobs[job_id]["status"] = "error"
                _set_job_error(jobs[job_id], e)
                save_job(jobs[job_id])
        else:
            # Job completed cleanly. Run post-success hooks:
            #   1. Auto lip-sync (Wav2Lip) if the job opted in.
            #   2. Showcase stitch if this was the last sibling in a showcase batch.
            # Both are best-effort — failures log but don't fail the job.
            try:
                if job_id in jobs and jobs[job_id].get("lip_sync"):
                    log.info(f"[queue] post-hook: running auto lip-sync on {job_id}")
                    await asyncio.get_event_loop().run_in_executor(
                        None, _run_wav2lip_sync, job_id)
            except Exception as e:
                log.warning(f"[lipsync] auto hook failed: {e}", exc_info=True)
            try:
                if job_id in jobs and jobs[job_id].get("batch_kind") == "showcase":
                    await _maybe_assemble_showcase(jobs[job_id].get("batch_id", ""))
            except Exception as e:
                log.warning(f"[showcase] post-process hook failed: {e}", exc_info=True)
        finally:
            _job_queue.task_done()
            # Release sleep lock once nothing else is pending.
            # Next enqueue will re-acquire automatically.
            if _job_queue.empty():
                _apply_sleep_prevention(False)


async def _enqueue_upload(job_id: str) -> None:
    """Put an approved publish on the upload queue (creates it if needed)."""
    global _upload_queue
    if _upload_queue is None:
        _upload_queue = asyncio.Queue()
    await _upload_queue.put(job_id)
    log.info(f"[publish] Job {job_id} queued for upload "
             f"(position {_upload_queue.qsize()})")


async def _upload_worker():
    """Background worker — uploads approved publishes serially (Phase 3C).

    Mirrors _job_queue_worker's shape: dequeue, run, persist the outcome.
    Uploads are pure aiohttp awaits, so shutdown cancellation lands at the
    in-flight await and unwinds cleanly — the CancelledError handler marks
    the job failed (re-approvable) before the task exits, matching the
    interrupt-in-flight-work behaviour the job worker got in 432de20.
    """
    from pipeline.publisher import PublishError, PublishMeta, get_uploader

    log.info("[publish] Upload worker started")
    while True:
        try:
            job_id = await _upload_queue.get()
        except asyncio.CancelledError:
            log.info("[publish] Upload worker cancelled, exiting")
            return
        job = jobs.get(job_id)
        pub = (job or {}).get("publish")
        try:
            if not job or not isinstance(pub, dict):
                log.warning(f"[publish] {job_id}: no publish state — skipping")
                continue
            if pub.get("status") != "approved":
                # Cancelled (or re-staged) between approve and dequeue.
                log.info(f"[publish] {job_id}: status is "
                         f"{pub.get('status')!r}, not 'approved' — skipping")
                continue

            pub["status"] = "uploading"
            pub["error"] = None
            save_job(job)

            file_rel = pub.get("file") or ""
            file_path = str((BASE / file_rel).resolve())
            meta = PublishMeta(
                title=pub.get("title") or "",
                description=pub.get("description") or "",
                tags=list(pub.get("tags") or []),
            )
            log.info(f"[publish] {job_id}: uploading {file_rel} to "
                     f"{pub.get('platform', 'vk')}")
            uploader = get_uploader(pub.get("platform", "vk"))
            result = await uploader.upload(file_path, meta)

            pub.update(
                status="uploaded",
                url=result.get("url"),
                platform_id=result.get("platform_id"),
                uploaded_at=time.time(),
                error=None,
            )
            save_job(job)
            log.info(f"[publish] {job_id}: uploaded — {result.get('url')}")
        except asyncio.CancelledError:
            # Server shutdown mid-upload. VK upload URLs are single-use, so
            # the transfer cannot resume — park it as failed + re-approvable.
            if isinstance(pub, dict):
                pub["status"] = "failed"
                pub["error"] = ("Upload interrupted by server shutdown — "
                                "approve again to retry")
                if job:
                    save_job(job)
            log.warning(f"[publish] {job_id}: upload interrupted by shutdown")
            raise
        except PublishError as e:
            pub["status"] = "failed"
            pub["error"] = str(e)
            save_job(job)
            log.warning(f"[publish] {job_id}: upload failed: {e}")
        except Exception as e:
            pub["status"] = "failed"
            pub["error"] = f"{type(e).__name__}: {e}"
            save_job(job)
            log.error(f"[publish] {job_id}: upload crashed: {e}", exc_info=True)
        finally:
            _upload_queue.task_done()


def load_jobs_from_disk():
    loaded = load_all_jobs()
    jobs.update(loaded)
    # Mark any jobs that appear to still be "running" as error+resumable.
    # After a server restart the in-memory queue is empty, so these jobs
    # aren't actually being processed anymore. Without this fix, the
    # History tab shows them as permanently "transcribing..." / "queued".
    # The user can still click Resume (if a checkpoint exists) to pick
    # up where the pipeline left off.
    stale_count = 0
    for jid, job in jobs.items():
        if job.get("status") in ACTIVE_STATUSES:
            job["status"] = "error"
            job["error"] = (
                job.get("error")
                or "Interrupted by server restart — click Resume to continue"
            )
            job["stale_from_restart"] = True
            save_job(job)
            stale_count += 1
    if stale_count:
        log.info(
            f"Marked {stale_count} stale job(s) as 'error' "
            f"(left over from previous server run)"
        )
    # Same treatment for publishes the previous process left in flight: the
    # in-memory upload queue is empty now, so "uploading"/"approved" would
    # otherwise read as live forever. Mark them failed + re-approvable.
    stale_pub = 0
    for job in jobs.values():
        pub = job.get("publish")
        if isinstance(pub, dict) and pub.get("status") in ("uploading", "approved"):
            pub["status"] = "failed"
            pub["error"] = ("server restarted during upload — "
                            "approve again to retry")
            save_job(job)
            stale_pub += 1
    if stale_pub:
        log.info(f"Marked {stale_pub} stale publish(es) as 'failed' "
                 f"(left over from previous server run)")
    log.info(f"Loaded {len(jobs)} jobs from disk")


def _job_checkpoint_info(job_id: str) -> dict:
    """Return which checkpoint stages exist on disk for a job.

    Used by list_jobs so the History UI knows whether an errored or
    cancelled job is resumable (and from where). Purely filesystem
    inspection — cheap enough to do on every /api/jobs poll."""
    work_dir = OUTPUT_DIR / job_id
    if not work_dir.exists():
        return {"has_checkpoint": False, "latest_checkpoint_stage": None}
    # Check most-advanced first so 'latest' reflects how far the
    # pipeline got before stopping. CHECKPOINT_ORDER_DESC covers every
    # stage in PIPELINE_STAGES (newest → oldest), including the fine-grained
    # download/extract/transcribe checkpoints added for per-stage retry.
    for stage in CHECKPOINT_ORDER_DESC:
        if (work_dir / f"checkpoint_{stage}.json").exists():
            return {"has_checkpoint": True, "latest_checkpoint_stage": stage}
    # Legacy single-file pipeline state
    if (work_dir / "pipeline_state.json").exists():
        try:
            with open(work_dir / "pipeline_state.json", "r", encoding="utf-8") as f:
                d = json.load(f)
            return {
                "has_checkpoint": True,
                "latest_checkpoint_stage": d.get("stage") or "unknown",
            }
        except Exception:
            pass
    return {"has_checkpoint": False, "latest_checkpoint_stage": None}


def save_job(job: dict):
    save_job_sync(job)


# ─────────────────────────────────────────────────────────────
# Pipeline checkpoint system
# ─────────────────────────────────────────────────────────────
# Instead of one big pipeline_state.json, we save multiple checkpoint
# files — one per completed stage. This lets the user rewind to any
# earlier stage (e.g., "regenerate translation from scratch") without
# redoing the stages before it.
#
# File naming:
#   pipeline_state.json              — legacy; last-stage checkpoint
#   checkpoint_transcription_done.json
#   checkpoint_translation_done.json
#   checkpoint_tts_done.json
#
# Each file is self-contained — loading it is enough to resume from
# the corresponding stage. "stage" field says which stage just finished.

def _save_checkpoint(job_id: str, work_dir: Path, stage: str, data: dict) -> None:
    data["stage"] = stage
    data["job_id"] = job_id
    data["saved_at"] = time.time()

    # Named checkpoint (never overwritten by later stages)
    cpath = work_dir / f"checkpoint_{stage}.json"
    try:
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"[checkpoint] Saved: {stage} -> {cpath.name}")
    except Exception as e:
        log.warning(f"[checkpoint] Save failed ({stage}): {e}")

    # Legacy pipeline_state.json — always the MOST RECENT checkpoint
    try:
        with open(work_dir / "pipeline_state.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"[checkpoint] Legacy save failed: {e}")


def _load_checkpoint(job_id: str, stage: str) -> Optional[dict]:
    """Load a specific checkpoint. Returns None if not found."""
    work_dir = OUTPUT_DIR / job_id
    cpath = work_dir / f"checkpoint_{stage}.json"
    if not cpath.exists():
        # Fallback to legacy pipeline_state.json if it matches the stage
        legacy = work_dir / "pipeline_state.json"
        if legacy.exists():
            try:
                with open(legacy, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("stage") == stage:
                    return d
            except Exception:
                pass
        return None
    try:
        with open(cpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"[checkpoint] Load failed ({stage}): {e}")
        return None


def _latest_checkpoint(job_id: str) -> Optional[dict]:
    """Return the most advanced checkpoint available for a job."""
    for stage in CHECKPOINT_ORDER_DESC:
        cp = _load_checkpoint(job_id, stage)
        if cp:
            return cp
    return None


# ─────────────────────────────────────────────────────────────
# TTS engine selection
# ─────────────────────────────────────────────────────────────

# Target languages the cloning engines were not trained on. VoxCPM2's official
# coverage (HF model card) does not include Bulgarian, Ukrainian, Czech,
# Romanian or Hungarian — cross-lingual synthesis there produces unreliable
# output at best. These targets synthesize via edge-tts instead: a generic
# Microsoft voice instead of a clone, but intelligible words. The set only
# gates the cloning engines; edge-tts jobs pass through untouched.
_EDGE_ONLY_TARGET_LANGS = {
    # Original wave: not in VoxCPM2's official coverage (HF model card)
    "bg", "uk", "cs", "ro", "hu",
    # Wave 2 — everything new except he/sw/tl/ms/my/km/lo, which ARE
    # covered by VoxCPM2 and keep voice cloning
    "bn", "ur", "fa", "ta", "te", "mr", "gu", "kn", "ml",
    "sk", "hr", "sr", "sl", "lt", "lv", "et", "ca", "is", "af",
    "mk", "sq", "bs", "cy",
    "kk", "az", "uz", "ka", "mn", "ne", "si",
}


def _tts_engine_for_lang(tts, target_lang: str):
    """Return the engine that should actually synthesize `target_lang`.

    Cloning engines are kept for languages they cover; anything in
    _EDGE_ONLY_TARGET_LANGS drops to a fresh EdgeTTSFallback instance.
    """
    if not isinstance(tts, EdgeTTSFallback) and \
            (target_lang or "")[:2].lower() in _EDGE_ONLY_TARGET_LANGS:
        log.info(f"[tts] '{target_lang}' is outside {tts.name}'s training "
                 f"coverage — using edge-tts for this job (no voice cloning)")
        return EdgeTTSFallback()
    return tts


# Settings that are baked into the TTS engine at construction time, so a
# change only takes effect once the cached engine is dropped and rebuilt.
# Everything else (cross-lingual values, QA, stretch) is read per synthesis.
TTS_REBUILD_KEYS = frozenset({
    "tts_engine", "voxcpm_model", "voxcpm_cfg", "voxcpm_steps",
    "voxcpm_denoise_refs",
})


def validate_voxcpm_overrides(voxcpm_cfg, voxcpm_steps):
    """Clean a per-job VoxCPM override pair. Returns (cfg, steps, error).

    0 / None / "" means "not overridden — use the global setting", so it
    skips validation; anything else is bounded by the same FIELD_SPECS the
    Settings tab uses. These two numbers are handed to VoxCPM directly, so an
    out-of-range value is refused at the door rather than clamped quietly.
    """
    out = []
    for key, raw in (("voxcpm_cfg", voxcpm_cfg), ("voxcpm_steps", voxcpm_steps)):
        if raw is None or raw == "" or float(raw or 0) == 0:
            out.append(0)
            continue
        try:
            out.append(coerce_field(key, raw))
        except ValueError as e:
            return 0, 0, str(e)
    return out[0], out[1], ""


def reset_tts_engine() -> bool:
    """Drop the cached TTS engine so the next job rebuilds it from config.

    Returns True if an engine was actually unloaded. Callers must ensure no
    job is mid-synthesis — see ACTIVE_STATUSES — because unloading frees the
    model a running job is using.
    """
    global _tts_engine
    if _tts_engine is None:
        return False
    try:
        unload = getattr(_tts_engine, "unload", None)
        if callable(unload):
            unload()
    except Exception as e:
        log.warning(f"[tts] unload during settings change failed: {e}")
    _tts_engine = None
    _free_gpu_memory()
    log.info("[tts] engine released — next job rebuilds it from current settings")
    return True


def get_tts_engine():
    """TTS engine factory. Priority: VoxCPM2 → F5-TTS → Edge-TTS.

    Selection follows cfg.tts_engine preference, falling back through
    the tier chain when a higher-tier engine isn't installed or fails to load.
    """
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    _free_gpu_memory()

    requested = cfg.tts_engine  # "voxcpm" | "f5tts" | "edge-tts"

    # Tier 1: VoxCPM2
    if requested in ("voxcpm", "auto"):
        try:
            import voxcpm  # noqa
            _tts_engine = VoxCPMSynthesizer(
                model_id=cfg.voxcpm_model,
                load_denoiser=cfg.voxcpm_denoise_refs,
                cfg_value=cfg.voxcpm_cfg,
                inference_timesteps=cfg.voxcpm_steps,
            )
            log.info("TTS engine: VoxCPM2 (voice cloning)")
            import atexit
            atexit.register(lambda: _tts_engine.unload() if _tts_engine else None)
            return _tts_engine
        except ImportError:
            log.info("VoxCPM2 not installed, trying F5-TTS...")

    # Tier 2: F5-TTS
    if requested in ("f5tts", "auto", "voxcpm"):
        try:
            import f5_tts  # noqa
            _tts_engine = F5TTSEngine()
            log.info("TTS engine: F5-TTS (voice cloning, lighter than VoxCPM2)")
            return _tts_engine
        except ImportError:
            log.info("F5-TTS not installed, falling back to Edge-TTS")

    # Tier 3: Edge-TTS (always available)
    _tts_engine = EdgeTTSFallback()
    log.warning("TTS engine: Edge-TTS fallback (no voice cloning)")
    return _tts_engine


# ─────────────────────────────────────────────────────────────
# Voice Preset Library
# ─────────────────────────────────────────────────────────────
# Each preset = a voice-design description + a fixed seed.
# The seed locks VoxCPM's random state so all segments in a job
# sound like the SAME voice (voice design is non-deterministic
# by default per the VoxCPM docs).
VOICE_PRESETS = {
    "auto": {
        "name": "Auto (use video voice if possible)",
        "style": "",
        "seed": None,  # random per-job
    },
    "male_warm": {
        "name": "Male — warm, middle-aged, calm",
        "style": "middle-aged male voice, warm and calm, clear articulation",
        "seed": 101,
    },
    "male_deep": {
        "name": "Male — deep, authoritative narrator",
        "style": "deep mature male voice, authoritative narrator, slow pace",
        "seed": 202,
    },
    "male_young": {
        "name": "Male — young, energetic",
        "style": "young adult male voice, energetic and friendly",
        "seed": 303,
    },
    "male_sports": {
        "name": "Male — sports instructor",
        "style": "grown adult male sports instructor, clear and steady, confident",
        "seed": 404,
    },
    "female_calm": {
        "name": "Female — warm, gentle",
        "style": "warm female voice, gentle and soothing, mid-tone",
        "seed": 505,
    },
    "female_narrator": {
        "name": "Female — professional narrator",
        "style": "professional female narrator, clear articulation, neutral tone",
        "seed": 606,
    },
    "female_young": {
        "name": "Female — young, cheerful",
        "style": "young adult female voice, friendly and cheerful",
        "seed": 707,
    },
}


VOICE_PRESETS_DIR = BASE / "presets" / "voices"
VOICE_PRESETS_DIR.mkdir(parents=True, exist_ok=True)

# User-editable glossary overrides. Loaded by translator.py when building
# prompts; editable via /api/glossary endpoints in the Settings tab.
USER_GLOSSARY_FILE = BASE / "presets" / "user_glossary.json"

# User preferences — persisted across sessions. Simple JSON blob edited
# by the UI; not schema-validated server-side (it's just a KV store).
PREFS_FILE = BASE / "user_prefs.json"


_VOICE_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def _voice_metadata_path(audio_path: Path) -> Path:
    """Sidecar metadata file for a voice preset (<name>.json next to audio)."""
    return audio_path.with_suffix(".json")


def _read_voice_metadata(audio_path: Path) -> dict:
    """Load JSON sidecar with structured metadata, falling back to legacy
    `<name>.txt` description if JSON doesn't exist. Always returns a dict."""
    j = _voice_metadata_path(audio_path)
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"[voice_presets] bad JSON in {j.name}: {e}")
    # Backward-compat: legacy .txt description file
    txt = audio_path.with_suffix(".txt")
    if txt.exists():
        try:
            return {"description": txt.read_text(encoding="utf-8").strip()[:300]}
        except Exception:
            pass
    return {}


def scan_file_presets() -> dict:
    """Scan presets/voices/ folder for voice references + their metadata.

    Each `<name>.{wav,mp3,flac,ogg,m4a}` becomes a file-based preset.
    Optional `<name>.json` sidecar carries structured metadata
    (description, gender, language, tags, created_at). Legacy `<name>.txt`
    is still read as the description for backward-compat.

    Re-scanned on every endpoint call — drop a file into the folder and
    it's available immediately, no restart needed.
    """
    presets = {}
    if not VOICE_PRESETS_DIR.exists():
        return presets
    for path in sorted(VOICE_PRESETS_DIR.iterdir()):
        if path.suffix.lower() not in _VOICE_AUDIO_EXTS:
            continue
        meta = _read_voice_metadata(path)
        pid = f"file:{path.stem}"
        presets[pid] = {
            "id": pid,
            "name": meta.get("display_name") or path.stem,
            "style": meta.get("style", ""),
            "seed": meta.get("seed"),
            "reference_file": str(path),
            "description": meta.get("description", ""),
            # Structured metadata for the Voices tab UI
            "gender": meta.get("gender", ""),         # 'male' | 'female' | 'neutral' | ''
            "language": meta.get("language", ""),     # iso code or empty
            "tags": meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
            "created_at": meta.get("created_at"),
            # File facts (computed, not stored)
            "file_size": path.stat().st_size if path.exists() else 0,
            "file_ext": path.suffix.lower().lstrip("."),
            "audio_url": f"/api/voice_presets/{pid}/audio",
        }
    return presets


def resolve_voice_config(voice_preset: str, voice_style: str, job_id: str):
    """Return (effective_voice_style, voice_seed, reference_file) for a run.

    reference_file is set ONLY when user picked a file-based preset
    from presets/voices/ folder. Otherwise it's empty string.
    """
    import hashlib
    # Check file presets first (they live in a folder, re-scanned each call)
    file_presets = scan_file_presets()
    if voice_preset in file_presets:
        p = file_presets[voice_preset]
        return "", 0, p["reference_file"]

    preset = VOICE_PRESETS.get(voice_preset, VOICE_PRESETS["auto"])
    # Priority: explicit voice_style beats preset style (lets user override)
    eff_style = (voice_style or "").strip() or preset["style"]
    # Priority: preset seed > hash of voice_style > hash of job_id (random-ish)
    if preset.get("seed") is not None:
        seed = preset["seed"]
    elif eff_style:
        seed = int(hashlib.md5(eff_style.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
    else:
        seed = int(hashlib.md5(job_id.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
    return eff_style, seed, ""


# ─────────────────────────────────────────────────────────────
# Designed voices: one draw, then clone
# ─────────────────────────────────────────────────────────────
# A style preset ("middle-aged male voice, warm and calm") is a prompt, not a
# voice. VoxCPM draws a new speaker from it on every generate call, and the
# seed does not pin that draw — the sampled speaker latent is conditioned on
# the text as well, so the same style and seed on two different lines is two
# different people. Measured on a 1602-segment dub with seed, tier and every
# inference parameter held constant: f0 across 40 segments of one speaker
# ranged 85-312 Hz. The dub sounded like a different narrator each line.
#
# Cloning is stable, which is why the `auto` path has never had this problem.
# So a style preset is materialised: drawn ONCE into a reference clip, then
# cloned from for every segment. The clip is cached by (style, seed, language)
# under presets/voices/.designed/, so a preset also sounds the same across
# jobs instead of being re-rolled per run.
#
# The cache lives in a dotted subdirectory of presets/voices/ on purpose:
# scan_file_presets() iterates that folder for audio FILES, so a directory is
# skipped and these never show up as user voices in the picker.
DESIGNED_VOICES_DIR = VOICE_PRESETS_DIR / ".designed"

# Internal routing mode -> the coarse label the UI, the job record and the
# publisher have always shown. "designed_ref" is a designed voice that was
# rendered once and cloned from, so it is still the user's "custom" choice;
# "cast" is its own thing because it is the only mode with more than one
# answer for a job.
_VOICE_MODE_LABEL = {
    "file_ref": "upload",
    "voice_design": "custom",
    "designed_ref": "custom",
    "cast": "cast",
    "source_refs": "source",
}


def _designed_voice_path(style: str, seed, target_lang: str) -> Path:
    """Cache location for one (style, seed, language) triple."""
    import hashlib
    key = f"{style.strip().lower()}|{seed}|{target_lang}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return DESIGNED_VOICES_DIR / f"{digest}.wav"


def _materialize_designed_voice(engine, style: str, seed,
                                target_lang: str = "en") -> str:
    """Return a reference clip speaking in `style`, rendering it if needed.

    Blocking — it drives the TTS worker subprocess. Call through _blocking().
    Returns "" on failure; callers fall back to per-segment voice design
    rather than failing the run, because a wobbly voice still beats no dub.
    """
    style = (style or "").strip().strip("()")
    if not style:
        return ""
    dest = _designed_voice_path(style, seed, target_lang)
    # >1 KB guards against a truncated write from an interrupted run being
    # served forever as a "cached" voice.
    if dest.exists() and dest.stat().st_size > 1000:
        log.info(f"[design] Reusing cached voice for {style!r} -> {dest.name}")
        return str(dest)
    DESIGNED_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        made = engine.design_reference(
            style, str(dest), voice_seed=seed, target_lang=target_lang)
    except Exception as e:
        log.warning(f"[design] Materialising {style!r} failed: {e}")
        return ""
    return made or ""


def _voice_choices() -> dict:
    """Every assignable voice, keyed by the id the casting map stores.

    "source" is not in here — it is per-speaker (whatever was cut from the
    video for that person) and so has no single reference to describe.
    """
    out = {}
    for pid, p in VOICE_PRESETS.items():
        if pid == "auto" or not p.get("style"):
            continue
        out[pid] = {"id": pid, "name": p["name"], "kind": "design",
                    "style": p["style"], "seed": p.get("seed")}
    for pid, p in scan_file_presets().items():
        out[pid] = {"id": pid, "name": p["name"], "kind": "library",
                    "style": p.get("style", ""), "seed": p.get("seed"),
                    "audio_url": p.get("audio_url", "")}
    return out


def _source_ref_for(speaker: str, ctx: dict, job_id: str) -> str:
    """The reference cut from the video for this speaker, if there is one.

    Prefers ctx["speaker_refs"] so a clip the user replaced through
    edit_speaker_ref wins — but only when that path is still inside this
    job's own speaker_refs/ folder. After a run that used a preset, the same
    key holds a path in presets/voices/, which is emphatically not the source
    voice; that case falls through to the untouched source_speaker_refs copy.
    """
    job_refs = str((OUTPUT_DIR / job_id / "speaker_refs").resolve())
    cur = (ctx.get("speaker_refs") or {}).get(speaker) or ""
    if cur and os.path.exists(cur):
        try:
            if str(Path(cur).resolve()).startswith(job_refs):
                return cur
        except OSError:
            pass
    fallback = (ctx.get("source_speaker_refs") or {}).get(speaker) or ""
    return fallback if fallback and os.path.exists(fallback) else ""


async def _resolve_casting(cast: dict, speakers, ctx: dict, job_id: str,
                           engine, target_lang: str) -> tuple:
    """Turn {speaker: voice_id} into ({speaker: ref_wav}, {speaker: ""}, report).

    Every speaker present in the audio gets an entry, whether or not the map
    mentions them: an unassigned speaker keeps their own voice, which is the
    only default that cannot surprise anyone. A voice that will not resolve
    (deleted preset file, failed render) degrades to the source voice for
    that speaker alone and says so in the report, rather than taking the
    whole dub down or silently recasting everyone.
    """
    choices = _voice_choices()
    refs, transcripts, report = {}, {}, []

    for sp in speakers:
        want = (cast.get(sp) or "source").strip()
        src = _source_ref_for(sp, ctx, job_id)
        chosen, resolved_id = "", want

        if want and want != "source":
            spec = choices.get(want)
            if spec is None and want.startswith("design:"):
                free = want[len("design:"):].strip()
                spec = {"kind": "design", "style": free, "seed": None}
            if spec is None:
                report.append({"speaker": sp, "voice": want,
                               "status": "unknown", "used": "source"})
                resolved_id = "source"
            elif spec["kind"] == "library":
                ref_file = scan_file_presets().get(want, {}).get("reference_file", "")
                if ref_file and os.path.exists(ref_file):
                    chosen = ref_file
                else:
                    report.append({"speaker": sp, "voice": want,
                                   "status": "missing_file", "used": "source"})
                    resolved_id = "source"
            else:
                seed = spec.get("seed")
                if seed is None:
                    import hashlib
                    seed = int(hashlib.md5(spec["style"].encode()).hexdigest()[:8],
                               16) & 0x7FFFFFFF
                chosen = await _blocking(
                    _materialize_designed_voice, engine, spec["style"], seed,
                    target_lang,
                )
                if not chosen:
                    report.append({"speaker": sp, "voice": want,
                                   "status": "render_failed", "used": "source"})
                    resolved_id = "source"

        if not chosen:
            chosen = src
            if not chosen:
                report.append({"speaker": sp, "voice": resolved_id,
                               "status": "no_source_ref", "used": "none"})
        if chosen:
            refs[sp] = chosen
            # Empty transcript = Controllable Cloning (reference only). A
            # prompt transcript would have to match the reference audio word
            # for word, and none of these references have one.
            transcripts[sp] = ""
            if not any(r["speaker"] == sp for r in report):
                report.append({"speaker": sp, "voice": resolved_id,
                               "status": "ok", "used": resolved_id})

    return refs, transcripts, report


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init SQLite store (creates table + migrates legacy JSON files)
    init_db(BASE / "gochidubb.db")
    # Beta stage-reuse store, same file as the job store.
    artifact_store.init_store(BASE / "gochidubb.db")
    load_jobs_from_disk()

    # Initialize job queue + start serial worker. Using a single worker
    # ensures GPU-heavy pipelines don't collide and OOM the card.
    global _job_queue, _queue_worker_task, _scheduler_task
    global _upload_queue, _upload_worker_task
    _job_queue = asyncio.Queue()
    # Intake starts open. A pause is a live operational decision, never a
    # state that survives a restart and silently strands the queue.
    global _intake_gate, _intake_changed
    _intake_gate = asyncio.Event()
    _intake_gate.set()
    _intake_changed = asyncio.Event()
    _queue_worker_task = asyncio.create_task(_job_queue_worker())
    # Scheduler: polls every 30s looking for jobs with status='scheduled'
    # whose scheduled_at time has arrived. Survives restarts — status and
    # scheduled_at persist in the job dict on disk.
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    # Upload worker: drains the separate publish queue (network-bound, so it
    # runs beside — never behind — the GPU job queue).
    _upload_queue = asyncio.Queue()
    _upload_worker_task = asyncio.create_task(_upload_worker())

    # update() runs on the event loop for most stages but from a worker thread
    # for the blocking ones, and asyncio.create_task() only works on the
    # former. Holding the serving loop lets either path schedule a delivery.
    global _MAIN_LOOP
    _MAIN_LOOP = asyncio.get_running_loop()

    # First line of the activity feed's System stream, and a useful one: it
    # dates the buffer, so "nothing before this" is explained rather than
    # looking like lost history.
    activity.record_system(
        f"Server started · mode {cfg.mode} · {len(jobs)} job(s) loaded")

    if os.getenv("GOCHIDUBB_OPEN_BROWSER", "1") == "1" and not os.getenv("DOCKER"):
        async def open_browser():
            await asyncio.sleep(1.5)
            try:
                webbrowser.open("http://localhost:8910")
            except Exception:
                pass
        asyncio.create_task(open_browser())

    # Warm up the TTS subprocess so the first dub doesn't pay the 40-60s
    # model-load tax. We send a tiny dummy job to the daemon worker which
    # loads VoxCPM and then idles waiting for real jobs. Runs in background;
    # failures are non-fatal (engine will load lazily on first real use).
    # VoxCPM warmup is now OFF by default. Reasoning: VoxCPM holds ~4 GB of
    # VRAM for its lifetime, and on 12 GB cards that prevents Ollama from
    # loading larger translation models (gemma4:e4b = 9.6 GB). By deferring
    # VoxCPM load until AFTER translation, Ollama gets the full 12 GB to
    # itself, translates fast, unloads via keep_alive, then VoxCPM loads
    # for TTS. Cost: first dub is ~20s slower (one-time VoxCPM load).
    # To opt back in (fast multi-user servers with abundant VRAM): set
    # GOCHIDUBB_WARMUP=1 in .env or environment.
    if os.getenv("GOCHIDUBB_WARMUP", "0") == "1":
        async def warmup_tts():
            await asyncio.sleep(2.0)  # let server finish binding port
            try:
                import tempfile as _tmp
                import wave as _wave
                import struct as _struct
                log.info("[warmup] Pre-spawning persistent TTS worker...")
                t0 = time.time()
                tts = get_tts_engine()
                if not isinstance(tts, VoxCPMSynthesizer):
                    return  # Edge-TTS fallback doesn't need warmup
                # Create a minimal valid WAV file to serve as dummy reference
                dummy_dir = _tmp.mkdtemp(prefix="gochidubb_warmup_")
                dummy_ref = os.path.join(dummy_dir, "dummy_ref.wav")
                with _wave.open(dummy_ref, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                    # 1 second of low-volume noise (so ref validation passes)
                    w.writeframes(_struct.pack("<" + "h"*16000,
                                               *([0]*16000)))
                dummy_segs = [{
                    "idx": 0, "start": 0.0, "end": 1.0,
                    "text": "привет", "translated_text": "привет",
                    "speaker": "SPEAKER_00",
                }]
                # Run in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: tts.synthesize_segments(
                    dummy_segs, dummy_dir,
                    speaker_refs={"SPEAKER_00": dummy_ref},
                    speaker_transcripts={"SPEAKER_00": ""},
                    tts_speed="balanced",
                ))
                log.info(f"[warmup] TTS worker ready in {time.time()-t0:.1f}s "
                         f"— first dub will skip model-load")
                # Clean up dummy artifacts silently
                try:
                    shutil.rmtree(dummy_dir, ignore_errors=True)
                except Exception:
                    pass
            except Exception as e:
                log.warning(f"[warmup] Pre-load failed "
                            f"(engine will load on first real use): {e}")
        asyncio.create_task(warmup_tts())

    yield

    # Shutdown. Flag first so in-process work (demucs) bails at its next
    # checkpoint, then children, then the asyncio tasks. Cancelling the queue
    # worker only raises CancelledError at the `await` — the thread underneath
    # stays blocked in run_ffmpeg's proc.wait() or inside torch until it
    # finishes, and asyncio.run() joins that thread with no timeout. Without
    # this a restart during a long extract or render hangs until SIGKILLed.
    global _shutting_down
    _shutting_down = True
    try:
        _maybe_terminate_tts_worker()
        _terminate_child_processes()
    except Exception as e:
        log.warning(f"[shutdown] child cleanup failed: {e}")

    for t in (_queue_worker_task, _scheduler_task, _upload_worker_task):
        if t and not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(title="GoChiDUBB Studio", version="2.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Agent attribution ────────────────────────────────────────────────
# The MCP server and CLI are separate processes that drive this one over HTTP,
# so an agent's gochidubb.dub(...) arrives as an ordinary POST /api/dub —
# identical to the browser's. GoChiDUBBClient tags its requests with
# X-GoChiDUBB-Client, and this records the tagged ones as tool calls for the
# activity feed. Untagged requests (the UI) are deliberately not recorded.
#
# The header is informational only: it is never consulted for authorization,
# so a forged one can add a feed line and nothing else.

# Route prefix → the MCP tool it corresponds to. Longest match wins, so
# /api/dub/{id}/retry_stage/... is a retry_stage, not a dub.
_TOOL_ROUTES = [
    ("/api/dub/batch", "compare"),
    ("/api/dub", "dub"),
    ("/api/showcase", "showcase"),
    ("/api/quick_test", "quick_test"),
    ("/api/jobs", "list_jobs"),
    ("/api/job", "get_status"),
    ("/api/system", "system_status"),
    ("/api/voices", "list_voices"),
    ("/api/languages", "list_languages"),
    ("/api/scout/trending", "scout_trending"),
    ("/api/scout/dub", "scout_dub"),
    ("/api/publish/pending", "publish_pending"),
]


def _tool_for_path(path: str) -> str:
    if "/retry_stage/" in path:
        return "retry_stage"
    if path.endswith("/cancel"):
        return "cancel"
    if path.endswith("/redub"):
        return "redub"
    if path.endswith("/quality"):
        return "quality_report"
    best = ""
    for prefix, tool in _TOOL_ROUTES:
        if path.startswith(prefix) and len(prefix) > len(best):
            best, name = prefix, tool
    return name if best else path


_MAIN_LOOP = None

# Job status -> the webhook event it corresponds to. Only these three fire;
# they are the ones the design names.
_WEBHOOK_EVENT_FOR_STATUS = {
    "complete": "job.completed",
    "error": "job.failed",
    "awaiting_translation_review": "job.awaiting_review",
    "awaiting_voice_review": "job.awaiting_review",
}


def _fire_webhooks(status: str, job: dict) -> None:
    """Schedule deliveries for a job transition. Never raises, never blocks.

    A dub must not slow down or fail because someone's endpoint is down, so
    this only ever schedules detached tasks; app.webhooks handles timeouts,
    the single retry, and turning failures into delivery records.
    """
    event = _WEBHOOK_EVENT_FOR_STATUS.get(status)
    if not event:
        return
    hooks = app_webhooks.hooks_for(event)
    if not hooks:
        return
    payload = app_webhooks.payload_for_job(job)

    async def _run():
        for h in hooks:
            await app_webhooks.deliver_one(h, event, payload)
        activity.record_system(
            f"Webhook {event} · {len(hooks)} endpoint(s) · job {job.get('id')}")

    try:
        asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        # Called from a worker thread — hand it to the serving loop instead.
        if _MAIN_LOOP is not None:
            asyncio.run_coroutine_threadsafe(_run(), _MAIN_LOOP)


@app.middleware("http")
async def _record_agent_calls(request, call_next):
    client_id = request.headers.get("X-GoChiDUBB-Client")
    if not client_id or not request.url.path.startswith("/api/"):
        return await call_next(request)
    started = time.time()
    response = await call_next(request)
    try:
        activity.record_tool_call(
            _tool_for_path(request.url.path),
            client_id[:64],
            status=response.status_code,
            ms=(time.time() - started) * 1000.0,
        )
    except Exception:
        # A feed line is never worth failing a request over.
        log.debug("[activity] could not record tool call", exc_info=True)
    return response



# ─────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Pipeline stage registry
# ─────────────────────────────────────────────────────────────
# The pipeline is a linear list of stages. Each stage reads from and writes
# to a single mutable context dict (`ctx`), and the driver snapshots that
# ctx to `checkpoint_<name>.json` after the stage succeeds.
#
# That single rule is what makes every stage independently retryable: to
# re-run stage N you only need the checkpoint written by stage N-1. The
# retry endpoint loads it, applies the caller's overrides on top, and runs
# forward from there — no earlier stage is recomputed.
#
# "checkpoint" names are deliberately NOT all `<id>_done`: diarize/translate
# write `transcription_done` / `translation_done` because the older wizard,
# retry-TTS, redub and showcase endpoints read those exact filenames. The
# payload is a superset of what they used to contain, so they keep working.
PIPELINE_STAGES = [
    {
        "id": "download", "label": "Acquire video",
        "hint": "Download or copy the source media",
        "checkpoint": "download_done", "status": "downloading",
        "progress": (2, 8), "artifacts": ["video_path"],
    },
    {
        "id": "extract", "label": "Extract audio",
        "hint": "Demux, denoise, VAD-filter, split background",
        "checkpoint": "extract_done", "status": "extracting",
        "progress": (10, 16), "artifacts": ["audio_16k", "bg_audio_path"],
        "artifact_files": ["audio_16k.wav"],
    },
    {
        "id": "transcribe", "label": "Transcribe",
        "hint": "WhisperX speech-to-text",
        "checkpoint": "transcribe_done", "status": "transcribing",
        "progress": (18, 35), "artifacts": [],
    },
    {
        "id": "diarize", "label": "Diarize",
        "hint": "Identify speakers and cut voice references",
        "checkpoint": "transcription_done", "status": "diarizing",
        "progress": (38, 43), "artifacts": [], "artifact_dir": "speaker_refs",
    },
    {
        "id": "translate", "label": "Translate",
        "hint": "LLM translation of every segment",
        "checkpoint": "translation_done", "status": "translating",
        "progress": (45, 62), "artifacts": ["srt_path"],
        "artifact_files": ["subtitles.srt"],
    },
    {
        "id": "tts", "label": "Synthesize",
        "hint": "Voice-clone each segment",
        "checkpoint": "tts_done", "status": "synthesizing",
        "progress": (65, 85), "artifacts": [], "artifact_dir": "tts_segments",
    },
    {
        "id": "assemble", "label": "Assemble",
        "hint": "Time-align and loudness-normalize the dub track",
        "checkpoint": "assemble_done", "status": "assembling",
        "progress": (88, 92), "artifacts": ["dubbed_wav"],
        "artifact_files": ["dubbed_audio.wav"],
    },
    {
        "id": "merge", "label": "Render",
        "hint": "Mux the dubbed audio back onto the video",
        "checkpoint": "merge_done", "status": "merging",
        "progress": (93, 100), "artifacts": ["output_mp4"],
        "artifact_files": ["dubbed_video.mp4"],
    },
]

STAGE_ORDER = [s["id"] for s in PIPELINE_STAGES]
STAGE_BY_ID = {s["id"]: s for s in PIPELINE_STAGES}
# Checkpoint filenames newest-first — used to find "how far did this job get".
CHECKPOINT_ORDER_DESC = [s["checkpoint"] for s in reversed(PIPELINE_STAGES)]


def _stage_index(stage_id: str) -> int:
    try:
        return STAGE_ORDER.index(stage_id)
    except ValueError:
        return -1


# ── Job modes (Phase 3E) ────────────────────────────────────────────────
# "dub" (default) walks every pipeline stage. "reupload" — for music videos
# and other content that must not be dubbed — walks ONLY the download stage,
# then _finalize_reupload() produces outputs/<id>/dubbed_video.mp4 (copy or
# -c copy remux of the source) so publish/download/export keep working
# unchanged. Everything else (transcribe → merge) never runs.
JOB_MODES = ("dub", "reupload")


def normalize_job_mode(mode) -> Optional[str]:
    """Canonical job mode, or None when the value is not a known mode.

    Empty/missing input means the default "dub" — only an explicit unknown
    string is rejected.
    """
    m = str(mode or "dub").strip().lower()
    return m if m in JOB_MODES else None


def stages_for_mode(mode: str) -> list:
    """Pipeline stage ids the runner walks for a job mode. Pure."""
    if (str(mode or "dub").strip().lower()) == "reupload":
        return ["download"]
    return list(STAGE_ORDER)


# Keys that must never be written to a checkpoint: transient handles and
# internal control flags that would otherwise be replayed on resume.
_CTX_PRIVATE_PREFIX = "_"


def _ctx_for_checkpoint(ctx: dict) -> dict:
    """JSON-safe snapshot of the pipeline context."""
    out = {}
    for k, v in ctx.items():
        if k.startswith(_CTX_PRIVATE_PREFIX):
            continue
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


def _serialize_segments(segments: list) -> list:
    """Normalize segments for checkpointing — stable idx, known fields only.

    `idx` must be unique: downstream stages key dicts by it (see the
    translation merge in `_stage_translate`), so a duplicate silently
    overwrites one segment with another and a line of dialogue disappears from
    the dub with nothing logged. That happened for real — pipeline/segment_post
    split a long segment into two `dict(seg)` copies that both kept the
    parent's idx.
    """
    out = []
    seen: dict = {}
    for i, s in enumerate(segments):
        idx = s.get("idx", i)
        if idx in seen:
            # Re-key rather than drop, and say so — losing a segment quietly
            # is the failure mode this guard exists to prevent.
            new_idx = max(max(seen), i) + 1
            while new_idx in seen:
                new_idx += 1
            log.warning(
                f"[segments] duplicate idx={idx} at position {i} "
                f"(text={str(s.get('text',''))[:60]!r}) — reassigned to "
                f"{new_idx}. Upstream should emit unique indices."
            )
            idx = new_idx
        seen[idx] = True
        item = {
            "idx": idx,
            "start": s.get("start", 0.0),
            "end": s.get("end", 0.0),
            "text": s.get("text", ""),
            "speaker": s.get("speaker", "SPEAKER_00"),
        }
        for opt in ("translated_text", "audio_path", "qa_score", "tts_tier",
                    "qa", "avg_logprob", "no_speech_prob",
                    "word_conf_mean", "word_conf_min"):
            if s.get(opt) is not None:
                item[opt] = s.get(opt)
        # Aggregate per-word ASR confidences the first time through (the
        # words list itself is NOT serialized — only its summary survives
        # checkpointing; later passes carry the stats via the loop above).
        if s.get("words") and "word_conf_mean" not in item:
            from pipeline.transcriber import word_confidence_stats
            item.update(word_confidence_stats(s.get("words")))
        out.append(item)
    return out


def _stage_input_state(job_id: str, stage_id: str) -> Optional[dict]:
    """Load the checkpoint that a given stage needs as its input.

    Walks backwards from the stage's predecessor until a checkpoint exists,
    so a job that never wrote (say) `extract_done` because it predates this
    feature can still be retried from `translate` using `transcription_done`.
    """
    idx = _stage_index(stage_id)
    if idx <= 0:
        return None
    for prev in reversed(PIPELINE_STAGES[:idx]):
        cp = _load_checkpoint(job_id, prev["checkpoint"])
        if cp:
            return cp
    return None


def _resolve_voice_into_ctx(job: dict, ctx: dict) -> None:
    """(Re-)resolve the voice preset/style/seed and stash it on the ctx.

    Called on every pipeline run — including retries — so that a retry which
    overrides `voice_preset` picks up that preset's reference file and style
    instead of the one baked into the checkpoint.
    """
    eff_style, voice_seed, preset_ref_file = resolve_voice_config(
        ctx.get("voice_preset", "auto"), ctx.get("voice_style", ""), job["id"],
    )
    ctx["voice_style_effective"] = eff_style
    ctx["voice_seed"] = voice_seed
    if preset_ref_file and os.path.exists(preset_ref_file):
        log.info(f"[ref] File-preset selected: {preset_ref_file}")
        ctx["reference_audio"] = preset_ref_file
    job.update(
        voice_preset=ctx.get("voice_preset", "auto"),
        voice_style_effective=eff_style,
        voice_seed=voice_seed,
    )


# ─────────────────────────────────────────────────────────────
# Stage implementations
# ─────────────────────────────────────────────────────────────
# Every handler has the same shape:
#     async def _stage_x(job, work, ctx, update, perf) -> None
# `perf` is the stage_timer's detail dict — anything put in it lands in
# metrics.json next to the timing, which is what makes a slow stage
# diagnosable after the fact (how many segments? which model? which device?).


async def _blocking(fn, *args, **kwargs):
    """Run a synchronous pipeline call off the event loop.

    Handlers are `async def`, but yt-dlp, ffmpeg, WhisperX and pyannote are all
    plain blocking calls. Awaiting nothing while they run pins the single
    asyncio thread for minutes at a time, and *every* HTTP request queues behind
    it — so during a job the UI could not fetch /api/jobs or
    /api/dub/{id}/stages at all. The panel sat on "Loading stages…", a freshly
    submitted job did not appear as running until whatever stage was in flight
    finished, and even this file's own transcribe watchdog (an asyncio task
    that reports elapsed time) never got scheduled.

    Only the blocking call moves to the worker thread. Everything that touches
    `ctx`, `job` or `perf` stays on the loop, so there is no new shared state.
    Jobs remain serialized by the queue worker, which awaits this.
    """
    return await asyncio.to_thread(lambda: fn(*args, **kwargs))


async def _stage_download(job, work, ctx, update, perf):
    update(step_detail="Getting video...")
    # Full yt-dlp info dict probed at submit time (URL sources only) —
    # popped here so the transient blob doesn't linger on the job dict;
    # download_video persists it as <work>/source_info.json.
    source_info = job.pop("_source_info", None)
    video_path = await _blocking(
        download_video, ctx["source"], str(work), info=source_info)
    duration = await _blocking(get_duration, video_path)
    ctx["video_path"] = video_path
    ctx["duration"] = duration
    perf["media_sec"] = round(duration, 1)
    try:
        perf["size_mb"] = round(os.path.getsize(video_path) / 1024 / 1024, 1)
    except OSError:
        pass
    update(duration=round(duration, 1), progress=8)


async def _stage_extract(job, work, ctx, update, perf):
    update(step_detail="Extracting audio tracks...")
    audio_16k = str(work / "audio_16k.wav")
    await _blocking(extract_audio, ctx["video_path"], audio_16k)

    # Optional denoise for noisy source audio.
    # BJJ/cooking/sports videos often have mat noise, background music,
    # crowd, or equipment hum that WhisperX mistakes for words. We
    # apply an ffmpeg filter chain to clean the audio WITHOUT removing
    # voice quality. Filter chain reasoning:
    #   - afftdn: FFT-based noise reduction (safe, preserves speech)
    #   - highpass=80: drop sub-bass rumble (room noise, AC)
    #   - lowpass=10000: drop tweeter noise (mic hiss, digital artifacts)
    # This is conservative; aggressive denoise can hurt Whisper accuracy.
    perf["denoise"] = bool(ctx.get("auto_denoise", True))
    if ctx.get("auto_denoise", True):
        try:
            import subprocess
            denoise_start = time.time()
            audio_clean = str(work / "audio_16k_clean.wav")
            update(progress=12, step_detail="Cleaning audio...")
            await _blocking(
                subprocess.run,
                ["ffmpeg", "-y", "-i", audio_16k,
                 "-af", "afftdn=nr=10:nf=-25,highpass=f=80,lowpass=f=10000",
                 "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                 audio_clean],
                check=True, capture_output=True, timeout=180,
            )
            log.info(
                f"[perf] · denoise took {time.time()-denoise_start:.1f}s "
                f"(audio_16k_clean.wav) — feeding to WhisperX"
            )
            audio_16k = audio_clean
        except Exception as e:
            log.warning(f"Denoise failed (using raw audio): {e}")
            perf["denoise"] = "failed"

    bg_audio_path = ""
    if ctx.get("keep_bg"):
        try:
            bg_start = time.time()
            audio_hq = str(work / "audio_hq.wav")
            update(step_detail="Separating background audio...")
            await _blocking(extract_audio_hq, ctx["video_path"], audio_hq)
            # separate_background() falls back to a *silent* background rather
            # than failing, which looks identical to success in the output —
            # so it reports what it could not do instead of only logging it.
            sep_notices = []
            _, bg_audio_path = await _blocking(
                separate_background, audio_hq, str(work), notices=sep_notices,
                should_abort=_should_abort_long_work(job["id"]))
            log.info(f"[perf] · bg separation took {time.time()-bg_start:.1f}s")
            perf["bg_separated"] = not sep_notices
            if sep_notices:
                perf["notices"] = sep_notices
                diag.record_runtime_notices(sep_notices, job_id=job["id"])
        except SeparationAborted:
            # Demucs stopped early because we asked it to. Must not be
            # swallowed like a separation failure — the job is going away.
            perf["bg_separated"] = "aborted"
            if _cancel_requested(job["id"]):
                raise JobCancelled("cancelled during background separation")
            raise
        except Exception as e:
            log.warning(f"BG separation skipped: {e}")
            perf["bg_separated"] = "failed"
    ctx["bg_audio_path"] = bg_audio_path
    update(progress=15)

    # VAD filtering — strip long silence/music before Whisper.
    # silero-vad is optional (graceful fallback to full audio).
    if cfg.vad_enabled:
        try:
            vad_start = time.time()
            vad_out = str(work / "audio_16k_vad.wav")
            update(progress=16, step_detail="Filtering non-speech regions...")
            audio_16k, speech_ratio, vad_intervals = await _blocking(
                apply_vad_filter, audio_16k, vad_out,
                threshold=cfg.vad_threshold,
            )
            # The filter concatenates speech regions, so whisper's timestamps
            # come back on a compressed timeline. Keep the intervals so
            # _stage_transcribe can map them back onto the video's timeline —
            # without this the dub ends as early as the silence removed.
            ctx["vad_intervals"] = vad_intervals
            perf["speech_ratio"] = round(speech_ratio, 3)
            log.info(
                f"[perf] · vad took {time.time()-vad_start:.1f}s "
                f"(speech ratio {speech_ratio:.0%})"
            )
            if speech_ratio < 0.15:
                log.warning(
                    f"[vad] Low speech ratio ({speech_ratio:.0%}) — "
                    f"consider disabling VAD or checking audio source"
                )
        except Exception as e:
            log.warning(f"VAD skipped: {e}")
            perf["vad"] = "failed"

    ctx["audio_16k"] = audio_16k


async def _stage_transcribe(job, work, ctx, update, perf):
    # WhisperX doesn't expose internal progress, so during long transcription
    # (e.g. 20-min podcasts take 5-6min) the UI would just show "Transcribing..."
    # forever. We spawn a watchdog that updates step_detail with elapsed
    # seconds so the user can see it's still alive.
    duration = ctx.get("duration", 0.0)
    whisper_model = ctx.get("whisper_model") or cfg.whisper_model
    source_lang = ctx.get("source_lang", "auto")
    update(step_detail="Transcribing speech...")
    _t_start = time.time()
    _done_flag = {"done": False}

    async def _trans_watchdog():
        while not _done_flag["done"]:
            elapsed = int(time.time() - _t_start)
            if elapsed > 10:  # only show after 10s to avoid noise on short videos
                mins, secs = divmod(elapsed, 60)
                hint = f"Transcribing ({duration:.0f}s audio)... elapsed {mins}m{secs:02d}s"
                if elapsed > 120:
                    hint += " · try smaller whisper model for faster transcribe"
                update(step_detail=hint)
            await asyncio.sleep(5)

    _watchdog_task = asyncio.create_task(_trans_watchdog())
    try:
        segments, detected_lang = await _blocking(
            transcribe, ctx["audio_16k"], source_lang, whisper_model,
        )
    finally:
        _done_flag["done"] = True
        _watchdog_task.cancel()
        try:
            await _watchdog_task
        except (asyncio.CancelledError, Exception):
            pass

    took = max(time.time() - _t_start, 0.001)
    log.info(
        f"[perf] · transcribe {duration:.0f}s audio in {took:.1f}s "
        f"({duration / took:.1f}x realtime, model={whisper_model})"
    )
    if not segments:
        raise RuntimeError("No speech detected in video")

    effective_src = detected_lang if source_lang == "auto" else source_lang
    ctx["segments"] = _serialize_segments(segments)
    # Undo the VAD timeline compression before anything downstream sees these
    # times. Diarization, TTS placement and the SRT all assume the video's own
    # timeline; skipping this shifts every segment earlier by the silence cut
    # before it, so a dub of a 60s clip whose speech starts at 14s finishes
    # around the 30s mark and drifts out of sync the whole way.
    vad_intervals = ctx.get("vad_intervals")
    if vad_intervals:
        vad_remap_segments(ctx["segments"], vad_intervals)
        log.info(
            f"[vad] remapped {len(ctx['segments'])} segments onto the original "
            f"timeline (last end {ctx['segments'][-1]['end']:.1f}s)"
        )
    ctx["effective_src"] = effective_src
    ctx["source_lang_detected"] = detected_lang
    ctx["whisper_model"] = whisper_model
    perf.update(
        segments=len(segments), whisper_model=whisper_model,
        realtime_x=round(duration / took, 2), lang=detected_lang,
    )
    update(
        source_lang_detected=detected_lang,
        segment_count=len(segments),
        progress=35,
    )


async def _stage_diarize(job, work, ctx, update, perf):
    """Assign speakers and cut per-speaker voice references.

    `skip_diarization` is the escape hatch for the common failure mode:
    pyannote needs an HF_TOKEN and a working model download, and when it
    fails there's nothing wrong with the transcript. Retrying this stage
    with skip_diarization=true drops straight to the single-speaker
    fallback reference (Case C) and the run continues.
    """
    segments = [dict(s) for s in ctx["segments"]]
    audio_16k = ctx["audio_16k"]
    effective_src = ctx.get("effective_src", "en")
    target_lang = ctx.get("target_lang", "ru")
    reference_audio = ctx.get("reference_audio", "")
    speaker_mode = ctx.get("speaker_mode", "main")
    skip = bool(ctx.get("skip_diarization"))

    update(step_detail=("Skipping diarization (single speaker)..." if skip
                        else "Identifying speakers..."))

    speaker_turns = []
    if skip:
        log.info("[diarize] Skipped by request — using single-speaker fallback")
        perf["skipped"] = True
    else:
        hf_token = effective_hf_token()
        # diarize_speakers never raises — it returns [] for everything from
        # "no token" to "one speaker in the video". `notices` is how it tells
        # the difference, and without it a broken setup is indistinguishable
        # from a monologue.
        notices = []
        try:
            speaker_turns = await _blocking(
                diarize_speakers, audio_16k, hf_token=hf_token,
                notices=notices,
            )
            segments = assign_speakers_to_segments(segments, speaker_turns)
            perf["speaker_turns"] = len(speaker_turns)
        except Exception as e:
            # Diarization is genuinely optional — a failure here must not
            # cost the caller the transcription they already paid for. We
            # fall through to the Case C single-speaker reference and
            # record the reason so the UI can offer a targeted retry.
            log.warning(f"[diarize] Failed ({e}) — falling back to single speaker")
            perf["diarize_error"] = str(e)[:300]
            notices.append(pnotice(
                code="pyannote.crashed",
                severity="error",
                subsystem="diarize",
                title="Diarization crashed",
                detail=f"{type(e).__name__}: {e}",
                remediation=["Retry the Diarize stage",
                             "Or retry with 'Skip diarization' to continue"],
            ))
            speaker_turns = []
        if notices:
            perf["notices"] = notices
            diag.record_runtime_notices(notices, job_id=job.get("id", ""))

    # Store raw transcript preview for UI
    ctx["transcript_raw"] = [
        {
            "idx": i,
            "start": s["start"], "end": s["end"],
            "text": s["text"],
            "speaker": s.get("speaker", ""),
        }
        for i, s in enumerate(segments)
    ]
    update(transcript_raw=ctx["transcript_raw"])

    speaker_refs = {}
    speaker_transcripts = {}
    # Always-preserved copy of refs extracted from the SOURCE video.
    # Even when user picks a preset/upload (which overrides speaker_refs),
    # we still stash the original source refs here so that "retry TTS"
    # without re-choosing a ref can go back to the source voice.
    source_speaker_refs = {}

    # Case A: user uploaded a reference voice -> use it for ALL speakers.
    # Pure Controllable Cloning: just reference_wav_path, nothing else.
    if reference_audio and os.path.exists(reference_audio):
        log.info(f"[ref] Using USER-UPLOADED reference: {reference_audio}")
        unique_speakers = {s.get("speaker", "SPEAKER_00") for s in segments}
        for sp in unique_speakers:
            speaker_refs[sp] = reference_audio
            # NOT populating speaker_transcripts — we want clean Controllable
            # Cloning (reference_wav_path only), not Ultimate Cloning which
            # needs an exact transcript of the reference audio.
            speaker_transcripts[sp] = ""
        if speaker_turns:
            try:
                refs_dir = str(work / "speaker_refs")
                source_speaker_refs = await _blocking(
                    extract_speaker_audio,
                    audio_16k, speaker_turns, refs_dir, main_only=False,
                ) or {}
                log.info(f"[ref] Also extracted {len(source_speaker_refs)} "
                         f"source-voice refs for potential retry use")
            except Exception as e:
                log.warning(f"[ref] Source-ref extraction failed (ok to skip): {e}")

    # Case B: diarization worked -> per-speaker refs from the source
    elif speaker_turns:
        log.info("[ref] No user upload; extracting speaker refs from source video")
        refs_dir = str(work / "speaker_refs")
        main_only = speaker_mode == "main"
        speaker_refs = await _blocking(
            extract_speaker_audio,
            audio_16k, speaker_turns, refs_dir, main_only=main_only,
        )
        source_speaker_refs = dict(speaker_refs)
        if speaker_refs:
            # Populate speaker_transcripts (enables Tier 1 Ultimate Cloning)
            # ONLY for same-language dubbing. Cross-lingual Tier 1 makes
            # VoxCPM "continue" the source-language phonetics, so Russian
            # text comes out with English phonemes = gibberish.
            if effective_src == target_lang:
                for spk in speaker_refs:
                    texts = [s["text"] for s in segments if s.get("speaker") == spk]
                    if texts:
                        speaker_transcripts[spk] = " ".join(texts[:3])
            else:
                log.info(f"[ref] Cross-lingual dub ({effective_src}→{target_lang}); "
                         f"clearing prompt_text to force Controllable Cloning")
                for spk in speaker_refs:
                    speaker_transcripts[spk] = ""

        # If main_only: remap EVERY segment to the sole extracted speaker
        if main_only and speaker_refs:
            primary = next(iter(speaker_refs))
            for s in segments:
                s["speaker"] = primary

    # Case C: diarization failed/skipped -> ONE clean reference from long segments
    if not speaker_refs:
        log.info("[ref] No user upload + diarization unavailable - "
                 "building fallback single-speaker reference from source")
        fb_path = str(work / "speaker_refs" / "ref_fallback.wav")
        (work / "speaker_refs").mkdir(exist_ok=True)
        fb = await _blocking(extract_fallback_reference,
                             audio_16k, segments, fb_path, duration=30.0)
        if fb:
            speaker_refs["SPEAKER_00"] = fb
            source_speaker_refs["SPEAKER_00"] = fb
            if effective_src == target_lang:
                speaker_transcripts["SPEAKER_00"] = " ".join(
                    s["text"] for s in segments[:5]
                )
            else:
                speaker_transcripts["SPEAKER_00"] = ""
            for s in segments:
                s["speaker"] = "SPEAKER_00"
        perf["fallback_ref"] = bool(fb)

    # ─── POST-PROCESS SEGMENTS ───────────────────────────────
    # WhisperX cuts on VAD (breath) boundaries, not sentence boundaries,
    # so natural sentences often get split at pauses. The resulting
    # micro-fragments give TTS too little context to clone voice correctly.
    # See pipeline/segment_post.py for details.
    before = len(segments)
    try:
        from pipeline.segment_post import postprocess_segments
        segments = await _blocking(postprocess_segments, segments)
        perf["segments_merged"] = before - len(segments)
    except Exception as e:
        log.warning(f"Segment postprocess failed (continuing with raw): {e}")

    n_speakers = len(set(s.get("speaker", "?") for s in segments))
    ctx["segments"] = _serialize_segments(segments)
    ctx["speaker_refs"] = dict(speaker_refs)
    ctx["source_speaker_refs"] = dict(source_speaker_refs)
    ctx["speaker_transcripts"] = dict(speaker_transcripts)
    perf.update(speakers=n_speakers, segments=len(segments))
    update(speaker_count=n_speakers, segment_count=len(segments), progress=42)


def _is_untranslated(seg: dict) -> bool:
    """A segment counts as untranslated when the LLM gave us nothing new."""
    tt = (seg.get("translated_text") or "").strip()
    src = (seg.get("text") or "").strip()
    return (not tt) or tt == src


async def _stage_translate(job, work, ctx, update, perf):
    target_lang = ctx.get("target_lang", "ru")
    model = ctx.get("model") or cfg.translation_model
    context_hint = ctx.get("context_hint", "")
    segments = [dict(s) for s in ctx["segments"]]

    # ─── PARTIAL RETRY ────────────────────────────────────────────────
    # `translate_failed_only` is the answer to "the model died halfway and
    # I want to finish the job with a different one": we keep every segment
    # that already has a real translation and re-submit only the ones that
    # came back empty or identical to the source.
    prior_by_idx = {}
    if ctx.get("translate_failed_only"):
        prev = _load_checkpoint(job["id"], "translation_done") or {}
        candidates = {
            s.get("idx"): s for s in prev.get("segments", [])
            if not _is_untranslated(s)
        }
        # Carry the good translations onto our working copy — but only where
        # the SOURCE text still matches. If transcription was re-run since,
        # segment N is not the same sentence any more, and pasting the old
        # translation onto it would silently mistranslate the line.
        mismatched = 0
        for s in segments:
            good = candidates.get(s.get("idx"))
            if not good:
                continue
            if (good.get("text") or "").strip() != (s.get("text") or "").strip():
                mismatched += 1
                continue
            s["translated_text"] = good.get("translated_text", "")
            prior_by_idx[s["idx"]] = good
        if mismatched:
            log.info(
                f"[translate] {mismatched} prior translation(s) discarded — "
                f"source text changed since they were made"
            )
        todo = [s for s in segments if _is_untranslated(s)]
        log.info(
            f"[translate] Partial retry: keeping {len(prior_by_idx)} good "
            f"segment(s), re-translating {len(todo)} with model={model}"
        )
        if not todo:
            log.info("[translate] Nothing left to translate — all segments OK")
            perf.update(model=model, translated=0, reused=len(prior_by_idx))
            ctx["segments"] = _serialize_segments(segments)
            _finalize_translation(job, work, ctx, update, perf)
            await _maybe_translate_meta(job, ctx)
            return
    else:
        todo = segments

    update(step_detail=f"Translating to {target_lang} with {model}...")
    total_todo = len(todo)

    def _translate_progress(done, total, eta_sec):
        # Map translation progress into the overall pipeline 45→62% range.
        # `total` comes from the translator and counts unique lines to
        # translate, which is ≤ the segment count (repeats are translated
        # once), so don't substitute the segment count here.
        pct = 45 + int((done / max(total, 1)) * 17)
        eta_str = f" · ~{eta_sec // 60}m{eta_sec % 60}s left" if eta_sec > 30 else ""
        update(
            progress=min(pct, 62),
            step_detail=f"Translating line {done}/{total}{eta_str}",
        )

    # translate_segments(segments, target_lang, model, ...)
    # NOT (segments, source_lang, target_lang, model) — the translator
    # infers the source language from the prompt internally.
    t0 = time.time()
    translated = await translate_segments(
        todo, target_lang, model,
        context_hint=context_hint,
        progress_callback=_translate_progress,
        # Naming the source lets the prompt guard closely-related pairs
        # (uk->bg, ru->uk) where an untranslated word still looks like an
        # answer. ctx carries whatever whisper actually detected.
        source_lang=ctx.get("effective_src") or ctx.get("source_lang_detected") or "",
    )
    log.info(
        f"[perf] · translate {total_todo} segment(s) in {time.time()-t0:.1f}s "
        f"({total_todo / max(time.time()-t0, 0.001):.2f} seg/s, model={model})"
    )

    # Merge results back (identity for the full-run case, real merge for retries)
    by_idx = {s.get("idx"): s for s in segments}
    for s in translated:
        tgt = by_idx.get(s.get("idx"))
        if tgt is not None:
            tgt.update(s)
        else:
            by_idx[s.get("idx")] = s
    segments = [by_idx[k] for k in sorted(by_idx.keys())]

    # Sanity check: if a significant fraction of segments are still in the
    # source language, stop here instead of letting VoxCPM try to speak
    # English with Russian cross-lingual cfg (which crashes the worker).
    untranslated_count = sum(1 for s in segments if _is_untranslated(s))
    perf.update(
        model=model, translated=total_todo, reused=len(prior_by_idx),
        untranslated=untranslated_count, segments=len(segments),
    )
    if untranslated_count:
        log.warning(
            f"[translate] {untranslated_count}/{len(segments)} segments did not "
            f"translate — retry the 'translate' stage with a different model "
            f"and 'only failed segments' to fix just those"
        )

    # Translate the source title/description while the LLM is still loaded
    # (3D) — doing it after the unload below would reload the model for two
    # short strings.
    await _maybe_translate_meta(job, ctx)

    # Unload the LLM from VRAM before TTS. Without this, Ollama's 9+ GB model
    # sits in VRAM during TTS, leaving too little room for VoxCPM (~4 GB).
    try:
        await unload_ollama_model(model)
    except Exception as e:
        log.warning(f"Failed to unload Ollama model (non-fatal): {e}")

    ctx["segments"] = _serialize_segments(segments)
    ctx["model"] = model
    # Checkpoint BEFORE the pass/fail verdict below. A run that translated
    # some of the transcript must not throw that work away by raising —
    # the whole point of `translate_failed_only` is to resume from here.
    _finalize_translation(job, work, ctx, update, perf)

    # Sanity check: if a significant fraction of segments are still in the
    # source language, stop here instead of letting VoxCPM try to speak
    # English with Russian cross-lingual cfg (which crashes the worker).
    #
    # This used to fire only when EVERY segment failed, so a run that
    # translated 16 of 182 lines was recorded as status=ok and flowed into
    # TTS — producing a "finished" dub that is 91% the original language.
    # Anything past a coin flip is a failed stage, not a warning.
    if untranslated_count > len(segments) // 2:
        raise RuntimeError(
            f"Translation failed for {untranslated_count} of {len(segments)} "
            f"segments — they are still in the source language, so the dub "
            f"would be mostly untranslated. The partial result is saved: "
            f"retry the 'translate' stage with 'only failed segments' to "
            f"finish it, ideally with a faster/non-reasoning model than "
            f"'{model}' (check the logs above for timeouts or "
            f"budget-exhaustion warnings)."
        )


def _split_for_translation(text: str, limit: int = 1500) -> list:
    """Split a long text into translation-sized chunks on paragraph breaks.

    Descriptions can run to several thousand characters; sending one giant
    string back risks a truncated or summarised reply. Paragraphs are kept
    whole where possible, and an oversized paragraph is split on line breaks
    before being hard-cut as a last resort.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks, current = [], ""
    for para in text.split("\n\n"):
        if len(para) > limit:
            if current:
                chunks.append(current)
                current = ""
            line_buf = ""
            for line in para.split("\n"):
                if len(line) > limit:
                    # A long line of ordinary prose must not be cut
                    # mid-word: a chunk ending "…descripti" translates to
                    # nonsense, and the seam is visible in the output. Break
                    # on whitespace, and only slice blindly when a single
                    # token really is longer than the limit.
                    if line_buf:
                        chunks.append(line_buf)
                        line_buf = ""
                    for word in line.split(" "):
                        while len(word) > limit:
                            if line_buf:
                                chunks.append(line_buf)
                                line_buf = ""
                            chunks.append(word[:limit])
                            word = word[limit:]
                        if not word:
                            continue
                        if len(line_buf) + len(word) + 1 > limit:
                            if line_buf:
                                chunks.append(line_buf)
                            line_buf = word
                        else:
                            line_buf = f"{line_buf} {word}" if line_buf else word
                    continue
                if len(line_buf) + len(line) + 1 > limit:
                    if line_buf:
                        chunks.append(line_buf)
                    line_buf = line
                else:
                    line_buf = f"{line_buf}\n{line}" if line_buf else line
            if line_buf:
                current = line_buf
            continue
        if len(current) + len(para) + 2 > limit:
            if current:
                chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


async def _maybe_translate_meta(job, ctx) -> None:
    """Translate the source title + description head into the target language.

    Phase 3D: runs as a best-effort micro-step when the translate stage
    completes, so build_publish_meta() can prefer job["meta_translated"].
    Never fails the stage — a job without a translated title just publishes
    under its original one (editable at approval time anyway).
    """
    meta = job.get("meta") or {}
    title = (meta.get("title") or "").strip()
    if not title or job.get("meta_translated"):
        return
    target = (ctx.get("target_lang") or "").strip().lower()
    src = (ctx.get("effective_src") or ctx.get("source_lang")
           or meta.get("language") or "").strip().lower()
    if not target or target.split("-")[0] == src.split("-")[0]:
        return  # same language (or unknown target) — nothing to translate
    desc = (meta.get("description") or "").strip()
    chapters = meta.get("chapters") or []
    # One request per chunk, so a long description doesn't overflow the
    # model's context and come back truncated. Splitting on blank lines keeps
    # paragraphs intact; the fallback split is only for a wall of text.
    desc_chunks = _split_for_translation(desc) if desc else []
    chapter_titles = [c.get("title", "") for c in chapters if c.get("title")]
    try:
        from pipeline.translator import translate_texts
        texts = [title] + desc_chunks + chapter_titles
        out = await translate_texts(
            texts, target,
            model=ctx.get("model") or cfg.translation_model,
            context_hint=ctx.get("context_hint") or None,
        )
        n_desc = len(desc_chunks)
        translated_titles = out[1 + n_desc:]
        job["meta_translated"] = {
            "title": out[0],
            "description": "\n\n".join(out[1:1 + n_desc]) if n_desc else "",
            # Timings are unchanged by translation — only the labels move.
            "chapters": [
                {"start": c.get("start"), "title": t}
                for c, t in zip(
                    [c for c in chapters if c.get("title")], translated_titles)
            ],
        }
        save_job(job)
        log.info(f"[translate] meta translated for {job['id']}: "
                 f"{out[0][:80]!r} (+{n_desc} description chunk(s), "
                 f"{len(translated_titles)} chapter(s))")
    except Exception as e:
        log.warning(f"[translate] meta title/description translation failed "
                    f"(non-fatal): {e}")


def _finalize_translation(job, work, ctx, update, perf) -> None:
    """Shared tail of the translate stage: UI preview + subtitle file."""
    segments = ctx["segments"]
    job_id = job["id"]
    transcript_preview = [
        {
            "start": s["start"], "end": s["end"],
            "text": s["text"],
            "translated": s.get("translated_text", ""),
            "speaker": s.get("speaker", ""),
        }
        for s in segments
    ]
    srt_path = str(work / "subtitles.srt")
    write_srt(segments, srt_path)
    ctx["srt_path"] = srt_path
    update(
        transcript=transcript_preview, progress=62,
        srt_url=f"/outputs/{job_id}/subtitles.srt",
    )


async def _stage_tts(job, work, ctx, update, perf):
    tts = get_tts_engine()
    tts_dir = str(work / "tts_segments")
    segments = [dict(s) for s in ctx["segments"]]
    effective_src = ctx.get("effective_src", "en")
    target_lang = ctx.get("target_lang", "ru")
    # Languages the cloning engines don't cover (e.g. bg) drop to edge-tts.
    tts_used = _tts_engine_for_lang(tts, target_lang)
    eff_style = ctx.get("voice_style_effective", "")
    voice_seed = ctx.get("voice_seed")
    reference_audio = ctx.get("reference_audio", "")
    speaker_refs = dict(ctx.get("speaker_refs") or {})
    speaker_transcripts = dict(ctx.get("speaker_transcripts") or {})

    # ─── VOICE MODE ROUTING ──────────────────────────────────────────
    # VoxCPM has 3 mutually-exclusive modes; pick one based on the user's
    # choice. NEVER mix style prefix with speaker refs — the model will
    # literally read the style description out loud in the cloned voice.
    #
    # A casting map beats all of it. Every other mode here is a whole-job
    # decision, which is the flaw they share: a style preset silently
    # discarded diarization and collapsed a three-speaker interview onto one
    # voice, and an uploaded reference assigned the same clip to everyone.
    # A cast job resolves each speaker independently, so speakers survive.
    cast = dict(ctx.get("speaker_voice_map") or {})
    speakers = sorted({s.get("speaker") or "SPEAKER_00" for s in segments})
    if cast and isinstance(tts_used, VoxCPMSynthesizer):
        speaker_refs, speaker_transcripts, cast_report = await _resolve_casting(
            cast, speakers, ctx, job["id"], tts_used, target_lang,
        )
        ctx["voice_cast_report"] = cast_report
        perf["cast"] = cast_report
        mode = "cast"
        log.info(f"[tts] Cast {len(speaker_refs)}/{len(speakers)} speaker(s): "
                 + ", ".join(f"{r['speaker']}={r['used']}" for r in cast_report))
        if eff_style or (reference_audio and os.path.exists(reference_audio)):
            # Both were set. The cast is the more specific instruction and it
            # is the one the user auditioned, so it wins — but say so, because
            # a preset that appears to be selected and does nothing is exactly
            # the kind of thing that gets debugged for an hour.
            log.info("[tts] A whole-job voice preset/upload is also set; the "
                     "casting map overrides it")
        has_ref = any(speaker_refs.values())
    else:
        if cast:
            log.warning(
                f"[tts] Ignoring the casting map: {target_lang} synthesizes "
                f"with {type(tts_used).__name__}, which has fixed voices and "
                f"cannot clone")
        has_ref = any(speaker_refs.values())
        if reference_audio and os.path.exists(reference_audio):
            mode = "file_ref"          # user uploaded / file preset
        elif eff_style and eff_style.strip() and not has_ref:
            mode = "voice_design"
        elif eff_style and eff_style.strip() and has_ref:
            # User picked a style preset but we already extracted refs from
            # the source video. They presumably want a fresh designed voice.
            log.info("[pipeline] Style preset + source refs → dropping refs "
                     "for Voice Design")
            speaker_refs = {}
            speaker_transcripts = {}
            mode = "voice_design"
        else:
            mode = "source_refs"

    # Materialise the designed voice instead of designing it per segment.
    # See the DESIGNED_VOICES_DIR block: the style prompt draws a new speaker
    # on every call and the seed does not pin the draw, so designing 1602
    # times gives 1602 voices. Drawing once and cloning from the result gives
    # one. On failure we fall through to the old per-segment prefix — a
    # wobbly voice is still a dub, and no voice is not.
    if mode == "voice_design" and isinstance(tts_used, VoxCPMSynthesizer):
        designed = await _blocking(
            _materialize_designed_voice, tts_used, eff_style, voice_seed,
            target_lang,
        )
        if designed:
            speaker_refs = {sp: designed for sp in speakers}
            speaker_transcripts = {sp: "" for sp in speakers}
            mode = "designed_ref"
            has_ref = True
            perf["designed_voice"] = os.path.basename(designed)
            log.info(f"[tts] Designed voice materialised for {len(speakers)} "
                     f"speaker(s): {os.path.basename(designed)}")
        else:
            log.warning("[tts] Could not materialise the designed voice; "
                        "falling back to per-segment voice design")

    # On a retry, prefer the ORIGINAL source refs over whatever preset the
    # previous run baked into speaker_refs — otherwise "retry with source
    # voice" would silently keep using the last preset forever.
    if mode == "source_refs" and ctx.get("_is_retry"):
        stash = ctx.get("source_speaker_refs") or {}
        if stash:
            log.info(f"[tts] Retry: restoring ORIGINAL source refs: {list(stash)}")
            speaker_refs = dict(stash)
            speaker_transcripts = {sp: "" for sp in speaker_refs}
        # Re-seed so identical inputs don't yield byte-identical audio —
        # otherwise "click retry" would appear to do nothing.
        voice_seed = int(time.time() * 1000) % 2_147_483_647
        ctx["voice_seed"] = voice_seed
        log.info(f"[tts] Retry: rolled fresh voice_seed={voice_seed}")

    update(
        step_detail=f"Generating speech (mode={mode}, "
                    f"preset={ctx.get('voice_preset', 'auto')}, seed={voice_seed})...",
        voice_mode=_VOICE_MODE_LABEL.get(mode, "source"),
        voice_seed=voice_seed,
    )

    # Apply Voice Design prefix ONLY in voice_design mode (VoxCPM-only:
    # edge-tts would literally read the style description out loud).
    #
    # It goes on `tts_text`, NOT `translated_text`. The prefix is an
    # instruction to the model, not part of the dialogue, and
    # `translated_text` is what the rest of the system treats as the line:
    # the assemble stage rewrites subtitles.srt from it, the translation
    # editor shows it, and the partial-retry reuse check compares it. Writing
    # the prefix there put "(middle-aged male voice, warm and calm, clear
    # articulation)" at the head of all 1602 subtitle lines of a real dub,
    # and made every reuse comparison mismatch (prefixed checkpoint text vs
    # clean translation text), so a partial TTS retry re-synthesized
    # everything. `tts_text` is absent from _serialize_segments' whitelist,
    # so it never reaches a checkpoint.
    if mode == "voice_design" and isinstance(tts_used, VoxCPMSynthesizer):
        style = eff_style.strip().strip("()")
        for s in segments:
            base = s.get("translated_text") or s.get("text", "")
            if base and not base.startswith("("):
                s["tts_text"] = f"({style}){base}"

    # Keep already-rendered segments when this is a partial retry
    if ctx.get("tts_keep_existing"):
        # A retry of `tts` starts from the translation checkpoint, whose
        # segments carry no audio_path — so recover the previously rendered
        # files from the tts checkpoint. Reuse is keyed on the translated
        # text as well as the index: a line that was re-translated or edited
        # must be re-synthesized, not left speaking the old words.
        if not any(s.get("audio_path") for s in segments):
            prev = _load_checkpoint(job["id"], "tts_done") or {}
            prev_by_idx = {s.get("idx"): s for s in prev.get("segments", [])}
            recovered = 0
            for s in segments:
                old = prev_by_idx.get(s.get("idx"))
                if not old or not old.get("audio_path"):
                    continue
                if (old.get("translated_text") or "").strip() != \
                        (s.get("translated_text") or "").strip():
                    continue
                if os.path.exists(old["audio_path"]):
                    s["audio_path"] = old["audio_path"]
                    recovered += 1
            log.info(f"[tts] Recovered {recovered} rendered segment(s) from "
                     f"the previous TTS checkpoint")
        todo = [s for s in segments
                if not (s.get("audio_path") and os.path.exists(s["audio_path"]))]
        log.info(f"[tts] Partial retry: keeping {len(segments)-len(todo)} "
                 f"existing segment(s), synthesizing {len(todo)}")
    else:
        todo = segments

    def synth_progress(done, total):
        pct = 65 + int((done / max(total, 1)) * 20)
        update(progress=min(pct, 85), step_detail=f"Synthesizing: {done}/{total}")

    t0 = time.time()
    if todo:
        if isinstance(tts_used, VoxCPMSynthesizer):
            todo = tts_used.synthesize_segments(
                todo, tts_dir,
                speaker_refs=speaker_refs,
                speaker_transcripts=speaker_transcripts,
                progress_callback=synth_progress,
                voice_seed=voice_seed,
                tts_speed=ctx.get("tts_speed", "balanced"),
                is_cross_lingual=(effective_src != target_lang),
                target_lang=target_lang,
                cfg_override=ctx.get("voxcpm_cfg"),
                steps_override=ctx.get("voxcpm_steps"),
            )
        else:
            todo = await tts_used.synthesize_segments_async(
                todo, tts_dir, target_lang,
                progress_callback=synth_progress,
            )
        # Merge back (no-op when todo IS segments)
        by_idx = {s.get("idx"): s for s in segments}
        for s in todo:
            if s.get("idx") in by_idx:
                by_idx[s["idx"]].update(s)
        segments = [by_idx[k] for k in sorted(by_idx.keys())]

    synth_ok = sum(1 for s in segments if s.get("audio_path"))
    took = time.time() - t0
    log.info(
        f"[perf] · tts {synth_ok}/{len(segments)} segments in {took:.1f}s "
        f"({took / max(synth_ok, 1):.2f}s/segment, engine={type(tts_used).__name__}, "
        f"mode={mode})"
    )
    perf.update(
        engine=type(tts_used).__name__, mode=mode, seed=voice_seed,
        synthesized=synth_ok, failed=len(segments) - synth_ok,
        sec_per_segment=round(took / max(synth_ok, 1), 2),
    )
    # QA/tier aggregates from the worker's "done" event (tier_stats,
    # qa_regens, qa_measured_count, qa_unmeasured_count) — previously
    # log-only, now persisted with the stage metrics.
    if hasattr(tts_used, "last_run_stats"):
        try:
            perf.update({k: v for k, v in (tts_used.last_run_stats() or {}).items()
                         if v is not None})
        except Exception:
            pass
    if synth_ok == 0:
        # The worker reports per-segment failures as events rather than
        # raising, so name the actual cause here — "check model/GPU" sent us
        # hunting the GPU for what was a bad generation parameter.
        detail = ""
        if hasattr(tts_used, "last_failure_detail"):
            detail = tts_used.last_failure_detail()
        if detail:
            perf["failure"] = detail[:500]
        raise RuntimeError(
            f"All {len(segments)} TTS segments failed"
            + (f": {detail}" if detail else " - check model/GPU")
        )

    ctx["segments"] = _serialize_segments(segments)
    ctx["sample_rate"] = getattr(tts_used, "sample_rate", 48000)
    ctx.pop("tts_keep_existing", None)
    update(progress=85, step_detail=f"Synthesized {synth_ok}/{len(segments)}")


async def _stage_assemble(job, work, ctx, update, perf):
    update(step_detail="Assembling dubbed audio...")
    segments = [dict(s) for s in ctx["segments"]]
    dubbed_wav = str(work / "dubbed_audio.wav")
    asm_info = await _blocking(
        assemble_dubbed_audio,
        segments, ctx["duration"], dubbed_wav,
        ctx.get("sample_rate", 48000), apply_loudnorm=True,
    )
    _save_placements(work, segments)
    ctx["dubbed_wav"] = dubbed_wav
    # The assembler wrote placed_start/placed_end onto the local copies above.
    # Carry them back so everything downstream — the assemble_done checkpoint,
    # burn-in subtitles, showcase cuts — reads the timeline the audio actually
    # has rather than the source timings it was built from.
    for src_seg, placed in zip(ctx["segments"], segments):
        if "placed_start" in placed:
            src_seg["placed_start"] = placed["placed_start"]
            src_seg["placed_end"] = placed["placed_end"]
    # Rewrite the subtitles against that timeline. The first write happens
    # back in the translate stage, before a single segment has been placed,
    # so its timings leave the SRT out of sync with the audio by however far
    # the dub had to shift.
    try:
        write_srt(segments, str(work / "subtitles.srt"))
    except Exception as e:
        log.warning(f"Could not rewrite SRT against dubbed timings: {e}")
    # Persist what the assembler measured (previously log-only): how many
    # segments needed atempo stretching, and the ffmpeg loudnorm measurement
    # (input/output LUFS, true peak, LRA) parsed from print_format=json.
    if isinstance(asm_info, dict):
        perf["stretched_count"] = asm_info.get("stretched_count")
        perf["hard_compressed"] = asm_info.get("hard_compressed")
        perf["max_drift_sec"] = asm_info.get("max_drift_sec")
        if asm_info.get("loudnorm"):
            perf["loudnorm"] = asm_info["loudnorm"]
    try:
        perf["wav_mb"] = round(os.path.getsize(dubbed_wav) / 1024 / 1024, 1)
    except OSError:
        pass
    update(progress=92)


async def _stage_merge(job, work, ctx, update, perf):
    update(step_detail="Rendering final video...")
    output_mp4 = str(work / "dubbed_video.mp4")
    await _blocking(
        merge_audio_video,
        ctx["video_path"], ctx["dubbed_wav"], output_mp4,
        ctx.get("bg_audio_path", "") if ctx.get("keep_bg") else "",
    )
    ctx["output_mp4"] = output_mp4
    try:
        perf["mp4_mb"] = round(os.path.getsize(output_mp4) / 1024 / 1024, 1)
    except OSError:
        pass
    update(
        status="complete",
        progress=100,
        output_url=f"/outputs/{job['id']}/dubbed_video.mp4?v={int(time.time())}",
        completed_at=time.time(),
        step_detail="Done!",
    )
    log.info(f"Pipeline complete: {output_mp4}")


async def _finalize_reupload(job, work, ctx, update):
    """Reupload mode (3E): turn the downloaded source into the final output.

    Publish, download and export all read outputs/<id>/dubbed_video.mp4, so
    reupload jobs produce that same file: a hardlink/copy when the source is
    already an MP4, else a lossless `ffmpeg -c copy` remux. No transcode —
    a reupload must be bit-identical video/audio to the source.
    """
    import subprocess
    src = Path(ctx["video_path"])
    dst = work / "dubbed_video.mp4"
    update(progress=90, step_detail="Preparing file for reupload...")

    if src.resolve() != dst.resolve():
        if dst.exists():
            dst.unlink()
        if src.suffix.lower() == ".mp4":
            try:
                os.link(src, dst)  # instant, zero extra disk
            except OSError:
                await _blocking(shutil.copyfile, src, dst)
        else:
            # .mkv / .webm container — remux streams into MP4 unchanged.
            try:
                await _blocking(
                    subprocess.run,
                    ["ffmpeg", "-y", "-i", str(src), "-c", "copy",
                     "-movflags", "+faststart", str(dst)],
                    check=True, capture_output=True, timeout=600,
                )
            except subprocess.CalledProcessError as e:
                err = (e.stderr or b"").decode("utf-8", errors="replace")[-300:]
                raise RuntimeError(
                    f"Could not remux {src.name} into MP4 for reupload "
                    f"(codec not MP4-compatible?): {err}"
                )

    ctx["output_mp4"] = str(dst)
    update(
        status="complete",
        progress=100,
        output_url=f"/outputs/{job['id']}/dubbed_video.mp4?v={int(time.time())}",
        completed_at=time.time(),
        step_detail="Ready to publish (reupload mode — source kept as-is)",
    )
    log.info(f"Reupload pipeline complete: {dst}")


STAGE_HANDLERS = {
    "download": _stage_download,
    "extract": _stage_extract,
    "transcribe": _stage_transcribe,
    "diarize": _stage_diarize,
    "translate": _stage_translate,
    "tts": _stage_tts,
    "assemble": _stage_assemble,
    "merge": _stage_merge,
}


# ─────────────────────────────────────────────────────────────
# Pipeline driver
# ─────────────────────────────────────────────────────────────
def _wizard_pause_after(stage_id: str, ctx: dict) -> Optional[tuple]:
    """Return (status, detail, checkpoint) if the wizard pauses after a stage."""
    mode = ctx.get("wizard_mode", "auto")
    if stage_id == "diarize" and mode == "review_transcript":
        return ("awaiting_transcript_review",
                "Review transcription — edit or approve to continue",
                "transcription_done")
    if stage_id == "translate" and mode == "review_translation":
        return ("awaiting_translation_review",
                "Review translation — edit, retranslate, or approve to continue",
                "translation_done")
    if stage_id == "translate" and mode == "review_voices":
        # After translate, not after diarize, even though the speaker
        # references exist by then. Casting is only worth reviewing if you can
        # hear it, and a preview has to speak the lines the dub will actually
        # speak — at the diarize gate the text is still in the source
        # language, so a cross-lingual preview would demonstrate the wrong
        # phonetics. Translation is minutes; the stage this gate protects is
        # hours (a measured 12.25h for 1602 segments), so the gate still sits
        # in front of essentially all of the cost.
        return ("awaiting_voice_review",
                "Cast the voices — assign one per speaker, preview, then continue",
                "translation_done")
    return None


# Which stage's gate runs after which pipeline stage, and where it parks.
# Deliberately placed *before* the expensive stage each one protects:
# transcription is checked before a translator is paid for it, translation
# before the GPU synthesizes it. A gate after the last stage can only ask for
# a retry; a gate before one can save the work entirely.
_GATE_AFTER_STAGE = {
    "diarize": ("asr", "awaiting_transcript_review", "transcription_done"),
    "translate": ("translation", "awaiting_translation_review", "translation_done"),
}


def _quality_gate_after(stage_id: str, ctx: dict, job: dict,
                        job_id: str) -> Optional[tuple]:
    """Pause the job when this stage's output is not worth building on.

    Off unless `quality_gate` is enabled, and never fatal: a scorer that
    raises must not fail a dub that is otherwise fine, so anything unexpected
    is logged and the pipeline continues. The verdicts are stashed on the job
    so the UI, the webhook and the trends endpoint all describe the same
    decision rather than recomputing it.
    """
    # getattr, not attribute access: cfg is runtime-configurable and this
    # field is new, so an older config object (or a test double) must
    # degrade to "gate off" rather than failing the whole pipeline.
    if not getattr(cfg, "quality_gate", False):
        return None
    mapping = _GATE_AFTER_STAGE.get(stage_id)
    if not mapping:
        return None
    stage_name, status, cp_name = mapping
    try:
        from pipeline.quality import full_report, gate
        report = full_report(
            ctx.get("segments") or [],
            target_lang=ctx.get("target_lang") or job.get("target_lang") or "",
            source_lang=ctx.get("effective_src") or "",
        )
        result = gate(report, stage_name)
    except Exception as e:
        log.warning(f"[gate] {stage_name} gate skipped ({e})", exc_info=True)
        return None

    job["quality_gate"] = {
        "stage": stage_name,
        "passed": result["pass"],
        "score": result["score"],
        "threshold": result["threshold"],
        "reasons": result["reasons"],
        "verdicts": result["verdicts"],
        "checked_at": time.time(),
    }
    if result["pass"]:
        log.info(f"[gate] {stage_name} passed for {job_id} "
                 f"(score {result['score']})")
        return None

    why = "; ".join(result["reasons"])
    log.warning(f"[gate] {stage_name} gate FAILED for {job_id}: {why}")
    app_audit.record("job.quality_gate", target=job_id, detail=why)
    return (status,
            f"Paused by the {stage_name} quality gate — {why}. "
            f"Review, retranslate, or continue anyway.",
            cp_name)


async def run_pipeline_stages(
    job_id: str,
    ctx: dict,
    start_stage: str = "download",
    stop_after: str = "",
) -> None:
    """Run the pipeline from `start_stage` through to the end (or `stop_after`).

    Each stage is timed and resource-sampled, then its output context is
    checkpointed. A failure marks the job as errored but leaves every prior
    checkpoint intact, so the failed stage — and only the failed stage — can
    be retried via /api/dub/{id}/retry_stage/{stage}.
    """
    job = jobs[job_id]
    work = OUTPUT_DIR / job_id
    work.mkdir(exist_ok=True)

    def update(**kwargs):
        # Check the cancel flag at every stage transition. Any pipeline
        # path that calls update() will raise JobCancelled within ~1
        # instruction of the user clicking Cancel, and the outer handler
        # in _job_queue_worker will mark status=cancelled cleanly.
        if job.get("cancel_requested"):
            _maybe_terminate_tts_worker()
            raise JobCancelled(f"Job {job_id} cancelled by user")
        prev_status = job.get("status")
        job.update(kwargs)
        save_job(job)
        # Feed the activity stream, but only on a real transition: update() is
        # called for progress ticks an order of magnitude more often than for
        # status changes, and a feed of "still transcribing" is not a feed.
        new_status = kwargs.get("status")
        if new_status and new_status != prev_status:
            try:
                activity.record_job(
                    job_id, new_status,
                    title=job.get("title") or job.get("source_label"),
                    stage=kwargs.get("step_detail"),
                )
            except Exception:
                log.debug("[activity] could not record transition", exc_info=True)
            try:
                _fire_webhooks(new_status, job)
            except Exception:
                log.debug("[webhooks] could not schedule delivery", exc_info=True)

    _resolve_voice_into_ctx(job, ctx)
    start_i = max(_stage_index(start_stage), 0)
    stop_i = _stage_index(stop_after) if stop_after else len(PIPELINE_STAGES) - 1
    if stop_i < 0:
        stop_i = len(PIPELINE_STAGES) - 1

    run_t0 = time.time()
    log.info(
        f"[pipeline] job={job_id} running stages "
        f"{STAGE_ORDER[start_i]}→{STAGE_ORDER[stop_i]} "
        f"({stop_i - start_i + 1} of {len(PIPELINE_STAGES)})"
    )

    try:
        reuse_stages = reuse_runtime.enabled_stages(cfg)
        reused_here: list = ctx.setdefault("_reused_stages", [])

        for i in range(start_i, stop_i + 1):
            spec = PIPELINE_STAGES[i]
            sid = spec["id"]
            lo, hi = spec["progress"]
            # Anchor for the failure log window: if this stage dies, the UI
            # shows exactly the ring entries produced since this point.
            job["_stage_log_from"] = logbuf.current_seq()
            update(
                status=spec["status"], progress=lo, stage_id=sid,
                step_detail=spec["hint"],
            )

            # ── Stage reuse (beta, off unless cfg.reuse_enabled) ──────
            # Download/extract/transcribe/diarize are ~60% of pipeline time
            # and none of it depends on the target language, so a redub into
            # a new language recomputes all of it for nothing. When a
            # previous job already produced this exact output — same inputs,
            # same stage version — copy it instead.
            fingerprint = None
            if sid in reuse_stages:
                fingerprint = reuse_runtime.fingerprint_for(sid, ctx, cfg)
            hit = None
            if fingerprint:
                hit = reuse_runtime.try_reuse(
                    sid, fingerprint, ctx, work, OUTPUT_DIR, cfg)

            if hit:
                reused_here.append({"stage": sid, "from": hit["job_id"],
                                    "fingerprint": fingerprint})
                # A skipped stage never runs its own update(), so replay the
                # job fields it would have published. Without this a reused
                # download leaves job["duration"] unset and the publisher's
                # duplicate check quietly loses its duration signal.
                fields = dict(hit.get("job_fields") or {})
                if sid == "translate" and (work / "subtitles.srt").exists():
                    fields["srt_url"] = f"/outputs/{job_id}/subtitles.srt"
                    ctx["srt_path"] = str(work / "subtitles.srt")
                # Persisted on the job so the result is auditable after the
                # fact: a run that silently skipped work is impossible to
                # debug when its output looks wrong.
                update(progress=hi, reused_stages=list(reused_here),
                       step_detail=(
                           f"Reused from job {hit['job_id'][:8]} — "
                           f"{spec['label'].lower()} skipped"),
                       **fields)
                log.info(f"[pipeline] job={job_id} stage={sid} REUSED from "
                         f"{hit['job_id']}")
            else:
                with stage_timer(work, job_id, sid) as perf:
                    await STAGE_HANDLERS[sid](job, work, ctx, update, perf)
                if fingerprint:
                    reuse_runtime.record_stage(
                        sid, fingerprint, job_id, ctx, work)

            _save_checkpoint(job_id, work, stage=spec["checkpoint"],
                             data=_ctx_for_checkpoint(ctx))

            pause = _wizard_pause_after(sid, ctx) or _quality_gate_after(
                sid, ctx, job, job_id)
            if pause:
                status, detail, cp_name = pause
                update(status=status, progress=hi, step_detail=detail,
                       checkpoint_stage=cp_name)
                log.info(f"[wizard] Paused after '{sid}' for job {job_id}")
                return

        # A partial run (stop_after set, or a stage list that doesn't reach
        # `merge`) must land on a terminal status. Without this the job keeps
        # the last stage's in-flight status — "diarizing" forever — which
        # reads as a hung job and blocks further stage retries.
        if stop_i < len(PIPELINE_STAGES) - 1:
            if job.get("mode") == "reupload":
                # Reupload mode (3E): the download IS the product. Produce
                # dubbed_video.mp4 from the source and finish as complete —
                # "paused for review" is for partial dub runs, not this.
                await _finalize_reupload(job, work, ctx, update)
            else:
                last = STAGE_ORDER[stop_i]
                update(
                    status="paused",
                    step_detail=f"Stopped after '{STAGE_BY_ID[last]['label']}' — "
                                f"review the result, then retry or continue",
                    stage_id=None,
                    checkpoint_stage=PIPELINE_STAGES[stop_i]["checkpoint"],
                )

        log.info(
            f"[perf] ═ pipeline job={job_id} finished "
            f"{STAGE_ORDER[start_i]}→{STAGE_ORDER[stop_i]} "
            f"in {time.time() - run_t0:.1f}s total"
        )
    except JobCancelled:
        # Re-raise so _job_queue_worker marks the job as 'cancelled'
        # rather than 'error'. Logging is handled upstream.
        log.info(f"Pipeline cancelled for {job_id}")
        raise
    except (asyncio.CancelledError, SeparationAborted) as e:
        # Server shutdown mid-job. CancelledError is a BaseException on 3.11,
        # so without this clause it sails past `except Exception` below and
        # the job keeps whatever in-flight status it had — 'merging' forever,
        # which the UI reads as a live job that never finishes.
        job["failed_stage"] = job.get("stage_id", "")
        update(
            status="interrupted",
            error="Interrupted by server shutdown — retry from the last checkpoint",
            step_detail="Interrupted by server shutdown",
        )
        log.warning(f"[pipeline] job={job_id} interrupted at stage "
                    f"'{job.get('stage_id')}' ({type(e).__name__})")
        raise
    except Exception as e:
        job["failed_stage"] = job.get("stage_id", "")
        # Log BEFORE recording: _set_job_error snapshots the ring seq as the
        # end of the failure window, and the traceback must be inside it.
        log.exception(f"Pipeline failed at stage '{job.get('stage_id')}': {e}")
        _set_job_error(job, e)
        # A DownloadFailed carries a structured rescue hint (which yt-dlp
        # command to run by hand) — surface it for the UI's rescue panel.
        hint = getattr(e, "hint", None)
        if isinstance(hint, dict) and hint:
            job["download_hint"] = hint
        update(status="error")


async def run_pipeline(
    job_id: str,
    source: str,
    source_lang: str,
    target_lang: str,
    model: str,
    keep_bg: bool,
    whisper_model: str,
    reference_audio: str = "",
    speaker_mode: str = "main",
    context_hint: str = "",
    voice_style: str = "",
    voice_preset: str = "auto",
    tts_speed: str = "balanced",
    wizard_mode: str = "auto",  # "auto" | "review_translation"
                                # | "review_transcript" | "review_voices"
    auto_denoise: bool = True,  # apply ffmpeg denoise before WhisperX
    mode: str = "dub",  # "dub" | "reupload" (3E: reupload = download only)
    # Per-job VoxCPM overrides; 0 = follow Settings → Voice & TTS (CLD-189)
    voxcpm_cfg: float = 0.0,
    voxcpm_steps: int = 0,
):
    """Main dubbing pipeline entry point.

    Builds the initial stage context from the request and runs every stage.
    When wizard_mode != 'auto' the driver pauses at the matching checkpoint
    with status='awaiting_review' so the user can inspect/edit intermediate
    results before continuing."""
    job = jobs[job_id]
    job["wizard_mode"] = wizard_mode
    job["mode"] = normalize_job_mode(mode) or "dub"
    ctx = {
        "source": source,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": model,
        "keep_bg": keep_bg,
        "whisper_model": whisper_model,
        "reference_audio": reference_audio,
        "speaker_mode": speaker_mode,
        "context_hint": context_hint,
        "voice_style": voice_style,
        "voice_preset": voice_preset,
        "tts_speed": tts_speed,
        "wizard_mode": wizard_mode,
        "auto_denoise": auto_denoise,
        "voxcpm_cfg": voxcpm_cfg,
        "voxcpm_steps": voxcpm_steps,
    }
    # Reupload mode walks only the download stage; the driver's tail then
    # finalizes dubbed_video.mp4 from the source (see _finalize_reupload).
    stage_ids = stages_for_mode(job["mode"])
    stop_after = stage_ids[-1] if len(stage_ids) < len(STAGE_ORDER) else ""
    await run_pipeline_stages(
        job_id, ctx, start_stage=stage_ids[0], stop_after=stop_after)


# ─────────────────────────────────────────────────────────────
# API: System & Models
# ─────────────────────────────────────────────────────────────

@app.get("/api/lm_studio/models")
async def get_lm_studio_models():
    """Proxy to LM Studio API to get available models.

    The frontend calls this endpoint (instead of hitting LM Studio directly)
    to avoid CORS issues and centralize configuration. Queries LM Studio
    directly on every call so the model list is current when the user opens
    Settings / Models — this is not on a polling path, so it costs one request
    per visit.
    """
    if not USE_LM_STUDIO:
        return {"models": []}
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                LM_STUDIO_MODELS_ENDPOINT,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get("id", "") for m in data.get("data", [])]
                    return {"models": models}
                log.warning(f"LM Studio returned {response.status} from {LM_STUDIO_MODELS_ENDPOINT}")
                return {"models": []}
    except Exception as e:
        log.warning(f"Failed to fetch LM Studio models: {e}")
        return {"models": []}


@app.get("/api/system")
async def system_status():
    status = get_system_status()
    ollama_ok, ollama_models = await check_ollama()
    status["ollama"] = {
        "ok": ollama_ok,
        "models": ollama_models,
        "binary": status["ollama_binary"]["ok"],
    }
    # check_ollama() delegates to LM Studio when that's the configured
    # backend, but the UI reads `system.lm_studio` — a key nothing ever set,
    # so the System panel reported "Not connected" even while LM Studio was
    # happily serving models. Publish both spellings from the one probe.
    status["lm_studio"] = {
        "ok": ollama_ok if USE_LM_STUDIO else False,
        "models": ollama_models if USE_LM_STUDIO else [],
        "url": LM_STUDIO_MODELS_ENDPOINT,
    }
    status["catalog"] = MODEL_CATALOG
    # Deployment mode ("local" | "hosted") — the UI gates the Workspace
    # group on this without needing a second round trip.
    status["mode"] = cfg.mode if cfg.mode in ("local", "hosted") else "local"
    # Batch bounds, so the language picker enforces the same numbers the API
    # validates against instead of keeping its own copy in sync by hand.
    status["limits"] = {
        "min_target_langs": MIN_TARGET_LANGS,
        "max_target_langs": MAX_TARGET_LANGS,
    }

    # Live resource snapshot — the same probes the per-stage sampler uses,
    # so the System panel and the stage metrics always agree on what's
    # being measured (and on whether GPU telemetry is available at all).
    live = {"gpu_backend": gpu_backend()}
    gpu_now = gpu_snapshot()
    if gpu_now:
        live["gpu"] = {k: round(v, 1) for k, v in gpu_now.items()}
    try:
        import psutil as _ps
        live["cpu_pct"] = _ps.cpu_percent(interval=None)
        live["cpu_cores"] = _ps.cpu_count(logical=True)
        vm = _ps.virtual_memory()
        live["ram_used_gb"] = round(vm.used / 1024**3, 1)
        live["ram_total_gb"] = round(vm.total / 1024**3, 1)
        live["proc_rss_mb"] = round(
            _ps.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1
        )
    except Exception:
        pass
    status["resources"] = live

    tts_ready = status["voxcpm"]["ok"] or status["edge_tts"]["ok"]
    ready = (
        status["python"]["ok"] and
        status["ffmpeg"]["ok"] and
        status["yt_dlp"]["ok"] and
        status["whisper"]["ok"] and
        tts_ready and
        ollama_ok and
        len(ollama_models) > 0
    )
    status["ready"] = ready

    # Setup notices ride along on the poll the UI already runs, so the banner
    # needs no second timer. Passive checks only — nothing here touches the
    # network; the deep probe is behind POST /api/diagnostics/run.
    try:
        passive = diag.passive_checks(status)
        status["accelerator"] = passive["accelerator"]
        status["checks"] = passive["checks"]
        status["notices"] = diag.current_notices(status)
        status["notice_severity"] = worst_severity(status["notices"])
        last = diag.last_deep()
        status["diagnostics_ran_at"] = (last or {}).get("ran_at")
    except Exception as e:
        # Diagnostics must never be the reason the System panel breaks.
        log.debug(f"[system] diagnostics failed: {e}")
        status["notices"] = []
    return status


# ─────────────────────────────────────────────────────────────
# API: Trend scout — discover popular YouTube videos to dub
# ─────────────────────────────────────────────────────────────
from fastapi import Request as _ScoutRequest  # noqa: E402  (route-local import; top import block owned elsewhere)


@app.get("/api/scout/trending")
async def scout_trending(category: str = "", country: str = "",
                         limit: int = 20, include_shorts: bool = False):
    """Trending video candidates from mostviewed.today (yt-dlp fallback).

    Each candidate carries `already_processed` (+ `existing_job_id`) matched
    against jobs this server has already run, so agents can skip duplicates.
    """
    from app import scout
    try:
        result = await scout.fetch_trending(
            category=category or None,
            country=country or None,
            limit=max(1, min(int(limit or 20), 100)),
            include_shorts=include_shorts,
        )
    except Exception as e:
        log.warning(f"[scout] trending fetch failed: {e}")
        return JSONResponse({"error": str(e)}, 502)
    scout.annotate_already_processed(result["candidates"], jobs)
    return result


@app.post("/api/scout/dub")
async def scout_dub(request: _ScoutRequest):
    """Convenience: submit a scout candidate straight into the dub pipeline.

    JSON body: {video_id or url, target_lang, mode?: "dub"|"reupload"|"auto",
    is_music?, scheduled_at?, + optional pass-through of the usual dub params
    (source_lang, model, keep_bg, whisper_model, speaker_mode, context_hint,
    voice_style, voice_preset, tts_speed, wizard_mode, auto_denoise,
    lip_sync)}. mode "auto" resolves to "reupload" for music candidates
    (category 10), else "dub". Both modes reuse the exact POST /api/dub
    submission path — reupload jobs walk only the download stage (3E).
    """
    from app import scout
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
    except Exception:
        return JSONResponse({"error": "JSON object body required"}, 400)

    video_id = str(body.get("video_id") or "").strip()
    url = str(body.get("url") or "").strip()
    if not url and video_id:
        url = f"https://www.youtube.com/watch?v={video_id}"
    if not url:
        return JSONResponse({"error": "Provide video_id or url"}, 400)

    target_lang = str(body.get("target_lang") or "").strip()
    if not target_lang:
        return JSONResponse({"error": "target_lang is required"}, 400)

    mode = str(body.get("mode") or "auto").strip().lower()
    if mode not in ("dub", "reupload", "auto"):
        return JSONResponse(
            {"error": f"Unknown mode {mode!r} (dub|reupload|auto)"}, 400)
    if mode == "auto":
        is_music = body.get("is_music")
        if is_music is None and video_id:
            cand = scout.find_cached_candidate(video_id)
            is_music = bool(cand and cand.get("is_music"))
        mode = "reupload" if is_music else "dub"

    try:
        scheduled_at = float(body.get("scheduled_at") or 0.0)
    except (TypeError, ValueError):
        return JSONResponse({"error": "scheduled_at must be unix epoch seconds"}, 400)

    # Reuse the exact same submission path as POST /api/dub (no duplication:
    # start_dub is called directly with the URL branch inputs). video and
    # reference MUST be passed explicitly: calling the function directly
    # bypasses FastAPI's dependency injection, so their declared defaults
    # are `File(None)` sentinel objects — truthy, and without .filename.
    resp = await start_dub(
        source=url,
        video=None,
        reference=None,
        source_lang=str(body.get("source_lang") or "auto"),
        target_lang=target_lang,
        model=str(body.get("model") or "gemma4:e4b"),
        keep_bg=bool(body.get("keep_bg", True)),
        whisper_model=str(body.get("whisper_model") or "large-v3"),
        speaker_mode=str(body.get("speaker_mode") or "main"),
        context_hint=str(body.get("context_hint") or ""),
        voice_style=str(body.get("voice_style") or ""),
        voice_preset=str(body.get("voice_preset") or "auto"),
        tts_speed=str(body.get("tts_speed") or "balanced"),
        wizard_mode=str(body.get("wizard_mode") or "auto"),
        auto_denoise=bool(body.get("auto_denoise", False)),
        lip_sync=bool(body.get("lip_sync", False)),
        mode=mode,
        scheduled_at=scheduled_at,
        voxcpm_cfg=float(body.get("voxcpm_cfg") or 0.0),
        voxcpm_steps=int(body.get("voxcpm_steps") or 0),
    )
    if isinstance(resp, JSONResponse):
        return resp  # start_dub validation error — pass it through

    job_id = resp.get("job_id")
    if video_id and job_id and job_id in jobs:
        # Record provenance so already_processed matching works even before
        # the downloader-metadata probe populates job["meta"] on its own.
        jobs[job_id].setdefault("meta", {}).setdefault("video_id", video_id)
        jobs[job_id]["scout"] = {"video_id": video_id}
        save_job(jobs[job_id])
    return {**resp, "mode": mode}


@app.post("/api/diagnostics/run")
async def run_diagnostics(token: str = Form("")):
    """Deep environment check — the only place that touches the network.

    Asks Hugging Face whether the gated pyannote repos are really accessible,
    which is the one question that distinguishes "you never accepted the
    conditions" from "the download failed today". pyannote itself prints
    "private or gated" for any HTTP error, so its message cannot be trusted to
    tell them apart.
    """
    try:
        report = await diag.deep_checks(token or effective_hf_token())
        return {"ok": True, **report}
    except Exception as e:
        log.warning(f"[diagnostics] deep check failed: {e}")
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.get("/api/logs")
async def get_logs(level: str = "", limit: int = 300, since_seq: int = 0):
    """Recent server log lines, including third-party stdout/stderr.

    Secrets are scrubbed on the way *into* the buffer (app/logbuf.py), not
    here — GOCHIDUBB_HOST can expose this server to the network, and a token
    that reached the deque would already be readable there.
    """
    return logbuf.snapshot(level=level, limit=max(1, min(int(limit or 300), 2000)),
                           since_seq=int(since_seq or 0))


@app.get("/api/activity")
async def get_activity(kinds: str = "", limit: int = 80, since_id: int = 0):
    """The activity feed: agent tool calls and job transitions, newest first.

    `kinds` is a comma-separated subset of app.activity.KINDS, matching the
    design's filter tabs. `since_id` is a monotonic event id rather than a
    timestamp, so the UI can poll for only what it has not seen without two
    events in the same millisecond hiding each other.

    In-memory and bounded — this is "what is happening now", not an audit
    trail. The persisted audit log is a separate concern.
    """
    want = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None
    if want:
        bad = [k for k in want if k not in activity.KINDS]
        if bad:
            return JSONResponse(
                {"error": f"unknown kind(s): {', '.join(bad)}",
                 "valid": list(activity.KINDS)}, 400)
    return {
        "events": activity.recent(
            limit=max(1, min(int(limit or 80), 400)),
            kinds=want,
            since_id=int(since_id or 0),
        ),
        "last_id": activity.last_id(),
    }


# ── Develop: API keys ────────────────────────────────────────────────
# Keys are hashed at rest and the plaintext is returned exactly once, at
# creation. Scope enforcement is gated on hosted mode (see _require_scope):
# in local mode every route stays open, as it always has been.

@app.get("/api/apikeys")
async def list_api_keys():
    return {"keys": app_apikeys.list_keys(),
            "scopes": app_apikeys.SCOPES,
            "enforced": cfg.mode == "hosted"}


@app.post("/api/apikeys")
async def create_api_key(name: str = Form(...), scopes: str = Form(""),
                         environment: str = Form("live"),
                         expires_days: str = Form("")):
    want = [s.strip() for s in scopes.split(",") if s.strip()]
    try:
        days = int(expires_days) if str(expires_days).strip() else None
    except ValueError:
        return JSONResponse({"error": "expires_days must be a number"}, 400)
    try:
        rec, token = app_apikeys.create(name, want, environment=environment,
                                        expires_days=days)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 400)
    activity.record_system(f"API key created · {rec['name']} "
                           f"({', '.join(rec['scopes'])})")
    app_audit.record("apikey.create", target=rec["id"],
                     detail=f"{rec['name']} · {', '.join(rec['scopes'])}",
                     environment=rec["environment"])
    # The only time the token is ever returned.
    return {"key": rec, "token": token}


@app.post("/api/apikeys/{key_id}/revoke")
async def revoke_api_key(key_id: str):
    if not app_apikeys.revoke(key_id):
        return JSONResponse({"error": "not found or already revoked"}, 404)
    activity.record_system(f"API key revoked · {key_id}", severity="warn")
    app_audit.record("apikey.revoke", target=key_id)
    return {"ok": True}


@app.delete("/api/apikeys/{key_id}")
async def delete_api_key(key_id: str):
    if not app_apikeys.delete(key_id):
        return JSONResponse({"error": "not found"}, 404)
    return {"ok": True}


# ── Develop: webhooks ────────────────────────────────────────────────

@app.get("/api/webhooks")
async def list_webhooks():
    return {"hooks": app_webhooks.list_hooks(),
            "events": list(app_webhooks.EVENTS),
            "deliveries": app_webhooks.deliveries(50)}


@app.post("/api/webhooks")
async def add_webhook(url: str = Form(...), events: str = Form(""),
                      secret: str = Form("")):
    want = [e.strip() for e in events.split(",") if e.strip()]
    try:
        rec = app_webhooks.add(url, want, secret=secret)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 400)
    activity.record_system(f"Webhook added · {rec['url']}")
    app_audit.record("webhook.add", target=rec["id"], detail=rec["url"],
                     events=rec["events"])
    return {"hook": rec}


@app.delete("/api/webhooks/{hook_id}")
async def delete_webhook(hook_id: str):
    if not app_webhooks.remove(hook_id):
        return JSONResponse({"error": "not found"}, 404)
    app_audit.record("webhook.remove", target=hook_id)
    return {"ok": True}


@app.post("/api/webhooks/{hook_id}/test")
async def test_webhook(hook_id: str):
    """Send a sample delivery so a listener can be verified before a real run."""
    hook = next((h for h in app_webhooks._read() if h.get("id") == hook_id), None)
    if not hook:
        return JSONResponse({"error": "not found"}, 404)
    d = await app_webhooks.deliver_one(
        hook, "job.completed",
        {"job_id": "test_0000", "status": "complete", "title": "Test delivery",
         "target_lang": "fr", "duration_sec": 61.0, "test": True})
    return {"delivery": d}


# ── Workspace: usage, members, audit ─────────────────────────────────
# The minutes here are real — measured source duration × target languages, the
# way the design meters. The money is an ESTIMATE of what a hosted workspace
# would charge at the design's published rates; this server bills nobody. The
# UI must present it as such.

@app.get("/api/billing/usage")
async def billing_usage(window_days: int = 30):
    since = time.time() - max(1, int(window_days or 30)) * 86400
    try:
        # Defined further down; resolved at call time. It walks each job's
        # output dir, which is fine for an on-demand page but not for a poll.
        storage = await storage_stats()
    except Exception:
        storage = {}
    gb = float(storage.get("total_gb") or 0.0)
    summary = app_billing.summarize(list(jobs.values()), since=since,
                                    storage_gb=gb)
    summary["window_days"] = int(window_days or 30)
    summary["mode"] = cfg.mode
    # Spelled out so no caller can mistake this for an invoice.
    summary["disclaimer"] = (
        "Estimate only. Minutes are measured from real jobs; the cost applies "
        "the design's published hosted rates. This server bills nobody."
    )
    summary["tiers"] = [
        {"from": 0, "to": 500, "rate": 0.080},
        {"from": 500, "to": 2000, "rate": 0.065},
        {"from": 2000, "to": None, "rate": 0.050},
    ]
    return summary


# ─────────────────────────────────────────────────────────────
# API: Pre-flight estimate — price and wait, before Start is pressed
# ─────────────────────────────────────────────────────────────
# The wizard needs a dollar total and an ETA on the screen where the user
# commits. Both are estimates and both say so; see app/billing.py's honesty
# boundary and app/estimate.py on why the ETA multiplies by language count.
#
# There is no quota in this product — no account, no plan, no allowance — so
# this reports minutes USED, never "minutes left". Rendering a remaining
# balance would be inventing a number.

# URL -> (fetched_at, curated meta). A wizard re-renders on every language
# click and yt-dlp takes seconds per probe, so the same URL is probed once.
_ESTIMATE_META_TTL = 600.0
_ESTIMATE_META_MAX = 64
_estimate_meta_cache: dict = {}


async def _probe_meta_cached(url: str) -> dict:
    """curate_metadata(probe_metadata(url)) with a short TTL cache.

    Probe failures cache nothing and return {} — the download stage is where
    a bad URL should surface, not here.
    """
    url = (url or "").strip()
    if not url.startswith("http"):
        return {}
    now = time.time()
    hit = _estimate_meta_cache.get(url)
    if hit and (now - hit[0]) < _ESTIMATE_META_TTL:
        return hit[1]
    try:
        info = await asyncio.to_thread(probe_metadata, url)
    except Exception as e:
        log.warning(f"[estimate] probe failed (non-fatal): {e}")
        return {}
    if not info:
        return {}
    meta = curate_metadata(info) or {}
    if len(_estimate_meta_cache) >= _ESTIMATE_META_MAX:
        oldest = min(_estimate_meta_cache, key=lambda k: _estimate_meta_cache[k][0])
        _estimate_meta_cache.pop(oldest, None)
    _estimate_meta_cache[url] = (now, meta)
    return meta


def _spawn_background(coro):
    """Start detached work that outlives the request that asked for it.

    A named seam rather than a bare asyncio.create_task, so a test can run
    the coroutine deterministically instead of racing the event loop. That
    race is not theoretical: under `TestClient` used without its context
    manager the portal is torn down when the request ends, and measuring it
    showed a **one millisecond** delay inside the task is enough for it to
    be cancelled instead of finishing. Tests that spawn real tasks there
    pass only by winning a coin flip.
    """
    return asyncio.create_task(coro)


def _discard_download(path: Path) -> None:
    """Delete a downloaded source nothing is going to use, best effort.

    Only touches the per-download directory this module creates under
    uploads/ (``qt_<hex>/``), never an uploaded file a user supplied.
    """
    try:
        parent = path.parent
        path.unlink(missing_ok=True)
        if (parent != UPLOAD_DIR and parent.parent == UPLOAD_DIR
                and parent.name.startswith("qt_") and parent.is_dir()
                and not any(parent.iterdir())):
            parent.rmdir()
    except OSError as e:
        log.warning(f"[multidub] could not discard {path}: {e}")


def _human_duration(seconds: float) -> str:
    """"12 minutes", "1 hour 5 minutes" — for strings a creator reads."""
    total = int(max(0, round(float(seconds or 0))))
    if total < 60:
        return f"{total} second{'s' if total != 1 else ''}"
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if not hours:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    out = f"{hours} hour{'s' if hours != 1 else ''}"
    if minutes:
        out += f" {minutes} minute{'s' if minutes != 1 else ''}"
    return out


def _finite_seconds(value: object) -> float:
    """A duration in seconds, or 0.0 for anything that is not a real one.

    NaN and ±Infinity have to die here rather than downstream: they survive
    arithmetic and only blow up at `round()` or at `json.dumps(allow_nan=False)`,
    which turns a bad query parameter into a 500. Negatives floor to zero — a
    video cannot be minus four seconds long, and pricing one would be worse.
    """
    try:
        out = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out) or out <= 0:
        return 0.0
    return out


@app.get("/api/estimate")
async def get_estimate(source: str = "", duration_sec: float = 0.0,
                       langs: str = "", window_days: int = 30):
    """Price and wall-clock estimate for a job that has not started yet.

    `source` is a URL — its duration/title/thumbnail come from a yt-dlp probe.
    For an upload, pass `duration_sec` read client-side off a <video> element;
    nothing has to be uploaded to get a price.

    `langs` may be empty: step 1 of the wizard wants the title and thumbnail
    before any language is picked, and a zero-language quote is $0.00.
    """
    codes = [c.strip().lower() for c in langs.split(",") if c.strip()]
    if len(codes) > MAX_TARGET_LANGS:
        return JSONResponse(
            {"error": f"At most {MAX_TARGET_LANGS} languages (got {len(codes)})"}, 400)
    unknown = [c for c in codes if c not in _QUICK_TEST_KNOWN_LANGS]
    if unknown:
        return JSONResponse({"error": f"Unknown language code(s): {unknown}"}, 400)
    if len(set(codes)) != len(codes):
        return JSONResponse({"error": "Duplicate language codes"}, 400)

    meta = await _probe_meta_cached(source)
    # `duration_sec` is read client-side off a <video> element, so it is not
    # server-controlled: a browser that reports Infinity for a stream (which
    # they do, before metadata loads) used to reach round() and json.dumps'
    # allow_nan=False and answer 500. Anything not finite is no duration.
    duration = _finite_seconds(meta.get("duration")) or _finite_seconds(duration_sec)

    # Same gate the single-dub route applies, moved forward so the wizard can
    # refuse an over-long video before Start rather than after the download.
    gate_error = None
    if cfg.max_source_duration_sec > 0 and duration > cfg.max_source_duration_sec:
        # No config identifiers here: this string is rendered verbatim on a
        # consumer screen, and the UI should not have to launder it.
        gate_error = (
            f"This video is {_human_duration(duration)} long, and this "
            f"server's limit is {_human_duration(cfg.max_source_duration_sec)}."
            f" Pick a shorter video, or raise the limit in Settings."
        )

    new_minutes = app_estimate.billable_minutes(duration, len(codes))
    since = time.time() - max(1, int(window_days or 30)) * 86400
    used = app_billing.summarize(list(jobs.values()), since=since)["minutes"]
    priced = app_billing.marginal_cost(used, new_minutes)

    measured = app_estimate.realtime_factor(jobs.values())
    factor = measured if measured is not None else float(cfg.eta_realtime_factor)
    eta = app_estimate.eta_seconds(duration, len(codes), factor)

    return {
        "duration_sec": round(duration, 2),
        "langs": codes,
        "billable_minutes": round(new_minutes, 2),
        "used_minutes": priced["used_minutes"],
        "cost": priced["cost"],
        "rate": priced["rate"],
        "bands": priced["bands"],
        "eta_sec": int(round(eta)),
        # The queue is single-consumer, so N languages run one after another.
        # Say so rather than quoting single-language time for a batch.
        "eta_per_lang_sec": int(round(eta / len(codes))) if codes else 0,
        "eta_basis": "measured" if measured is not None else "default",
        "eta_realtime_factor": round(factor, 2),
        "title": meta.get("title") or "",
        "thumbnail": meta.get("thumbnail") or "",
        "channel": meta.get("channel") or "",
        "source_type": "url" if (source or "").strip().startswith("http") else "upload",
        "duration_gate_error": gate_error,
        "estimate": True,
        "disclaimer": (
            "Estimate only. Minutes are measured from the real source length; "
            "the cost applies the design's published hosted rates. This "
            "server bills nobody."
        ),
    }


@app.get("/api/audit")
async def get_audit(limit: int = 200):
    return {"entries": app_audit.recent(limit=max(1, min(int(limit or 200), 1000))),
            "total": app_audit.count()}


# ─────────────────────────────────────────────────────────────
# API: Vendor admin console
# ─────────────────────────────────────────────────────────────
# Backs static/admin.html — the design's `3a` screens. All aggregation lives
# in app/admin.py, which is pure; these routes only gather the inputs it
# cannot reach on its own (disk sizes, GPU telemetry, per-stage metrics
# files) and hand them over.
#
# Same honesty boundary as /api/billing/usage, and for the same reason: the
# minutes are real, every dollar figure is an estimate at the design's
# published rates, and this server bills nobody. app/admin.py's docstring
# lists what the console shows, what it estimates, and what it refuses to
# invent.
#
# No authentication, because nothing on this server has any — see the
# loopback-bind rationale at the bottom of this file. In a hosted deployment
# this surface is the one that must go behind staff SSO first: it reads the
# revenue estimate, every API key record and the audit trail.

# Per-stage durations are read from each job's metrics.json, one small file
# per job. The fleet screen polls, so the walk is cached: without this a 5s
# poll over a few hundred finished jobs is a few hundred file reads a second
# for numbers that move once a job finishes.
_ADMIN_STAGE_TTL = 30.0
_admin_stage_cache: dict = {}


def _admin_stage_samples(now: float) -> tuple:
    """(recent, baseline) maps of stage id -> [seconds per source-second, …].

    "Recent" is the last 24 hours and "baseline" is the 7 days before that,
    which is the comparison the design's pool table draws ("stage p95s vs 7d
    baseline"). Jobs are visited newest-first and capped, so an install with
    thousands of finished jobs still answers in bounded time.

    **Samples are normalised by source length, not raw stage seconds.** Every
    stage's wall clock scales with how long the video was, so comparing raw
    p95s across two windows mostly compares what got dubbed rather than how
    fast it ran: measured on a real install, a day of full videos against a
    week of short test clips reported *every* stage as 3-34x degraded, which
    is a wall of false alarms and worse than no signal. Dividing by the source
    duration asks the question the panel is actually for — is this stage
    slower per unit of work than it was — and it is the same normalisation
    `app/estimate.py` applies to whole jobs.

    A job with no measured duration contributes nothing rather than a
    division by zero or a guess.
    """
    hit = _admin_stage_cache.get("samples")
    if hit and (now - hit[0]) < _ADMIN_STAGE_TTL:
        return hit[1]

    recent: dict = {}
    baseline: dict = {}
    cutoff_recent = now - 86400.0
    cutoff_base = now - 8 * 86400.0

    candidates = sorted(
        (j for j in jobs.values() if (j.get("created") or 0) >= cutoff_base),
        key=lambda j: -(j.get("created") or 0),
    )[:400]

    for job in candidates:
        source_sec = _finite_seconds(job.get("duration"))
        if source_sec <= 0:
            continue
        work = OUTPUT_DIR / (job.get("id") or "")
        if not work.is_dir():
            continue
        try:
            stages = (load_metrics(work).get("stages") or {})
        except Exception:
            continue
        bucket = recent if (job.get("created") or 0) >= cutoff_recent else baseline
        for stage_id, rec in stages.items():
            # Only successful runs. A stage that raised after four seconds is
            # not evidence that the stage got faster.
            if (rec.get("status") or "") not in ("", "ok", "success", "complete"):
                continue
            secs = rec.get("duration_sec")
            if isinstance(secs, (int, float)) and secs > 0:
                bucket.setdefault(stage_id, []).append(float(secs) / source_sec)

    _admin_stage_cache["samples"] = (now, (recent, baseline))
    return recent, baseline


async def _admin_storage_gb() -> float:
    """Total output size in GB, or 0.0 if the walk fails.

    Storage is never the reason an admin page fails to render, so this
    swallows errors the way /api/billing/usage does.
    """
    try:
        return float((await storage_stats()).get("total_gb") or 0.0)
    except Exception as e:
        log.warning(f"[admin] storage walk failed (non-fatal): {e}")
        return 0.0


@app.get("/api/admin/overview")
async def admin_overview(window_days: int = 30):
    """Screen `3a Business overview` — tiles, revenue series, attention list."""
    now = time.time()
    recent, baseline = _admin_stage_samples(now)
    health = app_admin.stage_health(recent, baseline)
    payload = app_admin.overview(
        list(jobs.values()), app_apikeys.list_keys(),
        days=max(1, int(window_days or 30)), now=now,
        storage_gb=await _admin_storage_gb(),
        stage_alerts=app_admin.stage_alerts(health),
    )
    payload["mode"] = cfg.mode
    return payload


@app.get("/api/admin/accounts")
async def admin_accounts(window_days: int = 30):
    """Screen `3a Customer accounts`, mapped onto API keys — see app/admin.py."""
    return app_admin.accounts(
        list(jobs.values()), app_apikeys.list_keys(),
        mode=cfg.mode, days=max(1, int(window_days or 30)),
        enforced=cfg.mode == "hosted",
    )


@app.get("/api/admin/account/{key_id}")
async def admin_account(key_id: str):
    """One key, with the audit entries that mention it.

    The design's detail pane promises "every action here lands in the
    audit log". It does — app/audit.py records key creation and revocation —
    so the pane reads that trail back rather than claiming it exists.
    """
    rec = next((k for k in app_apikeys.list_keys() if k.get("id") == key_id), None)
    if rec is None:
        return JSONResponse({"error": "No such key"}, 404)
    entries = [e for e in app_audit.recent(limit=1000)
               if e.get("target") == key_id or key_id in str(e.get("detail") or "")]
    return {
        "key": rec,
        "state": app_admin.key_state(rec),
        "scopes": app_apikeys.SCOPES,
        "audit": entries[:50],
        "enforced": cfg.mode == "hosted",
        "note": app_admin.accounts([], [])["note"],
    }


@app.get("/api/admin/revenue")
async def admin_revenue(window_days: int = 30):
    """Screen `3a Revenue ops` — rate card, unbilled work, per-month roll-up."""
    payload = app_admin.revenue(
        list(jobs.values()), days=max(1, int(window_days or 30)),
        storage_gb=await _admin_storage_gb(),
    )
    payload["mode"] = cfg.mode
    return payload


@app.get("/api/admin/revenue.csv")
async def admin_revenue_csv(window_days: int = 30):
    """The design's `Export CSV ↓`, one row per job.

    Built in memory: the row count is the job count, which is bounded by
    what one machine has dubbed, and streaming would buy nothing.
    """
    days = max(1, int(window_days or 30))
    now = time.time()
    rows = app_admin.revenue_csv_rows(
        list(jobs.values()), since=now - days * 86400, until=now)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%d")
    return Response(
        buf.getvalue(), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition":
                 f'attachment; filename="gochidubb-usage-{days}d-{stamp}.csv"'},
    )


@app.get("/api/admin/fleet")
async def admin_fleet():
    """Screen `3a Fleet and abuse queue` — capacity, stage p95s, review queue."""
    now = time.time()
    recent, baseline = _admin_stage_samples(now)
    gpu = gpu_snapshot() or {}
    try:
        gpu_name = (get_system_status().get("gpu") or {}).get("name") or ""
    except Exception:
        gpu_name = ""
    return app_admin.fleet(
        list(jobs.values()), app_apikeys.list_keys(), now=now,
        queue_depth=_job_queue.qsize() if _job_queue is not None else 0,
        gpu={k: round(v, 1) for k, v in gpu.items()} if gpu else {},
        gpu_backend=gpu_backend(), gpu_name=gpu_name,
        health=app_admin.stage_health(recent, baseline),
        intake_paused=_intake_is_paused(),
    )


@app.post("/api/admin/intake")
async def admin_set_intake(paused: bool = Form(...)):
    """The design's `Pause new jobs ⏸` — and it really does pause them.

    Pausing stops the worker taking anything new off the queue. A job already
    running is left alone: killing work mid-pipeline to honour a pause would
    throw away GPU time that has already been spent, and cancel is the
    endpoint for that.

    The pause is in-process and deliberately does not persist — see
    `_intake_gate` — so a restart always comes back admitting work.
    """
    global _intake_paused_since
    if _intake_gate is None:
        return JSONResponse({"error": "Queue is not running"}, 503)
    want = bool(paused)
    if want:
        if _intake_gate.is_set():
            _intake_paused_since = time.time()
        _intake_gate.clear()
    else:
        _intake_gate.set()
        _intake_paused_since = 0.0
    # Wake a worker that is parked in `_job_queue.get()` so the new state
    # takes effect now rather than after the next job slips through.
    if _intake_changed is not None:
        _intake_changed.set()
    app_audit.record("admin.intake", target="queue",
                     detail="paused" if want else "resumed")
    activity.record_system(
        f"Job intake {'paused' if want else 'resumed'} from the admin console")
    return {"ok": True, "paused": want, "since": _intake_paused_since,
            "queue_depth": _job_queue.qsize() if _job_queue is not None else 0}


@app.post("/api/models/pull")
async def pull_model(model: str = Form(...)):
    async def stream():
        try:
            async for event in ollama_pull_stream(model):
                total = event.get("total", 0)
                completed = event.get("completed", 0)
                st = event.get("status", "")
                pct = int(completed / total * 100) if total else 0
                payload = {"status": st, "completed": completed, "total": total, "percent": pct}
                yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'status': 'success', 'percent': 100})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/models/delete")
async def delete_model(model: str = Form(...)):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.delete(
                f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/delete",
                json={"name": model},
            )
            if r.status_code == 200:
                return {"ok": True}
            return JSONResponse({"error": r.text}, 400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/api/voxcpm/warmup")
async def voxcpm_warmup():
    try:
        tts = get_tts_engine()
        if isinstance(tts, VoxCPMSynthesizer):
            if not tts.is_loaded:
                await asyncio.get_event_loop().run_in_executor(None, tts.load)
            return {"ok": True, "loaded": True}
        return {"ok": True, "loaded": False, "fallback": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ─────────────────────────────────────────────────────────────
# API: Dubbing
# ─────────────────────────────────────────────────────────────

# Preference order for an automatic model substitution: fast non-thinking
# translation-specialized models first, then general purpose, then thinking
# models last. gemma4:e4b/e2b work but hang on 12 GB GPUs due to thinking
# mode — kept as last-resort only.
_PREFERRED_TRANSLATION_MODELS = [
    "aya-expanse:8b",      # Cohere multilingual, best EN↔RU
    "mistral-nemo:12b",    # Mistral, strong for European langs
    "qwen2.5:14b",         # Qwen non-thinking, very good
    "qwen3:8b",            # Qwen3, thinking optional
    "qwen2.5:7b",          # Qwen smaller, fast
    "gemma3:12b",          # Gemma3 (no thinking) — good quality
    "gemma3:4b",           # Gemma3 small
    "llama3.2:3b",         # Tiny fallback
    "qwen3:14b",           # Larger qwen3 (thinking optional)
    "gemma4:e4b",          # Thinking — heavy on 12 GB GPU
    "gemma4:e2b",          # Thinking — smaller but same issue
]


def _resolve_translation_model(model: str, installed: list) -> tuple:
    """Pick a model that actually exists. Returns (model, error message).

    The preference list is Ollama-shaped ("aya-expanse:8b"), but
    `check_ollama()` delegates to `check_lm_studio()` when USE_LM_STUDIO=1 —
    the documented default — and LM Studio ids are hyphenated
    ("aya-expanse-8b"). Nothing in the list can ever match, so without the
    last-resort below every batch route refused to start on a machine with a
    perfectly good model loaded, and told the user to run an `ollama` command
    that was not the runtime they were using.

    So: honour the preference order when it matches, otherwise take whatever
    is loaded. Refusing is reserved for there being nothing at all.
    """
    fallback = next((m for m in _PREFERRED_TRANSLATION_MODELS if m in installed), None)
    if fallback is None and installed:
        # Any real model beats refusing to run. Creator mode has no model
        # picker at all — quick_test is its only submit path — so a refusal
        # here strands a creator at the last step of the wizard with no
        # control that could fix it.
        fallback = installed[0]
    if fallback:
        log.warning(f"Requested model '{model}' not installed; "
                    f"using '{fallback}' instead")
        return fallback, None
    if USE_LM_STUDIO:
        return model, ("No translation model is loaded in LM Studio. Load one "
                       "there, or set USE_LM_STUDIO=0 to use Ollama instead.")
    return model, ("No translation model installed. Pull one via "
                   "'ollama pull aya-expanse:8b' or use the Models panel.")


@app.post("/api/dub")
async def start_dub(
    source: str = Form(""),
    video: Optional[UploadFile] = File(None),
    reference: Optional[UploadFile] = File(None),
    source_lang: str = Form("auto"),
    target_lang: str = Form("ru"),
    model: str = Form("gemma4:e4b"),
    keep_bg: bool = Form(True),
    whisper_model: str = Form("large-v3"),
    speaker_mode: str = Form("main"),   # "main" | "all"
    context_hint: str = Form(""),
    voice_style: str = Form(""),
    voice_preset: str = Form("auto"),
    tts_speed: str = Form("balanced"),
    wizard_mode: str = Form("auto"),  # "auto" | "review_translation"
                                      # | "review_transcript" | "review_voices"
    auto_denoise: bool = Form(False),
    lip_sync: bool = Form(False),  # if True, auto-run Wav2Lip after pipeline completes
    mode: str = Form("dub"),  # "dub" | "reupload" (3E: reupload = no dubbing)
    scheduled_at: float = Form(0.0),  # unix epoch seconds; 0 = start immediately
    voxcpm_cfg: float = Form(0.0),    # 0 = use the global setting
    voxcpm_steps: int = Form(0),      # 0 = use the global setting
):
    mode = normalize_job_mode(mode)
    if mode is None:
        return JSONResponse(
            {"error": f"Unknown mode. Options: {list(JOB_MODES)}"}, 400)

    voxcpm_cfg, voxcpm_steps, _vox_err = validate_voxcpm_overrides(
        voxcpm_cfg, voxcpm_steps)
    if _vox_err:
        return JSONResponse({"error": _vox_err}, 400)

    # Validate translation model exists in Ollama - fall back gracefully
    # otherwise. Reupload jobs never translate, so they skip the check —
    # a machine with no LLM installed can still run reuploads.
    _ok, _installed = (await check_ollama()) if mode == "dub" else (False, [])
    if _ok and model not in _installed:
        model, _model_err = _resolve_translation_model(model, _installed)
        if _model_err:
            return JSONResponse({"error": _model_err}, 400)

    job_id = uuid.uuid4().hex[:8]
    work = OUTPUT_DIR / job_id
    work.mkdir(exist_ok=True)

    if video and video.filename:
        ext = Path(video.filename).suffix or ".mp4"
        vid_path = str(UPLOAD_DIR / f"{job_id}{ext}")
        with open(vid_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        actual_source = vid_path
        source_type = "upload"
        source_label = video.filename
        # Log the uploaded file's real name/extension/size. The pipeline
        # copies whatever the user uploads into source_video.mp4 regardless
        # of actual format, so a .webp image (no audio/video stream) is a
        # common "breaks silently" case — this line makes it obvious in the
        # server log before the job even starts.
        try:
            _up_bytes = os.path.getsize(vid_path)
        except OSError:
            _up_bytes = -1
        log.info(
            f"[dub] upload job={job_id} file={video.filename!r} ext={ext} "
            f"size={_up_bytes/1048576 if _up_bytes >= 0 else '?'}MB -> {vid_path}"
        )
    elif source:
        actual_source = source
        source_type = "url" if source.startswith("http") else "path"
        source_label = source
    else:
        return JSONResponse({"error": "Provide a YouTube URL or upload a video"}, 400)

    # ── Probe URL metadata (title/duration) before creating the job ────
    # Probe failures never block submission; the download stage will
    # surface real errors. `meta` is a small curated dict persisted on the
    # job; `source_info` is the full yt-dlp blob (transient, never hits DB).
    meta = None
    source_info = None
    if source_type == "url":
        source_info = await asyncio.to_thread(probe_metadata, actual_source)
        if source_info:
            meta = curate_metadata(source_info)
            dur = meta.get("duration") or 0
            if cfg.max_source_duration_sec > 0 and dur > cfg.max_source_duration_sec:
                return JSONResponse({
                    "error": (
                        f"Video is {int(dur)}s long — over the configured limit of "
                        f"{cfg.max_source_duration_sec}s (max_source_duration_sec). "
                        f"Pick a shorter video or raise the limit in Settings."
                    )
                }, 400)
            if meta.get("title"):
                source_label = meta["title"]

    # ── Duplicate pre-check (4C, warn-only) ────────────────────────────
    # If a VK token is configured, ask VK whether a similar video already
    # exists BEFORE spending GPU-hours on the dub. Strictly best-effort:
    # short timeout, any failure skips silently, never blocks submission.
    duplicate_check = None
    if meta and meta.get("title"):
        from app.secrets import get_secret
        if get_secret("vk_access_token"):
            try:
                from pipeline.publisher import VKUploader, classify_matches
                _dup_matches = await asyncio.wait_for(
                    VKUploader().search_similar(
                        meta["title"], meta.get("duration")),
                    timeout=6.0,
                )
                duplicate_check = {
                    "duplicate_warnings": _dup_matches,
                    "duplicate_verdict": classify_matches(_dup_matches),
                }
            except Exception as e:
                log.debug(f"[dub] duplicate pre-check skipped: {e}")

    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    # Deferred start (mirrors /api/dub/batch): when scheduled_at is in the
    # future, park the job as status='scheduled' with the full pipeline args
    # stashed in _pending_args — the _scheduler_loop enqueues it when the
    # time arrives (and survives restarts, since both persist to disk).
    now = time.time()
    is_scheduled = scheduled_at > now + 10  # 10s grace for clock skew
    pipeline_args = {
        "source": actual_source,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": model,
        "keep_bg": keep_bg,
        "whisper_model": whisper_model,
        "reference_audio": ref_path,
        "speaker_mode": speaker_mode,
        "context_hint": context_hint,
        "voice_style": voice_style,
        "voice_preset": voice_preset,
        "tts_speed": tts_speed,
        "wizard_mode": wizard_mode,
        "auto_denoise": auto_denoise,
        "mode": mode,
        "voxcpm_cfg": voxcpm_cfg,
        "voxcpm_steps": voxcpm_steps,
    }

    jobs[job_id] = {
        "id": job_id,
        "status": "scheduled" if is_scheduled else "queued",
        "progress": 0,
        "source": actual_source,
        "source_type": source_type,
        "source_label": source_label,
        "target_lang": target_lang,
        "model": model,
        "speaker_mode": speaker_mode,
        "context_hint": context_hint,
        "voice_style": voice_style,
        "voice_preset": voice_preset,
        "voice_mode": ("upload" if ref_path else
                       ("custom" if voice_style.strip() else "preset")),
        "tts_speed": tts_speed,
        "voxcpm_cfg": voxcpm_cfg,
        "voxcpm_steps": voxcpm_steps,
        "wizard_mode": wizard_mode,
        "lip_sync": bool(lip_sync),
        "mode": mode,
        "created": time.time(),
        "step_detail": ("Scheduled..." if is_scheduled else "Queued..."),
        "scheduled_at": scheduled_at if is_scheduled else 0,
        "_pending_args": (dict(pipeline_args) if is_scheduled else None),
        **({"meta": meta, "title": meta.get("title"),
            "_source_info": source_info} if meta else {}),
        **({"duplicate_check": duplicate_check} if duplicate_check else {}),
    }
    save_job(jobs[job_id])

    if is_scheduled:
        log.info(f"[schedule] Job {job_id} deferred until "
                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(scheduled_at))}")
        return {"job_id": job_id, "scheduled_at": scheduled_at,
                **(duplicate_check or {})}

    # Dispatch to pipeline — via queue if GPU is busy, else directly.
    # Multiple simultaneous dub requests would OOM the 12GB 3080 Ti
    # (WhisperX large-v3 + VoxCPM + pyannote = 9-10GB each). The queue
    # ensures only ONE GPU-heavy job runs at a time; others wait.
    await enqueue_job(job_id, pipeline_args)

    return {"job_id": job_id, **(duplicate_check or {})}


@app.post("/api/dub/batch")
async def start_batch_dub(
    sources: str = Form(""),  # newline-separated URLs OR json list
    videos: Optional[list[UploadFile]] = File(None),
    reference: Optional[UploadFile] = File(None),
    source_lang: str = Form("auto"),
    target_lang: str = Form("ru"),
    model: str = Form("aya-expanse:8b"),
    keep_bg: bool = Form(True),
    whisper_model: str = Form("large-v3"),
    speaker_mode: str = Form("main"),
    context_hint: str = Form(""),
    voice_style: str = Form(""),
    voice_preset: str = Form("auto"),
    tts_speed: str = Form("balanced"),
    wizard_mode: str = Form("auto"),  # Usually "auto" for batch — no pauses
    auto_denoise: bool = Form(False),
    batch_label: str = Form(""),  # optional: "BJJ Course Week 1" for summary
    scheduled_at: float = Form(0.0),  # unix epoch seconds; 0 = start immediately
    voxcpm_cfg: float = Form(0.0),    # 0 = use the global setting
    voxcpm_steps: int = Form(0),      # 0 = use the global setting
):
    """Enqueue multiple videos for night-mode processing.

    Intended flow: user drops 5-10 videos in UI, picks a preset, clicks
    "Queue all". Each video becomes a separate job sharing common
    settings (target lang, voice, context). Jobs run serially via the
    GPU queue. Sleep prevention auto-activates while queue is non-empty.

    When scheduled_at is set to a future timestamp, jobs are created
    with status="scheduled" and a background task enqueues them at the
    target time. Useful for "queue this now, start at 2 AM when
    electricity is cheap" workflows.

    Returns: {job_ids: [...], batch_id: str} so UI can track summary.
    """
    voxcpm_cfg, voxcpm_steps, _vox_err = validate_voxcpm_overrides(
        voxcpm_cfg, voxcpm_steps)
    if _vox_err:
        return JSONResponse({"error": _vox_err}, 400)

    # Validate Ollama model once (not per-job)
    _ok, _installed = await check_ollama()
    if _ok and model not in _installed:
        model, _model_err = _resolve_translation_model(model, _installed)
        if _model_err:
            return JSONResponse({"error": _model_err}, 400)

    # Save shared reference once — all batch jobs reuse it
    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"batch_{uuid.uuid4().hex[:8]}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        log.info(f"[batch] Saved shared reference: {ref_path}")

    # Collect sources: URLs from form + uploaded files
    batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    job_ids = []

    # 1. URLs (newline-separated or JSON list)
    url_list = []
    if sources.strip():
        s = sources.strip()
        if s.startswith("["):
            try:
                url_list = json.loads(s)
            except Exception:
                url_list = [ln.strip() for ln in s.splitlines() if ln.strip()]
        else:
            url_list = [ln.strip() for ln in s.splitlines() if ln.strip()]

    # When a scheduled_at is in the future, jobs are parked in the jobs
    # dict with status='scheduled' and a background task wakes them up
    # at the target time. Otherwise we enqueue immediately as before.
    now = time.time()
    is_scheduled = scheduled_at > now + 10  # 10s grace for clock skew
    initial_status = "scheduled" if is_scheduled else "queued"

    async def _enqueue_or_defer(jid, pipeline_args):
        if not is_scheduled:
            await enqueue_job(jid, pipeline_args)
        else:
            # Just leave status=scheduled; the scheduler task will pick it up
            log.info(f"[schedule] Job {jid} deferred until {scheduled_at}")

    skipped = []  # URLs rejected by the duration gate: [{source, error}]
    for url in url_list:
        if not url:
            continue
        # Probe metadata per URL — failures never block submission.
        meta = None
        source_info = await asyncio.to_thread(probe_metadata, url)
        if source_info:
            meta = curate_metadata(source_info)
            dur = meta.get("duration") or 0
            if cfg.max_source_duration_sec > 0 and dur > cfg.max_source_duration_sec:
                skipped.append({
                    "source": url,
                    "error": (f"Video is {int(dur)}s long — over the configured "
                              f"limit of {cfg.max_source_duration_sec}s "
                              f"(max_source_duration_sec)"),
                })
                log.warning(f"[batch] Skipping {url}: duration {dur}s > "
                            f"{cfg.max_source_duration_sec}s limit")
                continue
        url_label = ((meta or {}).get("title")
                     or url[:60] + ("..." if len(url) > 60 else ""))
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = {
            "id": jid,
            "status": initial_status,
            "progress": 0,
            "source": url,
            "source_type": "url",
            "source_label": url_label,
            **({"meta": meta, "title": meta.get("title"),
                "_source_info": source_info} if meta else {}),
            "target_lang": target_lang,
            "model": model,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "voice_mode": ("upload" if ref_path else
                          ("custom" if voice_style.strip() else "preset")),
            "tts_speed": tts_speed,
            "voxcpm_cfg": voxcpm_cfg,
            "voxcpm_steps": voxcpm_steps,
            "whisper_model": whisper_model,
            "keep_bg": keep_bg,
            "wizard_mode": wizard_mode,
            "auto_denoise": auto_denoise,
            "batch_id": batch_id,
            "batch_label": batch_label,
            "created": time.time(),
            "scheduled_at": scheduled_at if is_scheduled else 0,
            # When scheduled, stash full pipeline args on the job so the
            # scheduler can re-hydrate and enqueue later
            "_pending_args": ({
                "source": url, "source_lang": source_lang,
                "target_lang": target_lang, "model": model,
                "keep_bg": keep_bg, "whisper_model": whisper_model,
                "reference_audio": ref_path, "speaker_mode": speaker_mode,
                "context_hint": context_hint, "voice_style": voice_style,
                "voice_preset": voice_preset, "tts_speed": tts_speed,
                "wizard_mode": wizard_mode, "auto_denoise": auto_denoise,
                "voxcpm_cfg": voxcpm_cfg, "voxcpm_steps": voxcpm_steps,
            } if is_scheduled else None),
        }
        save_job(jobs[jid])
        await _enqueue_or_defer(jid, {
            "source": url, "source_lang": source_lang,
            "target_lang": target_lang, "model": model,
            "keep_bg": keep_bg, "whisper_model": whisper_model,
            "reference_audio": ref_path, "speaker_mode": speaker_mode,
            "context_hint": context_hint, "voice_style": voice_style,
            "voice_preset": voice_preset, "tts_speed": tts_speed,
            "wizard_mode": wizard_mode, "auto_denoise": auto_denoise,
            "voxcpm_cfg": voxcpm_cfg, "voxcpm_steps": voxcpm_steps,
        })
        job_ids.append(jid)

    # 2. Uploaded files
    for video in (videos or []):
        if not video.filename:
            continue
        jid = uuid.uuid4().hex[:8]
        video_ext = Path(video.filename).suffix or ".mp4"
        dest = UPLOAD_DIR / f"{jid}{video_ext}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(video.file, f)
        jobs[jid] = {
            "id": jid,
            "status": initial_status,
            "progress": 0,
            "source": str(dest),
            "source_type": "file",
            "source_label": video.filename,
            "target_lang": target_lang,
            "model": model,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "voice_mode": ("upload" if ref_path else
                          ("custom" if voice_style.strip() else "preset")),
            "tts_speed": tts_speed,
            "voxcpm_cfg": voxcpm_cfg,
            "voxcpm_steps": voxcpm_steps,
            "whisper_model": whisper_model,
            "keep_bg": keep_bg,
            "wizard_mode": wizard_mode,
            "auto_denoise": auto_denoise,
            "batch_id": batch_id,
            "batch_label": batch_label,
            "created": time.time(),
            "scheduled_at": scheduled_at if is_scheduled else 0,
            "_pending_args": ({
                "source": str(dest), "source_lang": source_lang,
                "target_lang": target_lang, "model": model,
                "keep_bg": keep_bg, "whisper_model": whisper_model,
                "reference_audio": ref_path, "speaker_mode": speaker_mode,
                "context_hint": context_hint, "voice_style": voice_style,
                "voice_preset": voice_preset, "tts_speed": tts_speed,
                "wizard_mode": wizard_mode, "auto_denoise": auto_denoise,
                "voxcpm_cfg": voxcpm_cfg, "voxcpm_steps": voxcpm_steps,
            } if is_scheduled else None),
        }
        save_job(jobs[jid])
        await _enqueue_or_defer(jid, {
            "source": str(dest), "source_lang": source_lang,
            "target_lang": target_lang, "model": model,
            "keep_bg": keep_bg, "whisper_model": whisper_model,
            "reference_audio": ref_path, "speaker_mode": speaker_mode,
            "context_hint": context_hint, "voice_style": voice_style,
            "voice_preset": voice_preset, "tts_speed": tts_speed,
            "wizard_mode": wizard_mode, "auto_denoise": auto_denoise,
            "voxcpm_cfg": voxcpm_cfg, "voxcpm_steps": voxcpm_steps,
        })
        job_ids.append(jid)

    if is_scheduled:
        log.info(f"[batch] {batch_id}: SCHEDULED {len(job_ids)} jobs for "
                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(scheduled_at))} "
                 f"(label: {batch_label or 'untitled'})")
    else:
        log.info(f"[batch] {batch_id}: enqueued {len(job_ids)} jobs "
                 f"(label: {batch_label or 'untitled'})")
    resp = {
        "batch_id": batch_id,
        "job_ids": job_ids,
        "count": len(job_ids),
        "label": batch_label,
    }
    if skipped:
        resp["skipped"] = skipped
    return resp


@app.get("/api/dub/batch/{batch_id}")
async def get_batch_summary(batch_id: str):
    """Summary of a batch run — how many complete, failed, still running.
    Used by UI to show 'night-mode' dashboard."""
    batch_jobs = [j for j in jobs.values() if j.get("batch_id") == batch_id]
    if not batch_jobs:
        return JSONResponse({"error": "Batch not found"}, 404)
    total = len(batch_jobs)
    complete = sum(1 for j in batch_jobs if j.get("status") == "complete")
    errored = sum(1 for j in batch_jobs if j.get("status") == "error")
    queued = sum(1 for j in batch_jobs if j.get("status") == "queued")
    running = total - complete - errored - queued
    started = min((j.get("created", 0) for j in batch_jobs), default=0)
    finished = max((j.get("completed_at", j.get("created", 0))
                    for j in batch_jobs if j.get("status") in ("complete", "error")),
                   default=0)
    elapsed = (finished - started) if finished > started else (time.time() - started)
    return {
        "batch_id": batch_id,
        "label": batch_jobs[0].get("batch_label", ""),
        "total": total,
        "complete": complete,
        "errored": errored,
        "queued": queued,
        "running": running,
        "started": started,
        "finished": finished if complete + errored == total else None,
        "elapsed_sec": int(elapsed),
        "jobs": [{
            "id": j["id"],
            "label": j.get("source_label", j["id"]),
            "status": j.get("status"),
            "progress": j.get("progress", 0),
            "error": j.get("error", ""),
            "dubbed_url": (f"/outputs/{j['id']}/dubbed_video.mp4"
                          if j.get("status") == "complete" else None),
        } for j in sorted(batch_jobs, key=lambda x: x.get("created", 0))],
    }


# ═══════════════════════════════════════════════════════════════════════
#  Waveform preview endpoint — visualize reference audio
# ═══════════════════════════════════════════════════════════════════════
# Lets the UI show a waveform of reference audio BEFORE the user commits
# to using it. Useful for spotting silences, clipping, or picking the
# cleanest 15-second window from a longer file. Uses ffmpeg's built-in
# `showwavespic` filter — fast, no Python audio libs required.
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/waveform")
async def generate_waveform(
    audio: UploadFile = File(...),
    width: int = Form(800),
    height: int = Form(120),
):
    """Returns a PNG of the audio waveform. Accepts any FFmpeg-readable
    audio file. Width/height in pixels; defaults suit a typical UI panel."""
    import subprocess
    tmp_id = uuid.uuid4().hex[:8]
    ext = Path(audio.filename or "in.wav").suffix or ".wav"
    src = UPLOAD_DIR / f"wf_{tmp_id}{ext}"
    dst = UPLOAD_DIR / f"wf_{tmp_id}.png"
    try:
        with open(src, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-filter_complex",
             f"aformat=channel_layouts=mono,"
             f"compand=0|0:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2,"
             f"showwavespic=s={width}x{height}:colors=0xfb923c:split_channels=0",
             "-frames:v", "1", str(dst)],
            check=True, capture_output=True, timeout=30,
        )
        with open(dst, "rb") as f:
            png_data = f.read()
        return Response(content=png_data, media_type="image/png")
    except subprocess.CalledProcessError as e:
        log.warning(f"[waveform] ffmpeg failed: {e.stderr[:200]}")
        return JSONResponse({"error": "Could not generate waveform"}, 500)
    finally:
        for p in (src, dst):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════
#  Per-speaker reference inspection — diagnostic tool for review screen
# ═══════════════════════════════════════════════════════════════════════
# After diarization the pipeline extracts ~30s of clean speech per
# detected speaker into speaker_refs/ref_SPEAKER_XX.wav. These are fed
# to VoxCPM as voice-cloning references — bad refs = bad dubbed voice.
#
# These endpoints let the review UI inspect/audition the extracted refs
# so the user can diagnose "why does SPEAKER_01 sound wrong" BEFORE
# running TTS. PNGs are cached under speaker_refs/wf_*.png to avoid
# re-rendering the same waveform on every UI open.
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/job/{job_id}/speakers")
async def list_job_speakers(job_id: str):
    """Return per-speaker reference metadata for a job: list of
    {speaker, ref_path, duration_sec, exists} so the UI can enumerate
    detected speakers and render a waveform panel per speaker."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    refs_dir = OUTPUT_DIR / job_id / "speaker_refs"
    if not refs_dir.exists():
        return {"speakers": [], "hint": "No speaker references extracted for this job"}

    out = []
    for p in sorted(refs_dir.glob("ref_*.wav")):
        name = p.stem  # "ref_SPEAKER_00" or "ref_fallback"
        speaker = name.replace("ref_", "", 1)
        duration = 0.0
        try:
            # Fast duration read via ffprobe
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
                capture_output=True, text=True, timeout=10,
            )
            duration = float(r.stdout.strip() or 0.0)
        except Exception:
            pass
        out.append({
            "speaker": speaker,
            "duration_sec": round(duration, 1),
            "audio_url": f"/api/job/{job_id}/speaker_ref/{speaker}/audio",
            "waveform_url": f"/api/job/{job_id}/speaker_ref/{speaker}/waveform",
        })
    return {"speakers": out}


@app.get("/api/job/{job_id}/speaker_ref/{speaker}/audio")
async def get_speaker_ref_audio(job_id: str, speaker: str):
    """Stream the speaker's reference WAV for in-browser playback.
    Lets the user quickly audition the extracted reference."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    # Strict filename validation — prevent path traversal via speaker id
    if not re.match(r"^[A-Za-z0-9_]+$", speaker):
        return JSONResponse({"error": "Invalid speaker id"}, 400)
    p = OUTPUT_DIR / job_id / "speaker_refs" / f"ref_{speaker}.wav"
    if not p.exists():
        return JSONResponse({"error": "Reference not found"}, 404)
    return FileResponse(str(p), media_type="audio/wav",
                        filename=f"ref_{speaker}.wav")


@app.get("/api/job/{job_id}/speaker_ref/{speaker}/waveform")
async def get_speaker_ref_waveform(
    job_id: str, speaker: str,
    width: int = 700, height: int = 80,
):
    """Return a cached PNG waveform for this speaker. Caches to
    speaker_refs/wf_{speaker}_{w}x{h}.png so repeated opens don't
    re-run ffmpeg. Cache invalidates only when the ref WAV's mtime
    changes (e.g. user re-uploaded via edit_speaker_ref)."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if not re.match(r"^[A-Za-z0-9_]+$", speaker):
        return JSONResponse({"error": "Invalid speaker id"}, 400)
    refs_dir = OUTPUT_DIR / job_id / "speaker_refs"
    src = refs_dir / f"ref_{speaker}.wav"
    if not src.exists():
        return JSONResponse({"error": "Reference not found"}, 404)

    cache = refs_dir / f"wf_{speaker}_{width}x{height}.png"
    # Cache hit only if cached file exists AND was modified after source.
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return FileResponse(str(cache), media_type="image/png")

    # Miss: render + cache
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-filter_complex",
             f"aformat=channel_layouts=mono,"
             f"compand=0|0:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2,"
             f"showwavespic=s={width}x{height}:colors=0xfb923c:split_channels=0",
             "-frames:v", "1", str(cache)],
            check=True, capture_output=True, timeout=30,
        )
    except subprocess.CalledProcessError as e:
        log.warning(f"[waveform] per-speaker render failed: {e.stderr[:200]}")
        return JSONResponse({"error": "Could not render waveform"}, 500)
    return FileResponse(str(cache), media_type="image/png")


# ═══════════════════════════════════════════════════════════════════════
#  Subtitle burn-in — overlay SRT onto dubbed video
# ═══════════════════════════════════════════════════════════════════════
# Optional post-processing: take the dubbed video + translated SRT and
# produce a version with hard-coded subtitles burned into the frame.
# Useful for YouTube where auto-generated CC is often wrong.
# Uses ffmpeg's `subtitles` filter (relies on libass).
# ═══════════════════════════════════════════════════════════════════════

# Subtitle styling presets — shared by burn-in and preview endpoints.
# ffmpeg 'subtitles' filter uses libass force_style syntax. BorderStyle=1
# is outline+shadow; 3 is opaque box.
SUB_STYLE_MAP = {
    "default": "Fontsize=22,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=28",
    "large":   "Fontsize=30,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=3,Shadow=1,MarginV=40,Bold=1",
    "minimal": "Fontsize=18,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=20",
    # Yellow "classic cinema" style — high-legibility for action footage
    "yellow":  "Fontsize=24,PrimaryColour=&H00FFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=2,MarginV=30,Bold=1",
    # Opaque box for noisy backgrounds (e.g. bright snow, chaotic action)
    "boxed":   "Fontsize=22,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BackColour=&HC0000000,BorderStyle=3,Outline=0,Shadow=0,MarginV=28",
}


@app.post("/api/dub/{job_id}/subs_preview")
async def preview_subtitle_style(
    job_id: str,
    style: str = Form("default"),
    timestamp: float = Form(-1.0),  # seconds into video; -1 = auto-pick
):
    """Render a single frame from the dubbed video with subs overlaid in
    the given style — lets the user preview styling instantly instead of
    waiting for a full re-encode of the whole video.

    Timestamp selection:
      - User can specify a timestamp (UI may tie this to a scrub bar)
      - If -1, we auto-pick the middle of a segment that has subtitle
        text so the preview actually shows text (not a silent frame)
    """
    import subprocess
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if style not in SUB_STYLE_MAP:
        return JSONResponse(
            {"error": f"Unknown style '{style}'. Options: {list(SUB_STYLE_MAP)}"}, 400)

    work = OUTPUT_DIR / job_id
    src_video = work / "dubbed_video.mp4"
    srt_file = work / "translated.srt"

    if not src_video.exists():
        return JSONResponse({"error": "Dubbed video not yet generated"}, 400)

    if not srt_file.exists():
        cp_path = work / "checkpoint_tts_done.json"
        if not cp_path.exists():
            cp_path = work / "checkpoint_translation_done.json"
        if cp_path.exists():
            try:
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                _write_srt_file(cp.get("segments", []), srt_file)
            except Exception as e:
                return JSONResponse({"error": f"Could not generate SRT: {e}"}, 500)
        else:
            return JSONResponse({"error": "No transcript data found"}, 400)

    # Auto-pick: find a segment with text that lasts at least 1s
    if timestamp < 0:
        try:
            cp_path = work / "checkpoint_tts_done.json"
            if not cp_path.exists():
                cp_path = work / "checkpoint_translation_done.json"
            if cp_path.exists():
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                for seg in cp.get("segments", []):
                    text = (seg.get("translated_text") or seg.get("text") or "").strip()
                    dur = seg.get("end", 0) - seg.get("start", 0)
                    if text and dur >= 1.0:
                        # Pick middle of segment so sub is guaranteed visible
                        timestamp = seg["start"] + dur / 2
                        break
        except Exception:
            pass
        if timestamp < 0:
            timestamp = 2.0  # fallback

    srt_arg = str(srt_file).replace("\\", "/").replace(":", r"\:")
    force_style = SUB_STYLE_MAP[style]

    # Render one frame at <timestamp> with subs overlaid. Use -ss BEFORE
    # -i for fast seek (less accurate but saves ~10x on long videos),
    # and -frames:v 1 to output just one PNG.
    out_png = work / f"subs_preview_{style}.png"
    try:
        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", f"{timestamp:.2f}",
             "-i", str(src_video),
             "-vf", f"subtitles='{srt_arg}':force_style='{force_style}'",
             "-frames:v", "1",
             "-q:v", "3",  # good quality JPEG-equivalent
             str(out_png)],
            check=True, capture_output=True, timeout=30,
        )
        # Return a JSON URL so the browser can cache-bust the image.
        # The PNG is already accessible via the /outputs static mount.
        return JSONResponse({
            "url": f"/outputs/{job_id}/subs_preview_{style}.png?t={int(time.time())}"
        })
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or b"").decode("utf-8", errors="replace")[-500:]
        log.warning(f"[subs_preview] ffmpeg failed: {err_msg}")
        return JSONResponse({"error": "Preview render failed",
                             "detail": err_msg[:300]}, 500)


@app.post("/api/dub/{job_id}/burn_subs")
async def burn_subtitles(
    job_id: str,
    style: str = Form("default"),  # "default" | "large" | "minimal" | "yellow" | "boxed"
):
    """Generate a version of the dubbed video with burned-in subtitles
    from the translated SRT. Produces dubbed_video_subs.mp4 in the job dir."""
    import subprocess
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    work = OUTPUT_DIR / job_id
    src_video = work / "dubbed_video.mp4"
    srt_file = work / "translated.srt"
    dst_video = work / "dubbed_video_subs.mp4"

    if not src_video.exists():
        return JSONResponse({"error": "Dubbed video not yet generated"}, 400)

    # Ensure SRT exists (it's written alongside translation checkpoint,
    # but regenerate if missing using current segments)
    if not srt_file.exists():
        cp_path = work / "checkpoint_tts_done.json"
        if not cp_path.exists():
            cp_path = work / "checkpoint_translation_done.json"
        if cp_path.exists():
            try:
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                segments = cp.get("segments", [])
                _write_srt_file(segments, srt_file)
            except Exception as e:
                return JSONResponse({"error": f"Could not generate SRT: {e}"}, 500)
        else:
            return JSONResponse({"error": "No transcript data found"}, 400)

    # Use the shared style map (preview + burn-in stay in sync)
    force_style = SUB_STYLE_MAP.get(style, SUB_STYLE_MAP["default"])

    # ffmpeg needs forward-slash path and escaped colons on Windows
    # (the subtitles filter parses its argument like a filter string)
    srt_arg = str(srt_file).replace("\\", "/").replace(":", r"\:")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_video),
             "-vf", f"subtitles='{srt_arg}':force_style='{force_style}'",
             "-c:a", "copy",  # don't re-encode audio
             "-preset", "fast",
             str(dst_video)],
            check=True, capture_output=True, timeout=600,
        )
        return {
            "ok": True,
            "url": f"/outputs/{job_id}/dubbed_video_subs.mp4?v={int(time.time())}",
        }
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or b"").decode("utf-8", errors="replace")[-500:]
        log.warning(f"[burn_subs] ffmpeg failed for {job_id}: {err_msg}")
        return JSONResponse({
            "error": "Subtitle burn-in failed",
            "detail": err_msg[:300],
        }, 500)


def _write_srt_file(segments: list, dst: Path):
    """Write segments as an SRT file (SubRip format).

    Prefers the dubbed timeline (placed_start/placed_end, set by the
    assembler) so burn-in subtitles track the audio; falls back to source
    timings for a transcript that was never assembled.
    """
    def _fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60)
        s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _fmt(seg.get("placed_start", seg.get("start", 0.0)))
        end = _fmt(seg.get("placed_end", seg.get("end", 0.0)))
        text = (seg.get("translated_text") or seg.get("text") or "").strip()
        # Strip emotion tags like "(happy)" that are TTS-only and shouldn't
        # appear in subtitle text; keep only the spoken part
        text = re.sub(r"^\s*\([^)]+\)\s*", "", text).strip()
        if not text:
            continue
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    dst.write_text("\n".join(lines), encoding="utf-8")


def _trim_video(src: Path, dst: Path, seconds: int) -> Path:
    """Trim src to the first `seconds` seconds into dst.

    Tries stream-copy first (fast, requires keyframe alignment); on failure
    falls back to re-encode with libx264. Raises subprocess.CalledProcessError
    if both attempts fail. Returns dst on success.
    """
    import subprocess
    seconds = max(1, int(seconds))
    # Attempt 1 — stream copy. Works when keyframes align with the cut point.
    try:
        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", "0",
             "-i", str(src),
             "-t", str(seconds),
             "-c", "copy",
             "-avoid_negative_ts", "make_zero",
             str(dst)],
            check=True, capture_output=True, timeout=60,
        )
        log.info(f"[trim] {src.name} -> {dst.name} ({seconds}s, stream-copy)")
        return dst
    except subprocess.CalledProcessError as e1:
        log.warning(f"[trim] stream-copy failed for {src.name}: "
                    f"{(e1.stderr or b'').decode('utf-8', errors='replace')[-200:]}")
    # Attempt 2 — re-encode. Slower but works on any source.
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(src),
         "-t", str(seconds),
         "-c:v", "libx264", "-preset", "veryfast",
         "-c:a", "aac", "-b:a", "128k",
         str(dst)],
        check=True, capture_output=True, timeout=120,
    )
    log.info(f"[trim] {src.name} -> {dst.name} ({seconds}s, re-encoded)")
    return dst


# Default language picks for the Quick-Test feature. Frontend allows
# per-run override; this is just the pre-selection.
# Bounds on a multi-language batch. One is the floor: a single-language dub
# through this route is a perfectly ordinary request, and letting it through
# is what gives every creator video a batch_id — /api/dub sets none, so a
# one-language job submitted there cannot be grouped with anything later.
# The ceiling is a guard against a mis-click queueing a very long run, not a
# technical limit — each language is an independent job and the runner
# processes them serially, so the only cost of more is time.
MIN_TARGET_LANGS = 1
MAX_TARGET_LANGS = 12

# Showcase stitches N languages into one reel and redub's compare/showcase
# modes exist to put languages side by side; one language makes neither of
# them mean anything. They get their own floor rather than sharing the one
# above, so lowering that could not quietly turn a one-language showcase
# into a valid request.
MIN_SHOWCASE_LANGS = 2

_QUICK_TEST_DEFAULT_LANGS = ("es", "fr", "de", "ja", "pt")
# Validation set for Quick-Test / Showcase / Redub target codes. Derived from
# the edge-tts VOICE_MAP — the canonical language registry that GET
# /api/languages also serves — so a language registered for dubbing can never
# be rejected at submission time. (That used to happen to "bg": it was in the
# voice map and /api/languages, but missing from this once-hardcoded set.)
_QUICK_TEST_KNOWN_LANGS = set(EdgeTTSFallback.VOICE_MAP)


@app.post("/api/quick_test")
async def start_quick_test(
    video: Optional[UploadFile] = File(None),
    source: str = Form(""),                # YouTube/direct URL
    reference: Optional[UploadFile] = File(None),
    trim_seconds: int = Form(0),           # 0 = dub the whole video
    target_langs: str = Form(""),          # comma-separated e.g. "es,fr,de,ja,pt"
    source_lang: str = Form("auto"),
    model: str = Form("aya-expanse:8b"),
    whisper_model: str = Form("large-v3"),
    speaker_mode: str = Form("main"),
    voice_preset: str = Form("auto"),
    voice_style: str = Form(""),
    tts_speed: str = Form("balanced"),
    keep_bg: bool = Form(True),
    auto_denoise: bool = Form(False),
    context_hint: str = Form(""),
    batch_label: str = Form(""),
    wizard_mode: str = Form("auto"),       # "auto" | "review_translation" | …
    lip_sync: bool = Form(False),
    scheduled_at: float = Form(0.0),
    voxcpm_cfg: float = Form(0.0),         # 0 = use the global setting
    voxcpm_steps: int = Form(0),           # 0 = use the global setting
    background: bool = Form(False),        # return before the download runs
):
    """Multi-language dub: fan one source out into N dub jobs, one per target
    language, sharing a batch_id so the UI can show them together.

    This is a primary workflow, not a smoke test — it takes the same options
    as a single dub. `trim_seconds=0` (the default) dubs the whole video;
    a non-zero value trims a clip first, which is the cheap way to audition
    voices and languages before committing GPU time to the full thing.

    `background=1` returns as soon as the jobs exist, in status "preparing",
    and does the download and trim in a task. Without it the caller waits out
    a full YouTube download inside its HTTP request — minutes for a long
    video, and a proxy in front of this server will time the request out
    before it finishes. The central download stays central either way: the
    fan-out jobs share one file, which is why it lives in the handler at all.
    """
    # ── Validate inputs ───────────────────────────────────────────────
    voxcpm_cfg, voxcpm_steps, _vox_err = validate_voxcpm_overrides(
        voxcpm_cfg, voxcpm_steps)
    if _vox_err:
        return JSONResponse({"error": _vox_err}, 400)
    if not video and not source.strip():
        return JSONResponse({"error": "Provide either a video file or a URL"}, 400)
    if video and source.strip():
        return JSONResponse({"error": "Provide only one of video or url"}, 400)

    # 0 means "no trim". Any other value is a real clip length, and the 15s
    # floor stops a clip so short that whisper has nothing to work with.
    if trim_seconds and (trim_seconds < 15 or trim_seconds > 120):
        return JSONResponse(
            {"error": f"trim_seconds must be 0 (full video) or between 15 "
                      f"and 120 (got {trim_seconds})"},
            400,
        )

    langs = [c.strip() for c in target_langs.split(",") if c.strip()]
    if not (MIN_TARGET_LANGS <= len(langs) <= MAX_TARGET_LANGS):
        return JSONResponse(
            {"error": f"Pick {MIN_TARGET_LANGS}-{MAX_TARGET_LANGS} target "
                      f"languages (got {len(langs)})"}, 400)
    unknown = [c for c in langs if c not in _QUICK_TEST_KNOWN_LANGS]
    if unknown:
        return JSONResponse(
            {"error": f"Unknown language code(s): {unknown}"}, 400)
    if len(set(langs)) != len(langs):
        return JSONResponse({"error": "Duplicate language codes"}, 400)

    # ── Validate Ollama model (same fallback logic as start_batch_dub) ─
    _ok, _installed = await check_ollama()
    if _ok and model not in _installed:
        model, _model_err = _resolve_translation_model(model, _installed)
        if _model_err:
            return JSONResponse({"error": _model_err}, 400)

    # Populated from the yt-dlp probe below for URL sources; stays empty for
    # local uploads, which have no page to read metadata from.
    src_meta: dict = {}

    # ── Save shared reference (one upload, reused by all jobs) ────────
    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"qt_{uuid.uuid4().hex[:8]}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        log.info(f"[multidub] Saved shared reference: {ref_path}")

    # ── Identify the source ───────────────────────────────────────────
    # An upload is already local. A URL is probed now — through the same
    # TTL cache /api/estimate uses, so a wizard that just priced this link
    # pays nothing for the probe — and downloaded later, once, centrally:
    # the fan-out jobs all read the same file.
    upload_path: Optional[Path] = None
    url = ""
    if video and video.filename:
        ext = Path(video.filename).suffix or ".mp4"
        upload_path = UPLOAD_DIR / f"qt_{uuid.uuid4().hex[:8]}{ext}"
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        src_label = video.filename
    else:
        url = source.strip()
        src_label = url[:60] + ("..." if len(url) > 60 else "")
        # Without this the fan-out jobs carried no meta at all, so the
        # translate stage had nothing to render into the target language and
        # the Results metadata panel came up empty.
        src_meta = await _probe_meta_cached(url)
        if src_meta.get("title"):
            src_label = src_meta["title"][:60]

    batch_id = f"qt_{uuid.uuid4().hex[:8]}"
    _scope = f"{trim_seconds}s clip" if trim_seconds else "full video"
    label_final = batch_label or f"{len(langs)} languages · {_scope}"

    def _create_jobs(status: str, src: str) -> list:
        """One job per language, all sharing one batch_id."""
        ids = []
        for idx, lang in enumerate(langs):
            jid = uuid.uuid4().hex[:8]
            jobs[jid] = {
                "id": jid,
                "status": status,
                "progress": 0,
                "source": src,
                "source_type": "file",
                "source_label": f"{src_label} -> {lang.upper()}",
                "target_lang": lang,
                "source_lang": source_lang,
                "model": model,
                "speaker_mode": speaker_mode,
                "context_hint": context_hint,
                "voice_style": voice_style,
                "voice_preset": voice_preset,
                "voice_mode": ("upload" if ref_path else
                               ("custom" if voice_style.strip() else "preset")),
                "tts_speed": tts_speed,
                "voxcpm_cfg": voxcpm_cfg,
                "voxcpm_steps": voxcpm_steps,
                "whisper_model": whisper_model,
                "keep_bg": keep_bg,
                "wizard_mode": wizard_mode,
                "lip_sync": lip_sync,
                "auto_denoise": auto_denoise,
                # Every sibling shares the source's metadata; the translate
                # stage then renders its own meta_translated per language.
                "meta": dict(src_meta) if src_meta else {},
                "title": src_meta.get("title") or "",
                "trim_seconds": trim_seconds,
                "batch_id": batch_id,
                "batch_label": label_final,
                # Stored value kept as "quick_test" so batches created before
                # this became a first-class multi-language mode keep rendering
                # and rebuilding correctly; the UI maps it to a display label.
                "batch_kind": "quick_test",
                "batch_position": idx,
                "batch_total": len(langs),
                "created": time.time(),
                "scheduled_at": scheduled_at,
                "step_detail": ("Fetching your video…" if status == "preparing"
                                else ""),
                "_pending_args": None,
            }
            save_job(jobs[jid])
            ids.append(jid)
        return ids

    def _pipeline_args(lang: str, path: Path) -> dict:
        return {
            "source": str(path),
            "source_lang": source_lang,
            "target_lang": lang,
            "model": model,
            "keep_bg": keep_bg,
            "whisper_model": whisper_model,
            "reference_audio": ref_path,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "wizard_mode": wizard_mode,
            # NOT lip_sync: these are run_pipeline's kwargs, and it has no
            # such parameter — passing it raised TypeError at dequeue and
            # failed every multi-language job. The post-success hook reads
            # the flag off the job dict (see _job_queue_worker), which is
            # where it belongs.
            "auto_denoise": auto_denoise,
            "voxcpm_cfg": voxcpm_cfg,
            "voxcpm_steps": voxcpm_steps,
        }

    async def _materialize():
        """(path, error dict, http status). Every blocking call runs in a
        thread — this used to run download_video() straight inside the async
        handler, which froze the whole server for the length of a YouTube
        download and made the submit request look dead."""
        if upload_path is not None:
            src = upload_path
        else:
            dl_dir = UPLOAD_DIR / f"qt_{uuid.uuid4().hex[:8]}"
            try:
                dl_dir.mkdir(parents=True, exist_ok=True)
                src = Path(await asyncio.to_thread(
                    download_video, url, str(dl_dir)))
            except Exception as e:
                # The directory is created before the download, so a failure
                # leaves an empty one behind on every bad URL. Nothing else
                # ever cleans uploads/, so they accumulate silently.
                try:
                    if dl_dir.is_dir() and not any(dl_dir.iterdir()):
                        dl_dir.rmdir()
                except OSError:
                    pass
                return None, {"error": f"Could not download URL: {e}"}, 400
        if not trim_seconds:
            return src, None, 0
        dst = src.parent / f"{src.stem}_qt{trim_seconds}s.mp4"
        try:
            await asyncio.to_thread(_trim_video, src, dst, trim_seconds)
        except Exception as e:
            err_msg = ""
            if hasattr(e, "stderr") and getattr(e, "stderr", None):
                err_msg = e.stderr.decode("utf-8", errors="replace")[-300:]
            log.warning(f"[multidub] trim failed: {e} :: {err_msg}")
            return None, {"error": "Could not trim video",
                          "detail": err_msg or str(e)}, 500
        return dst, None, 0

    async def _fan_out(ids: list, final_path: Path) -> None:
        for jid, lang in zip(ids, langs):
            j = jobs.get(jid)
            if not j or j.get("status") == "cancelled" or j.get("cancel_requested"):
                # Cancelled while the download was still running.
                if j and j.get("status") != "cancelled":
                    j["status"] = "cancelled"
                    j["step_detail"] = "Cancelled while preparing"
                    save_job(j)
                continue
            j["source"] = str(final_path)
            save_job(j)
            await enqueue_job(jid, _pipeline_args(lang, final_path))

    # ── Background submit ─────────────────────────────────────────────
    if background:
        try:
            job_ids = _create_jobs("preparing", "")
        except Exception as e:
            # save_job failing part-way leaves the siblings it already
            # registered in the in-memory store as "preparing" — a status
            # nothing can delete. The caller gets a 500 and would otherwise
            # never learn those exist.
            for jid in [j for j, v in list(jobs.items())
                        if v.get("batch_id") == batch_id]:
                jobs.pop(jid, None)
            log.exception(f"[multidub] {batch_id}: could not create jobs")
            return JSONResponse(
                {"error": f"Could not create jobs: {e}"}, 500)

        def _abandon(ids: list, detail: str) -> None:
            """Move every still-preparing sibling to a terminal status.

            Nothing else will. A job left in "preparing" never reaches a
            pipeline, never times out, and is skipped by bulk delete
            (_UNDELETABLE_STATUSES) — so it sits on the home screen claiming
            to be working until the server restarts. Single delete does
            still remove it, which is the only reason this is a defect
            rather than an emergency.
            """
            for jid in ids:
                j = jobs.get(jid)
                if not j or j.get("status") != "preparing":
                    continue
                j["status"] = "error"
                j["error"] = detail
                j["step_detail"] = ""
                try:
                    save_job(j)
                except Exception as e:
                    # The in-memory store is the source of truth during a
                    # live run, so the status change above already frees the
                    # job. Letting a failed persist abort the loop would
                    # strand every sibling after this one — which is the
                    # exact failure this function exists to prevent.
                    log.warning(f"[multidub] could not persist {jid}: {e}")

        async def _prepare(ids: list = job_ids) -> None:
            # Everything below runs detached in a task: an exception here is
            # not returned to anyone, it is swallowed by the event loop. So
            # the whole body is guarded — the failure mode without this is
            # silent, permanent, and undeletable.
            try:
                final_path, err, _code = await _materialize()
                if err:
                    detail = (err.get("detail") or err.get("error")
                              or "Preparation failed")
                    _abandon(ids, detail)
                    log.warning(f"[multidub] {batch_id}: preparation failed "
                                f"— {detail}")
                    return
                if all((jobs.get(j) or {}).get("status") == "cancelled"
                       for j in ids):
                    # Every sibling was cancelled while this downloaded.
                    # Nothing will ever read the file, and uploads/ has no
                    # other sweeper.
                    _discard_download(final_path)
                    log.info(f"[multidub] {batch_id}: all jobs cancelled "
                             f"while preparing — download discarded")
                    return
                await _fan_out(ids, final_path)
                log.info(f"[multidub] {batch_id}: prepared and enqueued "
                         f"{len(ids)} job(s) ({_scope}, langs={langs})")
            except asyncio.CancelledError:
                _abandon(ids, "Server shut down while preparing this video.")
                raise
            except Exception as e:
                log.exception(f"[multidub] {batch_id}: preparation crashed")
                _abandon(ids, f"Preparation failed: {e}")
            finally:
                # A sibling can also be left behind by _fan_out itself —
                # enqueue_job raising halfway through leaves the ones after
                # it untouched. Nothing is still legitimately "preparing" by
                # the time this task ends.
                _abandon(ids, "Preparation ended without starting this job.")

        _spawn_background(_prepare())
        log.info(f"[multidub] {batch_id}: {len(job_ids)} job(s) accepted, "
                 f"preparing in background ({_scope}, langs={langs})")
        return {
            "ok": True,
            "batch_id": batch_id,
            "batch_kind": "quick_test",
            "job_ids": job_ids,
            "count": len(job_ids),
            "background": True,
            "status": "preparing",
            # Not known yet — the source has not been fetched.
            "trimmed_file": None,
            "trim_seconds": trim_seconds,
            "target_langs": langs,
        }

    # ── Synchronous submit (unchanged semantics) ──────────────────────
    trimmed_path, err, code = await _materialize()
    if err:
        return JSONResponse(err, code)

    job_ids = _create_jobs("queued", str(trimmed_path))
    await _fan_out(job_ids, trimmed_path)

    log.info(f"[multidub] {batch_id}: enqueued {len(job_ids)} jobs "
             f"({_scope}, langs={langs})")

    return {
        "ok": True,
        "batch_id": batch_id,
        "batch_kind": "quick_test",
        "job_ids": job_ids,
        "count": len(job_ids),
        "background": False,
        "trimmed_file": f"/uploads/{trimmed_path.name}",
        "trim_seconds": trim_seconds,
        "target_langs": langs,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Multilingual Showcase — same source split into N language segments
#  stitched back into one continuous video with corner language badges.
# ═══════════════════════════════════════════════════════════════════════
# Flow:
#   1. Same as Quick Test — fan out N full-length dubs, one per language.
#   2. When all sibling jobs finish, _maybe_assemble_showcase() reads the
#      source transcript, picks ~equal time slices snapped to the nearest
#      sentence-end boundary, trims each dub to its slice, overlays a
#      "· LL ·" badge in the top-right, and concatenates into one mp4.

_SHOWCASE_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


def _find_drawtext_font() -> str:
    """Locate a usable TTF/TTC for ffmpeg drawtext. Empty string = let
    ffmpeg fall back to its default (may fail on some Windows builds)."""
    for c in _SHOWCASE_FONT_CANDIDATES:
        if Path(c).exists():
            return c
    return ""


_FILTER_CACHE: dict = {}


def _ffmpeg_has_filter(name: str) -> bool:
    """Whether this ffmpeg build ships a given filter.

    `drawtext` needs libfreetype, and plenty of builds skip it — Homebrew's
    ffmpeg 8.x on macOS is one, which turns the showcase stitch into
    "No such filter: 'drawtext'" after every dub has already been rendered.
    Checked once and cached; `ffmpeg -filters` costs ~50ms.
    """
    if name in _FILTER_CACHE:
        return _FILTER_CACHE[name]
    import subprocess as _sp   # module-level import is not available here
    ok = False
    try:
        r = _sp.run(["ffmpeg", "-hide_banner", "-filters"],
                    capture_output=True, text=True, timeout=20)
        # Lines look like " ..C drawtext V->V  Draw text on top of video."
        ok = any(ln.split()[1:2] == [name]
                 for ln in r.stdout.splitlines() if len(ln.split()) > 1)
    except Exception as e:
        log.warning(f"[showcase] could not probe ffmpeg filters: {e}")
    _FILTER_CACHE[name] = ok
    return ok


def _render_label_png(text: str, dest: Path, font_path: str = "") -> bool:
    """Draw the showcase language label to an RGBA PNG.

    The fallback for builds without `drawtext`. Pillow does the text
    rendering, so ffmpeg only has to `overlay` a bitmap — a filter every
    build has. Deliberately mirrors the drawtext styling (22px white on a
    black box at 55%, 8px padding) so the two paths look the same.

    Pillow is not a declared dependency — it arrives via matplotlib — so a
    missing import is an expected outcome, not an error.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    try:
        size, pad = 22, 8
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size)
        except Exception:
            font = ImageFont.load_default(size)
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        box = probe.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 140))
        ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text,
                                 font=font, fill=(255, 255, 255, 255))
        img.save(dest)
        return True
    except Exception as e:
        log.warning(f"[showcase] label render failed: {e}")
        return False


def _snap_boundaries_to_sentences(segments: list, total_dur: float, n_parts: int) -> list:
    """Compute n_parts contiguous slices that together cover [0, total_dur].

    Slices are equal time chunks (`total_dur / n_parts`), with each interior
    boundary snapped to the nearest segment END time. Returns a list of
    (start, end) tuples — guaranteed contiguous, non-empty, sorted.
    """
    if n_parts <= 1 or not segments:
        return [(0.0, float(total_dur))]

    seg_ends = sorted({float(s.get("end", 0.0)) for s in segments if s.get("end")})
    seg_ends = [e for e in seg_ends if 0 < e < total_dur]

    target_each = total_dur / n_parts
    boundaries = []
    prev = 0.0
    for i in range(1, n_parts):
        target = i * target_each
        # Snap to the nearest segment end, but never go backwards past `prev`
        # (otherwise we'd get a zero-length or negative slice).
        candidates = [e for e in seg_ends if e > prev + 0.5]
        if candidates:
            snapped = min(candidates, key=lambda e: abs(e - target))
        else:
            snapped = target
        snapped = max(snapped, prev + 0.5)
        snapped = min(snapped, total_dur - (n_parts - i) * 0.5)
        boundaries.append(snapped)
        prev = snapped

    slices = []
    last = 0.0
    for b in boundaries:
        slices.append((last, b))
        last = b
    slices.append((last, float(total_dur)))
    return slices


def _save_placements(work_dir: Path, segments: list) -> None:
    """Write tts_placements.json next to dubbed_video.mp4. Records where each
    segment ended up in the final dubbed track (placed_start/end), which may
    differ from the source-time start/end due to overlap-pushing and atempo
    stretching in the assembler. Showcase reels need these to cut each dub
    at its OWN word boundaries instead of at fixed source timestamps."""
    try:
        rows = []
        for seg in segments:
            ps = seg.get("placed_start")
            pe = seg.get("placed_end")
            if ps is None or pe is None:
                continue
            rows.append({
                "idx": seg.get("idx"),
                "src_start": float(seg.get("start", 0.0)),
                "src_end": float(seg.get("end", 0.0)),
                "dub_start": float(ps),
                "dub_end": float(pe),
            })
        out = work_dir / "tts_placements.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if rows:
            log.info(f"[placements] saved {len(rows)} rows -> {out.name} "
                     f"(dub range [{rows[0]['dub_start']:.1f}–{rows[-1]['dub_end']:.1f}s])")
        else:
            log.warning(f"[placements] 0 rows written to {out} — "
                        "assembler did not set placed_start/end on any segment")
    except Exception as e:
        log.warning(f"[placements] failed to save: {e}")


def _load_placements(work_dir: Path) -> list:
    """Inverse of _save_placements. Returns [] if file missing/unparseable."""
    p = work_dir / "tts_placements.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning(f"[placements] failed to load {p}: {e}")
        return []


def _probe_duration(path: Path) -> float:
    """ffprobe a media file and return duration in seconds (0.0 on failure).

    Tries ffprobe first (fast), falls back to parsing `ffmpeg -i` stderr
    if ffprobe isn't available. Logs the reason on every failure so we
    don't get silent zero-returns.
    """
    import subprocess  # follow existing per-function-import pattern
    if not Path(path).exists():
        log.warning(f"[probe] file not found: {path}")
        return 0.0
    # Try ffprobe
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True, timeout=15,
        )
        d = float((r.stdout or "0").strip() or 0)
        if d > 0:
            return d
    except FileNotFoundError:
        log.warning("[probe] ffprobe not on PATH — falling back to ffmpeg -i")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        log.warning(f"[probe] ffprobe failed for {path}: {err[-200:].strip() or e}")
    except Exception as e:
        log.warning(f"[probe] ffprobe error for {path}: {type(e).__name__}: {e}")

    # Fallback: parse "Duration: HH:MM:SS.ss" from ffmpeg -i stderr.
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        import re as _re
        m = _re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", r.stderr or "")
        if m:
            h, mn, s = m.groups()
            return float(h) * 3600 + float(mn) * 60 + float(s)
        log.warning(f"[probe] ffmpeg fallback: no Duration in stderr for {path}")
    except FileNotFoundError:
        log.warning("[probe] ffmpeg not on PATH either — both probes failed")
    except Exception as e:
        log.warning(f"[probe] ffmpeg fallback failed: {type(e).__name__}: {e}")
    return 0.0


def _validate_attached_video(path: Path) -> Optional[str]:
    """Why `path` is not a usable video, or None when it is.

    Two checks: a positive probed duration (rejects images and corrupt
    files) and an actual video stream (rejects audio-only files — a WAV
    has a duration but nothing to dub onto).
    """
    import subprocess  # follow existing per-function-import pattern
    if _probe_duration(path) <= 0:
        return ("File has no readable duration — it doesn't look like a "
                "video (an image, or corrupt?)")
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except Exception as e:
        log.warning(f"[rescue] stream probe failed for {path}: "
                    f"{type(e).__name__}: {e}")
        return f"Could not inspect the file's streams: {type(e).__name__}"
    if "video" not in (r.stdout or ""):
        return ("File has no video stream (audio-only?) — attach the "
                "downloaded video file itself")
    return None


_showcase_assembling: set = set()  # in-progress batch IDs (de-dupe re-entry)
_showcase_tasks: set = set()       # strong refs so asyncio GC doesn't kill them


async def _maybe_assemble_showcase(batch_id: str) -> None:
    """Hook called after each job finishes. If all jobs in `batch_id` are
    complete and this batch is a 'showcase', assemble the combined reel.
    No-op otherwise. Safe to call multiple times — guarded by status check
    and an in-progress set."""
    log.info(f"[showcase] _maybe_assemble_showcase('{batch_id}') called")
    if not batch_id:
        log.info("[showcase] empty batch_id, skipping")
        return
    if batch_id in _showcase_assembling:
        log.info(f"[showcase] {batch_id} already assembling, skipping")
        return

    # Collect sibling jobs
    siblings = [j for j in jobs.values()
                if j.get("batch_id") == batch_id
                and j.get("batch_kind") == "showcase"]
    if not siblings:
        log.warning(f"[showcase] no siblings found for {batch_id}")
        return
    expected_total = max((j.get("batch_total") or 0) for j in siblings)
    if expected_total and len(siblings) < expected_total:
        log.info(f"[showcase] {batch_id}: only {len(siblings)}/{expected_total} jobs registered, waiting")
        return

    # All must be complete (not error/queued/running)
    statuses = [j.get("status") for j in siblings]
    if not all(s == "complete" for s in statuses):
        log.info(f"[showcase] {batch_id}: not all complete (statuses={statuses})")
        return

    # Already assembled? Bail.
    showcase_dir = OUTPUT_DIR / f"showcase_{batch_id}"
    out_mp4 = showcase_dir / "showcase.mp4"
    if out_mp4.exists():
        log.info(f"[showcase] {batch_id}: already assembled at {out_mp4}")
        return

    log.info(f"[showcase] {batch_id}: all checks passed, kicking off ffmpeg")
    _showcase_assembling.add(batch_id)
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, _assemble_showcase_sync, batch_id, siblings, showcase_dir
        )
    except Exception as e:
        log.error(f"[showcase] {batch_id}: assembler crashed: {e}", exc_info=True)
    finally:
        _showcase_assembling.discard(batch_id)


def _assemble_showcase_sync(batch_id: str, siblings: list, showcase_dir: Path) -> None:
    """Synchronous worker for showcase assembly. Runs in a thread to keep
    the event loop responsive (ffmpeg is blocking)."""
    import subprocess  # follow existing per-function-import pattern
    siblings = sorted(siblings, key=lambda j: j.get("batch_position", 0))
    n = len(siblings)
    log.info(f"[showcase] {batch_id}: assembling {n} language segments…")

    # ── Load source segment times (any sibling has them; pick the first) ─
    segments: list = []
    for j in siblings:
        work = OUTPUT_DIR / j["id"]
        for cp_name in ("checkpoint_translation_done.json",
                        "checkpoint_transcription_done.json"):
            cp = work / cp_name
            if cp.exists():
                try:
                    data = json.loads(cp.read_text(encoding="utf-8"))
                    segments = data.get("segments", []) or []
                    if segments:
                        break
                except Exception as e:
                    log.warning(f"[showcase] couldn't parse {cp}: {e}")
        if segments:
            break

    # ── Verify all dub files exist ────────────────────────────────────
    missing = [OUTPUT_DIR / j["id"] / "dubbed_video.mp4"
               for j in siblings
               if not (OUTPUT_DIR / j["id"] / "dubbed_video.mp4").exists()]
    if missing:
        log.error(f"[showcase] {batch_id}: {len(missing)} dub file(s) missing — aborting")
        return

    # ── Pre-load ALL placements once, compute effective dub duration ──
    # The dubbed audio for each language is typically shorter than the source
    # video because TTS may run faster (atempo-stretched) and the assembler
    # trims trailing silence. The VIDEO container duration == source duration
    # (we pad the last frame), so probing the mp4 gives ~120s for a 120s
    # source — but the AUDIO ends at 95-106s. If we slice into [95-120s] of
    # a dub that has no audio there, we get silence in the showcase.
    # Solution: compute total_dur from the ACTUAL last placed segment in every
    # dub (max dub_end across placements), then take min so every language
    # has content throughout the full showcase.
    all_placements: dict = {}   # job_id -> list of placement rows
    effective_ends: list = []   # max dub_end per sibling
    for j in siblings:
        pl = _load_placements(OUTPUT_DIR / j["id"])
        all_placements[j["id"]] = pl
        if pl:
            effective_ends.append(max(p["dub_end"] for p in pl))
        else:
            # Fallback: probe dubbed audio track duration (not the mp4 container)
            # using ffprobe's stream-level query which returns audio stream duration.
            import subprocess as _sp
            audio_dur = 0.0
            try:
                r = _sp.run(
                    ["ffprobe", "-v", "error",
                     "-select_streams", "a:0",
                     "-show_entries", "stream=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1",
                     str(OUTPUT_DIR / j["id"] / "dubbed_video.mp4")],
                    check=True, capture_output=True, text=True, timeout=15,
                )
                audio_dur = float((r.stdout or "0").strip() or 0)
            except Exception:
                pass
            if audio_dur > 0:
                effective_ends.append(audio_dur)
            else:
                log.warning(f"[showcase] {j['id']}: no placements and audio probe failed; "
                            "showcase may include silent tail for this language")

    if not effective_ends:
        # True last-resort: use source video duration
        first_source = Path(siblings[0].get("source", ""))
        fallback_dur = _probe_duration(first_source) if first_source.exists() else 0.0
        if fallback_dur <= 0:
            log.error(f"[showcase] {batch_id}: could not determine any dub duration — aborting")
            return
        total_dur = fallback_dur
        log.warning(f"[showcase] using source duration as total_dur fallback ({total_dur:.1f}s)")
    else:
        total_dur = min(effective_ends)
        log.info(f"[showcase] effective dub durations: "
                 f"{[f'{d:.1f}' for d in effective_ends]}s → using min={total_dur:.1f}s")

    slices = _snap_boundaries_to_sentences(segments, total_dur, n)
    log.info(f"[showcase] source-time slices: {[f'{s:.1f}-{e:.1f}' for s, e in slices]}")

    # ── Map each source-time slice to per-dub time using placements ────
    # Key fix: use OVERLAP matching — a segment spans [src_start, src_end]
    # in source time and [dub_start, dub_end] in dub time. We want every
    # segment that OVERLAPS the slice [g_s, g_e], not just those whose start
    # falls inside. Without overlap matching, a long merged segment (e.g.
    # src [30-85s]) will cover several source slices but only be found for
    # the one whose boundary contains 30s. This caused 30s source slices to
    # map to only 7s of dub time (one tail segment found instead of all).
    LEAD = 0.05   # 50ms lead so we don't clip a word's onset
    TRAIL = 0.15  # 150ms trail for a clean release
    per_dub_ranges: list = []
    for slice_idx, (g_s, g_e) in enumerate(slices):
        job = siblings[slice_idx]
        placements = all_placements.get(job["id"], [])
        # Effective audio end for this dub (from placements or fallback)
        eff_end = (max(p["dub_end"] for p in placements)
                   if placements else effective_ends[slice_idx]
                   if slice_idx < len(effective_ends) else total_dur)
        # Overlap matching: segment overlaps slice if src_start < g_e AND src_end > g_s
        in_slice = [
            p for p in placements
            if p["src_start"] < g_e + 0.001 and p["src_end"] > g_s - 0.001
        ]
        if in_slice:
            # Key: include the SOURCE time range as well as the dub placement
            # range. In the dubbed video, non-speech gaps keep source timing
            # (assembler places segments at their source timestamps). So a
            # slice [67.5-79s] that has a speech segment placed at dub[70-72s]
            # should show dub[67.5-79s] — not just [70-72s] — to include the
            # gap/background content before and after the speech burst.
            d_start = max(0.0, min(min(p["dub_start"] for p in in_slice), g_s) - LEAD)
            d_end = min(eff_end, max(max(p["dub_end"] for p in in_slice), g_e) + TRAIL)
            # Sanity: never go past eff_end or before 0
            if d_end <= d_start + 0.1:
                d_start = max(0.0, g_s)
                d_end = min(eff_end, g_e)
            log.info(f"[showcase] slice {slice_idx} ({job.get('target_lang')}): "
                     f"src [{g_s:.2f}-{g_e:.2f}] -> dub [{d_start:.2f}-{d_end:.2f}] "
                     f"({len(in_slice)} segs, eff_end={eff_end:.1f}s)")
        else:
            # No speech in this slice — show source-equivalent gap content,
            # clamped to effective audio end.
            d_start = max(0.0, g_s)
            d_end = min(g_e, eff_end)
            if d_end <= d_start + 0.05:
                log.warning(f"[showcase] slice {slice_idx} ({job.get('target_lang')}): "
                            f"no speech and eff_end={eff_end:.1f}s < slice_start={g_s:.1f}s "
                            f"— skipping (will use 0.1s stub)")
                d_end = d_start + 0.1  # avoid zero-length segment crashing ffmpeg
            else:
                log.warning(f"[showcase] slice {slice_idx} ({job.get('target_lang')}): "
                            f"no speech in slice, showing gap [{d_start:.2f}-{d_end:.2f}]")
        per_dub_ranges.append((job, d_start, d_end))

    # ── Build ffmpeg filter_complex ───────────────────────────────────
    showcase_dir.mkdir(parents=True, exist_ok=True)
    font_path = _find_drawtext_font()
    # ffmpeg filter syntax requires escaped colons on Windows paths
    font_arg = font_path.replace("\\", "/").replace(":", r"\:") if font_path else ""

    # Which labelling path this build can support. drawtext needs libfreetype;
    # when it is absent we render the same label with Pillow and overlay it,
    # which needs no special ffmpeg build. Losing the labels entirely is the
    # last resort — a stitched reel without captions still beats an error
    # after every dub has already been rendered.
    use_drawtext = _ffmpeg_has_filter("drawtext")
    label_pngs: dict = {}
    if not use_drawtext:
        for idx, (job, _s, _e) in enumerate(per_dub_ranges):
            code = (job.get("target_lang") or "").upper()
            png = showcase_dir / f"_label_{idx}_{code or 'x'}.png"
            if _render_label_png(f"· {code} ·", png, font_path):
                label_pngs[idx] = png
        if label_pngs:
            log.info(f"[showcase] ffmpeg has no drawtext filter — overlaying "
                     f"{len(label_pngs)} rendered label(s) instead")
        else:
            log.warning("[showcase] no drawtext filter and no Pillow — "
                        "stitching without language labels")

    filter_parts = []
    inputs = []
    for idx, (job, start, end) in enumerate(per_dub_ranges):
        src = OUTPUT_DIR / job["id"] / "dubbed_video.mp4"
        inputs.extend(["-i", str(src)])
        lang_code = (job.get("target_lang") or "").upper()
        label = f"· {lang_code} ·"  # · LL ·
        seg_dur = max(0.1, end - start)

        # Escape single quotes and special chars for drawtext text= field
        text_safe = label.replace("'", r"\'")

        if use_drawtext:
            drawtext = (
                "drawtext="
                + (f"fontfile='{font_arg}':" if font_arg else "")
                + f"text='{text_safe}':"
                "fontsize=22:fontcolor=white:"
                "box=1:boxcolor=black@0.55:boxborderw=8:"
                "x=w-tw-24:y=24"
            )
            label_step = f",{drawtext}"
        else:
            label_step = ""   # overlay is wired up after all inputs are known

        filter_parts.append(
            f"[{idx}:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS{label_step}[v{idx}]"
        )
        # apad+atrim guarantees the audio is EXACTLY seg_dur long: if the
        # dub's audio stream ends before `end` (TTS finished early) apad
        # fills with silence; atrim caps any overflow. Without this, ffmpeg
        # concat hands us a shorter audio stream than video and you get
        # the "audio cuts off at 50.6s while video runs 60s" bug.
        filter_parts.append(
            f"[{idx}:a]atrim=start={start:.3f}:end={end:.3f},"
            f"asetpts=PTS-STARTPTS,"
            f"apad=whole_dur={seg_dur:.3f},"
            f"atrim=duration={seg_dur:.3f}[a{idx}]"
        )

    # Overlay path: the PNGs are appended after every video input, so their
    # stream indices start at n. Each label is composited onto its own
    # trimmed segment at the same top-right position drawtext would use.
    if label_pngs:
        png_order = sorted(label_pngs)
        for slot, idx in enumerate(png_order):
            inputs.extend(["-i", str(label_pngs[idx])])
            filter_parts.append(
                f"[v{idx}][{n + slot}:v]overlay=x=W-w-24:y=24:"
                f"format=auto[v{idx}L]"
            )
    # Concat all trimmed pieces
    concat_inputs = "".join(
        f"[v{i}L][a{i}]" if i in label_pngs else f"[v{i}][a{i}]"
        for i in range(n)
    )
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[vout][aout]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(showcase_dir / "showcase.mp4"),
    ]

    out_dur = sum(e - s for _, s, e in per_dub_ranges)
    log.info(f"[showcase] running ffmpeg ({n} inputs, {out_dur:.1f}s out)")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-1200:]
        log.error(f"[showcase] ffmpeg failed:\n{err}")
        # Write a marker so the UI can surface the failure
        try:
            (showcase_dir / "error.txt").write_text(err, encoding="utf-8")
        except Exception:
            pass
        return
    finally:
        # The rendered label bitmaps were only ever ffmpeg inputs.
        for _png in label_pngs.values():
            try:
                _png.unlink(missing_ok=True)
            except Exception:
                pass

    # Write a manifest for the UI
    try:
        manifest = {
            "batch_id": batch_id,
            "created": time.time(),
            "n_segments": n,
            "slices": [
                {
                    "lang": j.get("target_lang"),
                    "src_start": float(src_s),
                    "src_end": float(src_e),
                    "dub_start": float(d_s),
                    "dub_end": float(d_e),
                    "job_id": j["id"],
                }
                for (j, d_s, d_e), (src_s, src_e) in zip(per_dub_ranges, slices)
            ],
            "total_seconds": float(out_dur),
        }
        (showcase_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        log.warning(f"[showcase] manifest write failed: {e}")

    log.info(f"[showcase] {batch_id}: done -> {showcase_dir / 'showcase.mp4'}")


# Default language picks for the Showcase feature — same defaults as
# Quick Test but kept separate so they can diverge if needed.
_SHOWCASE_DEFAULT_LANGS = _QUICK_TEST_DEFAULT_LANGS


@app.post("/api/showcase")
async def start_showcase(
    video: Optional[UploadFile] = File(None),
    source: str = Form(""),
    reference: Optional[UploadFile] = File(None),
    trim_seconds: int = Form(60),
    target_langs: str = Form(""),
    source_lang: str = Form("auto"),
    model: str = Form("aya-expanse:8b"),
    whisper_model: str = Form("large-v3"),
    speaker_mode: str = Form("main"),
    voice_preset: str = Form("auto"),
    voice_style: str = Form(""),
    tts_speed: str = Form("balanced"),
    keep_bg: bool = Form(True),
    auto_denoise: bool = Form(False),
    context_hint: str = Form(""),
    batch_label: str = Form(""),
    voxcpm_cfg: float = Form(0.0),         # 0 = use the global setting
    voxcpm_steps: int = Form(0),           # 0 = use the global setting
):
    """Showcase: trim a short clip, fan out into N normal dub jobs, and
    when all finish, automatically stitch them into one multilingual reel
    (each segment in a different language with a corner badge)."""
    # ── Input validation — identical to /api/quick_test ───────────────
    voxcpm_cfg, voxcpm_steps, _vox_err = validate_voxcpm_overrides(
        voxcpm_cfg, voxcpm_steps)
    if _vox_err:
        return JSONResponse({"error": _vox_err}, 400)
    if not video and not source.strip():
        return JSONResponse({"error": "Provide either a video file or a URL"}, 400)
    if video and source.strip():
        return JSONResponse({"error": "Provide only one of video or url"}, 400)
    if trim_seconds < 15 or trim_seconds > 120:
        return JSONResponse(
            {"error": f"trim_seconds must be between 15 and 120 (got {trim_seconds})"}, 400)

    langs = [c.strip() for c in target_langs.split(",") if c.strip()]
    if not (MIN_SHOWCASE_LANGS <= len(langs) <= MAX_TARGET_LANGS):
        return JSONResponse({"error": f"Pick {MIN_SHOWCASE_LANGS}-{MAX_TARGET_LANGS} "
                                      f"target languages (got {len(langs)})"}, 400)
    unknown = [c for c in langs if c not in _QUICK_TEST_KNOWN_LANGS]
    if unknown:
        return JSONResponse({"error": f"Unknown language code(s): {unknown}"}, 400)
    if len(set(langs)) != len(langs):
        return JSONResponse({"error": "Duplicate language codes"}, 400)

    # Validate ollama model with fallback
    _ok, _installed = await check_ollama()
    if _ok and model not in _installed:
        model, _model_err = _resolve_translation_model(model, _installed)
        if _model_err:
            return JSONResponse({"error": _model_err}, 400)

    # Shared reference (one upload, reused by all jobs)
    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"sc_{uuid.uuid4().hex[:8]}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        log.info(f"[showcase] Saved shared reference: {ref_path}")

    # Materialize source (file or yt-dlp URL)
    src_path: Path
    src_label: str
    if video and video.filename:
        ext = Path(video.filename).suffix or ".mp4"
        src_path = UPLOAD_DIR / f"sc_{uuid.uuid4().hex[:8]}{ext}"
        with open(src_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        src_label = video.filename
    else:
        url = source.strip()
        src_label = url[:60] + ("..." if len(url) > 60 else "")
        try:
            dl_dir = UPLOAD_DIR / f"sc_{uuid.uuid4().hex[:8]}"
            dl_dir.mkdir(parents=True, exist_ok=True)
            # In a thread, not inline: called directly inside this async
            # handler it froze the entire event loop — every other request,
            # including the UI's own polling — for the length of the
            # download. (/api/quick_test had the same bug and goes further,
            # returning before the download even starts; a showcase is
            # trimmed to 15-120s and is not the consumer front door, so it
            # keeps the simpler synchronous shape.)
            src_path = Path(await asyncio.to_thread(
                download_video, url, str(dl_dir)))
        except Exception as e:
            return JSONResponse({"error": f"Could not download URL: {e}"}, 400)

    # Trim
    trimmed_path = src_path.parent / f"{src_path.stem}_sc{trim_seconds}s.mp4"
    try:
        await asyncio.to_thread(_trim_video, src_path, trimmed_path, trim_seconds)
    except Exception as e:
        err_msg = ""
        if hasattr(e, "stderr") and getattr(e, "stderr", None):
            err_msg = e.stderr.decode("utf-8", errors="replace")[-300:]
        log.warning(f"[showcase] trim failed: {e} :: {err_msg}")
        return JSONResponse(
            {"error": "Could not trim video", "detail": err_msg or str(e)}, 500)

    # Fan out — one job per language, all sharing the batch_id
    batch_id = f"sc_{uuid.uuid4().hex[:8]}"
    label_final = batch_label or f"Showcase · {trim_seconds}s · {len(langs)} langs"
    job_ids: list = []

    for idx, lang in enumerate(langs):
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = {
            "id": jid,
            "status": "queued",
            "progress": 0,
            "source": str(trimmed_path),
            "source_type": "file",
            "source_label": f"{src_label} -> {lang.upper()} [showcase]",
            "target_lang": lang,
            "model": model,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "voice_mode": ("upload" if ref_path else
                          ("custom" if voice_style.strip() else "preset")),
            "tts_speed": tts_speed,
            "voxcpm_cfg": voxcpm_cfg,
            "voxcpm_steps": voxcpm_steps,
            "whisper_model": whisper_model,
            "keep_bg": keep_bg,
            "wizard_mode": "auto",
            "auto_denoise": auto_denoise,
            "batch_id": batch_id,
            "batch_label": label_final,
            "batch_kind": "showcase",
            "batch_position": idx,
            "batch_total": len(langs),
            "created": time.time(),
            "scheduled_at": 0,
            "_pending_args": None,
        }
        save_job(jobs[jid])
        await enqueue_job(jid, {
            "source": str(trimmed_path),
            "source_lang": source_lang,
            "target_lang": lang,
            "model": model,
            "keep_bg": keep_bg,
            "whisper_model": whisper_model,
            "reference_audio": ref_path,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "wizard_mode": "auto",
            "auto_denoise": auto_denoise,
            "voxcpm_cfg": voxcpm_cfg,
            "voxcpm_steps": voxcpm_steps,
        })
        job_ids.append(jid)

    log.info(f"[showcase] {batch_id}: enqueued {len(job_ids)} jobs "
             f"({trim_seconds}s, langs={langs})")

    return {
        "ok": True,
        "batch_id": batch_id,
        "batch_kind": "showcase",
        "job_ids": job_ids,
        "count": len(job_ids),
        "trimmed_file": f"/uploads/{trimmed_path.name}",
        "trim_seconds": trim_seconds,
        "target_langs": langs,
    }


@app.get("/api/showcase/{batch_id}")
async def get_showcase(batch_id: str):
    """Status + URL for an assembled showcase. Returns 404 if no showcase
    exists for this batch (either never started or still in progress)."""
    showcase_dir = OUTPUT_DIR / f"showcase_{batch_id}"
    mp4 = showcase_dir / "showcase.mp4"
    manifest = showcase_dir / "manifest.json"
    err_file = showcase_dir / "error.txt"

    # Sibling jobs (for progress reporting)
    siblings = [j for j in jobs.values()
                if j.get("batch_id") == batch_id
                and j.get("batch_kind") == "showcase"]

    if mp4.exists():
        man = {}
        if manifest.exists():
            try:
                man = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "ok": True,
            "status": "ready",
            "url": f"/outputs/showcase_{batch_id}/showcase.mp4",
            "manifest": man,
            "sibling_count": len(siblings),
        }

    if err_file.exists():
        try:
            err = err_file.read_text(encoding="utf-8")[-800:]
        except Exception:
            err = "Assembly failed (see server logs)"
        return JSONResponse(
            {"status": "error", "error": err}, 500)

    # Still in progress — count how many sibling jobs are done
    done = sum(1 for j in siblings if j.get("status") == "complete")
    errored = sum(1 for j in siblings if j.get("status") == "error")
    if siblings:
        return {
            "ok": True,
            "status": "assembling" if (done == len(siblings) and batch_id in _showcase_assembling)
                       else ("waiting_for_jobs" if errored == 0 else "jobs_failed"),
            "completed_jobs": done,
            "errored_jobs": errored,
            "total_jobs": len(siblings),
        }

    return JSONResponse({"status": "not_found"}, 404)


@app.post("/api/job/{job_id}/redub")
async def redub_job(
    job_id: str,
    target_langs: str = Form(""),
    mode: str = Form("compare"),          # 'single' | 'compare' | 'showcase'
    model: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form(None),
    voice_preset: Optional[str] = Form(None),
    voice_style: Optional[str] = Form(None),
    tts_speed: Optional[str] = Form(None),
    keep_bg: Optional[bool] = Form(None),
    speaker_mode: Optional[str] = Form(None),
    # None = inherit whatever the original job ran with; 0 = force back to
    # the global setting even if the original had an override.
    voxcpm_cfg: Optional[float] = Form(None),
    voxcpm_steps: Optional[int] = Form(None),
):
    """Re-dub an existing video into new language(s). Reuses the original
    source (file path or URL) — no re-upload required, just specify which
    new languages you want and the mode.

    Modes:
      single   — one new dub in one new language
      compare  — N new dubs (2-6), each in a different language (like Quick Test)
      showcase — N dubs, then stitched into one multilingual reel
    """
    orig = jobs.get(job_id)
    if not orig:
        return JSONResponse({"error": f"Job {job_id} not found"}, 404)

    # Validate mode + langs
    if mode not in ("single", "compare", "showcase"):
        return JSONResponse({"error": f"Invalid mode '{mode}'"}, 400)

    langs = [c.strip().lower() for c in target_langs.split(",") if c.strip()]
    if not langs:
        return JSONResponse({"error": "Specify at least one target_lang"}, 400)
    if mode == "single" and len(langs) != 1:
        return JSONResponse({"error": "mode=single requires exactly 1 language"}, 400)
    if mode in ("compare", "showcase") and not (
            MIN_SHOWCASE_LANGS <= len(langs) <= MAX_TARGET_LANGS):
        return JSONResponse({"error": f"mode={mode} needs {MIN_SHOWCASE_LANGS}-"
                                      f"{MAX_TARGET_LANGS} langs (got {len(langs)})"}, 400)
    unknown = [c for c in langs if c not in _QUICK_TEST_KNOWN_LANGS]
    if unknown:
        return JSONResponse({"error": f"Unknown language codes: {unknown}"}, 400)
    if len(set(langs)) != len(langs):
        return JSONResponse({"error": "Duplicate language codes"}, 400)

    # ── Locate the source (file path or URL) ──────────────────────────
    # Preference order:
    #   1) Original source path if file still exists (uploads/...)
    #   2) source_video.mp4 in the original job's output dir (always copied)
    #   3) Original URL — yt-dlp will re-fetch (cached if possible)
    orig_source = orig.get("source", "")
    source_type = orig.get("source_type", "file")
    src_for_new_jobs: str

    if source_type == "file":
        if orig_source and Path(orig_source).exists():
            src_for_new_jobs = orig_source
        else:
            backup = OUTPUT_DIR / job_id / "source_video.mp4"
            if backup.exists():
                src_for_new_jobs = str(backup)
            else:
                return JSONResponse({
                    "error": "Original source file is gone — can't redub. "
                             "Re-upload it instead.",
                    "original_source": orig_source,
                }, 400)
    else:
        # URL source — reuse the already-downloaded source_video.mp4 from
        # the original job's work dir when it still exists, instead of
        # hitting the network again. Falls back to the URL otherwise.
        cached = OUTPUT_DIR / job_id / "source_video.mp4"
        if cached.exists():
            src_for_new_jobs = str(cached)
            log.info(f"[redub] Reusing downloaded source: {cached}")
        elif orig_source:
            src_for_new_jobs = orig_source
        else:
            return JSONResponse({"error": "Original job has no source URL"}, 400)

    # ── Validate / fallback the Ollama model ──────────────────────────
    chosen_model = model or orig.get("model", "aya-expanse:8b")
    _ok, _installed = await check_ollama()
    if _ok and chosen_model not in _installed:
        chosen_model, _model_err = _resolve_translation_model(
            chosen_model, _installed)
        if _model_err:
            return JSONResponse({"error": _model_err}, 400)

    # ── Build settings (inherit from original, accept overrides) ──────
    settings = {
        "model": chosen_model,
        "whisper_model": whisper_model or orig.get("whisper_model", "large-v3"),
        "voice_preset": voice_preset or orig.get("voice_preset", "auto"),
        "voice_style": voice_style if voice_style is not None else orig.get("voice_style", ""),
        "tts_speed": tts_speed or orig.get("tts_speed", "balanced"),
        "keep_bg": keep_bg if keep_bg is not None else bool(orig.get("keep_bg", False)),
        "speaker_mode": speaker_mode or orig.get("speaker_mode", "main"),
        "auto_denoise": bool(orig.get("auto_denoise", False)),
        "context_hint": orig.get("context_hint", ""),
        "source_lang": orig.get("source_lang", "auto"),
        "voxcpm_cfg": (orig.get("voxcpm_cfg", 0) if voxcpm_cfg is None
                       else voxcpm_cfg),
        "voxcpm_steps": (orig.get("voxcpm_steps", 0) if voxcpm_steps is None
                         else voxcpm_steps),
    }
    settings["voxcpm_cfg"], settings["voxcpm_steps"], _vox_err = (
        validate_voxcpm_overrides(settings["voxcpm_cfg"],
                                  settings["voxcpm_steps"]))
    if _vox_err:
        return JSONResponse({"error": _vox_err}, 400)
    label_base = (orig.get("title") or orig.get("source_label", "")
                  or Path(orig_source).name or job_id)

    def _build_job_dict(jid: str, lang: str, extra: dict | None = None) -> dict:
        d = {
            "id": jid,
            "status": "queued",
            "progress": 0,
            "source": src_for_new_jobs,
            "source_type": source_type,
            "source_label": f"{label_base} -> {lang.upper()} [redub]",
            "target_lang": lang,
            "model": settings["model"],
            "speaker_mode": settings["speaker_mode"],
            "context_hint": settings["context_hint"],
            "voice_style": settings["voice_style"],
            "voice_preset": settings["voice_preset"],
            "voice_mode": orig.get("voice_mode", "preset"),
            "tts_speed": settings["tts_speed"],
            "voxcpm_cfg": settings["voxcpm_cfg"],
            "voxcpm_steps": settings["voxcpm_steps"],
            "whisper_model": settings["whisper_model"],
            "keep_bg": settings["keep_bg"],
            "wizard_mode": "auto",
            "auto_denoise": settings["auto_denoise"],
            "redubbed_from": job_id,
            "created": time.time(),
            "scheduled_at": 0,
            "_pending_args": None,
        }
        # Carry curated metadata / real title over from the original job
        if orig.get("meta"):
            d["meta"] = orig["meta"]
        if orig.get("title"):
            d["title"] = orig["title"]
        if extra:
            d.update(extra)
        return d

    def _pipeline_args(lang: str) -> dict:
        return {
            "source": src_for_new_jobs,
            "source_lang": settings["source_lang"],
            "target_lang": lang,
            "model": settings["model"],
            "keep_bg": settings["keep_bg"],
            "whisper_model": settings["whisper_model"],
            "reference_audio": "",
            "speaker_mode": settings["speaker_mode"],
            "context_hint": settings["context_hint"],
            "voice_style": settings["voice_style"],
            "voice_preset": settings["voice_preset"],
            "tts_speed": settings["tts_speed"],
            "wizard_mode": "auto",
            "auto_denoise": settings["auto_denoise"],
            "voxcpm_cfg": settings["voxcpm_cfg"],
            "voxcpm_steps": settings["voxcpm_steps"],
        }

    # ── Single mode: one job, no batch wrapper ────────────────────────
    if mode == "single":
        jid = uuid.uuid4().hex[:8]
        lang = langs[0]
        jobs[jid] = _build_job_dict(jid, lang)
        save_job(jobs[jid])
        await enqueue_job(jid, _pipeline_args(lang))
        log.info(f"[redub] queued single job {jid} ({lang}) from {job_id}")
        return {"ok": True, "job_id": jid, "redubbed_from": job_id, "target_lang": lang}

    # ── Compare / Showcase: batched fan-out ───────────────────────────
    batch_kind = "showcase" if mode == "showcase" else "quick_test"
    prefix = "sc" if mode == "showcase" else "rd"
    batch_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    batch_label = f"Re-dub · {len(langs)} langs · from {job_id[:6]}"
    job_ids: list = []

    for idx, lang in enumerate(langs):
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = _build_job_dict(jid, lang, extra={
            "batch_id": batch_id,
            "batch_label": batch_label,
            "batch_kind": batch_kind,
            "batch_position": idx,
            "batch_total": len(langs),
        })
        save_job(jobs[jid])
        await enqueue_job(jid, _pipeline_args(lang))
        job_ids.append(jid)

    log.info(f"[redub] {batch_id} ({batch_kind}): enqueued {len(job_ids)} jobs "
             f"from {job_id} ({langs})")
    return {
        "ok": True,
        "batch_id": batch_id,
        "batch_kind": batch_kind,
        "job_ids": job_ids,
        "redubbed_from": job_id,
        "target_langs": langs,
        "mode": mode,
    }


@app.post("/api/showcase/{batch_id}/rebuild")
async def rebuild_showcase(batch_id: str):
    """Manually re-trigger showcase assembly. Useful when the auto-hook
    failed (e.g. ffprobe issue) and the user doesn't want to re-run all
    N dubs from scratch. Deletes any prior showcase.mp4 / error.txt
    first so _maybe_assemble_showcase() will actually rebuild."""
    siblings = [j for j in jobs.values()
                if j.get("batch_id") == batch_id
                and j.get("batch_kind") == "showcase"]
    if not siblings:
        return JSONResponse(
            {"error": f"No showcase batch with id {batch_id}"}, 404)

    incomplete = [j["id"] for j in siblings if j.get("status") != "complete"]
    if incomplete:
        return JSONResponse({
            "error": f"{len(incomplete)} of {len(siblings)} child jobs aren't "
                     f"complete yet — can't assemble",
            "incomplete_job_ids": incomplete,
        }, 400)

    showcase_dir = OUTPUT_DIR / f"showcase_{batch_id}"
    for marker in ("showcase.mp4", "error.txt"):
        p = showcase_dir / marker
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                log.warning(f"[showcase] couldn't remove {p}: {e}")
    _showcase_assembling.discard(batch_id)

    # Fire-and-forget — keep a strong reference in _showcase_tasks so the
    # event loop's weak-ref GC doesn't kill the task mid-execution.
    task = asyncio.create_task(_maybe_assemble_showcase(batch_id))
    _showcase_tasks.add(task)
    task.add_done_callback(_showcase_tasks.discard)
    log.info(f"[showcase] {batch_id}: rebuild scheduled (task={task!r})")
    return {"ok": True, "status": "rebuilding", "batch_id": batch_id}


@app.post("/api/showcase/from_batch/{batch_id}")
async def stitch_batch_as_showcase(batch_id: str):
    """Convert any complete batch (quick_test, compare, …) into a showcase
    reel — no re-dubbing needed. Re-marks child jobs as batch_kind='showcase'
    so the normal assembly path picks them up, then triggers assembly.
    Idempotent: safe to call again if you want to re-stitch."""
    all_in_batch = [j for j in jobs.values() if j.get("batch_id") == batch_id]
    if not all_in_batch:
        return JSONResponse({"error": f"Batch {batch_id!r} not found"}, status_code=404)

    incomplete = [j["id"] for j in all_in_batch if j.get("status") != "complete"]
    if incomplete:
        return JSONResponse({
            "error": f"{len(incomplete)} of {len(all_in_batch)} jobs aren't complete yet",
            "incomplete_job_ids": incomplete,
        }, status_code=400)

    # Upgrade batch_kind so _maybe_assemble_showcase finds these siblings
    for j in all_in_batch:
        if j.get("batch_kind") != "showcase":
            j["batch_kind"] = "showcase"
            save_job(j)

    # Clear any stale showcase output so assembly runs fresh
    showcase_dir = OUTPUT_DIR / f"showcase_{batch_id}"
    for marker in ("showcase.mp4", "error.txt"):
        p = showcase_dir / marker
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                log.warning(f"[showcase] couldn't clear {p}: {e}")
    _showcase_assembling.discard(batch_id)

    task = asyncio.create_task(_maybe_assemble_showcase(batch_id))
    _showcase_tasks.add(task)
    task.add_done_callback(_showcase_tasks.discard)
    log.info(f"[showcase] {batch_id}: stitch-from-batch triggered "
             f"({len(all_in_batch)} jobs)")
    return {"ok": True, "status": "assembling", "batch_id": batch_id,
            "jobs": len(all_in_batch)}


# ═══════════════════════════════════════════════════════════════════════
#  Export for Platform — re-encode dubbed video for specific platforms
# ═══════════════════════════════════════════════════════════════════════
# Each preset defines an ffmpeg -vf filter chain, optional fps override,
# and whether to auto-burn translated subtitles into the frame.
#
# Crop/pad strategy:
#   16:9 outputs → letterbox (black bars) to preserve all content
#   9:16 / 1:1   → scale-to-fill + centre-crop (standard for social)
# ═══════════════════════════════════════════════════════════════════════

_EXPORT_PRESETS: dict = {
    "youtube_1080p": {
        "vf": "scale=1920:1080:force_original_aspect_ratio=decrease,"
              "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
        "fps": None, "burn_subs": False,
    },
    "youtube_4k": {
        "vf": "scale=3840:2160:force_original_aspect_ratio=decrease,"
              "pad=3840:2160:(ow-iw)/2:(oh-ih)/2:black",
        "fps": None, "burn_subs": False,
    },
    "tiktok": {
        "vf": "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "fps": 30, "burn_subs": True,
    },
    "shorts": {
        "vf": "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "fps": 30, "burn_subs": True,
    },
    "reels": {
        "vf": "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "fps": 30, "burn_subs": True,
    },
    "instagram_square": {
        "vf": "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080",
        "fps": None, "burn_subs": False,
    },
    "twitter": {
        "vf": "scale=1280:720:force_original_aspect_ratio=decrease,"
              "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
        "fps": None, "burn_subs": False,
    },
}


@app.post("/api/dub/{job_id}/export")
async def export_for_platform(
    job_id: str,
    preset: str = Form("youtube_1080p"),
    style: str = Form("default"),  # subtitle style — used when preset auto-burns subs
):
    """Re-encode the dubbed video for a specific platform preset.

    Returns JSON: {"ok": True, "url": "..."}  on success,
                  {"error": "...", "detail": "..."}  on failure.
    """
    import subprocess
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if preset not in _EXPORT_PRESETS:
        return JSONResponse(
            {"error": f"Unknown preset '{preset}'. Options: {list(_EXPORT_PRESETS)}"}, 400)

    work = OUTPUT_DIR / job_id
    src_video = work / "dubbed_video.mp4"
    if not src_video.exists():
        return JSONResponse({"error": "Dubbed video not yet generated"}, 400)

    pc = _EXPORT_PRESETS[preset]
    dst_video = work / f"export_{preset}.mp4"
    vf = pc["vf"]

    # For presets that burn subs, append the subtitle filter to the chain
    if pc["burn_subs"]:
        srt_file = work / "translated.srt"
        if not srt_file.exists():
            cp_path = work / "checkpoint_tts_done.json"
            if not cp_path.exists():
                cp_path = work / "checkpoint_translation_done.json"
            if cp_path.exists():
                try:
                    cp = json.loads(cp_path.read_text(encoding="utf-8"))
                    _write_srt_file(cp.get("segments", []), srt_file)
                except Exception as e:
                    return JSONResponse({"error": f"Could not generate SRT: {e}"}, 500)
        if srt_file.exists():
            force_style = SUB_STYLE_MAP.get(style, SUB_STYLE_MAP["default"])
            srt_arg = str(srt_file).replace("\\", "/").replace(":", r"\:")
            vf = f"{vf},subtitles='{srt_arg}':force_style='{force_style}'"

    cmd = [
        "ffmpeg", "-y", "-i", str(src_video),
        "-vf", vf,
        "-c:a", "copy",   # audio pass-through — no re-encode
        "-preset", "fast",
    ]
    if pc["fps"]:
        cmd += ["-r", str(pc["fps"])]
    cmd.append(str(dst_video))

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        return JSONResponse({
            "ok": True,
            "url": f"/outputs/{job_id}/export_{preset}.mp4?v={int(time.time())}",
            "preset": preset,
        })
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or b"").decode("utf-8", errors="replace")[-500:]
        log.warning(f"[export] ffmpeg failed for {job_id}/{preset}: {err_msg}")
        return JSONResponse({
            "error": f"Export failed for preset '{preset}'",
            "detail": err_msg[:300],
        }, 500)


# ═══════════════════════════════════════════════════════════════════════
#  Publish pipeline (Phase 3C) — stage → approve → upload, human-gated
# ═══════════════════════════════════════════════════════════════════════
# job["publish"] is the single source of truth for one platform upload:
#   {platform, status, title, description, tags, file, duplicate_warnings,
#    duplicate_verdict, url, platform_id, error, staged_at, uploaded_at}
# Status flow: staged → approved → uploading → uploaded | failed, with
# cancel legal any time before the bytes start moving. Transitions are
# validated by pipeline.publisher.can_transition; the actual upload runs on
# the separate _upload_queue / _upload_worker (network-bound, never behind
# the GPU job queue). Nothing here ever uploads — /approve is the only way
# to reach the worker, which is the human gate this design exists for.

async def _json_body(request) -> dict:
    """Lenient JSON body parse — publish routes accept an empty body."""
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


@app.post("/api/dub/{job_id}/publish/stage")
async def publish_stage(job_id: str, request: _ScoutRequest):
    """Stage a finished job for publishing — builds metadata, runs the
    duplicate check, optionally exports a platform preset first. NEVER
    uploads. Body JSON (all optional): {platform: "vk", export_preset: str}.

    Re-staging over an existing publish is allowed (it overwrites) unless an
    upload is in flight. A missing VK token downgrades the duplicate check
    to verdict "unchecked" with a "warning" in the response — staging is
    about preparing metadata for human review, not about VK being ready.
    """
    from pipeline.publisher import (
        PublishError, build_publish_meta, can_transition, classify_matches,
        get_uploader, score_match,
    )
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, 404)
    if job.get("status") != "complete":
        return JSONResponse(
            {"error": f"Job is '{job.get('status')}' — only completed jobs "
                      f"can be staged for publishing"}, 400)

    prev = job.get("publish")
    prev_status = prev.get("status") if isinstance(prev, dict) else None
    if not can_transition(prev_status, "stage"):
        return JSONResponse(
            {"error": "An upload is in flight for this job — wait for it "
                      "to finish before re-staging"}, 409)

    body = await _json_body(request)
    platform = str(body.get("platform") or "vk").strip().lower()
    try:
        uploader = get_uploader(platform)
    except PublishError as e:
        return JSONResponse({"error": str(e)}, 400)

    # Optional platform export first (reuses the export route wholesale).
    export_preset = str(body.get("export_preset") or "").strip()
    if export_preset:
        exp = await export_for_platform(job_id, preset=export_preset,
                                        style="default")
        if exp.status_code != 200:
            return exp  # unknown preset / missing video / ffmpeg failure
        file_rel = f"outputs/{job_id}/export_{export_preset}.mp4"
    else:
        file_rel = f"outputs/{job_id}/dubbed_video.mp4"
    if not (BASE / file_rel).is_file():
        return JSONResponse({"error": f"Publish file missing: {file_rel}"}, 400)

    meta = build_publish_meta(job)

    # Duplicate check (4C, warn-only). Score each hit against the staged
    # (usually translated) title AND the original one — recall on VK search
    # is much better in the video's own language.
    warning = None
    dup_matches: list = []
    verdict = "unchecked"
    duration = (job.get("meta") or {}).get("duration") or job.get("duration")
    try:
        dup_matches = await uploader.search_similar(meta.title, duration)
        alt = (job.get("title") or "").strip()
        if alt and alt != meta.title:
            for m in dup_matches:
                m["score"] = round(max(
                    m.get("score", 0.0),
                    score_match(m.get("title", ""), m.get("duration"),
                                meta.title, duration, alt_title=alt),
                ), 3)
            dup_matches.sort(key=lambda m: m.get("score", 0.0), reverse=True)
        verdict = classify_matches(dup_matches)
    except PublishError as e:
        warning = f"Duplicate check skipped: {e}"
        log.info(f"[publish] {job_id}: {warning}")
    except Exception as e:
        warning = f"Duplicate check failed: {type(e).__name__}: {e}"
        log.warning(f"[publish] {job_id}: {warning}")

    job["publish"] = {
        "platform": platform,
        "status": "staged",
        "title": meta.title,
        "description": meta.description,
        "tags": meta.tags,
        "file": file_rel,
        "duplicate_warnings": dup_matches,
        "duplicate_verdict": verdict,
        "url": None,
        "platform_id": None,
        "error": None,
        "staged_at": time.time(),
        "uploaded_at": None,
    }
    save_job(job)
    log.info(f"[publish] {job_id}: staged for {platform} "
             f"(verdict={verdict}, file={file_rel})")
    resp = {"publish": job["publish"]}
    if warning:
        resp["warning"] = warning
    return resp


@app.post("/api/dub/{job_id}/publish/approve")
async def publish_approve(job_id: str, request: _ScoutRequest):
    """The human gate: approve a staged (or failed) publish for upload.

    Body JSON (optional): {title, description} — edits applied before the
    upload is enqueued. This is the ONLY route that feeds the upload worker.
    """
    from pipeline.publisher import can_transition
    job = jobs.get(job_id)
    pub = (job or {}).get("publish")
    if not job or not isinstance(pub, dict):
        return JSONResponse({"error": "Nothing staged for this job"}, 404)
    if not can_transition(pub.get("status"), "approve"):
        return JSONResponse(
            {"error": f"Cannot approve from status '{pub.get('status')}' — "
                      f"only 'staged' or 'failed' publishes can be approved"},
            409)

    body = await _json_body(request)
    if str(body.get("title") or "").strip():
        pub["title"] = str(body["title"]).strip()
    if str(body.get("description") or "").strip():
        pub["description"] = str(body["description"]).strip()

    pub["status"] = "approved"
    pub["error"] = None
    save_job(job)
    await _enqueue_upload(job_id)
    return {"publish": pub}


@app.post("/api/dub/{job_id}/publish/cancel")
async def publish_cancel(job_id: str):
    """Withdraw a staged/approved/failed publish. 409 while uploading."""
    from pipeline.publisher import can_transition
    job = jobs.get(job_id)
    pub = (job or {}).get("publish")
    if not job or not isinstance(pub, dict):
        return JSONResponse({"error": "Nothing staged for this job"}, 404)
    status = pub.get("status")
    if status == "uploading":
        return JSONResponse(
            {"error": "Upload already in flight — it cannot be cancelled "
                      "mid-transfer"}, 409)
    if not can_transition(status, "cancel"):
        return JSONResponse(
            {"error": f"Cannot cancel from status '{status}'"}, 409)
    pub["status"] = "cancelled"
    save_job(job)
    return {"publish": pub}


@app.get("/api/dub/{job_id}/publish")
async def get_publish(job_id: str):
    job = jobs.get(job_id)
    pub = (job or {}).get("publish")
    if not job or not isinstance(pub, dict):
        return JSONResponse({"error": "Nothing staged for this job"}, 404)
    return {"publish": pub}


@app.get("/api/publish/pending")
async def publish_pending():
    """Review inbox: every publish that still needs (or is doing) work."""
    from pipeline.publisher import PUBLISH_PENDING_STATUSES
    pending = []
    for jid, job in jobs.items():
        pub = job.get("publish")
        if isinstance(pub, dict) and pub.get("status") in PUBLISH_PENDING_STATUSES:
            pending.append({
                "job_id": jid,
                "title": (job.get("title") or job.get("source_label")
                          or jid),
                "publish": pub,
            })
    pending.sort(key=lambda p: p["publish"].get("staged_at") or 0, reverse=True)
    return {"pending": pending}


@app.post("/api/scout/check_duplicate")
async def scout_check_duplicate(request: _ScoutRequest):
    """Standalone duplicate probe (4C): does a similar video already exist?

    JSON body: {title, duration_sec?, alt_title?}. Warn-only by design —
    returns {matches, verdict}; a missing/invalid token is a 502 with the
    friendly PublishError message.
    """
    from pipeline.publisher import (
        PublishError, VKUploader, classify_matches, score_match,
    )
    body = await _json_body(request)
    title = str(body.get("title") or "").strip()
    if not title:
        return JSONResponse({"error": "title is required"}, 400)
    duration = body.get("duration_sec")
    alt_title = str(body.get("alt_title") or "").strip() or None
    try:
        matches = await VKUploader().search_similar(title, duration)
    except PublishError as e:
        return JSONResponse({"error": str(e)}, 502)
    if alt_title:
        for m in matches:
            m["score"] = round(max(
                m.get("score", 0.0),
                score_match(m.get("title", ""), m.get("duration"),
                            title, duration, alt_title=alt_title),
            ), 3)
        matches.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    return {"matches": matches, "verdict": classify_matches(matches)}


# ─────────────────────────────────────────────────────────────
# Stage helpers — reusable from run_pipeline AND from resume endpoints
# ─────────────────────────────────────────────────────────────

async def _run_translate_stage(
    job: dict, work: Path, segments: list, effective_src: str,
    target_lang: str, model: str, context_hint: str,
) -> list:
    """Run translation on raw segments + return segments with translated_text."""
    def update(**kwargs):
        if job.get("cancel_requested"):
            raise JobCancelled(f"Job {job['id']} cancelled by user")
        job.update(kwargs); save_job(job)
    update(status="translating", progress=45,
           step_detail=f"Translating to {target_lang}...")
    # translate_segments(segments, target_lang, model, ...)
    # NOT (segments, source_lang, target_lang, model)
    translated = await translate_segments(
        segments, target_lang, model,
        context_hint=context_hint,
        source_lang=effective_src or "",
    )
    # See comment on unload in main pipeline — free VRAM for VoxCPM
    try:
        await unload_ollama_model(model)
    except Exception as e:
        log.warning(f"Failed to unload Ollama model (non-fatal): {e}")
    return translated


async def _run_tts_and_merge_stage(
    job: dict, work: Path, state: dict,
    voice_style: str, voice_preset: str, tts_speed: str,
    ref_path_override: str = "",
    audio_output_name: str = "dubbed_audio.wav",
    tts_subdir: str = "tts_segments",
    preserve_existing_audio_paths: bool = False,
) -> dict:
    """Run TTS + assemble + merge. Returns dict with 'segments' and status.
    - preserve_existing_audio_paths: if True, segments that already have
      a valid audio_path keep their existing file (for per-segment regen).
    """
    job_id = job["id"]
    def update(**kwargs):
        if job.get("cancel_requested"):
            _maybe_terminate_tts_worker()
            raise JobCancelled(f"Job {job_id} cancelled by user")
        job.update(kwargs); save_job(job)

    eff_style, voice_seed, preset_ref = resolve_voice_config(
        voice_preset, voice_style, job_id
    )
    ref_path = ref_path_override
    if preset_ref and os.path.exists(preset_ref):
        ref_path = preset_ref
        log.info(f"[stage] Using file-preset reference: {preset_ref}")

    # Force a FRESH voice_seed whenever this is NOT the first-time run:
    #   - retry_tts (audio_output_name is "dubbed_audio_retry.wav")
    #   - per-segment regen (preserve_existing_audio_paths=True)
    # Without re-seeding, identical inputs to VoxCPM produce byte-identical
    # outputs — so "click retry" would silently do nothing visible to user.
    is_rerun = (audio_output_name != "dubbed_audio.wav"
                or preserve_existing_audio_paths)
    if is_rerun:
        voice_seed = int(time.time() * 1000) % 2_147_483_647
        log.info(f"[stage] Re-run: rolled fresh voice_seed={voice_seed}")

    # ─── VOICE MODE ROUTING ──────────────────────────────────────────
    # VoxCPM has 3 mutually-exclusive modes — the user's UI choice maps
    # to ONE of them. Previously, mixing refs + style prefix was producing
    # garbage: VoxCPM would try to clone the video voice AND literally read
    # out the "(deep male voice, narrator)" style description as text.
    #
    #   Mode 1: File/uploaded reference → Controllable Cloning
    #           Clean reference_wav_path only, NO style prefix.
    #
    #   Mode 2: Style preset (e.g. "male_deep") → Voice Design
    #           "(style description)<text>" — NO reference at all.
    #
    #   Mode 3: No change (auto preset, no upload) → keep state refs
    #           Controllable/Ultimate Cloning from video refs, no prefix.
    mode = "source_refs"
    if ref_path and os.path.exists(ref_path):
        mode = "file_ref"
    elif eff_style and eff_style.strip():
        mode = "voice_design"

    update(
        status="synthesizing", progress=65,
        voice_preset=voice_preset, voice_style=voice_style,
        voice_style_effective=eff_style, voice_seed=voice_seed,
        tts_speed=tts_speed,
        voice_mode=("upload" if mode == "file_ref" else
                    ("custom" if mode == "voice_design" else "source")),
        step_detail="Generating speech...",
    )
    log.info(f"[stage] Voice mode: {mode} "
             f"(ref={bool(ref_path)}, style={bool(eff_style)})")

    # Start from state's refs, then override per mode
    speaker_refs = dict(state.get("speaker_refs", {}))
    speaker_transcripts = dict(state.get("speaker_transcripts", {}))

    if mode == "file_ref":
        log.info(f"[stage] Using file/upload ref for all speakers: {ref_path}")
        target_keys = list(speaker_refs.keys()) or ["SPEAKER_00"]
        for sp in target_keys:
            speaker_refs[sp] = ref_path
            speaker_transcripts[sp] = ""  # Controllable Cloning only
    elif mode == "voice_design":
        # CRITICAL: Voice Design needs NO reference. Without this clear,
        # VoxCPM sees ref + style prefix and produces broken output.
        log.info("[stage] Clearing speaker refs for Voice Design mode")
        speaker_refs = {}
        speaker_transcripts = {}
    else:
        # source_refs mode: use refs extracted from the source video.
        # CRITICAL: speaker_refs in the checkpoint may have been overwritten
        # by an earlier preset/upload (if the user previously dubbed with
        # zhirik.wav, speaker_refs contains zhirik paths). The real source
        # refs were stashed separately as source_speaker_refs — prefer those
        # when available so "retry without changing anything" truly falls
        # back to the original video's voice, not the last-used preset.
        source_refs_stash = state.get("source_speaker_refs") or {}
        if source_refs_stash and is_rerun:
            log.info(f"[stage] Retry: restoring ORIGINAL source refs "
                     f"(not previous preset): {list(source_refs_stash.keys())}")
            speaker_refs = dict(source_refs_stash)
            # Clear transcripts — cross-lingual/controllable cloning only
            for sp in speaker_refs:
                speaker_transcripts[sp] = ""
        else:
            log.info(f"[stage] Using original source speaker refs: "
                     f"{list(speaker_refs.keys())}")

    segments = [dict(s) for s in state["segments"]]
    tts = get_tts_engine()
    # Languages the cloning engines don't cover (e.g. bg) drop to edge-tts.
    tts_used = _tts_engine_for_lang(tts, state.get("target_lang", "ru"))
    target_lang_for_voice = state.get("target_lang", "ru")
    speakers = sorted({s.get("speaker") or "SPEAKER_00" for s in segments})

    # A casting map overrides every whole-job mode above — see the matching
    # block in _stage_tts. Retries and per-segment regens go through here, so
    # without this a re-run of a cast job would quietly recast it.
    cast = dict(state.get("speaker_voice_map") or {})
    if cast and isinstance(tts_used, VoxCPMSynthesizer):
        speaker_refs, speaker_transcripts, cast_report = await _resolve_casting(
            cast, speakers, state, job_id, tts_used, target_lang_for_voice,
        )
        mode = "cast"
        update(voice_mode="cast")
        log.info(f"[stage] Cast {len(speaker_refs)}/{len(speakers)} speaker(s): "
                 + ", ".join(f"{r['speaker']}={r['used']}" for r in cast_report))

    # Draw the designed voice once and clone from it, rather than designing
    # it afresh on every line. See DESIGNED_VOICES_DIR for why the seed alone
    # does not hold a voice still.
    if mode == "voice_design" and isinstance(tts_used, VoxCPMSynthesizer):
        designed = await _blocking(
            _materialize_designed_voice, tts_used, eff_style, voice_seed,
            target_lang_for_voice,
        )
        if designed:
            speaker_refs = {sp: designed for sp in speakers}
            speaker_transcripts = {sp: "" for sp in speakers}
            mode = "designed_ref"
            log.info(f"[stage] Designed voice materialised: "
                     f"{os.path.basename(designed)}")
        else:
            # Fall back to the per-segment style prefix. It goes on
            # `tts_text`, never `translated_text`: the SRT, the translation
            # editor and the assembler's emotion-tag heuristic all read the
            # latter and must not see a model instruction.
            log.warning("[stage] Could not materialise the designed voice; "
                        "falling back to per-segment voice design")
            style = eff_style.strip().strip("()")
            for s in segments:
                base = s.get("translated_text") or s.get("text", "")
                if base and not base.startswith("("):
                    s["tts_text"] = f"({style}){base}"

    # Preserve-mode: skip TTS for segments that already have valid audio
    if preserve_existing_audio_paths:
        todo, keep = [], []
        for s in segments:
            ap = s.get("audio_path", "")
            if ap and os.path.exists(ap):
                keep.append(s)
            else:
                todo.append(s)
        log.info(f"[stage] Preserving {len(keep)} existing segments, "
                 f"synthesizing {len(todo)} new")
        synth_input = todo
    else:
        synth_input = segments

    tts_dir = str(work / tts_subdir)
    total = len(synth_input)
    def synth_progress(done, total_inner):
        pct = 65 + int((done / max(total_inner, 1)) * 20)
        update(progress=min(pct, 85),
               step_detail=f"Synthesizing: {done}/{total_inner}")

    if total > 0:
        if isinstance(tts_used, VoxCPMSynthesizer):
            # Determine cross-lingual from state (may be missing from older
            # checkpoints — in that case assume cross-lingual as a safer default
            # since that's the common dubbing use-case)
            src_lang = state.get("effective_src") or state.get("source_lang", "en")
            tgt_lang = state.get("target_lang", "ru")
            synth_input = tts_used.synthesize_segments(
                synth_input, tts_dir,
                speaker_refs=speaker_refs,
                speaker_transcripts=speaker_transcripts,
                progress_callback=synth_progress,
                voice_seed=voice_seed,
                tts_speed=tts_speed,
                is_cross_lingual=(src_lang != tgt_lang),
                target_lang=tgt_lang,
                # Redub / retry_tts / per-segment regen all reach synthesis
                # through here, and each writes the override onto the job
                # before calling — so the job dict is the single place to
                # read it from, rather than a fourth positional parameter.
                cfg_override=job.get("voxcpm_cfg"),
                steps_override=job.get("voxcpm_steps"),
            )
        else:
            synth_input = await tts_used.synthesize_segments_async(
                synth_input, tts_dir, state.get("target_lang", "ru"),
                progress_callback=synth_progress,
            )

    # Re-merge synth_input back into segments list if preserve-mode
    if preserve_existing_audio_paths:
        by_idx = {s.get("idx"): s for s in segments}
        for s in synth_input:
            if s.get("idx") in by_idx:
                by_idx[s["idx"]].update(s)
        segments = list(by_idx.values())

    synth_ok = sum(1 for s in segments if s.get("audio_path"))
    if synth_ok == 0:
        raise RuntimeError("All TTS synthesis failed - check model/GPU")
    update(progress=85, step_detail=f"Synthesized {synth_ok}/{len(segments)}")

    update(status="assembling", progress=88, step_detail="Assembling dubbed audio...")
    dubbed_wav = str(work / audio_output_name)
    assemble_dubbed_audio(segments, state["duration"], dubbed_wav,
                          tts_used.sample_rate, apply_loudnorm=True)
    _save_placements(work, segments)

    update(status="merging", progress=93, step_detail="Rendering final video...")
    output_mp4 = str(work / "dubbed_video.mp4")
    merge_audio_video(
        state["video_path"], dubbed_wav, output_mp4,
        state.get("bg_audio_path", "") if state.get("keep_bg") else "",
    )

    # Update the tts_done checkpoint so next regen starts from current audio.
    # Existing qa_score/tts_tier on regenerated segments is preserved from the
    # worker (s.get("qa_score") is set by synthesizer); for non-regenerated
    # segments, read from the previous checkpoint to keep QA badges intact.
    prev = _load_checkpoint(job_id, stage="tts_done") or {}
    prev_by_idx = {seg["idx"]: seg for seg in prev.get("segments", [])}

    def _meta(s, i):
        # Prefer fresh worker values on regenerated segments; else pull from prev
        pidx = s.get("idx", i)
        fallback = prev_by_idx.get(pidx, {})
        qa = s.get("qa_score")
        if qa is None: qa = fallback.get("qa_score")
        tier = s.get("tts_tier")
        if tier is None: tier = fallback.get("tts_tier")
        return qa, tier

    _save_checkpoint(job_id, work, stage="tts_done", data={
        **state,
        "segments": [
            (lambda qa_tier: {
                "idx": s.get("idx", i),
                "start": s["start"], "end": s["end"],
                "text": s["text"],
                "translated_text": s.get("translated_text", ""),
                "speaker": s.get("speaker", "SPEAKER_00"),
                "audio_path": s.get("audio_path", ""),
                "qa_score": qa_tier[0],
                "tts_tier": qa_tier[1],
            })(_meta(s, i))
            for i, s in enumerate(segments)
        ],
    })

    update(
        status="complete", progress=100,
        step_detail="Done!",
        output_url=f"/outputs/{job_id}/dubbed_video.mp4?v={int(time.time())}",
        completed_at=time.time(),
    )
    log.info(f"[stage] Pipeline complete: {output_mp4}")
    return {"segments": segments, "output_url": f"/outputs/{job_id}/dubbed_video.mp4"}


async def retry_tts_pipeline(job_id: str, voice_style: str, voice_preset: str,
                              tts_speed: str, ref_path: str):
    """Re-runs ONLY TTS + assemble + merge using the previously-saved
    translation/transcription. Much faster than re-running the full pipeline."""
    job = jobs.get(job_id)
    if not job:
        return
    work = OUTPUT_DIR / job_id
    state = _load_checkpoint(job_id, "translation_done") or \
            _load_checkpoint(job_id, "tts_done")
    if not state:
        job["status"] = "error"
        _set_job_error(job, "No saved state to retry - run full pipeline once first",
                       stage_id="tts")
        save_job(job)
        return
    try:
        await _run_tts_and_merge_stage(
            job, work, state,
            voice_style=voice_style, voice_preset=voice_preset, tts_speed=tts_speed,
            ref_path_override=ref_path,
            audio_output_name="dubbed_audio_retry.wav",
            tts_subdir="tts_segments_retry",
        )
    except Exception as e:
        log.error(f"[retry] Failed: {e}", exc_info=True)
        _set_job_error(job, e, stage_id="tts")
        job["status"] = "error"
        save_job(job)





@app.post("/api/dub/{job_id}/retry_tts")
async def retry_tts(
    job_id: str,
    voice_style: str = Form(""),
    voice_preset: str = Form("auto"),
    tts_speed: str = Form("balanced"),
    reference: Optional[UploadFile] = File(None),
    # None = keep whatever the job already ran with; 0 = drop back to global.
    voxcpm_cfg: Optional[float] = Form(None),
    voxcpm_steps: Optional[int] = Form(None),
):
    """Re-runs only TTS + merge stages using saved state from a completed job.
    Much faster than re-running the full pipeline (no download, transcribe,
    translate steps)."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)

    # _run_tts_and_merge_stage reads the overrides off the job dict, so a
    # per-retry change is applied by persisting it before the run starts.
    if voxcpm_cfg is not None or voxcpm_steps is not None:
        _cfg, _steps, _vox_err = validate_voxcpm_overrides(
            jobs[job_id].get("voxcpm_cfg", 0) if voxcpm_cfg is None else voxcpm_cfg,
            jobs[job_id].get("voxcpm_steps", 0) if voxcpm_steps is None else voxcpm_steps,
        )
        if _vox_err:
            return JSONResponse({"error": _vox_err}, 400)
        jobs[job_id]["voxcpm_cfg"] = _cfg
        jobs[job_id]["voxcpm_steps"] = _steps
        save_job(jobs[job_id])

    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_retry_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    # A re-run must not display the previous run's failure while it works.
    _clear_job_error(jobs[job_id])
    save_job(jobs[job_id])

    asyncio.create_task(retry_tts_pipeline(
        job_id, voice_style, voice_preset, tts_speed, ref_path,
    ))
    return {"ok": True, "job_id": job_id}


# ─────────────────────────────────────────────────────────────
# API: Wizard / Checkpoint Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/dub/{job_id}/checkpoint/{stage}")
async def get_checkpoint(job_id: str, stage: str):
    """Return the contents of a saved checkpoint for the UI to display.
    Useful for editable transcript/translation review screens."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    cp = _load_checkpoint(job_id, stage)
    if not cp:
        return JSONResponse({"error": f"Checkpoint '{stage}' not found"}, 404)

    # Expose speaker_refs as {speaker_id: basename} — the UI shows the
    # filename so the user knows which ref is active and can replace it.
    # We don't send absolute paths (server-internal).
    refs_summary = {}
    for spk, path in (cp.get("speaker_refs") or {}).items():
        if path and os.path.exists(path):
            try:
                import soundfile as _sf
                info = _sf.info(path)
                refs_summary[spk] = {
                    "filename": os.path.basename(path),
                    "duration": round(info.frames / info.samplerate, 1),
                    "is_user_upload": os.path.basename(path).startswith("user_"),
                }
            except Exception:
                refs_summary[spk] = {
                    "filename": os.path.basename(path),
                    "duration": None, "is_user_upload": False,
                }

    return {
        "stage": cp.get("stage"),
        "saved_at": cp.get("saved_at"),
        "target_lang": cp.get("target_lang"),
        "duration": cp.get("duration"),
        "segments": cp.get("segments", []),
        "speaker_refs": refs_summary,
    }


# ─────────────────────────────────────────────────────────────
# API: Per-stage retry & observability
# ─────────────────────────────────────────────────────────────
# The pipeline checkpoints its context after every stage, so any stage can
# be re-run in isolation:
#
#   POST /api/dub/{id}/retry_stage/diarize   {"skip_diarization": true}
#       → reloads the transcribe checkpoint, skips speaker detection,
#         continues through translate/tts/merge. Nothing before diarize
#         is recomputed.
#
#   POST /api/dub/{id}/retry_stage/translate
#        {"model": "qwen2.5:14b", "translate_failed_only": true}
#       → keeps every segment that already translated cleanly and
#         re-submits only the failures to the new model.
#
# GET /api/dub/{id}/stages returns the same picture the UI renders: which
# stages are done/stale/failed, what each produced, how long it took, and
# what CPU/GPU it burned while doing it.

# Only these ctx keys may be set by a retry request. Anything else in the
# override payload is ignored — the ctx also holds resolved artifact paths
# (video_path, audio_16k, speaker_refs…) and letting a client overwrite
# those would point the pipeline at arbitrary files on disk.
_RETRY_OVERRIDE_KEYS = {
    "source_lang", "target_lang", "model", "keep_bg", "whisper_model",
    "speaker_mode", "context_hint", "voice_style", "voice_preset",
    "tts_speed", "auto_denoise", "wizard_mode", "reference_audio",
    "skip_diarization", "translate_failed_only", "tts_keep_existing",
    "voxcpm_cfg", "voxcpm_steps",
}

# Retry overrides that are numbers handed straight to VoxCPM rather than
# strings the pipeline interprets. Everything in _RETRY_OVERRIDE_KEYS is
# otherwise taken as-is from the request body, which is fine for a model
# name and is not for these.
_RETRY_NUMERIC_KEYS = ("voxcpm_cfg", "voxcpm_steps")

# Declarative retry controls, rendered generically by the UI so a new
# knob only has to be added here (plus honoured in the stage handler).
STAGE_RETRY_OPTIONS = {
    "extract": [
        {"key": "auto_denoise", "type": "bool", "label": "Denoise audio",
         "hint": "FFT noise reduction before transcription"},
        {"key": "keep_bg", "type": "bool", "label": "Keep background",
         "hint": "Separate and re-mix music/SFX"},
    ],
    "transcribe": [
        {"key": "whisper_model", "type": "select", "label": "Whisper model",
         "choices": ["large-v3", "large-v2", "medium", "small", "base", "tiny"],
         "hint": "Smaller = faster, less accurate"},
        {"key": "source_lang", "type": "select", "label": "Source language",
         "choices": ["auto", "en", "ru", "es", "fr", "de", "it", "pt", "ja",
                     "ko", "zh", "ar", "hi", "tr", "pl", "uk", "nl", "sv",
                     "th", "vi", "cs", "ro", "hu", "bg", "el", "fi", "id",
                     "no", "da", "bn", "ur", "fa", "he", "sw", "tl", "ms",
                     "ta", "te", "mr", "gu", "kn", "ml", "sk", "hr", "sr",
                     "sl", "lt", "lv", "et", "ca", "is", "af", "mk", "sq",
                     "bs", "cy", "kk", "az", "uz", "ka", "mn", "ne",
                     "si", "my", "km", "lo"]},
    ],
    "diarize": [
        {"key": "skip_diarization", "type": "bool", "label": "Skip diarization",
         "hint": "Use one fallback voice reference — the fix when pyannote "
                 "has no HF_TOKEN or fails to load"},
        {"key": "speaker_mode", "type": "select", "label": "Speakers",
         "choices": ["main", "all"], "hint": "main = dub everyone as the "
         "dominant speaker"},
    ],
    "translate": [
        {"key": "model", "type": "model", "label": "Translation model",
         "hint": "Swap the LLM without redoing transcription"},
        {"key": "translate_failed_only", "type": "bool",
         "label": "Only failed segments",
         "hint": "Keep good translations, retry the ones that came back empty"},
        {"key": "context_hint", "type": "text", "label": "Context hint",
         "hint": "Domain/terminology guidance for the model"},
    ],
    "tts": [
        {"key": "voice_preset", "type": "voice", "label": "Voice preset"},
        {"key": "tts_speed", "type": "select", "label": "Quality",
         "choices": ["fast", "balanced", "quality"]},
        {"key": "tts_keep_existing", "type": "bool",
         "label": "Only missing segments",
         "hint": "Keep rendered audio, synthesize just the gaps"},
        {"key": "voxcpm_cfg", "type": "number", "label": "Guidance (cfg)",
         "min": 1, "max": 3, "step": 0.1,
         "hint": "Higher sticks closer to the text. Blank = global setting"},
        {"key": "voxcpm_steps", "type": "number", "label": "Inference steps",
         "min": 4, "max": 24, "step": 1,
         "hint": "More steps, better quality, slower. Blank = global setting"},
    ],
    "assemble": [],
    "merge": [],
    "download": [],
}

# Statuses that mean "the pipeline currently owns this job" — retrying
# under them would race the running stage over the same files.
_BUSY_STATUSES = {
    "preparing", "queued", "running", "resuming", "downloading", "extracting",
    "transcribing", "diarizing", "translating", "synthesizing",
    "assembling", "merging",
}


def _artifact_entry(job_id: str, work: Path, path: str, label: str = "") -> Optional[dict]:
    """Describe one artifact file: size, mtime, and a URL when servable."""
    if not path:
        return None
    p = Path(path)
    exists = p.exists()
    entry = {
        "label": label or p.name,
        "name": p.name,
        "exists": exists,
        "size_mb": None,
        "url": None,
    }
    if exists:
        try:
            entry["size_mb"] = round(p.stat().st_size / 1024 / 1024, 2)
            entry["modified"] = p.stat().st_mtime
        except OSError:
            pass
        # Files inside the job's work dir are already served by the
        # /outputs static mount; anything else (uploads) stays server-side.
        try:
            rel = p.resolve().relative_to(work.resolve())
            entry["url"] = f"/outputs/{job_id}/{rel.as_posix()}"
        except (ValueError, OSError):
            pass
    return entry


def _stage_artifacts(job_id: str, work: Path, spec: dict, cp: Optional[dict]) -> list:
    """Artifacts a stage produced, read out of its own checkpoint."""
    out = []
    seen = set()
    if cp:
        for key in spec.get("artifacts", []):
            e = _artifact_entry(job_id, work, cp.get(key) or "", label=key)
            if e:
                out.append(e)
                seen.add(e["name"])
    # Well-known output filenames. Jobs created before stage checkpoints
    # existed have no `srt_path`/`dubbed_wav` key in their checkpoint, but
    # the files are sitting right there in the work dir — find them by name
    # so the panel isn't empty for every pre-existing job.
    for fname in spec.get("artifact_files", []):
        if fname in seen:
            continue
        p = work / fname
        if p.exists():
            e = _artifact_entry(job_id, work, str(p))
            if e:
                out.append(e)
                seen.add(fname)
    # Directory artifacts (speaker refs, per-segment TTS wavs) are summarized
    # rather than listed — a 400-segment job would otherwise return 400 rows.
    dirname = spec.get("artifact_dir")
    if dirname:
        d = work / dirname
        if d.is_dir():
            files = [f for f in d.iterdir() if f.is_file()]
            total = sum(f.stat().st_size for f in files if f.exists())
            out.append({
                "label": dirname,
                "name": dirname,
                "exists": True,
                "is_dir": True,
                "file_count": len(files),
                "size_mb": round(total / 1024 / 1024, 2),
                "url": f"/outputs/{job_id}/{dirname}/",
            })
    return out


# Which quality-rollup entry (pipeline/quality.py stage names) colors which
# pipeline stage in the UI timeline. Timing lives on `assemble` (stretch and
# drift happen there); loudness on `merge` (it measures the final track).
_STAGE_QUALITY_KEY = {
    "transcribe": "asr",
    "translate": "translation",
    "tts": "tts",
    "assemble": "timing",
    "merge": "loudness",
}


def build_stage_report(job_id: str) -> dict:
    """Per-stage status, artifacts and timings for one job.

    Stage state is derived from three sources that can disagree, in this
    priority order:
      1. the job's live status  → "running" for the stage in flight
      2. metrics.json           → "failed" if the last attempt errored
      3. checkpoint on disk     → "done", else "pending"

    A "stale" state is reported when a stage's checkpoint is older than an
    upstream one: retrying `translate` alone leaves the previous `tts_done`
    file on disk, and it would be misleading to show that as current.
    """
    job = jobs.get(job_id, {})
    work = OUTPUT_DIR / job_id
    metrics = load_metrics(work)
    stage_metrics = metrics.get("stages", {})
    live_stage = job.get("stage_id")
    job_status = job.get("status", "")
    is_busy = job_status in _BUSY_STATUSES

    stages = []
    newest_upstream = 0.0
    for i, spec in enumerate(PIPELINE_STAGES):
        sid = spec["id"]
        cp = _load_checkpoint(job_id, spec["checkpoint"])
        m = stage_metrics.get(sid) or {}
        saved_at = (cp or {}).get("saved_at") or 0.0

        if cp:
            # Within one run, checkpoints are always written in stage order,
            # so a downstream timestamp that predates an upstream one can only
            # mean the upstream stage was re-run on its own afterwards.
            state = "stale" if saved_at < newest_upstream else "done"
        else:
            state = "pending"
        if m.get("status") == "error" and not cp:
            state = "failed"
        if job.get("failed_stage") == sid and job_status == "error":
            state = "failed"
        if is_busy and live_stage == sid:
            state = "running"
        newest_upstream = max(newest_upstream, saved_at)

        # Retryable = we have (or don't need) the input state for this stage.
        if i == 0:
            can_retry = bool(job.get("source"))
        else:
            can_retry = _stage_input_state(job_id, sid) is not None

        detail = m.get("detail", {}) or {}
        # A stage can succeed and still not have done what the user assumes —
        # diarization falling back to a weaker model is the motivating case.
        # `degraded` is what lets the UI say "done, but…" without inventing a
        # fourth value for `state` that every existing consumer would have to
        # learn.
        st_notices = [n for n in detail.get("notices", []) if isinstance(n, dict)]
        stages.append({
            "id": sid,
            "label": spec["label"],
            "hint": spec["hint"],
            "index": i,
            "checkpoint": spec["checkpoint"],
            "state": state,
            "has_checkpoint": bool(cp),
            "saved_at": saved_at or None,
            "can_retry": can_retry and not is_busy,
            "options": STAGE_RETRY_OPTIONS.get(sid, []),
            "artifacts": _stage_artifacts(job_id, work, spec, cp),
            "duration_sec": m.get("duration_sec"),
            "attempt": m.get("attempt"),
            "last_status": m.get("status"),
            "last_run_at": m.get("finished_at"),
            "error": m.get("error"),
            "resources": m.get("resources", {}),
            "detail": detail,
            "notices": st_notices,
            "degraded": bool(st_notices) and state in ("done", "stale"),
        })

    # Per-stage quality chips for the UI timeline. Read ONLY the stored
    # rollup (job["quality"], persisted by GET /api/dub/{id}/quality) —
    # never recompute here: this report is polled while jobs run.
    qual = job.get("quality") or {}
    qstages = qual.get("stages") or {}
    for s in stages:
        q = qstages.get(_STAGE_QUALITY_KEY.get(s["id"], ""))
        if q and q.get("available"):
            s["quality_score"] = q.get("score")

    # The failed stage's row should carry the job's structured error even
    # when metrics.json has nothing for it (e.g. a crash before the stage
    # timer wrote its record).
    last_error = job.get("last_error")
    if last_error and job_status == "error":
        for s in stages:
            if s["id"] == last_error.get("stage") and not s["error"]:
                s["error"] = last_error.get("message")

    total = sum(s["duration_sec"] or 0 for s in stages)
    return {
        "job_id": job_id,
        "job_status": job_status,
        "busy": is_busy,
        "current_stage": live_stage,
        "failed_stage": job.get("failed_stage"),
        "last_error": last_error,
        "total_duration_sec": round(total, 3),
        "gpu_backend": gpu_backend(),
        "quality_overall": qual.get("overall"),
        "stages": stages,
        "notices": merge_notices(*[s["notices"] for s in stages]),
    }


def _read_user_glossary() -> dict:
    """Raw contents of presets/user_glossary.json, or {} if absent/broken."""
    if not USER_GLOSSARY_FILE.exists():
        return {}
    try:
        with open(USER_GLOSSARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning(f"[glossary] could not read for flags: {e}")
        return {}


@app.get("/api/dub/{job_id}/flags")
async def get_job_flags(job_id: str, max_flags: int = 5):
    """The handful of spans worth a human's attention before recording.

    Recomputed on every call rather than stored, so it always reflects edits
    already applied through /edit_translations — and so the heuristic in
    pipeline/flags.py can be tuned without invalidating anything on disk.
    An empty `flags` list is the expected result for a clean transcript.
    """
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, 404)
    cp = _load_checkpoint(job_id, "translation_done")
    if not cp:
        return JSONResponse(
            {"error": "No translation checkpoint yet — nothing to review"}, 404)

    target_lang = cp.get("target_lang") or job.get("target_lang") or ""
    source_lang = (cp.get("effective_src") or cp.get("source_lang")
                   or job.get("source_lang") or "")
    if source_lang == "auto":
        source_lang = ""
    try:
        flags = flag_segments(
            cp.get("segments") or [],
            target_lang=target_lang,
            source_lang=source_lang,
            glossary=_read_user_glossary(),
            # Not `max_flags or 5`: zero is a real request for none, and `0 or
            # 5` would quietly answer it with the default five instead.
            max_flags=max(0, min(int(max_flags), 20)),
        )
    except Exception as e:
        log.warning(f"[flags] job={job_id} failed: {e}")
        return JSONResponse({"error": f"Could not compute flags: {e}"}, 500)

    return {
        "job_id": job_id,
        "target_lang": target_lang,
        "source_lang": source_lang,
        "count": len(flags),
        "flags": flags,
        # The original audio, for the review screen's "hear this bit" —
        # #t=start,end on this file lands on the right moment because VAD
        # writes its trimmed copy to a different name and segment times are
        # remapped back to the original timeline.
        "audio_url": f"/outputs/{job_id}/audio_16k.wav",
    }


@app.get("/api/dub/{job_id}/stages")
async def get_job_stages(job_id: str):
    """Stage-by-stage state, artifacts, timings and resource usage."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    return build_stage_report(job_id)


@app.get("/api/dub/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Raw metrics.json — latest run per stage plus the full attempt history.

    Useful for answering "why was this run 3 minutes slower than the last
    one" without re-instrumenting anything: every attempt keeps its own
    duration and CPU/GPU sample summary.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    data = load_metrics(OUTPUT_DIR / job_id)
    stages = data.get("stages", {})
    ordered = [
        {"stage": sid, **stages[sid]} for sid in STAGE_ORDER if sid in stages
    ]
    total = sum(s.get("duration_sec") or 0 for s in ordered)
    return {
        "job_id": job_id,
        "total_duration_sec": round(total, 3),
        "gpu_backend": gpu_backend(),
        "gpu_now": gpu_snapshot(),
        "stages": ordered,
        "history": data.get("history", []),
    }


def _quality_inputs(job_id: str):
    """Gather everything the quality scorers need from disk.

    Segments come from the NEWEST checkpoint that has any — the same
    walk-backwards idea as `_latest_checkpoint`, but skipping checkpoints
    whose payload carries no segment list (download/extract). Placements and
    the loudnorm measurement are optional; scorers report available=False
    when they are missing.
    """
    segments: list = []
    seg_stage = None
    for stage in CHECKPOINT_ORDER_DESC:
        cp = _load_checkpoint(job_id, stage)
        if cp and cp.get("segments"):
            segments, seg_stage = cp["segments"], stage
            break
    work = OUTPUT_DIR / job_id
    placements = _load_placements(work)
    loudnorm = (((load_metrics(work).get("stages") or {}).get("assemble") or {})
                .get("detail") or {}).get("loudnorm")
    return segments, seg_stage, placements, loudnorm


@app.get("/api/dub/{job_id}/metadata")
async def get_job_metadata(job_id: str):
    """Source metadata plus its translation into this job's target language.

    Served on demand rather than on the job object: descriptions run to
    thousands of characters, and /api/jobs is polled every couple of seconds
    while a job runs — carrying them there would bloat every poll for text
    almost nobody is looking at.

    Shaped for copy-paste into a re-upload form: `source` is what yt-dlp
    found, `translated` is the same fields in the target language (absent
    when the dub is same-language, or when translation was skipped).
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    job = jobs[job_id]
    meta = job.get("meta") or {}
    mt = job.get("meta_translated") or {}
    return {
        "job_id": job_id,
        "target_lang": job.get("target_lang"),
        "source": {
            "title": meta.get("title") or "",
            "description": meta.get("description") or "",
            "chapters": meta.get("chapters") or [],
            "thumbnail": meta.get("thumbnail") or "",
            "channel": meta.get("channel") or "",
            "webpage_url": meta.get("webpage_url") or "",
            "duration": meta.get("duration"),
        },
        "translated": {
            "title": mt.get("title") or "",
            "description": mt.get("description") or "",
            "chapters": mt.get("chapters") or [],
        } if mt else None,
    }


@app.get("/api/dub/{job_id}/quality")
async def get_job_quality(job_id: str):
    """Full per-stage quality report + 0-100 rollup + actionable verdicts.

    Computed on demand from persisted artifacts (checkpoints,
    tts_placements.json, metrics.json) — nothing is re-transcribed or
    re-rendered. Each verdict's `suggested_action` names an existing route
    (retry_tts / edit_translations / regenerate_segment / retranslate) so
    agents can act on the report mechanically. The small rollup is stored on
    the job as `job["quality"]` once the job is complete, which is what
    `/api/quality/trends` and the stage-report chips read.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    from pipeline.quality import full_report
    segments, seg_stage, placements, loudnorm = await asyncio.to_thread(
        _quality_inputs, job_id)
    if not segments and not placements:
        return JSONResponse(
            {"error": "No checkpoints with segments — job has no artifacts "
                      "to score yet"}, 404)
    report = full_report(segments, placements, loudnorm)
    job = jobs[job_id]
    if job.get("status") == "complete":
        # Recompute-and-overwrite: cheap, and it picks up granular re-runs
        # (retry_tts / regenerate_segment) that changed the checkpoints.
        job["quality"] = report["rollup"]
        save_job(job)
    return {
        "job_id": job_id,
        "segments_from": seg_stage,
        "n_segments": len(segments),
        **report,
    }


@app.get("/api/quality/trends")
async def quality_trends(limit: int = 50):
    """Stored quality rollups across recent completed jobs.

    Reads ONLY `job["quality"]` (persisted by GET /api/dub/{id}/quality on
    completed jobs) — computing full reports for 50 jobs would mean loading
    50 checkpoint files, so jobs without a stored rollup are skipped rather
    than scored here.
    """
    limit = max(1, min(int(limit), 200))
    rows = [
        {
            "job_id": job.get("id"),
            "title": job.get("title") or job.get("source_label")
                     or job.get("source"),
            "target_lang": job.get("target_lang"),
            "created": job.get("created"),
            "quality": job.get("quality"),
        }
        for job in jobs.values()
        if job.get("status") == "complete" and job.get("quality")
    ]
    rows.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return {"jobs": rows[:limit]}


@app.get("/api/dub/{job_id}/audit")
async def get_job_audit(job_id: str):
    """Artifact-loss audit (word coverage, idx integrity, QA verdicts).

    Same engine as `python tools/audit_job.py <id>` — findings carry
    severity loss/warn/info plus the offending segments, and `counts`
    totals them per severity.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    from tools.audit_job import audit_job as _audit_job
    return await asyncio.to_thread(_audit_job, OUTPUT_DIR / job_id)


def _fresh_run_ctx(job: dict) -> dict:
    """Pipeline context for a from-scratch run of an existing job.

    The same fields the original /api/dub submission produced, rebuilt from
    the persisted job — used by retry_stage for a from-download re-run and
    by attach_source when a manually-downloaded file replaces the download
    stage entirely.
    """
    return {
        "source": job.get("source", ""),
        "source_lang": job.get("source_lang", "auto"),
        "target_lang": job.get("target_lang", "ru"),
        "model": job.get("model") or cfg.translation_model,
        "keep_bg": bool(job.get("keep_bg")),
        "whisper_model": job.get("whisper_model") or cfg.whisper_model,
        "reference_audio": "",
        "speaker_mode": job.get("speaker_mode", "main"),
        "context_hint": job.get("context_hint", ""),
        "voice_style": job.get("voice_style", ""),
        "voice_preset": job.get("voice_preset", "auto"),
        "tts_speed": job.get("tts_speed", "balanced"),
        "auto_denoise": bool(job.get("auto_denoise", True)),
        "voxcpm_cfg": job.get("voxcpm_cfg", 0),
        "voxcpm_steps": job.get("voxcpm_steps", 0),
    }


@app.post("/api/dub/{job_id}/retry_stage/{stage}")
async def retry_stage(
    job_id: str,
    stage: str,
    overrides: str = Form("{}"),
    stop_after: str = Form(""),
    reference: Optional[UploadFile] = File(None),
):
    """Re-run one stage (and everything after it) from the previous
    stage's checkpoint.

    `overrides` is a JSON object of pipeline settings to change for this
    run — see STAGE_RETRY_OPTIONS for what each stage accepts. `stop_after`
    optionally halts the run at a later stage instead of going to the end,
    so the user can inspect the result before paying for TTS.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if stage not in STAGE_ORDER:
        return JSONResponse(
            {"error": f"Unknown stage '{stage}'. Valid: {', '.join(STAGE_ORDER)}"},
            400,
        )
    job = jobs[job_id]
    if job.get("status") in _BUSY_STATUSES:
        return JSONResponse(
            {"error": f"Job is {job['status']} — cancel it before retrying a stage"},
            409,
        )
    if stop_after and stop_after not in STAGE_ORDER:
        return JSONResponse({"error": f"Unknown stop_after '{stop_after}'"}, 400)
    if stop_after and _stage_index(stop_after) < _stage_index(stage):
        return JSONResponse(
            {"error": f"stop_after '{stop_after}' comes before '{stage}'"}, 400
        )

    try:
        ov = json.loads(overrides or "{}")
        if not isinstance(ov, dict):
            raise ValueError("overrides must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Bad overrides JSON: {e}"}, 400)

    # Build the input context: the checkpoint written by the previous
    # stage, or the original request for a from-scratch re-run.
    if _stage_index(stage) == 0:
        ctx = _fresh_run_ctx(job)
        if not ctx["source"]:
            return JSONResponse({"error": "Job has no source to re-download"}, 409)
    else:
        state = _stage_input_state(job_id, stage)
        if not state:
            return JSONResponse({
                "error": f"No checkpoint before '{stage}' — this job predates "
                         f"stage checkpoints or never got that far. "
                         f"Re-run from an earlier stage."
            }, 409)
        ctx = dict(state)
        # `stage`/`job_id`/`saved_at` are checkpoint bookkeeping, not context.
        for k in ("stage", "job_id", "saved_at"):
            ctx.pop(k, None)

    # A retry always runs to the end unless told otherwise, so drop any
    # wizard pause inherited from the original run.
    ctx["wizard_mode"] = "auto"
    for k, v in ov.items():
        if k not in _RETRY_OVERRIDE_KEYS:
            log.warning(f"[retry] Ignoring non-overridable key '{k}'")
            continue
        if k in _RETRY_NUMERIC_KEYS:
            try:
                v = coerce_field(k, v) if v not in (None, "", 0, "0") else 0
            except ValueError as e:
                return JSONResponse({"error": str(e)}, 400)
        ctx[k] = v

    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_retry{stage}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        ctx["reference_audio"] = ref_path

    # Marks this as a re-run for the TTS stage: fresh voice seed + restore
    # the original source refs instead of a previously-baked preset.
    ctx["_is_retry"] = True

    _clear_job_error(job)
    job.pop("cancel_requested", None)
    job["status"] = "queued"
    job["step_detail"] = f"Queued — retrying from '{stage}'"
    # Surface the changed settings on the job so History/Result reflect them.
    for k in ("model", "target_lang", "voice_preset", "voice_style",
              "tts_speed", "whisper_model", "context_hint", "speaker_mode",
              "voxcpm_cfg", "voxcpm_steps"):
        if k in ov:
            # ctx carries the coerced value; ov still holds the raw one.
            job[k] = ctx.get(k, ov[k])
    save_job(job)

    log.info(
        f"[retry] job={job_id} stage='{stage}'"
        + (f" → '{stop_after}'" if stop_after else " → end")
        + (f" overrides={ {k: v for k, v in ov.items() if k in _RETRY_OVERRIDE_KEYS} }"
           if ov else "")
    )

    await enqueue_job(job_id, {"__stage_retry__": {
        "ctx": ctx, "start_stage": stage, "stop_after": stop_after,
    }})
    return {
        "ok": True, "job_id": job_id,
        "retry_from": stage, "stop_after": stop_after or None,
    }


@app.post("/api/job/{job_id}/attach_source")
async def attach_source(job_id: str, video: UploadFile = File(...)):
    """Rescue a job whose download failed: accept a manually-downloaded
    video, install it as the job's source, and resume from 'extract'.

    Pairs with the `download_hint` a failed download attaches to the job —
    the user runs one of the suggested yt-dlp commands locally, then drops
    the file here. Deliberately not restricted to errored jobs: any
    non-busy job can have its source replaced and re-run.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    job = jobs[job_id]
    if job.get("status") in _BUSY_STATUSES:
        return JSONResponse(
            {"error": "Job is busy — cancel it before attaching a source"},
            409,
        )
    if not (video and video.filename):
        return JSONResponse({"error": "Provide a video file"}, 400)

    ext = Path(video.filename).suffix or ".mp4"
    tmp_path = UPLOAD_DIR / f"{job_id}_attached{ext}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    why = _validate_attached_video(tmp_path)
    if why:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        log.info(f"[rescue] job={job_id} rejected {video.filename!r}: {why}")
        return JSONResponse({"error": why}, 400)
    duration = _probe_duration(tmp_path)

    work = OUTPUT_DIR / job_id
    work.mkdir(exist_ok=True)
    dest = work / "source_video.mp4"
    shutil.move(str(tmp_path), str(dest))
    size_mb = round(dest.stat().st_size / 1048576, 2)
    # Keep job["source"] as-is (provenance — a re-download retry stays
    # possible), just record how the video actually got here.
    job["rescued_with_upload"] = video.filename

    ctx = _fresh_run_ctx(job)
    ctx["video_path"] = str(dest)
    ctx["duration"] = duration
    ctx["wizard_mode"] = "auto"
    # Write the download checkpoint the download stage would have written,
    # so the stage panel shows it as done and retry_stage("extract") keeps
    # working later.
    _save_checkpoint(job_id, work, stage="download_done",
                     data=_ctx_for_checkpoint(ctx))

    _clear_job_error(job)
    job.pop("cancel_requested", None)
    job["status"] = "queued"
    job["duration"] = round(duration, 1)
    job["step_detail"] = "Queued — resuming from attached video"
    save_job(job)

    log.info(f"[rescue] job={job_id} attached {video.filename!r} "
             f"({size_mb} MB, {duration:.1f}s) — resuming from 'extract'")
    await enqueue_job(job_id, {"__stage_retry__": {
        "ctx": ctx, "start_stage": "extract", "stop_after": "",
    }})
    return {"ok": True, "job_id": job_id, "resumed_from": "extract",
            "duration": duration, "size_mb": size_mb}


@app.post("/api/dub/{job_id}/edit_translations")
async def edit_translations(job_id: str, edits: str = Form(...)):
    """Update the translated_text for one or more segments in the saved
    checkpoint. `edits` is a JSON string: {"<idx>": "new translation", ...}
    After editing, the user should call /continue to proceed to TTS."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    cp = _load_checkpoint(job_id, "translation_done")
    if not cp:
        return JSONResponse({"error": "No translation checkpoint to edit"}, 404)
    try:
        edit_map = json.loads(edits)
        if not isinstance(edit_map, dict):
            raise ValueError("edits must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid edits JSON: {e}"}, 400)

    # Apply edits by segment index (string keys from the JSON)
    n_edited = 0
    for s in cp.get("segments", []):
        key = str(s.get("idx"))
        if key in edit_map:
            new_text = str(edit_map[key]).strip()
            if new_text and new_text != s.get("translated_text"):
                s["translated_text"] = new_text
                n_edited += 1
    # Re-save the translation_done checkpoint with the edits
    work = OUTPUT_DIR / job_id
    _save_checkpoint(job_id, work, stage="translation_done", data=cp)

    # Re-export SRT with the user's edits so .srt download always matches
    # what gets spoken. Also update tts_done if it exists (per-segment
    # regen panel uses it for translated_text display).
    if n_edited > 0:
        try:
            srt_path = str(work / "subtitles.srt")
            write_srt(cp.get("segments", []), srt_path)
        except Exception as e:
            log.warning(f"[edit] SRT re-export failed: {e}")
        tcp = _load_checkpoint(job_id, "tts_done")
        if tcp:
            tcp_segs = tcp.get("segments", [])
            tcp_by_idx = {s.get("idx"): s for s in tcp_segs}
            for seg in cp.get("segments", []):
                t = tcp_by_idx.get(seg.get("idx"))
                if t is not None:
                    t["translated_text"] = seg.get("translated_text", "")
            _save_checkpoint(job_id, work, stage="tts_done", data=tcp)

    log.info(f"[edit] Applied {n_edited} translation edits for job {job_id}")
    return {"ok": True, "edited": n_edited}


@app.post("/api/dub/{job_id}/edit_speaker_ref/{speaker_id}")
async def edit_speaker_ref(
    job_id: str, speaker_id: str,
    reference: UploadFile = File(...),
):
    """Replace the voice-cloning reference for one speaker.
    Used in wizard mode when diarization found a second speaker but only
    got 3-5 seconds of their audio (too short for clean cloning) — the
    user can upload a longer, cleaner clip of them from elsewhere."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if not reference or not reference.filename:
        return JSONResponse({"error": "No file uploaded"}, 400)

    # Pick the earliest checkpoint that already has speaker_refs set; edit it
    # + later checkpoints so the new ref is picked up on /continue or regen.
    work = OUTPUT_DIR / job_id
    ref_dir = work / "speaker_refs"
    ref_dir.mkdir(exist_ok=True)
    ext = Path(reference.filename).suffix.lower() or ".wav"
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        return JSONResponse({"error": f"Unsupported format: {ext}"}, 400)

    # Always store the user's upload as a fresh file so we don't clobber
    # the diarizer-extracted one (allows user to revert later if needed).
    user_ref = str(ref_dir / f"user_{speaker_id}{ext}")
    with open(user_ref, "wb") as f:
        shutil.copyfileobj(reference.file, f)

    n_updated = 0
    for stage in ("transcription_done", "translation_done", "tts_done"):
        cp = _load_checkpoint(job_id, stage)
        if cp and "speaker_refs" in cp:
            cp["speaker_refs"][speaker_id] = user_ref
            _save_checkpoint(job_id, work, stage=stage, data=cp)
            n_updated += 1

    if n_updated == 0:
        return JSONResponse(
            {"error": "No checkpoint has speaker_refs yet"}, 400
        )
    log.info(f"[edit_spk] Replaced ref for {speaker_id} on job {job_id} "
             f"({n_updated} checkpoints updated)")
    return {"ok": True, "speaker_id": speaker_id,
            "new_ref_path": user_ref, "checkpoints_updated": n_updated}


# ═══════════════════════════════════════════════════════════════════════
#  Voice casting — one voice per speaker, auditioned before the GPU runs
# ═══════════════════════════════════════════════════════════════════════
# Every voice control that came before these routes was a whole-job switch:
# a style preset, or one uploaded clip, applied to everyone. On the common
# case — a video with more than one person in it — that is not a choice
# between voices, it is the loss of the other speakers. Diarization would run,
# references would be cut, and then _stage_tts threw them away.
#
# Casting replaces the switch with an assignment: {speaker: voice_id}, stored
# on the job's checkpoints and honoured by both the pipeline stage and the
# retry path. "source" (the default for any speaker left unassigned) keeps
# the voice cut from the video, which is the only default that cannot
# surprise anyone.
#
# The audition matters more than it looks. TTS is by far the most expensive
# stage — a measured 12.25 hours for 1602 segments — and until now the first
# time anyone heard the cast was after all of it. Three preview segments cost
# about a minute.
# ═══════════════════════════════════════════════════════════════════════

def _casting_state(job_id: str) -> Optional[dict]:
    """The newest checkpoint that can support casting, or None.

    Needs translated text (to preview real lines) and speaker labels, so the
    translation checkpoint is the earliest one that qualifies; tts_done is
    preferred when present because it carries the user's per-segment edits.
    """
    return (_load_checkpoint(job_id, "tts_done")
            or _load_checkpoint(job_id, "translation_done"))


def _speaker_rows(state: dict, job_id: str) -> list:
    """Per-speaker summary: how much they say, and a line to preview it with."""
    segs = state.get("segments") or []
    by_speaker: dict = {}
    for sg in segs:
        sp = sg.get("speaker") or "SPEAKER_00"
        row = by_speaker.setdefault(sp, {"segments": 0, "seconds": 0.0, "lines": []})
        row["segments"] += 1
        row["seconds"] += max(0.0, float(sg.get("end", 0)) - float(sg.get("start", 0)))
        text = (sg.get("translated_text") or "").strip()
        if text:
            row["lines"].append((sg.get("idx"), text))

    refs_dir = OUTPUT_DIR / job_id / "speaker_refs"
    out = []
    for sp, row in sorted(by_speaker.items(),
                          key=lambda kv: -kv[1]["segments"]):
        # A mid-length line auditions better than the longest or the
        # shortest: "Да." proves nothing about a voice, and a 40-second
        # paragraph makes the preview as slow as the thing it is meant to
        # save you from.
        lines = sorted(row["lines"], key=lambda t: len(t[1]))
        pick = lines[len(lines) // 2] if lines else (None, "")
        has_ref = (refs_dir / f"ref_{sp}.wav").exists()
        out.append({
            "speaker": sp,
            "segments": row["segments"],
            "seconds": round(row["seconds"], 1),
            "share": round(row["segments"] / max(len(segs), 1), 3),
            "sample_idx": pick[0],
            "sample_text": pick[1][:200],
            "has_source_ref": has_ref,
            "audio_url": (f"/api/job/{job_id}/speaker_ref/{sp}/audio"
                          if has_ref else ""),
            "waveform_url": (f"/api/job/{job_id}/speaker_ref/{sp}/waveform"
                             if has_ref else ""),
        })
    return out


def _validate_cast(cast: dict, speakers: set) -> tuple:
    """Return (clean_map, errors). Unknown speakers and voices are rejected
    rather than dropped — a typo that silently does nothing is worse than a
    400, because you only find out after the render."""
    choices = _voice_choices()
    clean, errors = {}, []
    for sp, vid in (cast or {}).items():
        vid = str(vid or "").strip()
        if sp not in speakers:
            errors.append(f"unknown speaker {sp!r}")
            continue
        if vid in ("", "source"):
            clean[sp] = "source"
        elif vid in choices or vid.startswith("design:"):
            clean[sp] = vid
        else:
            errors.append(f"unknown voice {vid!r} for {sp}")
    return clean, errors


def _persist_cast(job_id: str, cast: dict) -> int:
    """Write the map onto every checkpoint a later run might resume from.

    The same shape as edit_speaker_ref: whichever checkpoint /continue or a
    stage retry picks up, it has to carry the cast, or the re-run silently
    reverts to whole-job voice mode.
    """
    work = OUTPUT_DIR / job_id
    n = 0
    for stage in ("transcription_done", "translation_done", "tts_done"):
        cp = _load_checkpoint(job_id, stage)
        if cp:
            cp["speaker_voice_map"] = dict(cast)
            _save_checkpoint(job_id, work, stage=stage, data=cp)
            n += 1
    return n


@app.get("/api/dub/{job_id}/voice_casting")
async def get_voice_casting(job_id: str):
    """Who speaks in this job, what they are currently cast as, and what
    else they could be cast as."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    state = _casting_state(job_id)
    if not state:
        return JSONResponse(
            {"error": "Nothing to cast yet — this job has not been "
                      "translated. Casting opens at the translation "
                      "checkpoint."}, 409)
    rows = _speaker_rows(state, job_id)
    cast = dict(state.get("speaker_voice_map") or {})
    return {
        "job_id": job_id,
        "status": jobs[job_id].get("status", ""),
        "target_lang": state.get("target_lang", ""),
        "speakers": rows,
        "map": {r["speaker"]: cast.get(r["speaker"], "source") for r in rows},
        "voices": sorted(_voice_choices().values(), key=lambda v: v["name"]),
        "report": jobs[job_id].get("voice_cast_report") or [],
    }


@app.post("/api/dub/{job_id}/voice_casting")
async def set_voice_casting(job_id: str, request: _ScoutRequest):
    """Assign voices to speakers. Body: {"map": {"SPEAKER_00": "male_deep"}}.

    Voice ids are a built-in preset ("male_deep"), a library voice
    ("file:my_clip"), a free-text design ("design:gravelly old sailor"), or
    "source" to keep the voice from the video. Anything not named keeps its
    source voice.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    state = _casting_state(job_id)
    if not state:
        return JSONResponse({"error": "Nothing to cast yet"}, 409)
    body = await _json_body(request)
    speakers = {r["speaker"] for r in _speaker_rows(state, job_id)}
    clean, errors = _validate_cast(body.get("map") or {}, speakers)
    if errors:
        return JSONResponse({"error": "; ".join(errors)}, 400)

    n = _persist_cast(job_id, clean)
    job = jobs[job_id]
    job["speaker_voice_map"] = dict(clean)
    save_job(job)
    log.info(f"[cast] job={job_id} " +
             ", ".join(f"{k}={v}" for k, v in sorted(clean.items())) +
             f" ({n} checkpoints updated)")
    return {"ok": True, "map": clean, "checkpoints_updated": n}


@app.post("/api/dub/{job_id}/voice_preview")
async def preview_voice_casting(job_id: str, request: _ScoutRequest):
    """Synthesize a couple of real lines per speaker in the proposed cast.

    Body: {"map": {...}, "per_speaker": 1}. The map is optional — without it
    the saved cast is previewed. Nothing is persisted: this renders into
    outputs/<job>/voice_preview/ and hands back URLs, so auditioning a cast
    never touches the dub.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    job = jobs[job_id]
    # The preview and the dub want the same GPU, and the worker is a single
    # process holding a single model. Auditioning mid-synthesis would
    # contend for it and slow down the very stage this feature exists to
    # avoid re-running.
    if job.get("status") in ("synthesizing", "assembling", "merging"):
        return JSONResponse(
            {"error": f"Job is {job['status']} — wait for it to pause or "
                      f"finish before previewing voices"}, 409)
    state = _casting_state(job_id)
    if not state:
        return JSONResponse({"error": "Nothing to preview yet"}, 409)

    body = await _json_body(request)
    rows = _speaker_rows(state, job_id)
    speakers = {r["speaker"] for r in rows}
    raw = body.get("map")
    if raw is None:
        cast = dict(state.get("speaker_voice_map") or {})
    else:
        cast, errors = _validate_cast(raw, speakers)
        if errors:
            return JSONResponse({"error": "; ".join(errors)}, 400)
    per_speaker = max(1, min(3, int(body.get("per_speaker") or 1)))

    target_lang = state.get("target_lang", "ru")
    tts_used = _tts_engine_for_lang(get_tts_engine(), target_lang)
    if not isinstance(tts_used, VoxCPMSynthesizer):
        return JSONResponse(
            {"error": f"Voice casting needs a cloning engine; {target_lang} "
                      f"falls back to edge-tts, which has fixed voices"}, 409)

    refs, transcripts, report = await _resolve_casting(
        cast, sorted(speakers), state, job_id, tts_used, target_lang,
    )

    # Pick the lines: the same mid-length rule as the summary, widened when
    # more than one per speaker is asked for.
    by_speaker: dict = {}
    for sg in state.get("segments") or []:
        text = (sg.get("translated_text") or "").strip()
        if text:
            by_speaker.setdefault(sg.get("speaker") or "SPEAKER_00", []).append(sg)
    chosen = []
    for sp in sorted(speakers):
        pool = sorted(by_speaker.get(sp, []), key=lambda g: len(g.get("translated_text", "")))
        if not pool:
            continue
        mid = len(pool) // 2
        for off in range(per_speaker):
            i = mid + off - per_speaker // 2
            if 0 <= i < len(pool):
                chosen.append(pool[i])

    if not chosen:
        return JSONResponse({"error": "No translated lines to preview"}, 409)

    # Fresh directory each time: the worker names files by position, so a
    # shorter second preview would otherwise leave the first one's audio
    # behind and play it back as if it were the new cast.
    out_dir = OUTPUT_DIR / job_id / "voice_preview"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [{"idx": i, "speaker": sg.get("speaker") or "SPEAKER_00",
              "translated_text": sg.get("translated_text", ""),
              "text": sg.get("text", "")}
             for i, sg in enumerate(chosen)]
    src_lang = state.get("effective_src") or state.get("source_lang", "en")
    t0 = time.time()
    done = await _blocking(
        tts_used.synthesize_segments, specs, str(out_dir),
        speaker_refs=refs, speaker_transcripts=transcripts,
        voice_seed=state.get("voice_seed"),
        tts_speed=state.get("tts_speed", "balanced"),
        is_cross_lingual=(src_lang != target_lang),
        target_lang=target_lang,
    )
    took = round(time.time() - t0, 1)

    samples = []
    for spec, sg in zip(done or [], chosen):
        path = spec.get("audio_path") or ""
        if not path or not os.path.exists(path):
            continue
        samples.append({
            "speaker": spec.get("speaker"),
            "voice": cast.get(spec.get("speaker"), "source"),
            "text": spec.get("translated_text", "")[:200],
            "start": sg.get("start"),
            "url": f"/outputs/{job_id}/voice_preview/{os.path.basename(path)}"
                   f"?v={int(time.time())}",
        })
    log.info(f"[cast] Previewed {len(samples)}/{len(specs)} line(s) for "
             f"job={job_id} in {took}s")
    if not samples:
        return JSONResponse(
            {"error": "Preview produced no audio — check the TTS engine logs",
             "report": report}, 500)
    return {"ok": True, "samples": samples, "report": report,
            "took_sec": took}


# ═══════════════════════════════════════════════════════════════════════
#  Transcript export
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/dub/{job_id}/transcripts.txt")
async def download_transcripts_txt(job_id: str):
    """Export side-by-side transcript as plain text. Useful for content
    creators who want to copy/paste into captions, descriptions, etc.

    Format:
        === SEGMENT 1 (0.0s → 5.6s · SPEAKER_00) ===
        EN: Original source text
        RU: Translated text (with any user edits applied)
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)

    # Prefer tts_done (has user edits from per-segment regens), fall back
    # to translation_done. If neither exists, return 404.
    cp = _load_checkpoint(job_id, "tts_done") or _load_checkpoint(job_id, "translation_done")
    if not cp:
        return JSONResponse({"error": "No completed transcript yet"}, 404)

    segs = cp.get("segments", [])
    if not segs:
        return JSONResponse({"error": "No segments"}, 404)

    target = cp.get("target_lang", "ru").upper()
    lines = [
        f"GoChiDUBB Studio transcript export · job {job_id}",
        f"Target language: {target} · {len(segs)} segments",
        "=" * 60, "",
    ]
    for s in segs:
        idx = s.get("idx", 0) + 1
        start = s.get("start", 0.0)
        end = s.get("end", 0.0)
        spk = s.get("speaker", "SPEAKER_00")
        lines.append(f"=== #{idx} · {start:.1f}s → {end:.1f}s · {spk} ===")
        lines.append(f"EN: {s.get('text', '').strip()}")
        lines.append(f"{target}: {s.get('translated_text', '').strip()}")
        lines.append("")

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content="\n".join(lines),
        headers={
            "Content-Disposition": f'attachment; filename="transcripts_{job_id}.txt"'
        },
    )


# ═══════════════════════════════════════════════════════════════════════
#  Lip-sync (Wav2Lip) — optional post-processing
# ═══════════════════════════════════════════════════════════════════════
# Wav2Lip (Rudrabha 2020) warps mouth regions in the source video to
# match the dubbed audio. Makes dubs look less obviously mismatched —
# especially helpful for talking-head footage.
#
# Not shipped as a pip install because:
#   1. ~400MB checkpoint needs to be downloaded manually (the repo's
#      Google Drive link requires a browser)
#   2. Its dependencies conflict with newer torch; safest to run in a
#      subprocess so it can have its own venv if needed
#   3. Only useful when faces are visible — probably <30% of use cases
#
# Detection: we look for the checkpoint file existing at one of several
# common paths. If absent, the endpoint returns a friendly install guide.
# If present, we invoke Wav2Lip via subprocess (its inference.py is the
# standard entry point).
# ═══════════════════════════════════════════════════════════════════════

# The upstream README points at a Google Drive link that needs a browser and
# has rotted more than once. This mirror serves the same file — verified
# byte-identical (xet hash 40dfad7c…) against two other public mirrors.
WAV2LIP_CHECKPOINT_URL = (
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
)


def _find_wav2lip_setup():
    """Scan common paths for Wav2Lip repo + checkpoint. Returns dict with
    'repo_dir', 'checkpoint', 'python' — all strings, or None if missing.
    Checks paths relative to the GoChiDUBB Studio install, and a few user-home
    fallbacks since people often clone repos in various places."""
    # Candidate directories where the repo might be cloned
    candidate_dirs = [
        BASE / "Wav2Lip",
        BASE / "external" / "Wav2Lip",
        BASE.parent / "Wav2Lip",
        Path.home() / "Wav2Lip",
    ]
    # Also check if GOCHIDUBB_WAV2LIP_DIR env var was set
    env_dir = os.getenv("GOCHIDUBB_WAV2LIP_DIR", "")
    if env_dir:
        candidate_dirs.insert(0, Path(env_dir))

    for d in candidate_dirs:
        if not d.exists():
            continue
        inference = d / "inference.py"
        if not inference.exists():
            continue
        # Find the checkpoint — Wav2Lip ships two models (wav2lip.pth and
        # wav2lip_gan.pth). Prefer GAN — sharper mouth detail.
        ckpt_candidates = [
            d / "checkpoints" / "wav2lip_gan.pth",
            d / "checkpoints" / "wav2lip.pth",
            d / "checkpoints" / "Wav2Lip.pth",
        ]
        ckpt = next((c for c in ckpt_candidates if c.exists()), None)
        if ckpt is None:
            continue
        return {
            "repo_dir": str(d),
            "checkpoint": str(ckpt),
            # Wav2Lip's own requirements (librosa 0.7, numpy 1.17, numba 0.48)
            # are from 2020 and cannot coexist with the ones VoxCPM and
            # faster-whisper need — nor do they install at all on Python 3.11.
            # So it gets its own interpreter when one is configured, and only
            # falls back to ours for the rare case where the deps happen to
            # line up. tools/setup_wav2lip.py builds that venv and prints the
            # line to paste into .env.
            "python": os.getenv("GOCHIDUBB_WAV2LIP_PYTHON") or sys.executable,
            "checkpoint_name": ckpt.name,
        }
    return None


def _wav2lip_install_guide() -> dict:
    """Install guide returned when Wav2Lip isn't found. Structured so UI
    can render a copy-paste block + link to checkpoint download.

    The steps deliberately do *not* install anything into our own venv. The
    upstream repo pins librosa 0.7 / numpy 1.17 / numba 0.48; those cannot
    coexist with what VoxCPM and faster-whisper need, and numpy 1.17 does not
    even build on Python 3.11. tools/setup_wav2lip.py creates a separate venv
    and applies the small patches modern librosa and Apple Silicon need.
    """
    repo = BASE / "Wav2Lip"
    ckpt = repo / "checkpoints" / "wav2lip_gan.pth"
    return {
        "error": "wav2lip_not_installed",
        "message": "Lip-sync requires Wav2Lip. It's optional — install it only if you "
                   "want mouth movements to match the dubbed audio. Works best on "
                   "talking-head footage; useless for action / wide shots.",
        "install_steps": [
            {"label": "1. Clone, build an isolated venv, and patch (one command)",
             "cmd": f"{sys.executable} tools/setup_wav2lip.py"},
            {"label": "2. Download the GAN checkpoint (~400 MB) — not automatic, "
                      "it is a large download you should opt into",
             "cmd": f"curl -L -o {ckpt} {WAV2LIP_CHECKPOINT_URL}"},
            {"label": "3. Add the two lines setup_wav2lip.py prints to your .env, "
                      "then restart the server. The toggle lights up on its own."},
        ],
        "manual_equivalent": [
            f"git clone https://github.com/Rudrabha/Wav2Lip.git {repo}",
            "python3 -m venv <repo>/venv-w2l",
            "<repo>/venv-w2l/bin/pip install torch numpy scipy opencv-python "
            "librosa tqdm",
            "patch inference.py for MPS and audio.py for librosa>=0.10 "
            "(see tools/setup_wav2lip.py --show-patches)",
        ],
        "note": "Wav2Lip is picky about input — only helps when faces are clear "
                "and roughly front-facing. It fails on fast cuts, extreme angles, "
                "and low-res video. Expect 2-5x the video duration to process, "
                "and considerably more on CPU.",
        "macos": "Upstream Wav2Lip only knows CUDA and CPU, so on Apple Silicon it "
                 "runs on the CPU unless patched. setup_wav2lip.py adds the MPS "
                 "branch; PYTORCH_ENABLE_MPS_FALLBACK=1 (already in .env.example) "
                 "covers the ops MPS is missing.",
        "env_override": "GOCHIDUBB_WAV2LIP_DIR=<path> if the repo lives elsewhere; "
                        "GOCHIDUBB_WAV2LIP_PYTHON=<path> to point at its venv "
                        "interpreter instead of ours.",
    }


@app.get("/api/lip_sync/status")
async def lip_sync_status():
    """Quick detection endpoint — UI calls this to decide whether to show
    the lip-sync button lit (ready) or greyed (needs install)."""
    setup = _find_wav2lip_setup()
    if not setup:
        return {"installed": False, "guide": _wav2lip_install_guide()}
    # Which interpreter runs it matters enough to surface: falling back to our
    # own venv is the single most likely reason a "ready" install then dies at
    # `import librosa`, and there is no other way to see which one was picked.
    isolated = bool(os.getenv("GOCHIDUBB_WAV2LIP_PYTHON"))
    return {
        "installed": True,
        "checkpoint": setup["checkpoint_name"],
        "repo_dir": setup["repo_dir"],
        "python": setup["python"],
        "isolated_venv": isolated,
        "warning": None if isolated else
        "Running Wav2Lip with the GoChiDUBB interpreter. Upstream needs "
        "librosa 0.7 / numpy 1.17, which conflict with ours — set "
        "GOCHIDUBB_WAV2LIP_PYTHON to its own venv "
        "(see tools/setup_wav2lip.py).",
    }


def _run_wav2lip_sync(job_id: str) -> dict:
    """Synchronously apply Wav2Lip to a job's dubbed_video.mp4. Used by
    both the manual `POST /api/dub/{id}/lip_sync` endpoint AND the
    auto-lip-sync hook in the queue worker (when the job was submitted
    with `lip_sync=True`).

    Returns a dict with `ok`, `url`, `elapsed_sec` on success — or
    `error` + `message` (+ optional `stderr_tail`) on failure. Updates
    `jobs[job_id]['lipsync_status']` ('running' → 'done' | 'error') and
    `lipsync_url` so the UI can poll for completion.
    """
    import subprocess as _sp

    if job_id not in jobs:
        return {"error": "job_not_found", "message": f"Job {job_id} not found"}

    setup = _find_wav2lip_setup()
    if not setup:
        return {"error": "wav2lip_not_installed", "guide": _wav2lip_install_guide()}

    work = OUTPUT_DIR / job_id
    src_video = work / "dubbed_video.mp4"
    if not src_video.exists():
        return {"error": "no_dub", "message": "Dubbed video not generated yet"}

    dub_wav = work / "_lipsync_dub.wav"
    try:
        _sp.run(
            ["ffmpeg", "-y", "-i", str(src_video),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             str(dub_wav)],
            check=True, capture_output=True, timeout=120,
        )
    except _sp.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-400:]
        return {"error": "audio_extract_failed", "message": err}

    w2l_out = work / "_lipsync_raw.mp4"
    env = os.environ.copy()
    env["PYTHONPATH"] = setup["repo_dir"] + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        setup["python"], "inference.py",
        "--checkpoint_path", setup["checkpoint"],
        "--face", str(src_video),
        "--audio", str(dub_wav),
        "--outfile", str(w2l_out),
        "--pads", "0", "10", "0", "0",
        "--resize_factor", "1",
        "--nosmooth",
    ]

    log.info(f"[lipsync] Running Wav2Lip for job {job_id}")
    jobs[job_id]["lipsync_status"] = "running"
    save_job(jobs[job_id])
    elapsed = 0.0
    try:
        t0 = time.time()
        result = _sp.run(
            cmd, cwd=setup["repo_dir"], env=env,
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            err_tail = (result.stderr or "")[-600:]
            log.warning(f"[lipsync] Wav2Lip failed: {err_tail}")
            jobs[job_id]["lipsync_status"] = "error"
            jobs[job_id]["lipsync_error"] = err_tail
            save_job(jobs[job_id])
            return {
                "error": "wav2lip_runtime_error",
                "message": "Wav2Lip ran but failed. Likely no face detected, "
                           "GPU OOM, or missing deps.",
                "stderr_tail": err_tail,
            }
        elapsed = time.time() - t0
        log.info(f"[lipsync] Wav2Lip done in {elapsed:.0f}s")
    except _sp.TimeoutExpired:
        jobs[job_id]["lipsync_status"] = "error"
        save_job(jobs[job_id])
        return {"error": "wav2lip_timeout", "message": "Didn't finish in 30 minutes"}

    if not w2l_out.exists():
        jobs[job_id]["lipsync_status"] = "error"
        save_job(jobs[job_id])
        return {"error": "wav2lip_no_output",
                "message": "Wav2Lip reported success but produced no output"}

    # Re-mux with the original full-quality dub audio (Wav2Lip's output
    # has 16kHz mono audio which sounds terrible).
    final_out = work / "dubbed_video_lipsync.mp4"
    try:
        _sp.run(
            ["ffmpeg", "-y",
             "-i", str(w2l_out),
             "-i", str(src_video),
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             str(final_out)],
            check=True, capture_output=True, timeout=300,
        )
    except _sp.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-400:]
        jobs[job_id]["lipsync_status"] = "error"
        save_job(jobs[job_id])
        return {"error": "final_mux_failed", "message": err}
    finally:
        for p in (dub_wav, w2l_out):
            try:
                if p.exists(): p.unlink()
            except Exception:
                pass

    url = f"/outputs/{job_id}/dubbed_video_lipsync.mp4?v={int(time.time())}"
    jobs[job_id]["lipsync_status"] = "done"
    jobs[job_id]["lipsync_url"] = url
    jobs[job_id].pop("lipsync_error", None)
    save_job(jobs[job_id])
    return {"ok": True, "url": url, "elapsed_sec": round(elapsed, 1)}


@app.post("/api/dub/{job_id}/lip_sync")
async def lip_sync(job_id: str):
    """Apply Wav2Lip to the dubbed video (manual on-demand endpoint).

    Returns the new video URL when done. Same code path is also used
    automatically by the pipeline when a job was submitted with
    `lip_sync=true` on the original form.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    setup = _find_wav2lip_setup()
    if not setup:
        return JSONResponse(_wav2lip_install_guide(), 501)
    # Run in executor — wav2lip is heavy synchronous CPU/GPU work
    result = await asyncio.get_event_loop().run_in_executor(
        None, _run_wav2lip_sync, job_id)
    if "error" in result:
        return JSONResponse(result, 500)
    return result


@app.post("/api/dub/{job_id}/edit_transcript")
async def edit_transcript(job_id: str, edits: str = Form(...)):
    """Update the source `text` for one or more segments in the transcription
    checkpoint. Used in wizard review_transcript mode when the user spots
    ASR errors before they get baked into the translation.
    `edits` is a JSON string: {"<idx>": "corrected text", ...}"""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    cp = _load_checkpoint(job_id, "transcription_done")
    if not cp:
        return JSONResponse({"error": "No transcription checkpoint to edit"}, 404)
    try:
        edit_map = json.loads(edits)
        if not isinstance(edit_map, dict):
            raise ValueError("edits must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid edits JSON: {e}"}, 400)

    n_edited = 0
    for s in cp.get("segments", []):
        key = str(s.get("idx"))
        if key in edit_map:
            new_text = str(edit_map[key]).strip()
            if new_text and new_text != s.get("text"):
                s["text"] = new_text
                n_edited += 1
    work = OUTPUT_DIR / job_id
    _save_checkpoint(job_id, work, stage="transcription_done", data=cp)
    log.info(f"[edit] Applied {n_edited} transcript edits for job {job_id}")
    return {"ok": True, "edited": n_edited}


@app.post("/api/dub/{job_id}/continue")
async def continue_pipeline(
    job_id: str,
    voice_style: str = Form(""),
    voice_preset: str = Form(""),
    tts_speed: str = Form(""),
    reference: Optional[UploadFile] = File(None),
):
    """Continue the pipeline from the most recent checkpoint. Called after
    the user has reviewed (and possibly edited) the transcript/translation
    in wizard mode.

    - If stopped at translation_done: runs TTS + merge.
    - If stopped at transcription_done: runs translate + TTS + merge.

    Voice settings, if provided, override what was originally requested."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    job = jobs[job_id]

    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_continue_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    # Merge new voice settings with job defaults
    final_style = voice_style if voice_style else job.get("voice_style", "")
    final_preset = voice_preset if voice_preset else job.get("voice_preset", "auto")
    final_speed = tts_speed if tts_speed else job.get("tts_speed", "balanced")

    cp = _latest_checkpoint(job_id)
    if not cp:
        return JSONResponse({"error": "No checkpoint to continue from"}, 404)

    # Reset error/stale flags so the History UI immediately reflects that
    # the job is alive again. _continue_from_checkpoint will set status
    # to "translating"/"synthesizing" as it starts each stage.
    job["status"] = "resuming"
    _clear_job_error(job)
    save_job(job)

    asyncio.create_task(_continue_from_checkpoint(
        job_id, cp, final_style, final_preset, final_speed, ref_path,
    ))
    return {"ok": True, "job_id": job_id, "resuming_from": cp.get("stage")}


def _stage_after_checkpoint(checkpoint_name: str) -> str:
    """Which stage should run next, given the checkpoint we're resuming from.

    Returns "" when the checkpoint is the final one (nothing left to do).
    Unknown/legacy names fall back to `tts` — the old behaviour, where
    `pipeline_state.json` was assumed to hold a post-translation state.
    """
    for i, spec in enumerate(PIPELINE_STAGES):
        if spec["checkpoint"] == checkpoint_name:
            return STAGE_ORDER[i + 1] if i + 1 < len(STAGE_ORDER) else ""
    return "tts"


async def _continue_from_checkpoint(
    job_id: str, cp: dict,
    voice_style: str, voice_preset: str, tts_speed: str, ref_path: str,
):
    """Resume the pipeline from wherever the wizard (or a crash) left it.

    Runs through the same stage driver as a fresh job, so the resumed part
    of the run is timed and checkpointed exactly like the original."""
    job = jobs.get(job_id)
    if not job:
        return

    stage = cp.get("stage", "")
    next_stage = _stage_after_checkpoint(stage)
    if not next_stage:
        log.info(f"[continue] Job {job_id} is already at the final stage")
        job.update(status="complete", progress=100); save_job(job)
        return

    ctx = dict(cp)
    for k in ("stage", "job_id", "saved_at"):
        ctx.pop(k, None)
    # Voice settings from the review screen override the checkpointed ones.
    if voice_style:
        ctx["voice_style"] = voice_style
    if voice_preset:
        ctx["voice_preset"] = voice_preset
    if tts_speed:
        ctx["tts_speed"] = tts_speed
    if ref_path:
        ctx["reference_audio"] = ref_path
    # The user just approved this checkpoint — don't pause on it again.
    ctx["wizard_mode"] = "auto"

    log.info(
        f"[continue] Resuming job {job_id} from checkpoint '{stage}' "
        f"→ stage '{next_stage}'"
    )
    try:
        await run_pipeline_stages(job_id, ctx, start_stage=next_stage)
    except JobCancelled:
        log.info(f"[continue] Job {job_id} cancelled")
        job.update(status="cancelled"); job.pop("cancel_requested", None)
        save_job(job)
    except Exception as e:
        log.error(f"[continue] Failed: {e}", exc_info=True)
        _set_job_error(job, e)
        job["status"] = "error"
        save_job(job)


@app.post("/api/dub/{job_id}/retranslate")
async def retranslate(
    job_id: str,
    model: str = Form(""),
    context_hint: str = Form(""),
    target_lang: str = Form(""),
):
    """Re-run ONLY the translation step using the saved transcription
    checkpoint. Useful when the user wants to try a different model,
    adjust the context hint, or switch target language mid-flight."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    cp = _load_checkpoint(job_id, "transcription_done")
    if not cp:
        return JSONResponse(
            {"error": "No transcription checkpoint — start with wizard_mode"}, 404
        )

    final_model = model or cp.get("model", "gemma4:e4b")
    final_context = context_hint if context_hint else cp.get("context_hint", "")
    final_target = target_lang or cp.get("target_lang", "ru")

    asyncio.create_task(_retranslate_stage(
        job_id, cp, final_model, final_context, final_target,
    ))
    return {"ok": True, "job_id": job_id}


async def _retranslate_stage(job_id: str, cp: dict, model: str,
                               context_hint: str, target_lang: str):
    job = jobs.get(job_id)
    if not job:
        return
    work = OUTPUT_DIR / job_id
    def update(**kwargs):
        if job.get("cancel_requested"):
            raise JobCancelled(f"Job {job_id} cancelled by user")
        job.update(kwargs); save_job(job)
    try:
        update(status="translating", progress=45, model=model,
               context_hint=context_hint, target_lang=target_lang,
               step_detail=f"Retranslating with {model}...")
        effective_src = cp.get("effective_src", "en")
        # translate_segments(segments, target_lang, model, ...)
        # NOT (segments, source_lang, target_lang, model)
        segments = await translate_segments(
            cp["segments"], target_lang, model,
            context_hint=context_hint,
            source_lang=effective_src or "",
        )
        _save_checkpoint(job_id, work, stage="translation_done", data={
            **cp,
            "target_lang": target_lang,
            "segments": [
                {
                    "idx": i, "start": s["start"], "end": s["end"],
                    "text": s["text"],
                    "translated_text": s.get("translated_text", ""),
                    "speaker": s.get("speaker", "SPEAKER_00"),
                }
                for i, s in enumerate(segments)
            ],
        })
        update(
            status="awaiting_translation_review", progress=63,
            step_detail="Retranslated — review and continue",
            checkpoint_stage="translation_done",
        )
    except Exception as e:
        log.error(f"[retranslate] Failed: {e}", exc_info=True)
        update(status="error", error=str(e))


@app.post("/api/dub/{job_id}/regenerate_segment/{seg_idx}")
async def regenerate_segment(
    job_id: str, seg_idx: int,
    # Accept both 'translated_text' (UI form field name) and 'new_text'
    # (legacy) — whichever is populated wins.
    translated_text: str = Form(""),
    new_text: str = Form(""),
    voice_style: str = Form(""),
    voice_preset: str = Form(""),
    reference: Optional[UploadFile] = File(None),
):
    """Regenerate a SINGLE TTS segment. Optionally lets user edit the
    translated text and/or use a different voice for just this one line.
    After the new audio is rendered, the final video is rebuilt so the
    player immediately reflects the change."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    job = jobs[job_id]
    cp = _load_checkpoint(job_id, "tts_done")
    if not cp:
        return JSONResponse(
            {"error": "Per-segment regen requires tts_done checkpoint"}, 404
        )

    # Find the target segment
    segs = cp.get("segments", [])
    target = next((s for s in segs if s.get("idx") == seg_idx), None)
    if not target:
        return JSONResponse({"error": f"Segment {seg_idx} not found"}, 404)

    # Apply edits if any. Persist them to BOTH checkpoints immediately so
    # that even if the regen crashes, the user's edit isn't lost. Also
    # updates translation_done so a later full re-translate would only wipe
    # edits when the user explicitly requests it.
    edited = (translated_text or new_text).strip()
    if edited and edited != (target.get("translated_text") or "").strip():
        target["translated_text"] = edited
        # Also update the translation_done checkpoint so retranslate baseline
        # reflects the user's latest edit
        tcp = _load_checkpoint(job_id, "translation_done")
        if tcp:
            for ts in tcp.get("segments", []):
                if ts.get("idx") == seg_idx:
                    ts["translated_text"] = edited
                    break
            _save_checkpoint(job_id, OUTPUT_DIR / job_id,
                             stage="translation_done", data=tcp)
        log.info(f"[regen_seg] Edited segment {seg_idx} text: "
                 f"'{edited[:50]}{'...' if len(edited) > 50 else ''}'")

    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_seg{seg_idx}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    # Delete the old audio_path so _run_tts_and_merge_stage treats it as "to do"
    old_audio = target.get("audio_path", "")
    if old_audio and os.path.exists(old_audio):
        try:
            os.remove(old_audio)
        except Exception:
            pass
    target["audio_path"] = ""

    # Persist the updated tts_done checkpoint to disk BEFORE dispatching the
    # async regen task. The stage reloads checkpoints per-run, and the
    # background task runs against THIS modified cp dict in-memory anyway,
    # but saving now guards against server crash between dispatch and save.
    _save_checkpoint(job_id, OUTPUT_DIR / job_id, stage="tts_done", data=cp)

    final_style = voice_style if voice_style else job.get("voice_style", "")
    final_preset = voice_preset if voice_preset else job.get("voice_preset", "auto")
    final_speed = job.get("tts_speed", "balanced")

    asyncio.create_task(_regen_single_segment(
        job_id, cp, seg_idx, final_style, final_preset, final_speed, ref_path,
    ))
    return {"ok": True, "job_id": job_id, "seg_idx": seg_idx}


async def _regen_single_segment(
    job_id: str, cp: dict, seg_idx: int,
    voice_style: str, voice_preset: str, tts_speed: str, ref_path: str,
):
    job = jobs.get(job_id)
    if not job:
        return
    work = OUTPUT_DIR / job_id
    def update(**kwargs):
        if job.get("cancel_requested"):
            _maybe_terminate_tts_worker()
            raise JobCancelled(f"Job {job_id} cancelled by user")
        job.update(kwargs); save_job(job)
    try:
        update(status="synthesizing", progress=65,
               step_detail=f"Regenerating segment {seg_idx+1}...")
        # preserve_existing_audio_paths=True — only the cleared one will re-synth
        await _run_tts_and_merge_stage(
            job, work, cp,
            voice_style=voice_style, voice_preset=voice_preset,
            tts_speed=tts_speed, ref_path_override=ref_path,
            preserve_existing_audio_paths=True,
        )
    except Exception as e:
        log.error(f"[regen_seg] Failed: {e}", exc_info=True)
        _set_job_error(job, e, stage_id="tts")
        update(status="error")


def _voice_preset_payload() -> dict:
    """Build the full presets list response used by both /api/voices
    and /api/voice_presets endpoints."""
    style_presets = [
        {"id": k, "name": v["name"], "style": v["style"], "type": "style"}
        for k, v in VOICE_PRESETS.items()
    ]
    file_presets = []
    for k, v in scan_file_presets().items():
        file_presets.append({
            "id": k, "name": v["name"], "style": v.get("style", ""),
            "type": "file",
            "description": v.get("description", ""),
            "reference_file": os.path.basename(v["reference_file"]),
            "gender": v.get("gender", ""),
            "language": v.get("language", ""),
            "tags": v.get("tags", []),
            "created_at": v.get("created_at"),
            "file_size": v.get("file_size", 0),
            "file_ext": v.get("file_ext", ""),
            "audio_url": v.get("audio_url"),
        })
    return {"presets": file_presets + style_presets}


@app.get("/api/voices")
async def list_voice_presets():
    """List all available voice presets (built-in styles + user file presets)."""
    return _voice_preset_payload()


@app.get("/api/voice_presets")
async def list_voice_presets_v2():
    """Alias for /api/voices — preferred name for new clients (CLI, MCP)."""
    return _voice_preset_payload()


def _file_preset_path(preset_id: str) -> Optional[Path]:
    """Resolve a `file:NAME` preset id back to its actual audio file on
    disk, or None if not found / not a file preset."""
    if not preset_id.startswith("file:"):
        return None
    name = preset_id[len("file:"):]
    # No path traversal — the name is just a basename, must not contain separators
    if "/" in name or "\\" in name or ".." in name:
        return None
    for ext in _VOICE_AUDIO_EXTS:
        p = VOICE_PRESETS_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


@app.get("/api/voice_presets/{preset_id}/audio")
async def get_voice_preset_audio(preset_id: str):
    """Stream the audio file behind a file-based preset. Used by the
    Voices tab's inline player and the dub form's preview."""
    from fastapi.responses import FileResponse
    p = _file_preset_path(preset_id)
    if not p:
        return JSONResponse({"error": "Preset not found or not a file preset"}, 404)
    media = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".m4a": "audio/mp4",
    }.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), media_type=media, filename=p.name)


_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-]{0,49}$")


def _sanitize_voice_name(raw: str) -> Optional[str]:
    """Normalize a user-provided voice name to a filesystem-safe stem.
    Returns None if invalid (would let through path traversal or weird chars)."""
    if not raw:
        return None
    s = raw.strip()
    # Collapse internal whitespace runs, strip trailing dots
    s = re.sub(r"\s+", " ", s).rstrip(".")
    if not _VOICE_NAME_RE.match(s):
        return None
    return s


@app.post("/api/voice_presets")
async def create_voice_preset(
    audio: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    gender: str = Form(""),
    language: str = Form(""),
    tags: str = Form(""),                  # comma-separated
    style: str = Form(""),
):
    """Upload a new voice reference. Saves the audio as
    `presets/voices/<name>.<ext>` and writes a JSON sidecar with the
    structured metadata. Re-upload with the same name overwrites.

    The new preset is immediately usable in any dub form (id = `file:<name>`).
    """
    clean = _sanitize_voice_name(name)
    if not clean:
        return JSONResponse({
            "error": "Name must be 1-50 chars, letters/digits/space/dash/underscore, "
                     "starting with a letter or digit."
        }, 400)

    if not audio.filename:
        return JSONResponse({"error": "No audio file provided"}, 400)
    ext = Path(audio.filename).suffix.lower()
    if ext not in _VOICE_AUDIO_EXTS:
        return JSONResponse({
            "error": f"Unsupported audio extension '{ext}'. Use one of: "
                     f"{', '.join(_VOICE_AUDIO_EXTS)}"
        }, 400)

    VOICE_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = VOICE_PRESETS_DIR / f"{clean}{ext}"
    # If a file with same stem but different ext already exists, remove the
    # old one so we don't keep duplicates (e.g. user re-uploads as mp3).
    for old_ext in _VOICE_AUDIO_EXTS:
        old = VOICE_PRESETS_DIR / f"{clean}{old_ext}"
        if old.exists() and old != audio_path:
            try:
                old.unlink()
            except Exception as e:
                log.warning(f"[voice_presets] couldn't remove old {old.name}: {e}")

    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    except Exception as e:
        return JSONResponse({"error": f"Couldn't save audio: {e}"}, 500)

    # Normalize tags
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    meta = {
        "display_name": clean,
        "description": (description or "").strip()[:500],
        "gender": (gender or "").strip().lower(),
        "language": (language or "").strip().lower(),
        "tags": tag_list,
        "style": (style or "").strip()[:300],
        "created_at": time.time(),
    }
    try:
        _voice_metadata_path(audio_path).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"[voice_presets] couldn't write metadata: {e}")

    log.info(f"[voice_presets] created '{clean}' ({audio_path.stat().st_size} bytes)")
    pid = f"file:{clean}"
    return {"ok": True, "id": pid, "preset": scan_file_presets().get(pid, {})}


@app.put("/api/voice_presets/{preset_id}")
async def update_voice_preset(
    preset_id: str,
    name: Optional[str] = Form(None),       # rename
    description: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
):
    """Update metadata for a file-based preset. Optionally rename it
    (renames the audio file + sidecar). Fields left as None are preserved."""
    src_audio = _file_preset_path(preset_id)
    if not src_audio:
        return JSONResponse({"error": "File preset not found"}, 404)

    # Load existing metadata, then merge in updates
    meta = _read_voice_metadata(src_audio)
    if description is not None:
        meta["description"] = description.strip()[:500]
    if gender is not None:
        meta["gender"] = gender.strip().lower()
    if language is not None:
        meta["language"] = language.strip().lower()
    if tags is not None:
        meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if style is not None:
        meta["style"] = style.strip()[:300]

    # Rename if a new name was given
    final_audio = src_audio
    new_id = preset_id
    if name is not None:
        clean = _sanitize_voice_name(name)
        if not clean:
            return JSONResponse({"error": "Invalid name"}, 400)
        new_audio = VOICE_PRESETS_DIR / f"{clean}{src_audio.suffix}"
        if new_audio.exists() and new_audio != src_audio:
            return JSONResponse({"error": f"A preset named '{clean}' already exists"}, 409)
        if new_audio != src_audio:
            old_meta_path = _voice_metadata_path(src_audio)
            old_txt_path = src_audio.with_suffix(".txt")
            src_audio.rename(new_audio)
            if old_meta_path.exists():
                old_meta_path.rename(_voice_metadata_path(new_audio))
            if old_txt_path.exists():
                old_txt_path.rename(new_audio.with_suffix(".txt"))
            final_audio = new_audio
            new_id = f"file:{clean}"
            meta["display_name"] = clean

    try:
        _voice_metadata_path(final_audio).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        return JSONResponse({"error": f"Couldn't save metadata: {e}"}, 500)

    log.info(f"[voice_presets] updated '{final_audio.stem}'")
    return {"ok": True, "id": new_id, "preset": scan_file_presets().get(new_id, {})}


@app.delete("/api/voice_presets/{preset_id}")
async def delete_voice_preset(preset_id: str):
    """Delete a file-based preset (audio + metadata sidecars).
    Built-in style presets cannot be deleted."""
    p = _file_preset_path(preset_id)
    if not p:
        return JSONResponse({"error": "File preset not found"}, 404)
    try:
        p.unlink()
        for sidecar in (_voice_metadata_path(p), p.with_suffix(".txt")):
            if sidecar.exists():
                sidecar.unlink()
    except Exception as e:
        return JSONResponse({"error": f"Couldn't delete: {e}"}, 500)
    log.info(f"[voice_presets] deleted '{preset_id}'")
    return {"ok": True, "id": preset_id}


@app.get("/api/preferences")
async def get_preferences():
    """Return saved UI preferences (last-used models, voice, speed, etc).
    The UI uses localStorage as primary but falls back to this when
    localStorage is unavailable (private browsing, cross-device, etc)."""
    if not PREFS_FILE.exists():
        return {}
    try:
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not read prefs: {e}")
        return {}


@app.post("/api/preferences")
async def set_preferences(prefs: str = Form(...)):
    """Update UI preferences. Merges into existing prefs; doesn't replace."""
    try:
        new_prefs = json.loads(prefs)
        if not isinstance(new_prefs, dict):
            raise ValueError("prefs must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid prefs JSON: {e}"}, 400)
    existing = {}
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(new_prefs)
    try:
        with open(PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# Config fields that must never leave the process verbatim. The Settings UI
# only needs to know *whether* a value is set, so it gets a stub. This matters
# because GOCHIDUBB_HOST can expose this server beyond loopback (see
# uvicorn.run at the bottom of this file), and /api/config has no auth.
_SECRET_CONFIG_KEYS = ("hf_token",)


def _redacted_config() -> dict:
    d = cfg.to_dict()
    for k in _SECRET_CONFIG_KEYS:
        if d.get(k):
            d[k] = mask_secret(d[k])
    return d


@app.get("/api/config")
async def get_config():
    """Return current UserConfig as JSON. Editable fields shown in Settings tab."""
    return _redacted_config()


@app.patch("/api/config")
async def patch_config(body: str = Form(...)):
    """Update one or more UserConfig fields and persist to config-user.json."""
    try:
        updates = json.loads(body)
        if not isinstance(updates, dict):
            raise ValueError("body must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)
    # A masked value is what GET handed out — writing it back would replace a
    # working token with the literal string "hf_a…".
    for k in _SECRET_CONFIG_KEYS:
        v = updates.get(k)
        if isinstance(v, str) and v.endswith("…"):
            updates.pop(k)
    # Changing an engine-level setting means dropping the loaded model. Doing
    # that under a running job would pull the model out from under it, so the
    # write is refused outright rather than half-applied.
    needs_rebuild = bool(TTS_REBUILD_KEYS & set(updates))
    if needs_rebuild:
        busy = [j.get("id") for j in jobs.values()
                if j.get("status") in ACTIVE_STATUSES]
        if busy:
            return JSONResponse({
                "error": "A job is running — voice settings that reload the "
                         "model can't be applied right now. Wait for it to "
                         "finish, or cancel it, then save again.",
                "busy_jobs": busy[:5],
            }, 409)
    try:
        cfg.update(**updates)
    except ValueError as e:
        # Failed FIELD_SPECS validation — nothing was written.
        return JSONResponse({"error": str(e)}, 400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)
    try:
        if "hf_token" in updates:
            # diarizer reads the env first; keep the two in step so a token
            # saved here takes effect without a restart.
            os.environ["HF_TOKEN"] = cfg.hf_token or ""
            diag.clear_runtime(list(diag.REVALIDATES["hf"]))
        reloaded = reset_tts_engine() if needs_rebuild else False
        return {"ok": True, "config": _redacted_config(),
                "tts_reloaded": reloaded, "tts_rebuild_pending": needs_rebuild}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ═══════════════════════════════════════════════════════════════════════
#  Secrets — platform credentials (VK token etc.)
# ═══════════════════════════════════════════════════════════════════════
# Deliberately NOT part of UserConfig: GET /api/config is unauthenticated,
# so tokens live in secrets.json (chmod 600) via app/secrets.py instead.
# These routes never echo values — status is presence booleans only.
# ═══════════════════════════════════════════════════════════════════════

from fastapi import Request  # noqa: E402  (kept local to this route region)
from app import secrets as app_secrets  # noqa: E402
from app import bugreport  # noqa: E402


@app.post("/api/secrets")
async def set_secrets(request: Request):
    """Store platform credentials. Accepts a JSON object body
    ({"vk_access_token": "..."}), a form field `body` holding the same
    JSON, or plain form key/value pairs. Only keys known to app.secrets
    are accepted (400 otherwise). Values are never echoed back."""
    ctype = request.headers.get("content-type", "")
    try:
        if "application/json" in ctype:
            updates = await request.json()
        else:
            form = await request.form()
            if "body" in form:
                updates = json.loads(form["body"])
            else:
                updates = {k: v for k, v in form.items()}
        if not isinstance(updates, dict):
            raise ValueError("body must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid body: {e}"}, 400)
    unknown = sorted(k for k in updates if k not in app_secrets.KNOWN_SECRETS)
    if unknown:
        return JSONResponse({
            "error": f"Unknown secret key(s): {', '.join(unknown)}. "
                     f"Known: {', '.join(sorted(app_secrets.KNOWN_SECRETS))}",
        }, 400)
    for k, v in updates.items():
        app_secrets.set_secret(k, "" if v is None else str(v))
    return {"ok": True, "status": app_secrets.secret_status()}


@app.get("/api/secrets/status")
async def get_secrets_status():
    """Presence booleans per known secret key — never the values."""
    return app_secrets.secret_status()


# ═══════════════════════════════════════════════════════════════════════
#  Bug reports — package a failed job for an issue tracker
# ═══════════════════════════════════════════════════════════════════════
# Assembly and delivery live in app/bugreport.py; these routes only look up
# the in-memory job (the DB copy has large fields stripped) and take a cheap
# system snapshot — no subprocesses, no network probes.
# ═══════════════════════════════════════════════════════════════════════

def _bug_report_payload(job_id: str) -> dict:
    job = jobs[job_id]
    system = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu_backend": gpu_backend(),
        "gpu": gpu_snapshot() or {},
        "translation_backend": "lm_studio" if USE_LM_STUDIO else "ollama",
        "mode": cfg.mode,
    }
    logs = bugreport.select_log_window(logbuf.snapshot, job.get("last_error"))
    return bugreport.build_bug_report(job, system=system, logs=logs)


@app.get("/api/bugreport/{job_id}")
async def get_bug_report(job_id: str):
    """Preview: the full report plus whether a Linear sink is configured."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    payload = _bug_report_payload(job_id)
    return {"report": payload, "signature": payload["signature"],
            "linear_configured": bugreport.sink_configured()}


@app.post("/api/bugreport/{job_id}")
async def send_bug_report(job_id: str, request: Request):
    """Deliver the report to the configured sink (Linear).

    Optional JSON body {"note": "..."} — a missing or malformed body is
    tolerated, and the note is redacted before it leaves the machine.
    """
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, 404)
    if not (job.get("last_error") or job.get("error")):
        return JSONResponse({"error": "Job has no recorded error"}, 400)
    note = ""
    try:
        body = await request.json()
        if isinstance(body, dict):
            note = bugreport.redact(str(body.get("note") or ""))[:2000]
    except Exception:
        pass
    sink = bugreport.get_sink()
    if sink is None:
        return JSONResponse({"error": "Linear is not configured. Set "
                             "linear_api_key and linear_team_id via "
                             "/api/secrets."}, 400)
    report = _bug_report_payload(job_id)
    result = await sink.deliver(report, note=note)
    log.info(f"[bugreport] {job_id} -> {result.get('action')} "
             f"{result.get('issue') or ''}")
    if not result.get("ok"):
        return JSONResponse({"ok": False,
                             "error": result.get("error") or "Delivery failed"},
                            502)
    return {"ok": True, "action": result.get("action"),
            "url": result.get("url"), "issue": result.get("issue"),
            "signature": result.get("signature")}


# ═══════════════════════════════════════════════════════════════════════
#  Glossary — user-editable term overrides
# ═══════════════════════════════════════════════════════════════════════
# The translator ships with a built-in BJJ glossary baked into
# translator.py. Users doing non-BJJ courses (cooking, tech, music, etc)
# need a way to add their own domain terms without editing Python code.
# Solution: sidecar JSON file at presets/user_glossary.json. Translator
# loads it lazily on each call via _load_user_glossary().
#
# Format (validated lightly, not strictly):
#   { "domains": [
#       { "name": "Cooking EN→RU",
#         "triggers": ["cooking", "recipe"],
#         "target_lang": "ru",
#         "terms": { "sear": "запекать", ... } },
#       ...
#   ] }
# ═══════════════════════════════════════════════════════════════════════

_GLOSSARY_EXAMPLE = {
    "domains": [
        {
            "name": "Example: Cooking EN→RU",
            "triggers": ["cooking", "recipe", "food"],
            "target_lang": "ru",
            "terms": {
                "sear": "обжарить до корочки",
                "simmer": "томить",
                "al dente": "аль денте",
            },
        },
    ],
}


@app.get("/api/glossary")
async def get_glossary():
    """Return the current user glossary JSON + metadata. If the file
    doesn't exist, return an example structure so the UI has something
    sensible to show as the starting template."""
    if not USER_GLOSSARY_FILE.exists():
        return {
            "exists": False,
            "data": _GLOSSARY_EXAMPLE,
            "hint": "File will be created on first save. Built-in BJJ glossary stays active.",
        }
    try:
        with open(USER_GLOSSARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"exists": True, "data": data}
    except Exception as e:
        return JSONResponse({
            "exists": True,
            "error": f"Could not parse glossary file: {e}",
            "raw_text": USER_GLOSSARY_FILE.read_text(encoding="utf-8", errors="replace"),
        }, 500)


@app.post("/api/glossary")
async def set_glossary(body: str = Form(...)):
    """Replace the user glossary. Validates structure server-side so a
    malformed save doesn't break the translator.

    Accepts:
      { "domains": [ { name, triggers, target_lang, terms }, ... ] }
    """
    try:
        data = json.loads(body)
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Top level must be an object"}, 400)
    domains = data.get("domains", [])
    if not isinstance(domains, list):
        return JSONResponse({"error": "'domains' must be an array"}, 400)
    # Light validation: each domain must have terms + triggers
    for i, d in enumerate(domains):
        if not isinstance(d, dict):
            return JSONResponse({"error": f"domains[{i}] must be an object"}, 400)
        if not isinstance(d.get("terms", {}), dict):
            return JSONResponse({"error": f"domains[{i}].terms must be an object"}, 400)
        if not isinstance(d.get("triggers", []), list):
            return JSONResponse({"error": f"domains[{i}].triggers must be an array"}, 400)

    # Ensure parent dir exists
    USER_GLOSSARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(USER_GLOSSARY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        total_terms = sum(len(d.get("terms", {})) for d in domains)
        # The translator caches the flattened glossary for the life of the
        # process. Without this the save is inert until a restart — which is
        # exactly what "we'll remember this for future videos" must not mean.
        clear_glossary_cache()
        log.info(f"[glossary] Saved {len(domains)} domain(s), {total_terms} term(s) total")
        return {
            "ok": True,
            "domains": len(domains),
            "total_terms": total_terms,
            "path": str(USER_GLOSSARY_FILE),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# One writer at a time. Per-term saves come from a review screen where the
# user is confirming several terms in a row, and read-modify-write of a whole
# JSON file from two overlapping requests loses one of them.
_glossary_write_lock = asyncio.Lock()

# Domain a per-term save lands in when the caller does not name one. Kept
# separate from hand-authored domains so a user editing the JSON by hand can
# see which entries the review screen created.
_GLOSSARY_REVIEW_DOMAIN = "creator-review"

# A glossary entry is a word or a short phrase somebody confirmed on a review
# card. These caps are far above anything real and far below anything that
# could crowd out a translation prompt.
_GLOSSARY_MAX_TERM = 200
_GLOSSARY_MAX_DOMAIN = 64
# Domain names become JSON keys and are shown in the Settings editor. No
# traversal is possible through a dict key, but there is no reason to store
# "../../etc/passwd" either.
_GLOSSARY_DOMAIN_RE = re.compile(r"^[\w .\-]{1,64}$", re.UNICODE)


@app.post("/api/glossary/term")
async def set_glossary_term(term: str = Form(""),
                            translation: str = Form(""),
                            target_lang: str = Form(""),
                            domain: str = Form("")):
    """Merge one term into the glossary, server-side.

    The review screen saves terms one at a time. POST /api/glossary is a
    whole-file replace, so doing this from the browser would mean
    read-modify-write across the network — two confirmations in quick
    succession and one silently loses. The merge happens here instead, under
    a lock, and the translator's cache is cleared so the *next* language of
    the same batch actually picks the decision up.
    """
    # Form("") rather than Form(...): a missing or blank field is a plain
    # mistake with a readable answer, and Form(...) made FastAPI answer it
    # with a 422 pydantic blob before any of the messages below could run.
    term = (term or "").strip()
    translation = (translation or "").strip()
    lang = (target_lang or "").strip().lower()
    if not term or not translation:
        return JSONResponse({"error": "term and translation are required"}, 400)
    if not lang:
        return JSONResponse({"error": "target_lang is required"}, 400)
    if lang not in _QUICK_TEST_KNOWN_LANGS:
        return JSONResponse({"error": f"Unknown language code: {lang}"}, 400)
    # Bounds, because every term here is rendered into the prompt of every
    # later translation for this language (_glossary_block, 60-term budget).
    # An unbounded term is a way to corrupt translations, not just to fill
    # a disk — and nothing a human confirms on a review card is this long.
    if len(term) > _GLOSSARY_MAX_TERM:
        return JSONResponse(
            {"error": f"Term is too long ({len(term)} characters, "
                      f"maximum {_GLOSSARY_MAX_TERM})."}, 400)
    if len(translation) > _GLOSSARY_MAX_TERM:
        return JSONResponse(
            {"error": f"Translation is too long ({len(translation)} "
                      f"characters, maximum {_GLOSSARY_MAX_TERM})."}, 400)
    domain_name = (domain or "").strip() or _GLOSSARY_REVIEW_DOMAIN
    if len(domain_name) > _GLOSSARY_MAX_DOMAIN or not _GLOSSARY_DOMAIN_RE.match(
            domain_name):
        return JSONResponse(
            {"error": "Domain must be 1-64 characters of letters, digits, "
                      "spaces, dots, dashes or underscores."}, 400)

    async with _glossary_write_lock:
        data = _read_user_glossary() or {}
        domains = data.get("domains")
        if not isinstance(domains, list):
            domains = []
        target = None
        for d in domains:
            if (isinstance(d, dict) and d.get("name") == domain_name
                    and (d.get("target_lang") or "").lower() == lang):
                target = d
                break
        if target is None:
            target = {"name": domain_name, "triggers": [],
                      "target_lang": lang, "terms": {}}
            domains.append(target)
        if not isinstance(target.get("terms"), dict):
            target["terms"] = {}
        previous = target["terms"].get(term)
        target["terms"][term] = translation
        data["domains"] = domains

        USER_GLOSSARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = USER_GLOSSARY_FILE.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, USER_GLOSSARY_FILE)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            return JSONResponse({"error": str(e)}, 500)
        clear_glossary_cache(lang)

    log.info(f"[glossary] {domain_name}/{lang}: {term!r} -> {translation!r}"
             + (f" (was {previous!r})" if previous else ""))
    return {
        "ok": True,
        "term": term,
        "translation": translation,
        "target_lang": lang,
        "domain": domain_name,
        "replaced": previous,
        "total_terms": sum(len(d.get("terms", {})) for d in domains
                           if isinstance(d, dict)),
    }


@app.delete("/api/glossary")
async def delete_glossary():
    """Remove the user glossary file entirely. Built-in BJJ terms still
    apply; this just clears user additions."""
    if USER_GLOSSARY_FILE.exists():
        try:
            USER_GLOSSARY_FILE.unlink()
            clear_glossary_cache()
            log.info("[glossary] User glossary file removed")
            return {"ok": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, 500)
    return {"ok": True, "note": "File didn't exist"}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Not found"}, 404)
    return jobs[job_id]


@app.post("/api/dub/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running job.

    - Queued: removed from the asyncio queue immediately, status=cancelled.
    - Running: sets cancel_requested flag. The pipeline's update() closures
      check this flag at every stage boundary and raise JobCancelled, which
      the queue worker catches and marks as cancelled. For TTS synthesis we
      additionally terminate the persistent VoxCPM subprocess so cancel
      takes effect within 1-2 seconds instead of waiting for the current
      segment to finish rendering (can be 30+ seconds on a long segment).
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    j = jobs[job_id]
    status = j.get("status")

    if status in ("complete", "error", "cancelled"):
        return {"ok": False, "reason": f"Job is already {status}"}

    if status == "scheduled":
        # Not yet in the queue — just flip status; scheduler loop will
        # see status != 'scheduled' and skip it on next poll.
        j["status"] = "cancelled"
        j["step_detail"] = "Cancelled before scheduled start"
        j.pop("_pending_args", None)
        save_job(j)
        log.info(f"[scheduler] Job {job_id} cancelled before scheduled start")
        return {"ok": True, "cancelled_from": "scheduled"}

    if status == "preparing":
        # Its source is still downloading in a background task. Flip the
        # status now rather than waiting for that to finish: _fan_out skips
        # anything already cancelled, so the job will never start, and
        # "cancelled" is terminal, so the card stops claiming to be working
        # and bulk delete will take it. Same shape as "scheduled" above.
        #
        # The download itself keeps running — yt-dlp is a blocking
        # subprocess inside a worker thread with no cancellation channel —
        # but nothing waits on it and _prepare discards the file if every
        # sibling in the batch is gone.
        j["status"] = "cancelled"
        j["cancel_requested"] = True
        j["step_detail"] = "Cancelled while preparing"
        save_job(j)
        log.info(f"[multidub] Job {job_id} cancelled while preparing")
        return {"ok": True, "cancelled_from": "preparing"}

    if status == "queued":
        # Drain queue, drop target job, push rest back. asyncio.Queue
        # doesn't support random removal directly.
        drained = []
        while not _job_queue.empty():
            try:
                item = _job_queue.get_nowait()
                if item[0] != job_id:
                    drained.append(item)
            except asyncio.QueueEmpty:
                break
        for item in drained:
            await _job_queue.put(item)
        j["status"] = "cancelled"
        j["step_detail"] = "Cancelled before start"
        save_job(j)
        log.info(f"[queue] Job {job_id} cancelled (removed from queue)")
        return {"ok": True, "cancelled_from": "queue"}

    # Running job — set flag, terminate TTS subprocess if mid-synth.
    # The pipeline's update() closures will pick up the flag at the next
    # stage boundary and raise JobCancelled cleanly.
    j["cancel_requested"] = True
    j["step_detail"] = "Cancelling..."
    save_job(j)
    # Proactively kill TTS worker so synth segments don't have to finish
    if status == "synthesizing":
        _maybe_terminate_tts_worker()
    log.info(f"[queue] Job {job_id} cancel requested (was {status})")
    return {"ok": True, "cancelled_from": "running",
            "message": "Cancel requested. Job will stop at the next stage boundary "
                       "(usually within 1-5 seconds)."}


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Not found"}, 404)
    work = OUTPUT_DIR / job_id
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    delete_job_db(job_id)
    jobs.pop(job_id, None)
    app_audit.record("job.delete", target=job_id)
    return {"ok": True}


# Statuses whose output directory is being written to right now. Deleting one
# rmtree's the directory out from under a live ffmpeg or TTS worker, which
# fails the job in a way that looks like a pipeline bug rather than a deletion.
_UNDELETABLE_STATUSES = {"running", "processing", "queued", "downloading",
                         "preparing"}


@app.post("/api/jobs/bulk_delete")
async def bulk_delete_jobs(job_ids: str = Form(...)):
    """Delete many jobs in one call.

    The UI needs this to be one round trip rather than N: selecting thirty
    jobs and firing thirty DELETEs means thirty chances to half-finish, and
    no single answer to show afterwards.

    Returns per-id outcomes rather than failing the whole batch on the first
    problem — a selection that happens to include one running job should
    delete the other twenty-nine and say so.
    """
    ids = [s.strip() for s in job_ids.split(",") if s.strip()]
    if not ids:
        return JSONResponse({"error": "No job_ids given"}, 400)
    if len(ids) > 500:
        return JSONResponse({"error": "Too many ids in one call (max 500)"}, 400)

    deleted, skipped, freed = [], [], 0
    for job_id in ids:
        job = jobs.get(job_id)
        if job is None:
            skipped.append({"id": job_id, "reason": "not found"})
            continue
        if (job.get("status") or "") in _UNDELETABLE_STATUSES:
            skipped.append({"id": job_id, "reason": f"job is {job['status']} — cancel it first"})
            continue
        work = OUTPUT_DIR / job_id
        try:
            if work.exists():
                freed += _dir_size_bytes(work)
                shutil.rmtree(work, ignore_errors=True)
            delete_job_db(job_id)
            jobs.pop(job_id, None)
            deleted.append(job_id)
        except Exception as e:
            log.warning(f"[delete] {job_id} failed: {e}")
            skipped.append({"id": job_id, "reason": str(e)[:120]})

    if deleted:
        app_audit.record("job.bulk_delete", target=f"{len(deleted)} jobs",
                         detail=",".join(deleted[:20]) + ("…" if len(deleted) > 20 else ""))
        log.info(f"[delete] removed {len(deleted)} job(s), freed {freed/1e6:.0f} MB")
    return {"ok": True, "deleted": deleted, "skipped": skipped,
            "freed_bytes": freed, "freed_mb": round(freed / 1e6, 1)}


# ═══════════════════════════════════════════════════════════════════════
#  Storage management — keep outputs/ from eating your drive
# ═══════════════════════════════════════════════════════════════════════
# Each dub leaves ~100-500 MB in outputs/{job_id}/ (source video, audio
# extracts, per-segment WAVs, dubbed video). Over a 5-course semester
# that's easily 20-50 GB.
#
# Design:
#   - Jobs can be "starred" to protect them from bulk cleanup. Stars
#     persist in the job dict (jobs[id].starred = True).
#   - /api/storage/stats returns total disk usage per job + aggregate.
#   - /api/storage/cleanup takes { older_than_days, mode } and deletes
#     jobs matching the criteria, skipping starred jobs. mode can be:
#       "all_files"  — rm -rf the whole outputs/{job_id} dir
#       "intermediate" — keep dubbed_video.mp4 + .srt, delete the rest
#         (source video, per-segment WAVs, intermediate audio ~90% saving)
#   - Dry-run by default; caller must pass dry_run=false to actually delete.
# ═══════════════════════════════════════════════════════════════════════

def _dir_size_bytes(path: Path) -> int:
    """Fast recursive directory size via os.scandir. Returns 0 on error."""
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += _dir_size_bytes(Path(entry.path))
            except OSError:
                continue
    except OSError:
        pass
    return total


# Files we keep when cleaning "intermediate" artifacts. These are the
# user-visible deliverables; everything else is regenerable from checkpoints.
_KEEP_ON_INTERMEDIATE_CLEAN = {
    "dubbed_video.mp4",
    "dubbed_video_subs.mp4",  # burn-in output
    "translated.srt",
    "checkpoint_translation_done.json",
    "checkpoint_tts_done.json",
    # Keep final so Resume stays possible even on cleaned jobs
}


@app.get("/api/storage/stats")
async def storage_stats():
    """Aggregate disk usage across all job outputs. Per-job breakdown so
    the UI can show a sortable list and identify the biggest offenders."""
    rows = []
    total_bytes = 0
    for jid, job in jobs.items():
        work = OUTPUT_DIR / jid
        if not work.exists():
            continue
        size = _dir_size_bytes(work)
        total_bytes += size
        rows.append({
            "id": jid,
            "label": job.get("source_label", jid),
            "status": job.get("status", ""),
            "target_lang": job.get("target_lang", ""),
            "created": job.get("created", 0),
            "completed_at": job.get("completed_at", 0),
            "starred": bool(job.get("starred")),
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 1),
            "age_days": round((time.time() - job.get("created", time.time())) / 86400, 1),
        })
    rows.sort(key=lambda r: -r["size_bytes"])
    return {
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 1),
        "total_gb": round(total_bytes / 1024 / 1024 / 1024, 2),
        "job_count": len(rows),
        "jobs": rows,
    }


@app.post("/api/storage/star/{job_id}")
async def toggle_star(job_id: str, starred: bool = Form(...)):
    """Star/unstar a job to protect it from bulk cleanup."""
    if job_id not in jobs:
        return JSONResponse({"error": "Not found"}, 404)
    jobs[job_id]["starred"] = bool(starred)
    save_job(jobs[job_id])
    return {"ok": True, "starred": jobs[job_id]["starred"]}


@app.post("/api/storage/cleanup")
async def cleanup_storage(
    older_than_days: int = Form(30),
    mode: str = Form("intermediate"),  # "intermediate" | "all_files"
    dry_run: bool = Form(True),
    include_errored: bool = Form(True),
    include_cancelled: bool = Form(True),
):
    """Bulk-remove old job outputs. Starred jobs are always skipped.

    - older_than_days=N — affects jobs created more than N days ago
    - mode=intermediate — keep the dubbed mp4 + srt + checkpoints, drop
      source video + per-segment WAVs + intermediate audio (~90% saving
      on typical jobs, job stays "viewable" but Regenerate may need
      re-download)
    - mode=all_files — rm -rf the whole job output dir (job record kept;
      UI will show "(files deleted)" and no View button)
    - dry_run=True — report what would be deleted, don't touch disk.
    """
    if mode not in ("intermediate", "all_files"):
        return JSONResponse({"error": f"Invalid mode: {mode}"}, 400)

    cutoff = time.time() - older_than_days * 86400
    candidates = []
    for jid, job in jobs.items():
        if job.get("starred"):
            continue
        if job.get("created", 0) > cutoff:
            continue
        status = job.get("status", "")
        # Only touch jobs in a settled state. Don't clean a running job.
        if status not in ("complete", "error", "cancelled"):
            continue
        if status == "error" and not include_errored:
            continue
        if status == "cancelled" and not include_cancelled:
            continue
        work = OUTPUT_DIR / jid
        if not work.exists():
            continue
        candidates.append((jid, job, work))

    bytes_freed = 0
    deleted_files = []
    errors = []
    for jid, job, work in candidates:
        try:
            if mode == "all_files":
                size = _dir_size_bytes(work)
                if not dry_run:
                    shutil.rmtree(work, ignore_errors=True)
                    # Mark job as files-deleted so UI can show it without
                    # trying to link to missing mp4. Keep db entry so
                    # history preserves the metadata.
                    job["files_deleted"] = True
                    save_job(job)
                deleted_files.append({"id": jid, "mode": "all_files",
                                       "size_mb": round(size/1024/1024, 1)})
                bytes_freed += size
            else:  # intermediate
                freed_here = 0
                for entry in list(os.scandir(work)):
                    if entry.name in _KEEP_ON_INTERMEDIATE_CLEAN:
                        continue
                    try:
                        if entry.is_file(follow_symlinks=False):
                            sz = entry.stat().st_size
                            if not dry_run:
                                Path(entry.path).unlink()
                            freed_here += sz
                        elif entry.is_dir(follow_symlinks=False):
                            sz = _dir_size_bytes(Path(entry.path))
                            if not dry_run:
                                shutil.rmtree(entry.path, ignore_errors=True)
                            freed_here += sz
                    except OSError as e:
                        errors.append(f"{jid}: {entry.name}: {e}")
                if freed_here > 0:
                    deleted_files.append({"id": jid, "mode": "intermediate",
                                          "size_mb": round(freed_here/1024/1024, 1)})
                    bytes_freed += freed_here
                if not dry_run:
                    job["intermediate_cleaned"] = True
                    save_job(job)
        except Exception as e:
            errors.append(f"{jid}: {e}")

    action = "would delete" if dry_run else "deleted"
    log.info(f"[cleanup] {action} {len(deleted_files)} jobs, "
             f"{round(bytes_freed/1024/1024/1024, 2)} GB freed "
             f"(mode={mode}, older_than={older_than_days}d, dry_run={dry_run})")
    return {
        "dry_run": dry_run,
        "mode": mode,
        "candidates": len(candidates),
        "affected": len(deleted_files),
        "bytes_freed": bytes_freed,
        "mb_freed": round(bytes_freed / 1024 / 1024, 1),
        "gb_freed": round(bytes_freed / 1024 / 1024 / 1024, 2),
        "details": deleted_files[:50],  # cap to keep payload small
        "errors": errors[:10],
    }


# The keys a job list actually needs to render a card. The rest of a job
# dict is dead weight on a poll: _strip_large_fields (app/db.py) drops the
# transcript but not meta["description"], which curate_metadata caps at
# 10,000 characters — per job, on every request.
_COMPACT_JOB_KEYS = (
    "id", "status", "progress", "step_detail", "error",
    "target_lang", "source_label", "title", "duration",
    "created", "started_at", "completed_at",
    "batch_id", "batch_label", "batch_kind", "batch_position", "batch_total",
    "output_url", "srt_url",
    "has_checkpoint", "latest_checkpoint_stage",
)
_COMPACT_META_KEYS = ("title", "thumbnail", "duration", "channel", "webpage_url")


def _compact_job(job: dict) -> dict:
    out = {k: job[k] for k in _COMPACT_JOB_KEYS if k in job}
    meta = job.get("meta") or {}
    if isinstance(meta, dict):
        out["meta"] = {k: meta[k] for k in _COMPACT_META_KEYS if k in meta}
    return out


@app.get("/api/jobs")
async def list_jobs(
    status: str | None = None,
    batch_id: str | None = None,
    limit: int = 0,
    since: float = 0,
    compact: bool = False,
):
    # Annotate each job with checkpoint info so the History UI can
    # decide whether to show a Resume button. This is intentionally
    # done at read time (not stored on the job object) because users
    # can delete output dirs manually — reading live keeps UI honest.
    selected = jobs.values()
    if status:
        wanted = {s.strip() for s in status.split(",") if s.strip()}
        selected = [j for j in selected if j.get("status") in wanted]
    if batch_id:
        selected = [j for j in selected if j.get("batch_id") == batch_id]
    if since > 0:
        selected = [j for j in selected if j.get("created", 0) >= since]
    sorted_jobs = sorted(selected, key=lambda j: j.get("created", 0), reverse=True)
    if limit > 0:
        sorted_jobs = sorted_jobs[:limit]
    enriched = []
    for j in sorted_jobs:
        info = _job_checkpoint_info(j["id"])
        # Shallow-copy so we don't mutate the in-memory job store
        row = {**j, **info}
        enriched.append(_compact_job(row) if compact else row)
    return {"jobs": enriched}


@app.get("/api/languages")
async def list_supported_languages():
    """Canonical supported target-language codes (65).

    Derived from the edge-tts fallback voice map in pipeline/synthesizer.py
    — the one place every dubbable language must be registered, since
    edge-tts is the floor every target language falls back to when VoxCPM2
    is unavailable. The CLI/MCP `languages` listing reads this instead of
    hardcoding a stale subset.
    """
    codes = list(EdgeTTSFallback.VOICE_MAP.keys())
    # `languages` stays a bare code list — tools/gochidubb_client.py returns
    # it verbatim and the CLI joins it into a string, so changing its shape
    # would break both. Names are added alongside, read from the translator's
    # LANGUAGE_NAMES rather than copied, so a newly registered language shows
    # up here without a second list to remember to update.
    names = {c: LANGUAGE_NAMES.get(c, c) for c in codes}
    return {
        "languages": codes,
        "names": names,
        "catalog": [{"code": c, "name": names[c]} for c in codes],
    }


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs or jobs[job_id].get("status") != "complete":
        return JSONResponse({"error": "Not ready"}, 400)
    path = str(OUTPUT_DIR / job_id / "dubbed_video.mp4")
    if not os.path.exists(path):
        return JSONResponse({"error": "File missing"}, 404)
    return FileResponse(path, filename=f"dubbed_{job_id}.mp4")


# ── Which front door ─────────────────────────────────────────────────
# Creator mode and Pro mode are two surfaces on the same account, the same
# jobs table and the same meter — a mode, not a tier. GET / honours a
# preference between them, and every failure mode of reading that preference
# resolves to Pro, which is what this route has always served. An existing
# user must never be stranded on a page they did not ask for by a missing
# file, an unreadable one, or a value nobody wrote.

def _ui_mode() -> str:
    """"creator" only on an explicit, readable preference; "pro" otherwise."""
    try:
        if not PREFS_FILE.exists():
            return "pro"
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            prefs = json.load(f)
        if isinstance(prefs, dict) and prefs.get("ui_mode") == "creator":
            # A preference pointing at a page that is not on disk would be a
            # blank screen with no way back except knowing /pro exists.
            if (STATIC_DIR / "creator.html").exists():
                return "creator"
            log.warning("[ui] ui_mode=creator but static/creator.html is "
                        "missing — serving Pro mode")
    except Exception as e:
        log.warning(f"[ui] could not read ui_mode preference ({e}) — "
                    f"serving Pro mode")
    return "pro"


@app.get("/")
async def index():
    if _ui_mode() == "creator":
        return FileResponse(str(STATIC_DIR / "creator.html"))
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/pro")
async def pro_index():
    """Pro mode, unconditionally — the escape hatch.

    Never redirects and never reads the preference: if someone's preference
    is wrong, or the creator page is broken, this is how they get their tool
    back. /creator is unconditional in the same way, in the other direction.
    """
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Stage reuse (beta) ───────────────────────────────────────────────
# Kept on their own /api/beta/ prefix and served by their own page, so the
# feature can be evaluated — or removed — without touching the main UI.

@app.get("/creator")
async def creator_index():
    """Creator mode — the consumer front door.

    Its own page rather than a mode inside index.html: that page loads React
    and Babel from a CDN and transpiles ~6,000 lines of JSX in the browser on
    every load, which is the wrong first paint for the screen a new user
    lands on.

    Unconditional, like /pro: it serves creator.html whatever the stored
    preference says, so each mode always has a direct address that works.
    """
    return FileResponse(str(STATIC_DIR / "creator.html"))


@app.get("/admin")
async def admin_index():
    """The vendor admin console — the design's `3a` screens.

    Unconditional, like /pro and /creator. It is a separate surface from the
    customer workspace (in the design, a separate host behind staff SSO), so
    it gets its own address rather than a tab inside either mode: a page that
    reads the revenue estimate, every API key and the audit trail should not
    be one mis-click away from the screen a creator uses.
    """
    return FileResponse(str(STATIC_DIR / "admin.html"))


@app.get("/beta")
async def beta_index():
    return FileResponse(str(STATIC_DIR / "beta.html"))


@app.get("/api/beta/reuse/status")
async def beta_reuse_status():
    """Config, per-stage cache contents, and what the gates are set to."""
    st = artifact_store.stats(OUTPUT_DIR)
    allowed = reuse_runtime.enabled_stages(cfg)
    by_stage = {s["stage"]: s for s in st.get("stages", [])}
    stages = [{
        "stage": stage,
        "allowed": stage in allowed,
        "cacheable": True,
        "entries": by_stage.get(stage, {}).get("entries", 0),
        "hits": by_stage.get(stage, {}).get("hits", 0),
        "version": app_reuse.STAGE_VERSIONS.get(stage),
        "inputs": list(app_reuse.STAGE_INPUTS.get(stage, ())),
    } for stage in app_reuse.CACHEABLE_STAGES]
    return {
        "enabled": bool(getattr(cfg, "reuse_enabled", False)),
        "reuse_stages": getattr(cfg, "reuse_stages", ""),
        "gates": app_reuse.gates_from_config(cfg),
        "stages": stages,
        "total_entries": st.get("total", 0),
        "total_hits": st.get("hits", 0),
    }


@app.get("/api/beta/reuse/entries")
async def beta_reuse_entries(stage: str = "", limit: int = 100):
    return {"entries": artifact_store.entries(stage=stage, limit=limit)}


@app.get("/api/beta/reuse/job/{job_id}")
async def beta_reuse_job(job_id: str):
    """What a given finished job reused, and what it contributed."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": f"Job {job_id} not found"}, 404)
    return {
        "job_id": job_id,
        "reused": job.get("reused_stages") or [],
        "contributed": [e for e in artifact_store.entries(limit=500)
                        if e["job_id"] == job_id],
    }


@app.get("/api/beta/reuse/plan/{job_id}")
async def beta_reuse_plan(job_id: str, target_lang: str = ""):
    """What re-running this job would reuse — without running anything.

    Answers the question the feature exists for: "if I dub this into another
    language, what do I actually pay for?" Reads the job's own checkpoints, so
    it only works once the job has produced them.
    """
    if job_id not in jobs:
        return JSONResponse({"error": f"Job {job_id} not found"}, 404)
    ctx: dict = {}
    for spec in PIPELINE_STAGES:
        cp = _load_checkpoint(job_id, stage=spec["checkpoint"])
        if cp:
            ctx.update(cp)
    if not ctx:
        return JSONResponse(
            {"error": "No checkpoints for this job yet — run it first"}, 409)
    if target_lang:
        ctx["target_lang"] = target_lang
    return {
        "job_id": job_id,
        "target_lang": ctx.get("target_lang"),
        "enabled": bool(getattr(cfg, "reuse_enabled", False)),
        "stages": reuse_runtime.plan(ctx, OUTPUT_DIR, cfg),
    }


@app.post("/api/beta/reuse/purge")
async def beta_reuse_purge(stage: str = Form(""), job_id: str = Form("")):
    """Forget cached artifacts. Only drops index rows — never job files."""
    removed = artifact_store.purge(stage=stage, job_id=job_id)
    log.info(f"[artifacts] purged {removed} row(s) "
             f"(stage={stage or 'all'}, job={job_id or 'all'})")
    return {"ok": True, "removed": removed}


if __name__ == "__main__":
    import uvicorn
    import atexit

    # ── PID file management ──────────────────────────────────────────
    # Write a PID file so the server manager (gochidubb_serverctl.py)
    # can track and manage this process. The PID file is cleaned up on
    # normal exit via atexit. If the process is killed with SIGKILL,
    # the stale PID file is detected and cleaned by the manager.
    _PID_FILE = Path(__file__).parent / ".gochidubb.pid"
    try:
        _PID_FILE.write_text(str(os.getpid()))
        atexit.register(lambda: _PID_FILE.unlink(missing_ok=True))
    except OSError:
        pass  # Non-fatal — server manager will find us via `ps` fallback

    # Backstop for exits that skip the lifespan teardown (crash, --reload
    # respawn, SystemExit from the signal handler below): make sure no ffmpeg
    # or TTS worker outlives us. atexit must never raise.
    def _atexit_reap_children():
        try:
            _terminate_child_processes(timeout=2.0)
        except Exception:
            pass

    atexit.register(_atexit_reap_children)

    # ── Signal handling for graceful shutdown ────────────────────────
    # When the server manager sends SIGTERM (or the user presses Ctrl+C),
    # uvicorn handles it internally and runs the lifespan shutdown phase.
    # We also register a fallback to ensure the PID file is cleaned up
    # even if uvicorn's signal handler doesn't fire atexit.
    def _signal_cleanup(signum, frame):
        _PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_cleanup)
    signal.signal(signal.SIGINT, _signal_cleanup)

    # ── Parse --reload flag for development ──────────────────────────
    _reload = "--reload" in sys.argv

    # Loopback by default. This server has no authentication of any kind: any
    # caller can start jobs, read every transcript, delete jobs and reach
    # /api/logs and /api/config. Binding 0.0.0.0 offered all of that to every
    # device on the network — including cafe and hotel wifi — as the default
    # for a tool most people run on a laptop.
    #
    # Exposing it deliberately (dubbing box in the corner, phone on the couch)
    # is still one env var away, and now it's a decision rather than a surprise.
    _host = os.getenv("GOCHIDUBB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    _port = int(os.getenv("GOCHIDUBB_PORT", "") or cfg.server_port or 8910)

    print("")
    print("+====================================================+")
    print("|  GoChiDUBB Studio - AI Video Dubbing               |")
    print(f"|  http://localhost:{_port}                             |")
    print("|  Press Ctrl+C to stop                              |")
    print("+====================================================+")
    if _host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[warn] Listening on {_host} — this server has no authentication.")
        print("[warn] Anyone who can reach this port can read and delete your jobs.")
    print("")
    uvicorn.run(app, host=_host, port=_port, log_level="info", reload=_reload)
