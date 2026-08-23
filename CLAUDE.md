# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GoChiDUBB Studio: a local, offline AI video dubbing pipeline (YouTube URL or file → voice-cloned dub in 65 languages). Everything runs on the user's machine — no cloud, no API keys required by default. FastAPI server on port 8910 with a static UI (`static/index.html`), an MCP server, and a CLI — all three drive the same backend.

## Common commands

The `venv/` python is the one to use (`venv/bin/python` on macOS/Linux, `venv\Scripts\python.exe` on Windows). `tools/gochidubb_serverctl.py` picks it up automatically.

```bash
# Run server (foreground dev mode with auto-reload)
python tools/gochidubb_serverctl.py foreground --reload

# Run server (background, PID-tracked; survives terminal close)
python tools/gochidubb_serverctl.py start
python tools/gochidubb_serverctl.py stop        # graceful SIGTERM, force-kills after 10s
python tools/gochidubb_serverctl.py restart
python tools/gochidubb_serverctl.py status      # detects orphaned processes
python tools/gochidubb_serverctl.py logs --follow

# Lint (CI-equivalent — this is all CI runs on push/PR)
ruff check .

# Syntax check what CI compiles
python -m compileall -q server.py pipeline/ tools/ app/

# Tests
pytest                                # full suite
pytest tests/test_translator.py       # single file
pytest tests/test_translator.py::test_name -v
pytest -k "assembler and not slow"    # filter by expression
```

Always use `tools/gochidubb_serverctl.py` rather than raw `python server.py &` — the manager writes `.gochidubb.pid` and cleans up orphans left behind by agentic restarts. `restart` is idempotent even if the previous process crashed.

`tests/manual_test_*.py` files are exploratory scripts, not pytest tests — don't run them under pytest.

## Architecture

### The pipeline is a linear chain of drop-in stages

Each file in `pipeline/` is one stage. They compose in `server.py`'s job runner in this fixed order:

```
downloader.py     yt-dlp → local file
audio.py          ffmpeg extract, hq extract, optional demucs/audio-separator background split
transcriber.py    faster-whisper → segments + word timestamps
diarizer.py       pyannote → speaker turns (optional; needs HF_TOKEN)
segment_post.py   sentence-boundary repair, split long segments, drop micro-fragments
translator.py     LM Studio (OpenAI /v1) OR Ollama → target-language text, length-matched, batched N lines/request
tts_worker.py     orchestrates synthesizer.py with tts_qa.py retry policy
synthesizer.py    VoxCPM2 / F5-TTS / CosyVoice / edge-tts (BaseTTSEngine + subclasses)
tts_qa.py         whisper-roundtrip QA check on each synthesized segment
assembler.py      time-align + mix with background + ffmpeg render final MP4 + write SRT
vad.py            silero VAD, used by audio.py to trim silence before whisper
models.py         system status + MODEL_CATALOG (Ollama pull streaming, disk usage)
```

Extending the pipeline usually means adding a subclass to `synthesizer.py` (implement `load`/`unload`/`synthesize_segments`) or `translator.py`, then wiring it into the factory. Don't add global state; use `app.config.cfg` for anything runtime-configurable.

### Server: `server.py` is a single 5.6 KLoC FastAPI module

Not a mistake — it's the coordinator that owns:
- The in-memory `jobs` dict (keyed by job_id, mirrored to SQLite by `app.db.save_job_sync`)
- All ~60 HTTP routes under `/api/*`
- Static file serving from `static/` at `/`
- The single-consumer async job runner that walks the pipeline stages above
- Batch semantics: `batch_id` groups jobs; `batch_kind == "showcase"` triggers `_maybe_assemble_showcase` when the last sibling finishes

Job lifecycle: `POST /api/dub` (single), `/api/dub/batch` (compare — N independent dubs), or `/api/showcase` (stitched reel — N dubs + assembly). `/api/job/{id}/redub` clones a finished job's source and re-runs only translate+TTS+assemble. `/api/dub/{id}/retry_tts`, `/edit_translations`, `/edit_speaker_ref`, `/regenerate_segment/{i}` allow granular re-runs from a checkpoint without re-transcribing.

### Persistence

`app/db.py` is a thin SQLite wrapper over `gochidubb.db` (project root). Schema is one `jobs` table with a JSON blob column and virtual columns (`status`, `created`, `batch_id`) extracted via `json_extract` for indexed queries. On startup it migrates any legacy `jobs_db/*.json` files. Large transient fields (`transcript`, `transcript_raw`, `_pending_args`) are stripped by `_strip_large_fields` before persist — but keep in mind the in-memory `jobs` dict is the source of truth during a live run.

### Configuration layering (highest wins)

1. **Environment variables** (via `.env` — loaded through `python-dotenv` in `server.py`)
2. **`config-user.json`** at project root (auto-written by `app.config.cfg.set(...)`; editable via the Settings tab in the UI)
3. **`UserConfig` dataclass defaults** in `app/config.py`

Env vars override `config-user.json` (see the `env_map` in `_load_config`). If you're adding a new tunable, extend the `UserConfig` dataclass — do NOT read `os.getenv` scattered across the codebase.

Translation backend is dual: `USE_LM_STUDIO=1` (default) hits an OpenAI-compatible endpoint at `LM_STUDIO_URL`; `USE_LM_STUDIO=0` uses Ollama at `OLLAMA_URL`. LM Studio serves completions *serially per model*, hence `LM_STUDIO_MAX_CONCURRENT` auto-tunes based on model size.

### Three parallel access paths

- **UI**: `static/index.html` (single-page JS, no build step, no framework) — talks to `/api/*`
- **CLI**: `tools/gochidubb_cli.py` (subcommands: `dub`, `compare`, `showcase`, `redub`, `status`, `jobs`, `wait`, `showcase-status`, `showcase-rebuild`, `cancel`, `rescue`, `delete`, `system`, `languages`, `models`, `voices`) — all HTTP calls via `tools/gochidubb_client.py`
- **MCP**: `tools/gochidubb_mcp.py` exposes `gochidubb_dub`, `gochidubb_compare`, `gochidubb_showcase`, `gochidubb_redub`, `gochidubb_get_job`, `gochidubb_list_jobs`, `gochidubb_get_showcase`, `gochidubb_rebuild_showcase`, `gochidubb_cancel_job`, `gochidubb_rescue_job`, `gochidubb_delete_job`, `gochidubb_system_status`, `gochidubb_list_languages`, `gochidubb_list_models`, `gochidubb_list_voices`

All three respect `GOCHIDUBB_URL` (default `http://localhost:8910`) so any of them can drive a remote box. The CLI and MCP both go through `GoChiDUBBClient` in `tools/gochidubb_client.py` — if you're adding a capability, add the API route to `server.py` first, then a client method, then wire up CLI/MCP.

## Non-obvious things to know

- **speechbrain/k2 stub block at the top of `server.py` is load-bearing.** It pre-populates `sys.modules` with empty stubs for `k2` and several `speechbrain.integrations.k2_fsa.*` paths BEFORE any pipeline import. WhisperX and pyannote transitively walk speechbrain's namespace, and on Windows the real `k2` wheel doesn't exist. Do not move imports above this block or remove the stubs.
- **Windows FFmpeg DLL registration block (also in `server.py`)** uses `os.add_dll_directory` because Python 3.8+ ignores PATH for DLL loading. Adds any dir containing `avformat*.dll` so `torchcodec` can find them.
- **VoxCPM2 warmup is off by default** (`GOCHIDUBB_WARMUP=0`). Reason: translation LLM gets full GPU first; VoxCPM2 loads lazily on first synth. Only turn on if you have VRAM to burn.
- **Translation is batched, and the batch reply is never trusted blindly.** `pipeline/translator.py` sends N numbered subtitle lines per request (auto-sized from `LM_STUDIO_CONTEXT_LENGTH`, override with `TRANSLATE_BATCH_SIZE`) instead of one request per segment. `_parse_numbered_lines` requires a strictly-sequential 1:1 reply; anything else (a merged line, an extra line, commentary) makes `_translate_group` split the batch in half and retry, down to single lines. Don't "fix" the parser to be lenient — a reply that's one line short would shift every later subtitle onto the wrong audio segment.
- **QA retries must not mutate the RNG seed mid-job** — this was a real bug that produced two different voices for retried segments in the same speaker. `pipeline/tts_worker.py` fixes it. If touching retry logic, keep seeds deterministic per (job, speaker, segment_idx).
- **Ollama model catalog is in `pipeline/models.py::MODEL_CATALOG`** — `POST /api/models/pull` streams progress from `ollama pull` back to the UI.
- **Do not add new dependencies without discussing** (per `CONTRIBUTING.md`). The stack is intentionally lean.
- **No `print` in pipeline code** — use `logging.getLogger("gochidubb.<module>")`.

## Testing notes

There is no full integration test suite — most of the pipeline is tested by running end-to-end. Pytest coverage focuses on pure functions (segment post-processing, VAD math, translator prompt building, assembler timing). Fixtures in `tests/conftest.py` provide realistic sample segments (`sample_segments`, `continuation_segments`, `micro_segments`, `long_segments`) — reuse them when adding tests for `pipeline/segment_post.py` or `pipeline/assembler.py`. `temp_audio_file` generates a valid 0.5s WAV without needing external files.

`tests/test_creator_routes.py` is the only suite that drives `server.app` itself, via `TestClient`. Two things about it are load-bearing:

- **It uses `TestClient(app)` without the context manager on purpose.** The lifespan is what calls `init_db()` and `load_all_jobs()`, so skipping it leaves `app.db._DB_PATH` as None and `save_job_sync` a no-op — a test run cannot touch the real `gochidubb.db`.
- **That also means background tasks are cancelled the moment a request ends.** Without the context manager the event-loop portal is torn down per request, so anything `asyncio.create_task` spawned dies with it — measured, a **1 ms** delay inside the task is enough for it to be cancelled instead of finishing. A test that spawns a real task there passes only when its stub wins that race, and would silently stop covering the real path the moment the stub got slower. Don't race it: `server._spawn_background` is a seam for exactly this — patch it to capture the coroutine and run it with `asyncio.run()`. No sleeps, no polling.

CI (`.github/workflows/lint.yml`) runs `ruff check .` + `python -m compileall` only. Landing a green CI is not evidence a feature works.
