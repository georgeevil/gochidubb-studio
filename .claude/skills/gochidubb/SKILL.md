---
name: gochidubb
description: Use when the user asks to dub a video, translate speech in a video, voice-clone across languages, build a multilingual reel, or compare dubbing quality. Wraps the GoChiDUBB Studio local AI dubbing pipeline (whisperx → Ollama → VoxCPM2 → ffmpeg) via either an MCP server (preferred when the `gochidubb_*` tools are available) or a CLI fallback. Handles single-language dubs, side-by-side comparisons across N languages, stitched multilingual showcase reels, and re-dubs of previously processed sources.
---

# GoChiDUBB — agentic dubbing

A local, offline AI video dubbing studio. The user runs the server on
their own machine — no upload to cloud, no per-minute fees. This skill
makes the pipeline drivable through natural language.

**For humans reading this file:** copy this folder into `~/.claude/skills/`
(so the path becomes `~/.claude/skills/gochidubb/SKILL.md`) and Claude
will know how to use the GoChiDUBB MCP or CLI when you ask. After that,
ask things like *"dub this YouTube short into French and German"* or
*"build a 5-language showcase reel of this clip"* — Claude picks the
right tool and waits for it.

Repo: https://github.com/georgeevil/gochidubb-studio
By [George Chigrichenko](https://x.com/GChigrichenko) ([@georgeevil](https://github.com/georgeevil)), built with Claude.

## When this skill applies

Trigger on requests like:
- "Dub this video into French"
- "Translate the speech in https://… to Spanish and German"
- "Make a showcase reel of this short in 5 languages"
- "Re-dub job 5038e404 into Japanese"
- "What's the status of my dub?"
- "List my completed dubs"
- "Compare voice quality across 3 languages on this clip"

## Two access paths — pick what's available

**Path A — MCP (preferred).** If tools named `gochidubb_*` are exposed by
an MCP server, use them directly. They return structured JSON.

**Path B — CLI fallback.** If MCP isn't connected, shell out:
```
python tools/gochidubb_cli.py <subcommand> ...
```
Same capabilities, JSON output. Works from anywhere a shell can run.

## Prerequisites — check first when a call fails

The user must have the GoChiDUBB server running (default `http://localhost:8910`).
If a tool call fails with a connection error, ask the user to start it:
```
python server.py        # from your gochidubb checkout (Windows: also start.bat)
./start.sh              # on Linux/macOS
```

Use `gochidubb_system_status` (MCP) or `gochidubb system` (CLI) to surface
missing dependencies before submitting work. Common failures:
- `ollama_binary.ok=False` → "User needs Ollama installed and running"
- `voxcpm.ok=False` → falls back to edge_tts (lower quality, no cloning)
- `gpu.ok=False` → still works on CPU but slow

## Core tools / commands

| Task | MCP tool | CLI |
|---|---|---|
| Dub in one language | `gochidubb_dub` | `gochidubb dub <src> --lang fr` |
| N separate dubs (compare) | `gochidubb_compare` | `gochidubb compare <src> --langs es,fr,de` |
| Stitched multilingual reel | `gochidubb_showcase` | `gochidubb showcase <src> --langs es,fr,de` |
| Re-dub existing job | `gochidubb_redub` | `gochidubb redub <jid> --langs ja,it --mode showcase` |
| Job status | `gochidubb_get_job` | `gochidubb status <jid>` |
| List jobs | `gochidubb_list_jobs` | `gochidubb jobs` |
| Showcase status | `gochidubb_get_showcase` | `gochidubb showcase-status <bid>` |
| Rebuild showcase stitch | `gochidubb_rebuild_showcase` | `gochidubb showcase-rebuild <bid>` |
| System health | `gochidubb_system_status` | `gochidubb system` |
| Supported languages | `gochidubb_list_languages` | `gochidubb languages` |
| Installed translation models | `gochidubb_list_models` | `gochidubb models` |

## Source argument — URL or local path

Every submission takes a `source`:
- **URL** (starts with `http://` or `https://`) — yt-dlp downloads it.
  Best for YouTube/Shorts.
- **Local file path** — uploaded to the server. Use absolute paths.

## Language codes

`en ru es pt fr de it pl tr ja ko zh ar hi nl` — and `auto` for source language.

## Models

Translation is via local Ollama. Default `aya-expanse:8b` (multilingual,
~5GB). If the user's machine doesn't have it, the server auto-falls back
to whichever installed model is best (`gemma3:12b`, `qwen3:8b`, etc.).
Check available models with `gochidubb_list_models` before suggesting a
specific one in a prompt.

## Blocking vs fire-and-forget

By default, submission tools queue work and return immediately with a
`job_id` / `batch_id`. Pass `wait=True` (MCP) or `--wait` (CLI) when the
user expects you to come back with a finished video URL in the same turn.
Realistic timings on a single GPU:
- One 60s dub: ~2-4 min (most time in transcribe + TTS)
- Compare 5 langs × 60s: ~10-20 min (serial GPU queue)
- Showcase 5 langs × 60s: same as compare + ~10s stitch

For long jobs, prefer submit-then-poll-then-summarize over a 10-minute
blocking wait inside a single tool call.

## Showcase vs Compare — when to suggest which

- **Compare** (`quick_test` on the server): 5 separate mp4 files, one
  per language, full length. Use when the user wants to A/B/C/D listen
  to quality across languages.
- **Showcase**: one single mp4 that switches language every ~12s with a
  small `· LL ·` badge in the top-right corner. Use when the user wants
  a deliverable to demo/share.

## Re-dub from history

When the user references an existing dub ("the French one I made
yesterday"), find it with `gochidubb_list_jobs(status='complete')`,
then `gochidubb_redub(job_id, target_langs=[...], mode=...)`. This
skips re-upload and re-download.

Modes:
- `single`: one new language → one new job
- `compare`: 2-6 new languages → N separate jobs (batch_kind=quick_test)
- `showcase`: 2-6 new languages → N jobs + stitch into one reel

## Failure recovery

- A job with `status="error"` and `has_checkpoint=True` can resume from
  the last completed pipeline stage — call `POST /api/dub/<id>/continue`
  (no MCP tool yet; suggest via CLI: `curl` to that endpoint or just
  hit Resume in the UI).
- Showcase batch where all child jobs completed but the stitch failed:
  use `gochidubb_rebuild_showcase(batch_id)` — much faster than rerunning
  the N dubs.

## Example multi-step flow

User: "Take https://youtu.be/abc, give me a 60-second multilingual
reel in 5 most popular languages and tell me when it's done."

1. `gochidubb_system_status` → confirm GPU + ollama_binary + voxcpm
2. `gochidubb_list_models` → pick a multilingual model from what's installed
3. `gochidubb_showcase(source="https://youtu.be/abc",
                       target_langs=["es","fr","de","ja","pt"],
                       trim_seconds=60, wait=True)`
4. Report URL: `{result.url}` to user
5. If `wait_timeout` exceeded → return `batch_id` and tell user to use
   `gochidubb_get_showcase(batch_id)` later.
