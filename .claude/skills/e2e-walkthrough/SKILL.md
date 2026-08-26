---
name: e2e-walkthrough
description: Execute one of the GoChiDUBB workflow runbooks (docs/runbooks/*.md) end-to-end in the user's own Chrome via the claude-in-chrome MCP, narrating each step so the human can watch live, while making every hard assertion against the API. Use when the user asks to "run the E2E", "walk through <workflow>", "verify the <x> flow", or names a runbook. Real pipeline, no mocks — a walkthrough takes minutes and shows real dubbing.
---

# E2E walkthrough — Claude drives, the human watches

Runbooks live in `docs/runbooks/`; `docs/runbooks/README.md` lists them
and defines the step format (Do / See / Assert / On failure). This skill
is the executor.

**For humans:** ask *"run the gated dub walkthrough"* or
*"/e2e-walkthrough sync-and-subtitles"* and watch your own browser — the
agent drives it tab-visible, narrates each step in chat, and reports a
step-by-step pass/fail at the end.

## Execution contract

1. **Read the runbook first**, whole. If it names a fixture
   (`docs/runbooks/fixtures/clip.mp4`) that is missing, stop and ask the
   user for a short clip rather than picking a random file from
   `outputs/` — those are their real jobs.
2. **Preconditions before browser**: `python tools/gochidubb_serverctl.py
   restart`, then `venv/bin/python tools/gochidubb_cli.py system`. If the
   server or a model is missing, report and stop.
3. **Browser**: claude-in-chrome MCP (load its tools via ToolSearch in ONE
   batched select). One tab, `http://localhost:8910/...`, `?cb=<ts>`
   cache-busters after any static-file change. The human is watching the
   same window: move deliberately, screenshot at each **See** row, and
   narrate what the screenshot should show before acting on it.
4. **Assertions**: every **Assert** row runs against the API (curl or the
   CLI) and is pass/fail — browser judgment never substitutes for one.
   Webhook assertions read the delivery ring at `GET /api/webhooks`
   (register a hook to an unreachable port in step 1 of runbooks that
   need one; attempts are recorded either way).
5. **Track a checklist** (one item per runbook step) and finish with a
   table: step · pass/fail · evidence (the measured value, not "ok").
6. **Never touch what isn't yours**: only jobs this run created may be
   continued, retried, edited, or deleted. Anything in `outputs/` that
   predates the run is read-only unless the runbook's cleanup section
   explicitly restores it.
7. **Cleanup always runs** — on failure too. Test jobs deleted, config
   restored to noted values, webhooks removed, `ui_mode` put back if the
   runbook flipped it.
8. **Optional recording**: on request, capture the walkthrough with the
   claude-in-chrome `gif_creator` and store per the ffmpeg palette
   convention in `docs/README.md`.

## Failure protocol

A failed Assert stops forward progress (cleanup still runs). Report the
step, the expected vs measured value, and — when the runbook's
"On failure" row names a suspect — that suspicion. File nothing
automatically; the human decides whether it's a bug or a drifted runbook.
When the Linear workspace is writable, offer to file; otherwise append to
the repo's `linear-drafts-*.md` pattern.

## Speed levers (documented, not defaults)

- `GOCHIDUBB_REUSE=1` — second run of the same clip skips
  download/extract/transcribe/diarize (~60% of wall-clock).
- A 30–60 s fixture keeps any full run under ~3 minutes; `compare`'s
  `--trim` exists for batch flows, single `dub` needs a short file.
