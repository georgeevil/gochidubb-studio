# Workflow runbooks — browser-visible E2E walkthroughs

Each file here defines one end-to-end walkthrough of a real workflow,
written so an AI agent (Claude with the claude-in-chrome MCP) can execute
it against `http://localhost:8910` **in the user's own browser, while the
user watches the same window**. The `/e2e-walkthrough` project skill
(`.claude/skills/e2e-walkthrough/SKILL.md`) is the executor; a human can
also follow any runbook by hand.

## Format

Every runbook is a numbered sequence of steps. Each step has up to four
parts:

| Part | Meaning |
|---|---|
| **Do** | The action — a CLI command, an API call, or a browser interaction |
| **See** | What should be visible in the browser (the human-watchable part) |
| **Assert** | A machine check against the API — the *hard* pass/fail |
| **On failure** | What a failure here usually means, when known |

Rules the executor follows:

- **Hard assertions live on the API side.** Browser checks are agent-judged
  (screenshot + page text) and exist for the human; a runbook passes or
  fails on its `Assert` rows.
- **Webhook assertions use the delivery ring**, not a catcher process:
  register a hook via `POST /api/webhooks`, then read
  `GET /api/webhooks` → `deliveries` — the same rows the human sees in the
  Webhooks panel.
- **Every runbook cleans up after itself**: test jobs deleted, any config
  it changed restored, webhooks it registered removed.
- **Real pipeline, no mocks.** A 30–90 s source clip runs the full gated
  pipeline in ~2–3 minutes on this machine; the wait is part of the demo.
  For faster repeat runs, `GOCHIDUBB_REUSE=1` skips the language-independent
  stages on the second pass (see `docs/stage-reuse.md`).

## Fixtures

Runbooks need a short (30–90 s) real video clip with speech. Put one at
`docs/runbooks/fixtures/clip.mp4` — media files are gitignored, so each
machine supplies its own (any short clip with clear speech works; a
completed job's `outputs/<id>/source_video.mp4` is a fine source to copy
from). Runbooks refer to it as `$CLIP`.

## The runbooks

| Runbook | Covers |
|---|---|
| `gated-dub-review-approve.md` | The flagship: three review gates in one run — translate → subtitles → final QC — with edits, glossary teaching, auto-fix, a loudness retarget, and delivery |
| `creator-flow.md` | Creator mode: submit → confidence-flagged review → approve → result |
| `voice-casting.md` | Per-speaker casting, audition, consent chip |
| `redub-and-reuse.md` | Redub into a new language; stage-reuse acceleration visible at /beta |
| `rescue.md` | A failed download rescued by attaching the file by hand |
| `sync-and-subtitles.md` | Sync-fit fixes and the cue editor, .srt/.vtt export |
| `wizard-regression.md` | The legacy `--wizard-mode` contract still pauses exactly once |
