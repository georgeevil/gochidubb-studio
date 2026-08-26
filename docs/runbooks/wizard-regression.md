# Wizard-mode regression — the legacy contract pauses exactly once

`wizard_mode` predates the gate set; every old client (creator, CLI
scripts, MCP callers) sends it. The mapping contract: a sent wizard_mode
(INCLUDING "auto") suppresses any configured gate defaults and arms at
most ONE gate.

**1. Arm gate defaults globally.** PATCH /api/config:
`review_gate_subtitles=on` (note prior value; restored at the end).

**2. Legacy single-pause.**
`CLI dub $CLIP --lang ru --wizard-mode review_translation`
- Assert: the job pauses ONCE at `awaiting_translation_review`; after
  `CLI continue <id>` it runs to `complete` — the subtitles default did
  NOT fire (wizard_mode suppressed it).

**3. Auto suppresses too.**
`CLI dub $CLIP --lang ru` (the CLI sends wizard_mode=auto)
- Assert: zero pauses; `complete` with `gates_cleared == []`.

**4. Explicit gates win over both.**
`CLI dub $CLIP --lang ru --review-gates subtitles=on`
- Assert: exactly one pause, at `awaiting_subtitle_review`.

**5. Clean up.** Restore `review_gate_subtitles`; delete the three jobs.
