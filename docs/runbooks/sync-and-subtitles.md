# Sync fit + subtitle studio — fix timing and cues without re-dubbing

Runs against any COMPLETED job with placements (make one from `$CLIP` if
none exists). Everything here re-runs only assemble+merge — seconds, no
re-synthesis, cost line says "free".

**1. Open the tabs.** `/pro#/review/<id>/sync` (deep link straight to the
Sync tab — works even on a completed job because the data exists).
- Assert: `GET /api/dub/<id>/sync_plan` rows match the table.

**2. Stage a fix.** On an off row: "speed ×N" (or "pad 0.3s" for an early
row); watch the plan chip appear.
- Do: "Apply plan & re-assemble ▸".
- Assert: `POST /sync_plan` stored `sync_overrides`; after the re-run,
  `GET /sync_plan` shows the fixed row inside tolerance (content
  permitting) and untouched segments' audio files are byte-identical.

**3. Shorten a line.** "shorten text" on a hot row → edit → "Regenerate
this line" (this one DOES re-synth one segment; the estimate line prices
it before the click).
- Assert: `POST /estimate_edits` for that idx returned a non-free price;
  after regeneration the segment's audio file changed, all others didn't.

**4. Cue editor.** Subtitles tab: edit a cue's display text (free), watch
the "edited" marker; auto-fix any warnings; download .srt and .vtt.
- Assert: `violation_count == 0` after auto-fix; the .vtt on disk carries
  the display override; `translated_text` in the checkpoint does NOT (the
  what-is-shown vs what-was-spoken contract).

**5. Clean up.** If this ran on a user's real job: restore by removing
`sync_overrides` (`POST /sync_plan` with `{}`) and the cue override
(edit with `null`), then re-assemble once more. On a throwaway job:
delete it.
