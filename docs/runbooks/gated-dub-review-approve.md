# Gated dub → review → approve (the flagship walkthrough)

Exercises the whole CLD-263 workbench: a dub armed with three review gates
pauses three times, each pause is reviewed and acted on in the Pro
workbench, and Approve delivers with exactly one `job.completed`.
Runtime: ~4–6 minutes on a 60–100 s clip. Cleans up after itself.

Conventions: `CLI` = `venv/bin/python tools/gochidubb_cli.py`,
`$CLIP` = `docs/runbooks/fixtures/clip.mp4`, `$JOB` = the job id step 2
returns. All URLs are on `http://localhost:8910`.

## Steps

**1. Preconditions.**
- Do: `python tools/gochidubb_serverctl.py restart`, then `CLI system`.
- Assert: server healthy; a TTS engine and a translation model are
  available. Record the current `GET /api/config` values for
  `loudness_target` (restored in step 12).
- Do: `POST /api/webhooks` with `url=http://localhost:9/e2e` and `events=job.awaiting_review,job.completed` (an
  unreachable port — delivery *attempts* are still recorded in the ring,
  which is all the assertion needs). Remember the hook id.

**2. Submit with three gates armed.**
- Do: `CLI dub $CLIP --lang ru --review-gates translation=on,subtitles=on,final_qc=on`
- See: browser at `/pro#/feed` — the job card appears in the Agent feed
  and walks import → analyze → translate.
- Assert: `GET /api/job/$JOB` → `review_gates.translation == "on"`.

**3. First gate: translation.**
- Do: wait (poll `GET /api/job/$JOB`) until `status == awaiting_translation_review`.
- See: `/pro#/review/$JOB` — the workbench opens on the Translation tab;
  the gates rail shows "Translation on · now".
- Assert: `pending_gate == "translation"`; the webhook ring
  (`GET /api/webhooks` → deliveries) has a `job.awaiting_review` attempt
  for `$JOB`.

**4. Edit a line.**
- Do: in the browser, click a segment's text, change one word, Save line —
  or `CLI edit-translations $JOB '{"0": "<edited line>"}'`.
- See: the card re-renders with the new text.
- Assert: `GET /api/dub/$JOB/subtitles` cue 0 text carries the edit.

**5. Teach the glossary.**
- Do: rail → Glossary quick-add: a term from the video + a rendering
  (or `CLI glossary-term <term> --lang ru --translation <rendering>`).
- Assert: `GET /api/dub/$JOB/flags` recomputes against the new term
  (count may rise — a fresh glossary_miss is CORRECT behavior).
- On failure: a stale flag count that ignores the new term means the
  glossary cache wasn't invalidated (CLD-234 regression).

**6. Approve → synthesize.**
- Do: header "Approve → synthesize ▸" (or `CLI continue $JOB`).
- See: feed shows synthesize ● then master; job re-parks.
- Assert: poll until `status == awaiting_subtitle_review`;
  `gates_cleared` contains `translation`.

**7. Second gate: subtitles.**
- See: workbench reopens on the Subtitles tab; cue table with CPS/lines/gap.
- Do: if any violation exists, click "Auto-fix warnings"; otherwise edit a
  cue long enough to create one, then auto-fix.
- Assert: `GET /api/dub/$JOB/subtitles` → `violation_count == 0`, and
  `vtt_url` is non-null (the .vtt exists on disk).

**8. Approve subtitles.**
- Do: "Approve subtitles ▸" (or `CLI continue $JOB`).
- Assert: poll until `status == awaiting_final_qc`; the ring now has ≥2
  `job.awaiting_review` attempts for `$JOB`.

**9. Final QC: read the checklist.**
- See: Final QC tab — rows for Loudness, Subtitles, Sync, Background bed,
  Glossary, Voice consent, each pass/warn/unavailable with real numbers.
- Assert: `GET /api/dub/$JOB/qc` → `rows` has the six ids above;
  `on_approve.deliverables` lists mp4 + srt + vtt.

**10. Retarget loudness and re-assemble (the gate re-arm).**
- Do: `PATCH /api/config` form `body={"loudness_target": -14.0}`, then
  `CLI retry-stage $JOB assemble --overrides '{"review_gates": {"final_qc": "on"}}'`.
- See: feed shows assemble/master re-running; job re-parks at Final QC.
- Assert: `GET /api/dub/$JOB/qc` loudness row moved to ≈ −14 (±1 LU) —
  the measured value, not the setting.

**11. Add a note and deliver.**
- Do: type a review note, Add; then "Approve & deliver ▸"
  (or `CLI continue $JOB`).
- See: job flips to Completed in the feed.
- Assert: `status == complete`; the ring has EXACTLY ONE `job.completed`
  attempt for `$JOB` (the single-firing-site contract); the note is in
  `GET /api/dub/$JOB/qc` → `review_notes`.

**12. Clean up.**
- Do: restore `loudness_target` to its step-1 value; `DELETE` the webhook;
  `CLI delete $JOB`. Confirm `outputs/$JOB/` is gone.
