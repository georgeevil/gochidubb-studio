# Creator flow — submit, review only what's uncertain, done

The consumer path: Creator mode shows a handful of confidence-flagged
spans, never a transcript editor. Runtime ~3–4 min.

**1. Preconditions.** Server up (`serverctl restart`); browser at
`/creator` (visiting it persists `ui_mode=creator` — note the value first
and restore it at the end if it was `pro`).

**2. Submit.** In the browser: New dub (`#/new`), drop `$CLIP`, pick one
language, note the price quote (from `/api/estimate`), start. Creator
submits with `wizard_mode=review_translation` — one pause, by design.
- Assert: job created with `review_gates.translation == "on"` and every
  other gate off (the wizard-mapping contract).

**3. Review the flags.** Wait for the review screen (`#/review/<id>`).
- See: flag cards only for uncertain spans; "hear this bit" plays the
  source audio around the span.
- Assert: `GET /api/dub/<id>/flags` count matches the cards shown.

**4. Decide one card.** Keep or fix a rendering; if fixed, the decision
lands in the glossary.
- Assert: `presets/user_glossary.json` (via `GET /api/glossary`) carries
  the term after a "use this" decision.

**5. Approve.** Continue → the job runs to complete with NO further
pauses (creator arms only the one gate).
- Assert: `status == complete`; `gates_cleared == ["translation"]`.

**6. Result.** `#/video/<key>`: play the dub; the background slider shows
×10 by default; Apply re-mixes in seconds.
- Assert: `GET /api/dub/<id>/qc` Background bed row is `pass`.

**7. Clean up.** Delete the job; restore `ui_mode` if changed; remove any
glossary term this run added (`GET /api/glossary`, write back without it).
