# Voice casting — one voice per speaker, auditioned before the long stage

Needs a MULTI-speaker clip (interview/podcast snippet) as `$CLIP`, or the
walkthrough degrades to single-speaker with a note. Runtime ~4 min.

**1. Submit with the voice gate.**
`CLI dub $CLIP --lang ru --review-gates voice_cast=on`
- Assert: job parks at `awaiting_voice_review` after translate
  (`pending_gate == "voice_cast"`).

**2. The casting rail.** Browser `/pro#/review/<id>` — the workbench
opens with the casting panel in the rail.
- See: one card per diarized speaker with minutes of speech; ▶ plays the
  speaker's reference clip.
- Assert: `GET /api/dub/<id>/voice_casting` speakers match the cards.

**3. Cast and audition.** Assign one speaker a preset voice, keep the
other on `source`; press the audition button.
- See/Assert: `POST /api/dub/<id>/voice_preview` returns sample URLs; the
  samples play in the browser.

**4. Consent chips.** Each speaker card shows "no clone consent recorded"
with an attest toggle.
- Do: attest one speaker.
- Assert: `GET /api/job/<id>` → `voice_consent` carries the record;
  `GET /api/dub/<id>/qc` consent row reflects policy (`off` ⇒
  unavailable — flip `voice_consent_policy=warn` via PATCH /api/config to
  see it engage, then restore).

**5. Cast & synthesize.** Approve — the cast commits BEFORE `/continue`
(the panel's one job).
- Assert: after completion, `speaker_voice_map` on the job matches what
  was cast; the two speakers sound different in the output.

**6. Speaker repair detour (optional, pre-TTS only).** If diarization
over-counted, merge the ghost on the Pronunciation & speakers tab.
- Assert: `POST /api/dub/<id>/speakers/edit` merge response counts > 0;
  the ghost is gone from `GET /api/job/<id>/speakers`.

**7. Clean up.** Delete the job; restore any config flipped in step 4.
