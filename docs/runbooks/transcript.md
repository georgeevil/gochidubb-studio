# Transcript — New → Transcript, fix the speakers, export

A transcript is its own goal on New, not a dub told to stop early. It
submits `mode=transcribe`, so every speaker is kept (no "main speaker"
fold), Whisper runs without self-conditioning, nothing is separated or
translated, and the job finishes `complete` with an export.

**1. Submit.** Browser `/pro#/home` → "What are you making?" → Transcript.
File = `$CLIP`, spoken language set, "Label who is speaking", Number of
voices = the real count (count jingles and idents too). Press Transcribe.
- Assert: `GET /api/job/<id>` has `mode == "transcribe"`,
  `stop_after == "diarize"`, `speaker_mode == "all"`, `keep_bg == false`,
  and `min_speakers == max_speakers` == the count chosen.
- Assert: the job ends `status == complete` with `transcript_url`, never
  `paused`.

**2. Result.** Jobs → the job → View.
- See: the Transcript result (no video player), a speaker row per voice
  with talk time and ▶, and the lines.

**3. Fix the speakers.** Name each speaker → Save names. Move one line to
another speaker with its select; move one to "+ new speaker"; mark one
"not speech". Merge two speakers.
- Assert: `GET /api/dub/<id>/transcript` shows the names as `label`, the
  moved lines under their new speakers, the non-speech line gone, and the
  merged speaker no longer listed.

**4. Re-detect.** "Detect them again with exactly N voices" → Re-run.
- Assert: the job runs only `diarize` (`GET /api/dub/<id>/stages`),
  returns to `complete`, and the page reloads the new speakers by itself.

**5. Export.** Download .txt, .srt, .vtt, .json.
- Assert: each one prints the names from step 3 ("Name: text").

**6. The old trap is gone.** New → Dub → Advanced → Run until.
- Assert: no "Transcript" options are offered.

**7. Clean up.** Delete the job.
