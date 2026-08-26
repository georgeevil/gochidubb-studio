# Redub + stage reuse — the second language is cheap, the third is fast

**1. Baseline job.** Complete a plain dub of `$CLIP` into one language
(`CLI dub $CLIP --lang ru`), or reuse a jobs-list entry that already
finished for this exact clip.

**2. Redub.** `CLI redub <job> --langs de` — re-runs translate+TTS+merge
from the saved transcript; no re-download, no re-transcription.
- See: `/pro#/jobs/live` — the new job starts at the translate stage.
- Assert: the redub job's stage timings (`GET /api/dub/<id>/stages`) show
  no download/transcribe entries.

**3. Reuse, measured.** Enable stage reuse (`GOCHIDUBB_REUSE=1` env or
`reuse_enabled` via PATCH /api/config; restart), then dub the SAME clip
fresh into another language.
- See: `/beta` — the reuse inspector lists fingerprint hits; the feed's
  job card says which stages were "Reused from job …".
- Assert: `GET /api/job/<id>` → `reused_stages` names
  download/extract/transcribe/diarize; wall-clock beats step 1's
  measurably (`GET /api/dub/<id>/metrics`).

**4. Clean up.** Delete the jobs this run created; restore
`reuse_enabled` to its prior value.
