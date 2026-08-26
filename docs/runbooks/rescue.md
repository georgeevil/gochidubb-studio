# Rescue — a failed download resumed by hand

The failure path is free and honest: a dead download job offers
classified hints and accepts the file from the user's own machine, then
resumes the SAME job.

**1. Break a download.** Submit a URL that will fail fast:
`CLI dub "https://www.youtube.com/watch?v=zzzzzzzzzzz" --lang ru`
- Assert: job reaches `status == error` with a `download_hint`
  (`GET /api/job/<id>` — summary + copyable commands).

**2. The rescue card.** Browser `/pro#/jobs` → the errored job.
- See: the error panel with the classified hint and a drop zone.

**3. Attach the file.** Drop `$CLIP` on the job's rescue drop zone (or
`CLI rescue <job> $CLIP`) — this is `POST /api/job/<id>/attach_source`.
- Assert: the SAME job id resumes (`rescued_with_upload` true), walks the
  pipeline, and completes. No second job is created.

**4. Clean up.** Delete the job.
