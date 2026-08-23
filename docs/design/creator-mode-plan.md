# Creator mode — implementation plan

Companion to [`creator-mode-spec.md`](creator-mode-spec.md). Every existing thing
referenced here is cited `file:line` against the tree at the time of writing.

## Headline

**Most of what the design needs already exists.** The review gate is fully built
(`wizard_mode="review_translation"`), one-source-to-N-languages is a first-class
endpoint (`POST /api/quick_test`), tiered pricing is implemented
(`app/billing.py`), and `batch_id` already groups per-language siblings.

The genuinely new backend work is three modules and five small routes. The
genuinely risky work is one pure function (flagged terms) and one architectural
truth the design ignores (the GPU queue is serial).

---

## 1. Screen → existing API map

### 1.1 Creator home (`2a Creator home`)

| Design element | Existing route | Notes |
|---|---|---|
| Active job card: title, progress, stage story | `GET /api/jobs` `server.py:8698` — full job dicts + `has_checkpoint`/`latest_checkpoint_stage` from `_job_checkpoint_info` `server.py:849` | Per-job fields present: `id, status, progress, step_detail, target_lang, source_label, title, meta, created, started_at, completed_at, batch_id, batch_label, batch_kind, batch_position, batch_total, output_url, srt_url, error`. Status vocabulary: `queued, scheduled, downloading, extracting, transcribing, translating, synthesizing, assembling, merging, awaiting_translation_review, paused, complete, error, cancelled, interrupted, uploaded`. Collapse every mid-pipeline status into one "Making your Japanese voice…" string. |
| Thumbnail | `job["meta"]["thumbnail"]` — remote URL from `curate_metadata` `pipeline/downloader.py:148` | URL sources only. Uploads have no thumbnail — render the dark placeholder tile the design already draws. |
| "2 of 3 languages ready" grouping | `batch_id`, set on every fan-out job (`/api/quick_test` `server.py:4547`) | Group client-side by `batch_id ?? id`. `GET /api/dub/batch/{batch_id}` `server.py:3950` gives a ready-made rollup (`total/complete/errored/queued/running/elapsed_sec` + per-job `dubbed_url`) but needs a `batch_id` you already have — a detail view, not the list. |
| Language chips (done / pending / needs attention) | `job["target_lang"]` + `job["status"]` | "needs attention" = `awaiting_translation_review`. |
| **Review** chip on a card | `status == "awaiting_translation_review"` | |
| Spend tile "$14.80 / 72 minutes dubbed" | `GET /api/billing/usage?window_days=30` `server.py:3395` → `{minutes, cost, rate, bands, by_lang, next_tier, storage_gb, estimate:true, disclaimer, tiers}` | Calls `storage_stats()`, which walks every output dir. **On-demand only — never poll it.** |
| "128 min left this month" | **Nothing.** See Risk 3. | |
| Watch preview / download | `job["output_url"]` = `/outputs/{id}/dubbed_video.mp4?v=…`; `job["srt_url"]` = `/outputs/{id}/subtitles.srt`. `/outputs` mounted `server.py:1371`. | |
| "Auto-dub new uploads" strip | **Nothing.** No channel-connect anything exists. Ship as a non-functional teaser or drop from v1. | |

### 1.2 Wizard (`2a Dub a video`)

| Design element | Existing route |
|---|---|
| Language chips (65) | `GET /api/languages` `server.py:8728` → `{"languages":["en","ru",…]}` — bare codes from `EdgeTTSFallback.VOICE_MAP`, exactly the set `/api/quick_test` validates against (`_QUICK_TEST_KNOWN_LANGS` `server.py:4403`). No display names on any route — Risk 7. |
| "My voice" / "A professional voice" | `GET /api/voice_presets` `server.py:7887` (alias of `/api/voices` `server.py:7881`) → `{"presets":[{id,name,style,type:"file"\|"style",description,gender,language,tags,audio_url,…}]}` via `_voice_preset_payload` `server.py:7856`. "My voice" = upload a `reference` file, or a saved `file:NAME` preset. |
| `▶ Hear it in Spanish` | `GET /api/voice_presets/{id}/audio` `server.py:7909` — streams the *reference* clip, not a Spanish sample. See Risk 8. |
| Thumbnail / title / "12 min 04 sec" | `probe_metadata` `pipeline/downloader.py:94` → `curate_metadata` `pipeline/downloader.py:148`. Currently only called inside `/api/dub` `server.py:3583`. Needs a route — Gap A. |
| `$2.34` / "ready in about 20 minutes" | **Nothing.** Gap A. |
| Toggle "Let me check the translation first" | `wizard_mode="review_translation"` form field, accepted by `/api/dub` `server.py:3494` and `/api/quick_test` `server.py:4425`. |
| **Start dubbing →** | `POST /api/quick_test` `server.py:4406`. Multipart: `video` (file) **or** `source` (URL), `reference` (file), `target_langs` (comma-separated), `trim_seconds=0` for full video, `source_lang`, `model`, `whisper_model`, `speaker_mode`, `voice_preset`, `voice_style`, `tts_speed`, `keep_bg`, `auto_denoise`, `context_hint`, `batch_label`, `wizard_mode`, `lip_sync`, `scheduled_at`, `voxcpm_cfg`, `voxcpm_steps`. Returns `{ok, batch_id, batch_kind:"quick_test", job_ids:[…], count, trimmed_file, trim_seconds, target_langs}`. |

`/api/dub/batch` `server.py:3701` is **N sources × 1 language** — not what Creator
mode wants. `/api/showcase` `server.py:5209` trims to 15–120s and stitches a reel.
`/api/quick_test` is the right one, and its docstring says so: *"This is a primary
workflow, not a smoke test."*

### 1.3 Review (`2a Review before recording`)

| Design element | Existing route |
|---|---|
| Job pauses before TTS | Already works. `_wizard_pause_after` `server.py:2694-2706` returns `("awaiting_translation_review", …, "translation_done")` after `translate`; applied at `server.py:2890-2896`, which `return`s (releasing the queue worker). |
| Read the segments | `GET /api/dub/{id}/checkpoint/translation_done` `server.py:6447` → `{stage, saved_at, target_lang, duration, segments:[…], speaker_refs:{}}` |
| Segment shape | From `_serialize_segments` `server.py:1611`: `{idx, start, end, text, speaker, translated_text, avg_logprob, no_speech_prob, word_conf_mean, word_conf_min}`. **The per-word `words` array is deliberately dropped** (`server.py:1650-1654`) — Risk 5. |
| Edit one term | `POST /api/dub/{id}/edit_translations` `server.py:7087`, form field `edits` = JSON `{"<idx>":"new text"}`. Re-saves the checkpoint, re-writes `subtitles.srt`, syncs `tts_done` if present. |
| **Looks good — record it →** | `POST /api/dub/{id}/continue` `server.py:7561`. Resumes from `_latest_checkpoint`, forces `wizard_mode="auto"` (`server.py:7656`) so it cannot re-pause. Optional `voice_style`/`voice_preset`/`tts_speed`/`reference` overrides. |
| **Skip review** | Same `/continue` call, without editing. |
| `▶ Hear this bit` | `<audio src="/outputs/{id}/audio_16k.wav#t={start},{end}">`. Verified present in real job dirs. Timeline is correct: VAD writes a *separate* `audio_16k_vad.wav` (`server.py:1822`) and segments are remapped back to the original timeline (`server.py:1901-1905` via `pipeline/vad.py:123`). |
| Which flagged terms to show | **Nothing.** Gap C. |
| "we'll remember this for future videos" | `GET/POST/DELETE /api/glossary` `server.py:8285/8308/8351` over `presets/user_glossary.json`, shape `{"domains":[{name,triggers,target_lang,terms:{src:tgt}}]}`. **Whole-file replace only, and writes are inert until restart.** Gap E. |
| LANGUAGES rail with ✓ per sibling | `GET /api/jobs?batch_id=…` |
| Notify | `job.awaiting_review` webhook already fires — `_WEBHOOK_EVENT_FOR_STATUS` `server.py:1425`. |

---

## 2. Backend gaps — verified against the code

### Gap A — cost/duration estimate. **Does not exist.**

`grep` for price/cost/rate/billing finds only `app/billing.py`, which is entirely
retrospective: `job_minutes` (`app/billing.py:105`) reads `job["duration"]`, and
that field is not set until the extract stage (`server.py:1744`) — i.e. after the
download. Nothing prices a job before it starts.

**Do not move the rates into `UserConfig`.** `TIERS` (`app/billing.py:34-38`) are
the design's *published hosted rates*, and the module docstring
(`app/billing.py:16-19`) draws an explicit honesty boundary: "GoChiDUBB in local
mode bills nobody… the rates below are the ones the design publishes."
`/api/billing/usage` re-publishes them verbatim (`server.py:3417-3421`). Making
them user-editable would let a local install invent prices while the response
still says `estimate: true` and "This server bills nobody". The only genuinely
new tunable is the ETA factor.

**Build:**

- `app/billing.py` — add `marginal_cost(used_minutes, new_minutes) -> dict`.
  **The one correctness detail that matters:** `price_minutes`
  (`app/billing.py:61`) prices cumulatively from zero, so a wizard estimate must
  be `price_minutes(used + new) - price_minutes(used)`. Getting this wrong shows
  $2.90 in the wizard and $2.34 on the meter.
- **new `app/estimate.py`** — pure. `realtime_factor(jobs)` = median of
  `(completed_at - started_at) / duration` over completed jobs with all three set
  (`started_at` `server.py:644`, `completed_at` `server.py:2624`).
  `eta_seconds(duration_sec, n_langs, factor)` — **multiplies by `n_langs`,
  because the queue is serial** (Risk 1).
- `app/config.py` — add to `UserConfig` (`app/config.py:115`):
  `eta_realtime_factor: float = 6.0` (fallback below 3 samples), plus
  `FIELD_SPECS["eta_realtime_factor"] = (float, 0.5, 60.0)` at `app/config.py:49`.
- `server.py` — `GET /api/estimate?source=<url>&duration_sec=<n>&langs=es,fr,ja`.
  URL path calls `await asyncio.to_thread(probe_metadata, url)` (mirroring
  `server.py:3583`) with a small URL→meta TTL cache so wizard re-renders don't
  re-probe. Upload path takes `duration_sec` read client-side from a `<video>`
  element — no upload needed to price. Response:
  `{duration_sec, langs, billable_minutes, cost, rate, bands, eta_sec,
  eta_basis:"measured"|"default", estimate:true, title, thumbnail, channel,
  duration_gate_error?}`. Reuse the `max_source_duration_sec` gate
  (`server.py:3586-3595`) so the wizard refuses over-long videos *before* Start.
- Tests: extend `tests/test_billing.py`; new `tests/test_estimate.py`.

### Gap B — review gate. **Already exists. Near-zero backend work.**

The whole chain is built (§1.3). The one real gap is multi-language semantics:
with N siblings each pauses independently, and the queue is single-consumer, so
language 2 does not reach its pause until language 1 has fully recorded. The
design's rail ("one under review, the others ✓") assumes all N pause together.

**Recommended — Option A, no runner change.** Creator mode submits the
review-gated job for **one** language first and holds the other N−1 back. On
approval it POSTs `/continue` for the primary *and then* submits the rest with
`wizard_mode="auto"`. One review, one approval. The glossary writes made during
review are what carry the decision into the sibling languages — which is exactly
what "we'll remember this for future videos" claims, taken literally.

Option B (a `POST /api/dub/batch/{batch_id}/continue` fan-out) is strictly worse:
every sibling must still reach `translate` before you can review anything, so
time-to-first-review is N× longer for no benefit.

### Gap C — flagged terms. **The high-risk piece. New pure module.**

**Signals actually available at `translation_done`:** `text`, `translated_text`,
`start`/`end`, `speaker`, `avg_logprob`, `no_speech_prob`, `word_conf_mean`,
`word_conf_min`, plus the glossary and `pipeline/quality.py`'s thresholds.
**Word-level probabilities are gone** — `_serialize_segments`
`server.py:1650-1654` keeps only the aggregates. Per-word highlighting must be
reconstructed by string matching, not read off a confidence array.

**New file: `pipeline/flags.py`** — pure functions, no I/O, no model loads,
GPU-free, testable:

```python
flag_segments(segments, *, target_lang, source_lang="",
              glossary=None, max_flags=5) -> list[dict]
```

Each flag: `{idx, kind, start, end, source_text, translated_text, source_span,
target_span, reason, score, variants}`.

Five detectors, cheapest and highest-signal first:

1. **`_inconsistent_renderings` — lead with this one.** Extract proper-noun
   candidates from `text` (regex `\b[A-Z][a-z''']{2,}\b`, dropping
   segment-initial tokens and a stopword set). Build
   `{source_name → set(target rendering)}` across the whole transcript. When one
   name maps to ≥2 distinct renderings → flag the first occurrence,
   `kind="name"`, `reason="name_inconsistent"`, `variants=[…]`. This is a *real,
   observable* failure of the batched translator
   (`pipeline/translator.py:1600-1640` — batches translate independently, so the
   same name genuinely comes back two ways), it needs no word aligner, and one
   glossary entry fixes every occurrence. The design's whole loop in one
   detector.
2. **`_untransliterated_name`** — cross-script only (`_LANG_SCRIPT`
   `pipeline/translator.py:1030`): a Latin-script proper noun surviving verbatim
   into a Cyrillic/CJK line → `kind="name"`. Same-script pairs skip this;
   `pipeline/quality.py:146 _source_bleed` already documents why (across scripts
   a repeated token is almost always a name that *should* carry over).
3. **`_glossary_miss`** — a glossary term present in `text` whose mapped value is
   absent from `translated_text` → `kind="term"`. Flatten with the same
   per-`target_lang` rule as `pipeline/translator.py:152`. Validates the glossary
   loop end to end.
4. **`_idiom_ratio`** — length-ratio outlier (reuse `LEN_RATIO_HIGH=2.0` /
   `LEN_RATIO_LOW=0.3`, `pipeline/quality.py:38-39`) **and** an idiom cue in the
   source (`[laughs]`, exclamation with a much shorter target, a small
   hand-written cue list) → `kind="joke"`. A bare ratio outlier is usually just a
   verbose translator and is not something a creator can act on — never flag it
   alone.
5. **`_low_asr`** — `word_conf_min < 0.35` or `avg_logprob < -0.9` on a segment of
   ≥3 words → `kind="unclear"`. Filler only: surface these solely when 1–4
   produce fewer than `max_flags`.

Rank by score, one flag per `idx`, cap at `max_flags=5`. **The zero-flag case must
be the common one** — if this routinely returns 5 flags it has become a
transcript editor with extra steps, the exact failure the design is avoiding.
Render "Nothing to check — we're confident" plus a single record button when the
list is empty.

- Route: `GET /api/dub/{job_id}/flags` — loads the checkpoint via
  `_load_checkpoint` `server.py:922`, calls `flag_segments`, returns
  `{job_id, target_lang, source_lang, count, flags,
  audio_url:"/outputs/{id}/audio_16k.wav"}`. Recomputed per call (cheap), so it
  always reflects edits already applied through `edit_translations`.
- Tests: **new `tests/test_flags.py`** — hand-built segment lists, no GPU. Reuse
  `tests/conftest.py` fixtures where the shape fits.

Ship the route first and iterate on the heuristic without touching the UI.

### Gap D — creator-shaped job list. **Grouping exists; two real problems.**

`batch_id` already groups, and `GET /api/jobs?batch_id=…` already filters. But:

1. **A one-language video is ungroupable.** `/api/dub` sets no `batch_id` at all
   (`server.py:3646-3676`), and `/api/quick_test` requires ≥2 languages
   (`MIN_TARGET_LANGS = 2`, `server.py:4394`). **Fix: `MIN_TARGET_LANGS = 1`.**
   Then Creator mode *always* goes through `/api/quick_test` and every creator
   video is a batch — the whole home screen falls out for free. Caution:
   `/api/showcase` (`server.py:5246`) and `/api/dub/{id}/redub`
   (`server.py:5473`) validate against the same constant, and a 1-language
   showcase is meaningless — give those two their own floor rather than sharing
   it.
2. **`/api/jobs` is fat.** `_strip_large_fields` (`app/db.py:184`) strips only the
   transcript, so `meta["description"]` (thousands of chars per job, from
   `curate_metadata`) ships on every poll. **Fix: a `compact=1` param on
   `server.py:8698`** projecting to the ~15 keys the home screen needs. Benefits
   Pro mode too.

### Gap E — mode preference, page route, glossary writes.

- **Mode preference: zero backend change.** `GET/POST /api/preferences`
  (`server.py:8091/8106`) is a free-form JSON merge into `user_prefs.json`
  (`server.py:1148`). `{"ui_mode":"creator"}` fits as-is. **Do not use
  `UserConfig`** — that is pipeline behaviour, validated against `FIELD_SPECS`,
  and `GET /api/config` is unauthenticated.
- **Page route:** `@app.get("/creator")` → `FileResponse(STATIC_DIR /
  "creator.html")`, mirroring `/beta` at `server.py:8760`. `/static` is already
  mounted (`server.py:1372`) so `creator.css` needs no new mount.
- **Glossary writes are broken for this use case, two ways:**
  1. `POST /api/glossary` (`server.py:8308`) is a **whole-file replace**.
     Per-term writes from the review screen mean read-modify-write from the
     browser, which races. **Add `POST /api/glossary/term`** taking
     `{term, translation, target_lang, domain?}` and merging server-side.
  2. `_GLOSSARY_CACHE` (`pipeline/translator.py:170-183`) caches the flattened
     glossary **for the process lifetime**. A term saved during review does not
     reach the next language's translation until a server restart. **Add
     `clear_glossary_cache()` to `pipeline/translator.py` and call it from all
     three glossary routes.** This is a live bug today; Creator mode is what
     would expose it, and without the fix the review screen's headline promise
     silently does nothing.

---

## 3. File layout

```
static/creator.html      one page, three views, hash routing (#/, #/new, #/review/<job_id>)
static/creator.css       light palette + components
pipeline/flags.py        flag_segments() — pure, GPU-free
app/estimate.py          realtime_factor(), eta_seconds() — pure
tests/test_flags.py
tests/test_estimate.py
```

Touched: `server.py`, `app/billing.py`, `app/config.py`,
`pipeline/translator.py`, `tests/test_billing.py`, `tests/test_translator.py`.

**Separate `creator.css`, not inline.** `theme.css` is already served from
`/static` and linked at `static/index.html:10`, so a second stylesheet costs
nothing, and it is the file a designer can hand back edited. `beta.html` inlines
its CSS (`static/beta.html:7-66`) but *justifies* it — a diagnostic page that must
stay readable "even if the main UI's CSS is mid-rewrite". Creator mode is the
front door; that reasoning does not carry.

**Own `:root` block, not a `theme.css` override.** `static/theme.css:1-22` is a
flat `:root` with no theme switch, and the two palettes disagree on what `--ink`,
`--line` and `--accent` mean. Give `creator.css` the spec's token names verbatim
(`--paper`, `--card`, `--ink-2`, `--accent-ink`, `--ok-fill`, …) so the design doc
and the stylesheet stay greppable against each other.

**Vanilla JS, not React.** `index.html` loads React 18 UMD + `@babel/standalone`
from unpkg and transpiles 5,989 lines of JSX in the browser on every load
(`static/index.html:15-18`). Defensible for a tool people leave open all day;
wrong for a consumer front door where first paint *is* the product. Three screens
with a handful of state each do not need a framework. Worth noting separately:
that CDN dependency means Pro mode does not work offline, in a product whose pitch
is "everything runs on your machine" — the creator page should not inherit it.
Structure: a `state` object, a `render()` that swaps one of three view functions
into `<main>`, `hashchange` routing, `fetch` helpers, polling only while a job is
active.

No new dependencies (`CONTRIBUTING.md`). CI is `ruff check .` + `python -m
compileall`.

---

## 4. Sequencing

| # | Side | Work | Depends on |
|---|---|---|---|
| 1 | Backend | **Estimate.** `app/billing.py::marginal_cost`, new `app/estimate.py`, `eta_realtime_factor` in `UserConfig` + `FIELD_SPECS`, `GET /api/estimate`. Tests. | — |
| 2 | Backend | **Flagged terms.** `pipeline/flags.py`, `GET /api/dub/{id}/flags`, `tests/test_flags.py`. | — |
| 3 | Backend | **Glossary write path.** `clear_glossary_cache()`, `POST /api/glossary/term`, wire cache-clear into all three routes. | — |
| 4 | Backend | **Batch-shaped submit.** `MIN_TARGET_LANGS = 1` (+ own floors for showcase/redub), `GET /api/jobs?compact=1`, **and the non-blocking submit path (Risk 4 — required, not optional).** | — |
| 5 | Backend | **Serving.** `@app.get("/creator")`. | — |
| 6 | Frontend | **The three screens.** | 4, 5 |
| 7 | Both | **Mode switch.** Menu items both ways POSTing `{"ui_mode":…}` to `/api/preferences`; *then* make `GET /` (`server.py:8751`) honour the pref, and add `/pro` as an unconditional escape hatch. | 6 |

**What the frontend can stub.** Chunks 1–3 are all read-only JSON with no side
effects, so a `MOCK = {…}` constant plus a `fetchOrMock()` wrapper lets all three
screens be built and design-reviewed before any of them land. Real from day one:
`/api/quick_test`, `/api/jobs`, `/api/job/{id}`,
`/api/dub/{id}/checkpoint/translation_done`, `/api/dub/{id}/edit_translations`,
`/api/dub/{id}/continue`, `/api/voice_presets`, `/api/languages`,
`/api/billing/usage`.

**Chunk 7 is deliberately last** — it is the only change that can strand an
existing Pro user on a page they did not ask for.

---

## 5. Risks, and where the design is wrong

1. **"Ready in about 20 minutes" is not achievable here.** The job queue is
   single-consumer by design — `enqueue_job` `server.py:558`, one
   `_job_queue_worker`, with the comment *"Multiple simultaneous dub requests
   would OOM the 12GB 3080 Ti."* Three 12-minute languages run **serially**. The
   honest ETA is N × single-language time; at a realistic realtime factor that is
   closer to 90 minutes than 20. The design is describing a hosted fleet that does
   not exist in this repo. Do not fake the number — rule 7 ("failures are free and
   silent") only works if the numbers are trustworthy.

2. **"we'll email you" — there is no email.** No `smtp`/`sendmail`/mail anywhere
   in `server.py`, `app/`, or `pipeline/`. What exists is webhooks
   (`app/webhooks.py`, event `job.awaiting_review` at `server.py:1425`). Honest
   v1 copy: "you can close this page — it keeps running", plus a browser
   Notification, which is free. SMTP means a new dependency and
   `CONTRIBUTING.md` forbids that without discussion.

3. **"128 min left this month" implies a quota that does not exist.**
   `/api/billing/usage` measures minutes *spent*; there is no account, no plan, no
   allowance — `app/billing.py:16-19` says so explicitly. Rendering a remaining
   balance is inventing a number. Show minutes *used* this month, which the data
   supports. Same for the avatar circle: no auth in local mode
   (`cfg.mode == "local"`, `app/config.py:129`).

4. **`/api/quick_test` blocks the entire event loop on the download.**
   `src_path = Path(download_video(url, str(dl_dir)))` at **`server.py:4527`** is
   called directly inside the `async def` — not wrapped in a thread.
   (`/api/showcase` has the identical bug at `server.py:5296`. The pipeline's own
   `_stage_download` `server.py:1734` does it correctly via `_blocking`.) A
   creator pasting a 12-minute YouTube link freezes the whole server for the
   length of the download, and the wizard's `fetch` looks dead. `asyncio.to_thread`
   unblocks the loop but leaves a multi-minute HTTP request that proxies will time
   out. **The right fix: a `background=1` flag on `/api/quick_test` that creates
   the `batch_id` + N jobs immediately in a `preparing` status, returns, and does
   download-then-fan-out in an `asyncio.create_task`.** Keeps the single central
   download (which is *why* it lives in the handler — the fan-out jobs share one
   file) and makes Start feel instant. A prerequisite, not a nice-to-have: it is
   the very first thing a non-technical user does.

5. **Word-level spans are not recoverable at review time.**
   `_serialize_segments` `server.py:1611-1656` deliberately drops the `words`
   array, keeping only `word_conf_mean`/`word_conf_min`. So the review screen
   cannot say "this *word* scored 0.3" — only "this *segment* did". Per-term
   highlights are reconstructed by string matching in `pipeline/flags.py`, which
   is precisely why the name-inconsistency detector (needs no alignment) should
   lead. **Do not "fix" `_serialize_segments` to keep words** — it would multiply
   checkpoint size on every job for a feature that touches a handful of segments.

6. **Glossary writes are inert until restart** (`pipeline/translator.py:170`).
   Covered in Gap E; restated here because the review screen's entire retention
   promise depends on it and it is silently broken today.

7. **`/api/languages` returns bare codes** (`server.py:8728`), with no
   code→display-name map on any route. `LANGUAGE_NAMES` exists at
   `pipeline/translator.py:200` but is not exposed. Cheapest fix: make
   `/api/languages` return `[{code, name}]` — additive, but check
   `tools/gochidubb_cli.py languages` and `tests/test_language_registry.py`
   first. Otherwise `creator.html` hardcodes 65 names, exactly the duplication
   CLAUDE.md's "one place every dubbable language must be registered" note warns
   against.

8. **`▶ Hear it in Spanish` does not exist.** `/api/voice_presets/{id}/audio`
   streams the *reference* clip, not a target-language sample. Real per-language
   samples mean a synthesis job per language per voice. Either reword to "Hear
   this voice" or scope sample generation separately — do not let the button imply
   something the backend cannot do.

---

# As built

Everything below was implemented and measured. Where it differs from the plan
above, this section wins.

## What shipped

    static/creator.html    the three screens + voices + usage, vanilla JS, no CDN
    static/creator.css     light palette, spec token names
    pipeline/flags.py      flag_segments() — pure, GPU-free
    app/estimate.py        realtime_factor(), eta_seconds() — pure
    tests/test_flags.py  tests/test_estimate.py

Touched: `server.py`, `app/billing.py`, `app/config.py`,
`pipeline/translator.py`, `static/index.html`, `tests/test_billing.py`,
`.gitignore`.

Routes added: `GET /api/estimate`, `GET /api/dub/{id}/flags`,
`POST /api/glossary/term`, `GET /creator`, `GET /pro`.
Changed: `POST /api/quick_test` (`background` flag; `MIN_TARGET_LANGS` 1),
`GET /api/jobs` (`compact=1`), `GET /api/languages` (`names`/`catalog`),
`GET /` (honours `ui_mode`, fail-safe to Pro).

## Where the plan was wrong

**Risk 5 understated the flag problem.** The plan treated flagged terms as the
risky piece; it was worse than that — the first implementation returned the cap
of 5 on half a real corpus, with the weakest detector producing 60% of output.
Two things only measurement could have found:

- **Similarity cannot separate true from false renderings.** `Киото`/`Кёто`
  (true) and `Потому`/`Почему` (false) both score 0.667. No threshold exists.
  The detector was rebuilt on a containment test: a name's rendering appears
  only in lines naming it; a word of the language appears everywhere.
- **Russian case inflection read as inconsistency** (`Кевин`/`Кевина`). Actively
  harmful, not just noisy — a creator "fixing" it produces ungrammatical Russian.

Also: `word_conf_min` is a *minimum*, a low-order statistic that drifts down
with segment length on its own (median 0.69), so a 0.35 gate matched 22.6% of
all real segments.

**A gap the plan missed entirely:** every batch route refused to start on an
LM Studio install — the documented default. `check_ollama()` delegates to
`check_lm_studio()`, whose ids are hyphenated, while the fallback list is
colon-shaped Ollama ids. Nothing could ever match. Pro mode survived on its
model dropdown; Creator mode has none, so a creator was stranded at the last
step of the wizard. Five duplicated call sites are now one
`_resolve_translation_model()`.

## Measured flag behaviour

53 checkpoints / 4,417 segments (20 distinct sources — most are one video fanned
out to several languages, so per-checkpoint counts flatter the result):

| | first cut | shipped |
|---|---|---|
| checkpoints showing nothing | 20.8% | **75.5%** |
| zero or one flag | 41.5% | **88.7%** |
| hit the cap of 5 | 26/53 | **1/53** |
| total flags | 156 | **23** |
| per 1,000 segments | 35.3 | **5.21** |

Per distinct source, worst language: 45% zero, 75% zero-or-one, mean 1.00.

Detectors shipped: `name_inconsistent` (10) · `name_untransliterated` (10) ·
`low_asr` (3, only onto an otherwise-empty screen, capped at 1).
`idiom_ratio` was **retired** — two flags corpus-wide, both unanswerable.

Roughly 75-80% of what a creator sees is worth seeing. Known residual:
`Freezy`→`Фордом`/`Форт` and `Jackson`→`Джейкоб`/`Джексона`, unrelated tokens
passing containment by coincidence (~80% precision on the flagship detector).
Deliberately not chased — over-fitting to a 20-source corpus is the bigger risk.

**Re-run the sweep before loosening any threshold in `pipeline/flags.py`.**
Every constant there cites the measurement that set it.

## Two lessons worth keeping

**Tuning a heuristic can silently invalidate copy written against it.** After
the retune, three pieces of UI copy still promised the review checks "names,
brands and jokes" — but brands were deliberately stripped from the detectors and
`idiom_ratio` was gone. The product was advertising two capabilities it no
longer had, on the screen 75% of creators land on. Changing detector behaviour
means re-reading every promise made about it.

**`GET /` fails safe toward Pro, structurally.** Missing file, missing key,
wrong type, unparseable JSON, and *a preference pointing at a creator.html that
is not on disk* all serve Pro. `/pro` and `/creator` never consult the
preference. This is the only change in the feature that can strand an existing
user on a page they did not ask for.

## What independent testing changed

An independent tester (which had built none of this) fuzzed the new routes and
drove the UI, and returned 12 defects. Three were the kind only an adversarial
pass finds, and all three were in the *frontend*, not the heuristic everyone was
worried about:

- **The record button was armed on a completed job.** `#/review/<id>` for a
  finished job showed a hardcoded "paused for your OK" chip and "Nothing has
  been recorded yet." — both false — beside a live button that would re-run TTS
  and assembly over finished output. Reachable with the Back button. Now gated
  in three layers, including a re-check inside the handler rather than trusting
  the DOM, because the action is irreversible.
- **A server outage served invented data.** A network failure was misclassified
  as "route missing", so the mock layer took over: seven fabricated videos and a
  spend tile reading **$14.80**. Same failure as "128 min left this month", but
  self-inflicted. The whole `MOCK` / `fetchOrMock` / `routeMissing` layer is now
  deleted and replaced with a real offline state.
- **26 of 51 cards showed raw internal errors** — one told the creator to run
  `ollama ps`, two told them to click a Resume button Creator mode does not
  have. Replaced prefix-stripping with an **allow-list plus a generic
  fallback**: a strip-list is always one new error message behind.

`tests/test_creator_routes.py` is the repo's first route-level coverage — before
it, `grep -rl TestClient tests/` returned nothing and all five new routes had
none. It is hermetic by construction (no lifespan, so `gochidubb.db` is never
opened; no http source, so no yt-dlp probe) and was proven runnable in a clean
`requirements-dev.txt` venv with no torch.

### Two testing lessons

**`TestClient` without its context manager tears down the event-loop portal when
a request ends**, cancelling anything `asyncio.create_task` spawned. Measured,
the slack is **one millisecond**: a 0ms stub yields `queued`, a 1ms stub yields
"Server shut down while preparing this video." So *every* background test was
passing by winning a coin flip — including ones that had passed 70+ times under
CPU contention, and which would have gone red on slower CI for reasons that have
nothing to do with the product. Green was green; the tell was invisible.

The fix is a named seam, `server._spawn_background(coro)`, in place of the bare
`create_task` — one line in production, identical behaviour, and tests patch it
to run the coroutine with `asyncio.run()` instead of racing the loop. No sleeps,
no polling. It also made outcomes *observable*, which allowed coverage that was
impossible before (partial cancellation; `CancelledError` at shutdown). Recorded
in CLAUDE.md under Testing notes.

The lesson generalises: **a flaky test is a symptom, and the ones that fail are
rarely the whole population.** This was found by going back through the tests
that were still passing.

**Cache before conclusions.** Two separate verification attempts against
`static/` changes read a cached page and showed pre-edit behaviour. If a
`static/` change appears not to work, rule out cache first, not last.

### One claim that was wrong, twice

A job stuck in `preparing` was described as **undeletable**, and that was used to
argue its severity up. `_UNDELETABLE_STATUSES` gates only
`POST /api/jobs/bulk_delete` (`server.py:9020`); `DELETE /api/job/{id}`
(`server.py:8976`) has no status guard. A test now pins the real behaviour so it
cannot be restated a third time.
