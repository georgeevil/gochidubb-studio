# Pro review workbench (4a) — implementation plan

Companion to the CLD-263 epic (children CLD-264…272, standalone CLD-273/274/275).
Every referenced thing is cited `file:line` against `feat/pro-review-workbench`
at 76b9d6f. Written for the implementer tasks #13–#18; nothing here needs
re-deriving, but line numbers WILL drift as tasks land — anchor by symbol name.

## Headline

Three of the four hard problems are already solved server-side: measurement
(`timing_quality` quality.py:296, loudnorm JSON assembler.py:107, flags
server.py:7901), partial re-runs (`retry_stage` server.py:8149,
`regenerate_segment` :9346), and pricing (`marginal_cost` billing.py:95). The
genuinely new mechanics are: **one gate evaluator** replacing two parallel pause
systems, **a subtitle module** (validator/cues/vtt — nothing exists), **assembler
fit overrides** (per-segment stretch + bounded pad slack), and **one tts_text
composition seam** (style prefix + pronunciation).

Also: **CLD-275 is already ~95% implemented** (commit 7dbe8a1) — see §10.

---

## 1. Gate model (CLD-264, foundation for everything)

### 1.1 Today's two systems, one call site

- `_wizard_pause_after()` server.py:3091 — single `wizard_mode` string, pauses
  after `diarize` or `translate` only.
- `_quality_gate_after()` server.py:3128 (map `_GATE_AFTER_STAGE` :3123) —
  score-driven auto-pause, same two stages.
- Both invoked at exactly one place in the driver, server.py:3302-3308, which
  `update()`s the pause status and returns.
- `/continue` (server.py:9160) resumes from `_latest_checkpoint()` and
  force-sets `ctx["wizard_mode"]="auto"` at :9254 — a resumed job can never
  pause again. `retry_stage` does the same at :8214.

**Decision: collapse both into one evaluator and make "cleared gates" explicit
state instead of destroying the pause config on resume.**

### 1.2 Schema

New pure module **`app/review_gates.py`** (pytest-friendly, no server import):

```python
GATES = ("transcript", "translation", "voice_cast", "subtitles", "final_qc")
MODES = ("off", "on", "flagged_only")

# Gate boundaries: which gates arm after which pipeline stage, in order.
# Two gates share the translate boundary — translation is reviewed before
# the cast, matching today's mutually-exclusive wizard modes.
BOUNDARY_GATES = {
    "diarize":   ["transcript"],
    "translate": ["translation", "voice_cast"],
    "assemble":  ["subtitles"],
    "merge":     ["final_qc"],
}

GATE_STATUS = {
    "transcript":  ("awaiting_transcript_review",  "transcription_done"),
    "translation": ("awaiting_translation_review", "translation_done"),
    "voice_cast":  ("awaiting_voice_review",       "translation_done"),
    "subtitles":   ("awaiting_subtitle_review",    "assemble_done"),
    "final_qc":    ("awaiting_final_qc",           "merge_done"),
}

def resolve_gates(explicit: dict|None, wizard_mode: str|None,
                  cfg_defaults: dict) -> dict:
    """Precedence: explicit review_gates > wizard_mode mapping > cfg defaults.
    A non-empty wizard_mode (INCLUDING "auto") suppresses cfg defaults — this
    is the backward-compat contract: creator.html and the CLI always send
    wizard_mode (creator.html:1220, client.py:111/138), so Pro-configured
    default gates can never surprise them."""

def first_pending(stage_id: str, gates: dict, cleared: list[str],
                  findings: dict[str, int|None]) -> str|None:
    """First gate at this boundary that is armed, not in `cleared`, and —
    for flagged_only — has findings > 0. findings[gate] is None when the
    caller could not compute (compute failure => pause: fail safe, same
    philosophy as today's quality gate being non-fatal but the inverse
    direction is deliberate — an uncomputable flagged_only check pauses
    rather than shipping unreviewed)."""
```

`wizard_mode` mapping: `review_transcript → transcript=on`,
`review_translation → translation=on`, `review_voices → voice_cast=on`,
`"auto"/"" → all off` (then cfg defaults only if wizard_mode was absent).

### 1.3 Where state lives

- `job["review_gates"]` — the resolved dict, set at submit, shown by the UI.
- `ctx["review_gates"]` — same dict; JSON-safe, no underscore, so it survives
  checkpoints (`_ctx_for_checkpoint` server.py:1910 drops `_`-prefixed keys
  only) and therefore resume/retry.
- `job["gates_cleared"]` + `ctx["gates_cleared"]` — list of gate names passed.
  **Reset rule**: when the driver runs stage `sid`, it removes from
  `gates_cleared` every gate at `sid`'s boundary and later ones (a
  retranslate re-arms the translation gate; a re-assemble re-arms the
  subtitle gate). Implemented in the driver loop, one line before the
  handler call at server.py:3294.
- `job["pending_gate"]` — the gate name currently blocking; cleared on resume.

### 1.4 Driver changes (server.py:3302-3308)

Replace the `_wizard_pause_after(...) or _quality_gate_after(...)` expression
with one call into an adapter `_evaluate_gate(sid, ctx, job, job_id)` that:

1. Computes `findings` lazily, only for gates present at this boundary in
   `flagged_only`:
   - `transcript`: existing quality-gate verdict (`full_report`+`gate`, kept
     verbatim from :3145-3155).
   - `translation`: `len(flag_segments(...))` (same call as `/flags` :7924)
     — plus the quality-gate verdict (fail ⇒ findings).
   - `voice_cast`: 1 if multi-speaker and `ctx["speaker_voice_map"]` empty,
     else 0.
   - `subtitles`: violation count from the subtitle validator (§4.2) over
     cues built from placed segments.
   - `final_qc`: warn-row count from the QC checklist builder (§4.4).
2. Preserves the legacy auto-gate: when a gate is `off` but
   `cfg.quality_gate` is true and the boundary is diarize/translate, the
   score check still runs and pauses exactly as today (status, audit record
   :3169, `job["quality_gate"]` stash all kept).
3. Returns `(status, detail, checkpoint)` or None. On pause, also sets
   `job["pending_gate"]`.

`_wizard_pause_after` and `_quality_gate_after` are deleted; their bodies move
into the adapter/pure module. **The voice-review placement comment at
server.py:3097-3106 (why voice casting gates after translate, not diarize)
must survive the move — it's load-bearing rationale.**

### 1.5 The final_qc special case (merge sets `complete` itself)

`_stage_merge` calls `update(status="complete", ...)` at server.py:3018-3024,
which fires `job.completed` webhooks via the driver's update closure
(`_fire_webhooks` call at :3224). A pause after merge must intercept *before*
that. Change `_stage_merge` to end with `update(progress=100, output_url=...,
completed_at=...)` **without** status, and let the driver do the final
transition: after the merge stage returns and its checkpoint is saved, the
driver either pauses (`awaiting_final_qc`) or sets `status="complete"` through
the same `update()` path (so `job.completed` fires exactly once, from one
place). `_finalize_reupload` (server.py:3028) keeps its own complete — reupload
jobs never gate.

**Showcase/batch exception**: jobs with `batch_kind == "showcase"` skip the
subtitle and final_qc gates entirely (force off at resolve time) —
`_maybe_assemble_showcase` (server.py:5914) fires when the last sibling
*completes*, and a sibling parked at QC would stall the reel. quick_test
batches gate normally (they're Pro/creator's main fan-out).

### 1.6 `/continue` advances exactly one gate

Rework `continue_pipeline` (server.py:9160) + `_continue_from_checkpoint`:

1. Derive the gate being passed from `job["status"]` (reverse of
   `GATE_STATUS`; jobs in non-gate statuses behave as today — crash-resume).
2. Append it to `gates_cleared` (job + persisted; mirrored into ctx on resume).
3. **Same-boundary check**: re-run `first_pending` for the current boundary.
   If another gate is pending (translation cleared, voice_cast armed), do NOT
   run any stage — flip status to the next gate's status, fire
   `job.awaiting_review`, return `{ok, now_awaiting: "voice_cast"}`.
4. Otherwise resume as today. Delete the `ctx["wizard_mode"]="auto"` line
   (:9254) — `gates_cleared` replaces it; later gates stay armed.
5. `merge_done` + final_qc: `_stage_after_checkpoint("merge_done")` returns ""
   and `_continue_from_checkpoint` marks complete directly (server.py:9236-9239)
   — that path bypasses the driver's webhook-firing update. Route it through
   `_fire_webhooks("complete", job)` explicitly (and record activity), so
   Approve fires `job.completed` exactly once.

`retry_stage` keeps "a retry always runs to the end" (:8213-8214) but instead
of forcing wizard_mode="auto" it sets `ctx["review_gates"]` all-off **unless**
the override payload carries `review_gates` — add `"review_gates"` to
`_RETRY_OVERRIDE_KEYS` (server.py:7599). The workbench's re-assemble passes
`{"review_gates": {"final_qc": "on", ...}}` to come back to the QC screen.

### 1.7 Status ripple (both new statuses)

| Surface | Change |
|---|---|
| `_WEBHOOK_EVENT_FOR_STATUS` server.py:1732 | add `awaiting_transcript_review` (missing today — a real bug vs webhooks.py:44's promise), `awaiting_subtitle_review`, `awaiting_final_qc` → `job.awaiting_review` |
| `webhooks.payload_for_job` webhooks.py:228 | add `pending_gate` |
| index.html `REVIEW_STATUSES` :105 | add both |
| index.html `stageForStatus` :127 | both → 5 (Master) |
| index.html `STATUS_META` :152 | `awaiting_subtitle_review` "Awaiting subtitle review", `awaiting_final_qc` "Awaiting final QC" (warn color) |
| creator.html awaiting list :77-79 | add both (defensive; creator can't arm them but a shared server must not show unknown-status jobs) |
| `_BUSY_STATUSES` :7682, `_UNDELETABLE_STATUSES` :10258 | no change — awaiting_* jobs are idle and deletable, same as existing review statuses |
| `ACTIVE_STATUSES` index.html:110 | no change |
| app/db.py virtual columns | no change (generic `json_extract` on status) |
| CLI `wait_for_job` client.py:469 | treat all `awaiting_*` as terminal-for-wait (check current set; extend if enumerated) |

### 1.8 UserConfig

Five fields + FIELD_SPECS enums + env_map entries (app/config.py:50, :360):

```python
review_gate_transcript: str = "off"    # GOCHIDUBB_GATE_TRANSCRIPT
review_gate_translation: str = "off"   # GOCHIDUBB_GATE_TRANSLATION
review_gate_voice_cast: str = "off"    # GOCHIDUBB_GATE_VOICE_CAST
review_gate_subtitles: str = "off"     # GOCHIDUBB_GATE_SUBTITLES
review_gate_final_qc: str = "off"      # GOCHIDUBB_GATE_FINAL_QC
```

Each `(str, ("off", "on", "flagged_only"))`. A helper in app/review_gates.py
assembles `cfg_defaults` from cfg.

---

## 2. API surface

New routes (all hermetically testable):

| Route | Req | Resp |
|---|---|---|
| `GET /api/dub/{id}/qc` | — | `{rows: [{id, label, state: "pass"\|"warn", value, detail}], warn_count, loudness_target, pending_fixes: {...}, on_approve: {resynth_segments, est_cost, deliverables, webhooks}}` — the one checklist document (CLD-270 §2); also what final_qc `flagged_only` consults |
| `POST /api/dub/{id}/review_notes` | form `text` | appends `{text, at, author: "local"}` to `job["review_notes"]` (list, capped 100) |
| `GET /api/dub/{id}/subtitles` | — | `{cues: [{idx, start, end, text, display_text?, cps, lines, chars_per_line, gap_ms, violations: [...]}], limits, violation_count, vtt_url, srt_url}` — render-ready, no client re-derivation |
| `POST /api/dub/{id}/subtitles/edit` | form `edits` JSON | per-cue `{display_text?, start_delta?, end_delta?}` → `job["subtitle_overrides"]`; rewrites .srt + .vtt; returns fresh cue list |
| `POST /api/dub/{id}/subtitles/autofix` | — | applies §4.3 fixes as overrides, post-validates, returns fresh cue list + remaining violations |
| `POST /api/dub/{id}/sync_plan` | JSON `{per_segment: {idx: {max_stretch?, pad_ms?}}, auto_fit_cap?}` | stores `job["sync_overrides"]`; server computes auto_fit expansions; returns the plan + which segments it will change. Re-assembly is then the existing `retry_stage/assemble` |
| `POST /api/dub/{id}/speakers/edit` | JSON `{ops: [{op: merge\|reassign\|rename\|mark_non_speech, ...}], apply_to_siblings: bool}` | applies to checkpoints (§5); returns updated speaker summary |
| `POST /api/dub/{id}/estimate_edits` | JSON `{edits: [...]}` (§7) | `{resynth_seconds, resynth_minutes, est_cost_usd, free, free_reason?, breakdown}` |
| `POST /api/dub/{id}/consent` | form `speaker`, `attested` | writes `job["voice_consent"][speaker]` (§6) |

Changed routes:

- `POST /api/dub` (:~3490) and `POST /api/quick_test` (:~5400): accept optional
  `review_gates` form field (JSON). Resolution per §1.2.
- `POST /api/dub/{id}/continue`: clears one gate (§1.6). Response gains
  `now_awaiting` when it lands on a same-boundary gate.
- `GET /api/job/{id}` ("get_status"): gains `cost_so_far_usd` (billing
  `job_minutes` × marginal position) and `eta_seconds` (app/estimate factor ×
  remaining fraction), plus `review_gates`/`pending_gate`/`gates_cleared` —
  this is how MCP/CLI get CLD-273's numbers without new tools for them.
- `POST /api/glossary/term` (:10047): optional `say` form field (§4.5).
- `POST /api/dub/{id}/voice_preview` (:8618): optional `text` in body — when
  present, synthesize that text (once per requested speaker) instead of picking
  lines. **CLD-266 claims this route already takes arbitrary text; it does not**
  (it picks mid-length real lines, server.py:8664-8681). Small change, needed
  for the pronunciation ▶ preview.
- `GET /api/dub/{id}/files` (:8737): add `"subtitles_vtt"` entry.
- `PATCH /api/config`: no code change — new UserConfig fields ride the
  existing settings machinery; gates panel + loudness preset use it.

---

## 3. Assembler & loudness (CLD-268, CLD-270)

### 3.1 Fit overrides + pad slack (pipeline/assembler.py)

`plan_segment_fit` (assembler.py:199) is already the pure timeline decision.
Extend, keeping it pure:

```python
def plan_segment_fit(seg_start, next_start, tts_dur, current_end,
                     max_stretch=_DEFAULT_MAX_STRETCH, pad_slack=0.0):
    # pad_slack: the segment may start up to this many seconds EARLY when
    # the previous placed segment's end + MIN_SEGMENT_GAP allows, spending
    # inter-segment silence to close "-8% early" drift. Never overlaps:
    # early start is bounded by current_end + MIN_SEGMENT_GAP exactly as
    # late start already is.
```

`assemble_dubbed_audio(...)` gains `fit_overrides: dict|None` — keyed by
segment `idx`: `{"max_stretch": float, "pad_ms": int}`. Per segment,
`max_stretch` overrides `_max_stretch_setting()` (:233, clamp 1.0–2.5 stays),
`pad_ms/1000` feeds `pad_slack`. The `+0.07` emotional bonus (:313) applies on
top of the per-segment value, same as the global one.

`_stage_assemble` (server.py:2954) passes
`fit_overrides=_sync_fit_overrides(job)` — a small server helper that merges
`job["sync_overrides"]["per_segment"]` with auto-fit expansion:
**auto_fit** computes, per placement row with stretch > cap⁻¹-tolerance…
concretely: for each `timing_quality` row whose `stretch` exceeds
`1 + drift_tol` but would close under `auto_fit_cap`, emit
`{idx: {max_stretch: needed}}`. Computed server-side in `/sync_plan` so the
table, the plan, and the re-assemble agree; segments already inside tolerance
get no override (CLD-268 acceptance).

### 3.2 Loudness target (CLD-270)

UserConfig: `loudness_target: float = -16.0` (FIELD_SPECS `(float, -31.0,
-8.0)`), `loudness_true_peak: float = -1.5` (`(float, -9.0, 0.0)`). Env:
`GOCHIDUBB_LOUDNESS_TARGET`/`_TRUE_PEAK`. UI presets: −16 default / −14
YouTube / −23 EBU (a select over the numeric field, not an enum — power users
can type).

`_normalize_loudness_inplace(wav_path, target_i=None, target_tp=None)`
(assembler.py:107) — None falls back to module constants LN_I/LN_TP (:16-18,
kept for standalone callers). `assemble_dubbed_audio` threads them;
`_stage_assemble` passes `cfg.loudness_target`/`cfg.loudness_true_peak`.
Measurement keeps flowing to `perf["loudnorm"]` → `loudness_quality()`
(quality.py:335) untouched.

---

## 4. Subtitles (CLD-269) + QC checklist (CLD-270)

### 4.1 Module placement

New **`pipeline/subtitles.py`** — pure functions, logging via
`logging.getLogger("gochidubb.subtitles")`, no server imports:

```python
DEFAULT_LIMITS = {"max_chars_per_line": 42, "max_lines": 2,
                  "max_cps": 17.0, "min_gap_ms": 120}

def build_cues(segments, overrides=None) -> list[Cue]
    # placed_start/placed_end preferred over start/end (same rule as
    # write_srt assembler.py:38); text = override display_text
    # or translated_text or text; non_speech segments are skipped.
def validate_cues(cues, limits) -> list[Violation]   # per-cue, typed kinds
def autofix_cues(cues, limits) -> dict[int, dict]    # returns override map
def write_srt_cues(cues, path); def write_vtt_cues(cues, path)
```

UserConfig: `subtitle_max_chars_per_line: int = 42`, `subtitle_max_lines: int
= 2`, `subtitle_max_cps: float = 17.0`, `subtitle_min_gap_ms: int = 120`
(+ FIELD_SPECS bounds, env entries).

### 4.2 display_text lives on the JOB, not in checkpoints

**Decision (resolves CLD-269's open question): a per-cue display override map
`job["subtitle_overrides"] = {"<idx>": {display_text, start_delta, end_delta}}`
— persisted with the job via `save_job`, never entering `_serialize_segments`'
whitelist.** Rationale: `translated_text` is the dialogue contract (SRT,
editor, retry-reuse, emotion-tag heuristic — CLAUDE.md); a CPS-shortening must
not change what was spoken NOR invalidate audio reuse, and keeping it off the
checkpoints means zero interaction with retry/reuse machinery. Timing nudges
bounded ±500 ms and clamped so cues never overlap neighbors.

Both SRT writers converge: `_stage_assemble`'s rewrite (server.py:2966) and
`edit_translations`' re-export (:8388) switch from `write_srt(segments, path)`
to building cues via `pipeline.subtitles.build_cues(segments,
job.get("subtitle_overrides"))` and writing **both** .srt and .vtt.
`write_srt` in assembler.py stays for the transcript-SRT path (translate
stage) — or is refactored to delegate; implementer's choice, output identical.

### 4.3 Auto-fix rules (post-validated, never creates a violation)

1. Line too long → re-wrap at last space ≤42 chars.
2. >2 lines → re-break at sentence/clause boundary; if impossible, merge to 2.
3. CPS > limit → cannot extend audio; shorten via display_text is a HUMAN
   edit — auto-fix only extends cue out-time into available gap (up to next
   cue − min_gap); if still violating, leave flagged.
4. gap < 120 ms → shave earlier cue's out-time.
Run `validate_cues` again after; only apply fix set if violations strictly
decrease.

### 4.4 QC checklist builder (server-side, render-ready)

`GET /api/dub/{id}/qc` assembles from existing data, no new computation stored:

- **loudness**: stage perf `loudnorm` (server.py:2985-2989) vs
  `cfg.loudness_target` ±1 LU; true peak vs cfg + clipping verdict.
- **subtitles**: `validate_cues` count.
- **sync**: `timing_quality()` rows with |stretch−1| > 0.05 (the design's ±5%)
  — count + idx list (drives the sync tab's deep link).
- **glossary honored**: for each glossary term whose trigger matched
  (flattened for target_lang), scan final `translated_text` — N/M.
- **consent**: speakers cloned without attestation when policy ≠ off (§6).
Each row `{id, label, state, value, detail}`. `warn_count` is the final_qc
`flagged_only` finding count.

Review notes: `job["review_notes"]` list; author "local" until CLD-247.

### 4.5 Pronunciation "say it like" (CLD-266)

Glossary term entries gain optional `say`. Storage stays backward-compatible:
`terms` values are today plain strings (`src: dst`); a term with pronunciation
becomes an **object** `{"dst": "...", "say": "GOH-chee"}` — the loader
normalizes both shapes (validator at server.py:9990 extended: term value must
be str or `{dst?, say?}` object). Translator's flattening
(`pipeline/translator.py` glossary cache) reads `dst` only; a term may have
either or both. `POST /api/glossary/term` gains `say` form field; same
per-target_lang scoping and `creator-review` default domain (:10029);
`clear_glossary_cache()` on write as today.

**One composition seam.** New `pipeline/pronounce.py`:

```python
def compose_tts_text(seg, style_prefix: str = "", say_map: dict = None) -> str|None:
    """The ONLY writer of seg['tts_text'] semantics: applies word-boundary,
    case-insensitive respelling of glossary `say` terms to the segment's
    translated_text, then the Voice-Design '(style)' prefix. Returns None
    when neither applies (caller leaves tts_text unset)."""
```

The two current write sites — server.py:2834-2839 (`_stage_tts`) and
:7315-7324 (`_run_tts_and_merge_stage`) — both call the seam; the seam is
also called when there is NO style prefix but the say_map matches (that's the
new part: pronunciation without voice design). say_map built per job:
flatten glossary for `target_lang` (+ all-language entries) → `{term: say}`.

Constraints honored:
- `tts_text` never enters `_serialize_segments`' whitelist (server.py:1953) —
  recomputed at every TTS run, which also means edits to the glossary
  automatically apply on retry.
- Only `VoxCPMSynthesizer` reads `tts_text` (synthesizer.py:545-550) — edge-tts
  and F5 never see it. **v1 is VoxCPM-only, documented in the UI tooltip.**
  (CLD-266's fear that other engines "would speak it" is structurally
  impossible today; keep it that way.)
- QA: `tts_worker._synth_one` synthesizes and QA-compares the same string
  (`seg["text"]`, tts_worker.py:175 → `_run_qa` :293) — whisper is compared
  against the respelled text it actually heard, so respellings don't
  mass-fail QA by construction. The spike (§8) confirms the CER impact.
- IPA: **not supported v1** unless the spike says otherwise; the field is
  free-text respelling.

Editor UI: term table + add form + ▶ preview via `voice_preview` with `text`
= a sentence containing the respelling (§2).

---

## 5. Speaker relabeling (CLD-267)

`POST /api/dub/{id}/speakers/edit`, allowed only pre-TTS-commit (status in
review/paused/error and a `transcription_done` checkpoint exists; refuse in
`_BUSY_STATUSES`). Ops applied to **both** `transcription_done` and
`translation_done` checkpoints when present (mirroring `edit_speaker_ref`'s
write-through, server.py:8406+):

- **merge {from, into}**: rewrite `speaker` on every matching segment; delete
  `speaker_refs[from]` in checkpoint data; `speaker_voice_map`: drop `from`
  (merged segments inherit `into`'s cast — "voice cast updates automatically").
- **reassign {segment_idx, to}**: one segment's `speaker`.
- **rename {speaker, label}**: display only → `job["speaker_labels"]`
  (persisted job field; APIs return it; the S-id stays stable everywhere).
- **mark_non_speech {segment_idxs}**: sets `non_speech: true` on the segment
  (add `"non_speech"` to `_serialize_segments`' optional whitelist,
  server.py:1957 — safe: it's pipeline-meaningful state, unlike tts_text).
  Consumers: TTS prep skips them (`_stage_tts`/`_run_tts_and_merge_stage`
  filter before building synth input), `build_cues` skips them, translator
  skips them if still untranslated. **Indices never shift** — segments stay in
  place, so flags/regenerate/placements idx-keying is unaffected (CLD-267
  acceptance).

Reuse hazard: the partial-retry recovery (server.py:2846-2864) reuses audio
keyed on idx + translated_text. **Add a speaker equality check** — a merged
segment must not reuse audio synthesized under the old speaker's voice
(`old.get("speaker") != s.get("speaker") → skip`). `speaker` is whitelisted in
checkpoints so old checkpoints carry it. Seeds stay deterministic per (job,
speaker, segment_idx) — merging before TTS is exactly the safe window; the
route's status guard enforces it.

Sibling fan-out: **explicit, not automatic** — `apply_to_siblings: true`
applies the same ops to every `batch_id` sibling that also passes the pre-TTS
status guard; response reports per-sibling outcome. UI checks the box by
default when siblings are pre-TTS (the issue's recommendation, with the
user-visible say-so it asks for).

UI: speaker cards (minutes + ▶ via existing `GET /speakers` :4915 + ref
audio/waveform :4951/:4967), low-minutes outlier callout (< 10% of total
speech or < 3 min), found-N/you-said-M banner when `speaker_mode` declared a
count, per-segment reassign list.

---

## 6. Voice consent (CLD-272 part 1) + pitch/pace (part 2)

### 6.1 Consent records

Two attachment points, one shape `{attested_by: "local", attested_at: ts,
scope: "dubbing"}`:

- **Uploaded presets**: consent captured at preset save (checkbox in the
  existing preset form) → stored in the preset record
  (`/api/voice_presets` CRUD server.py:9488-9674).
- **Per-job source cloning**: `POST /api/dub/{id}/consent` (form `speaker`,
  `attested=true`) → `job["voice_consent"][speaker]`. Serialized with the job
  (survives redub/retry — it's a plain job field through `save_job`).

Policy: UserConfig `voice_consent_policy: str = "off"` (`off|warn|require`,
FIELD_SPECS enum, env `GOCHIDUBB_CONSENT_POLICY`). Enforcement point:
`_resolve_casting` (used by `_stage_tts` :2737 and voice_preview) — for casts
resolving to `source` or `file:*` without an attestation (job field for
source; preset record for file):

- `warn`: proceed; cast_report entry gains `consent: "missing"`; a job notice.
- `require`: downgrade that speaker to the default preset; cast_report entry
  `used: <preset>, reason: "no clone consent · using preset"` — literally the
  design's guest-card string, surfaced by the cast rail.

Admin compliance panel: replace the "nothing to check against" copy
(app/admin.py:951, :988; static/admin.html:763) with real counts read from
presets + jobs. "Request consent link" stays out (CLD-247).

### 6.2 Pitch/pace: time-boxed investigation, not a feature

Protocol (task #15, ≤ half a day): `tests/manual_test_pitch_pace.py` — take
one cloned segment wav from an existing job, apply ffmpeg
`asetrate=48000*2^(±2/12),aresample=48000` (pitch) and `atempo=0.9/1.1`
(pace), run `tts_qa.check_segment_quality` against the original text + listen.
Go rule: whisper CER delta < 0.05 AND subjectively same-speaker at ±2
semitones/±10%. No-go (expected, per assembler.py:202's documented absence of
a rate knob): document the rejection in this file's changelog + CLD-272
comment; the design's sliders don't ship.

---

## 7. Cost-of-edits estimator (CLD-271)

`POST /api/dub/{id}/estimate_edits` — request:

```json
{"edits": [
  {"kind": "segment_text",   "idxs": [3, 17]},
  {"kind": "speaker_voice",  "speaker": "S2"},
  {"kind": "pronunciation",  "term": "GoChi"},
  {"kind": "subtitle_display", "idxs": [5]}
]}
```

Seconds-counting rules (pure helper `app/billing.py::edit_seconds(job_state,
edits)` or a small `app/estimate_edits.py` — pure, tested):

- Job has **no `tts_done` checkpoint** (or is parked at transcript/translation
  gate): everything is free → `{free: true, free_reason: "Text edits before
  TTS are free — nothing has been synthesized yet."}`.
- `segment_text` post-TTS: Σ (end − start) of the listed idxs.
- `speaker_voice`: Σ durations of all that speaker's speech segments.
- `pronunciation`: Σ durations of segments whose `translated_text` contains
  the term (server resolves term → idxs; word-boundary match, same matcher as
  the §4.5 seam so estimate and effect agree).
- `subtitle_display`: always 0 (display-only, audio untouched).
- Union, not sum, of overlapping idx sets (a segment counted once).

Price: `marginal_cost(month_used_minutes, resynth_minutes)` (billing.py:95),
month usage from `summarize()` — the figure matches the meter, marginal not
flat (CLD-271 acceptance). Response carries the local-mode disclaimer string
from app/billing's docstring. Shared UI: one `EstimateLine` component
(cost + Apply) used by sync-fit, subtitles, cast rail, QC approve panel.

---

## 8. VoxCPM respelling spike (decides CLD-266's ▶ shipping shape)

**Cheapest honest experiment** (task #15, before wiring UI polish; server may
be running but the spike doesn't need it):

`tests/manual_test_respelling.py` (exploratory script, not pytest — per repo
convention): load `VoxCPMSynthesizer` once, one fixed reference clip + seed,
synthesize pairs — plain vs respelled — for 4 term classes:
1. brand: "GoChi" vs "GOH-chee"
2. acronym: "SQL" vs "sequel"
3. foreign name in ru target: "Хачатурян" latin respelling
4. multi-word: "kubectl" vs "kube-control"

For each: `tts_qa.check_segment_quality(audio, <respelled text>)` + human
listen. **Go**: ≥3/4 terms audibly pronounced as intended AND surrounding-text
CER delta ≤ 0.1. **No-go fallback (ship regardless)**: the `say` field,
storage, seam, and UI all ship — they're harmless when unused — with the
editor labeled "phonetic respelling — works best with simple sound-alike
spellings; IPA not supported". Only the *claim* changes, not the code. If the
spike passes, remove the hedge and note measured results here.

Also try one IPA string; if VoxCPM speaks it as letters (expected), IPA stays
explicitly unsupported.

---

## 9. Activity persistence (CLD-274, option A — decided)

`app/activity.py` gains a store, keeping its API and in-memory deque:

- `attach_store(db_path)` — called from server lifespan right after
  `init_db()`. Creates `activity(id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, kind TEXT, data TEXT)` in **gochidubb.db** (own connection,
  `PRAGMA journal_mode=WAL`, thread-local; app/db.py's jobs connection
  untouched).
- Write path: `_append` (already redacts via `pipeline.notices.redact` — the
  redact-before-persist requirement is satisfied by construction) pushes onto
  a `queue.Queue(maxsize=2000)`; a daemon writer thread batches inserts (every
  ~1 s or 50 events). Full queue → drop oldest, count drops. The job-runner
  path never blocks on SQLite.
- `_seq`/`since_id`: on attach, `_seq` is initialized to MAX(id) and inserts
  use explicit ids from `_seq` — rowid == seq, so `GET /api/activity?since_id=`
  (server.py:3686) semantics survive restart unchanged.
- Startup backfill: load newest `DEFAULT_CAPACITY` rows into the deque —
  the feed home screen survives restart (the point of the ticket).
- Retention: on every 500th insert, `DELETE WHERE ts < now − 30d OR id <=
  max_id − 10000`.
- Docstring at activity.py:26-28 ("deliberately in-memory") must be rewritten
  to record the new decision and why (home screen).
- Tests: attach to a tmp db, append, assert rows + since_id continuity across
  a simulated restart (re-attach, deque backfilled). No server needed.

---

## 10. API keys (CLD-275) — mostly already done; verify, don't rebuild

Contradiction with the issue text (and with the explorer survey's item 10):
commit 7dbe8a1 already landed `environment` (`gcd_live_`/`gcd_test_` prefixes,
apikeys.py:64-67, create :131), `expires_days`/`expires` (create :158,
`is_active` refusing expired keys :198, `verify` honoring it :206), the route
accepting both (server.py:3728-3748, with `apikey.create` audit carrying
environment), and the UI form + rendering (index.html:5876-5877, :6025, :6033).

Remaining delta (task #16, small):

1. UI: `dead` (index.html:6025) conflates expired and revoked — render
   distinct badges ("expired" vs "revoked") and show the expiry date column
   (acceptance: "renders as expired, not revoked").
2. Form presets 30/90/365/never — verify present; add if not.
3. Audit on expiry-driven refusals: **deferred with CLD-197** — `verify()` is
   called from nowhere (no enforcement path exists; `_require_scope` at
   server.py:3717 is a comment about a function that doesn't exist), so there
   is no refusal to audit yet. Record this in the ticket, don't build dead code.
4. Existing keys without `environment` must list as `live` — check
   `list_keys`/`public` handles absence; one-line default if not.

---

## 11. Workbench UI (task #17)

### 11.1 Placement & component tree (index.html — React 18 UMD, no build step)

`ReviewView` (index.html:3916) is replaced by the workbench, mounted where it
already lives (the jobs view's review surface; `App` passes
`selectedReviewJobId` at :6962+). Match existing idioms: function components,
`useState`/`useEffect`, inline styles, `fetch` against `/api`.

```
ReviewWorkbench({jobs, selectedJobId, voicePresets, onContinued, onCancel, onPickJob})
├─ WorkbenchHeader
│   ├─ job picker (existing multi-job select, :3969-3977)
│   ├─ pipeline chip strip        — reuse STAGES/stageForStatus (:117/:127)
│   ├─ per-language sibling tabs  — batch_id siblings + flag counts (CLD-265)
│   └─ header actions             — context-dependent: Reject-to-X / Approve
├─ WorkbenchTabs  (Translation · Pronunciation & speakers · Sync fit ·
│                  Subtitles · Final QC — badges: flags / speaker warnings /
│                  off-sync count / violation count / qc warn_count)
├─ <tab body>
│   ├─ TranslationTab   — SegmentList (Flagged/All/Edited filters),
│   │                     FlagCard (kind+confidence from /flags, inline edit →
│   │                     /edit_translations, Accept, Add-to-glossary →
│   │                     /api/glossary/term, Apply-to-all-langs → same edit
│   │                     to each sibling's checkpoint), length-fit hint
│   │                     (len ratio vs cfg.tts_max_stretch)
│   ├─ PronSpeakersTab  — SayTermTable (+ add form + ▶ voice_preview w/ text),
│   │                     SpeakerCards + OutlierCallout + ReassignList
│   │                     (→ /speakers/edit)
│   ├─ SyncFitTab       — DriftTable (fed by /quality timing block — one
│   │                     source of truth, no client re-derivation),
│   │                     per-row fixes (shorten → inline edit +
│   │                     /regenerate_segment; allow speed-up / pad →
│   │                     /sync_plan), AutoFitBar (cap input + apply),
│   │                     ABPlayer (source vs dub, locked transport)
│   ├─ SubtitlesTab     — CueTable (from /subtitles, violations inline),
│   │                     AutoFixButton, video preview + HTML cue overlay,
│   │                     export .srt/.vtt links
│   └─ FinalQCTab       — QCChecklist (from /qc), ReviewNotes,
│   │                     ApprovePanel (on-approve summary + EstimateLine
│   │                     → /continue), Reject → jumps to sync/subtitle tab
└─ right rail
    ├─ GatesPanel       — the job's review_gates (read-only per job) + link
    │                     to Settings for defaults
    ├─ CastingPanel     — REUSED as-is (:3729+), shown when the voice_cast
    │                     gate is pending (consent chips added, §6)
    ├─ GlossaryQuickAdd
    └─ EstimateLine     — shared cost component (§7)
```

Shared components: `ABPlayer` and `EstimateLine` are used by two+ tabs.
Everything data-shaped comes render-ready from the server (`/flags`, `/qc`,
`/subtitles`, `/quality`) — that is the anti-duplication seam with
creator.html (vanilla JS): creator can adopt the same endpoints later without
sharing any client code.

**Tab-vs-status routing**: default tab from status
(`awaiting_transcript_review`/`awaiting_translation_review` → Translation,
`awaiting_voice_review` → Translation w/ rail casting focus,
`awaiting_subtitle_review` → Subtitles, `awaiting_final_qc` → Final QC). All
tabs stay reachable whenever their data exists (checkpoint/placements
present); tabs without data render an explanatory empty state. Approve always
means `/continue` (clears exactly the pending gate).

### 11.2 Nav consolidation (12 → 10)

Nav groups at index.html:686-706; render switch :6962+; TITLES :883.

- **Members → fold into API keys** (remove nav item, `MembersView` :6598, its
  TITLES entry). The view's own copy admits it: "Reference, not enforcement"
  (:6612). Its two real pieces move: the principals table (this machine + live
  keys) is redundant with ApiKeysView's key list; the ROLES×PERMISSIONS matrix
  (:6586-6596) becomes a collapsed "Roles reference" block at the bottom of
  ApiKeysView. Real members ride on CLD-247.
- **Audit → a tab inside Agent feed** (remove nav item). Both are event
  streams; with the feed persisted (§9) the split reads as arbitrary.
  `AuditView` (:6702) becomes a feed sub-tab keeping its own fetch of
  `/api/audit` — the audit log remains its separate append-only store
  (activity ≠ audit, per activity.py's docstring; this merges *screens*, not
  stores). The hosted-mode "Workspace" group then holds Billing alone — move
  Billing into Develop and drop the Workspace group (it returns with CLD-247).
- Keep: feed, home, jobs, library, discover (distinct scout workflow), apikeys,
  mcp, webhooks, billing, system. Verified each remaining view fetches distinct
  API surface; no other pair overlaps.

---

## 12. MCP + CLI parity (CLD-273, task #18)

Per the capability rule (route → client → CLI + MCP):

`tools/gochidubb_client.py` methods:
- `retry_stage(job_id, stage, stop_after="", overrides=None)` → POST
  retry_stage (overrides passed through; server whitelists via
  `_RETRY_OVERRIDE_KEYS`).
- `get_flags(job_id, max_flags=5)`, `add_glossary_term(term, translation,
  target_lang, domain="", say="")`, `edit_translations(job_id, edits: dict)`.
- `get_job` untouched — cost/ETA/gates arrive because the ROUTE gains them
  (§2), which is the design's `get_status` promise with zero new tools.

`tools/gochidubb_mcp.py`: `gochidubb_retry_stage`, `gochidubb_get_flags`,
`gochidubb_add_glossary_term`, `gochidubb_edit_translations`. Update
`gochidubb_quality_report`'s docstring (mcp:466) so its recommended next
actions are now real tools. `gochidubb_dub` gains optional `review_gates`.

CLI subcommands: `retry-stage`, `flags`, `glossary-term`, `edit-translations`;
`dub` gains `--review-gates translation=on,subtitles=flagged_only`; `status`
prints cost-so-far/ETA/pending gate when present.

Acceptance walk (scriptable): see paused gate in `get_job` → `get_flags` →
`edit_translations` → `continue_job` — no curl.

---

## 13. Work breakdown, ordering, interference

server.py is one file; the merge-chaos control is **region partition +
sequencing**. Regions by task:

| Task | server.py regions | Other files |
|---|---|---|
| #13 gates+QC | driver :3091-3330 (rewrite), `_stage_merge` :3005-3025, webhook map :1732, `/continue` :9160-9270, retry keys :7599 + :8214, submit routes (review_gates param), `/qc` + `/review_notes` (new, append at file end) | app/review_gates.py (new), app/config.py, pipeline/assembler.py (loudness only), webhooks.py:228, creator.html :77 |
| #14 subs+sync+cost | `_stage_assemble` :2954-2999 (cue writer + fit_overrides), `edit_translations` re-export :8388, new routes /subtitles*, /sync_plan, /estimate_edits (append) | pipeline/subtitles.py (new), pipeline/assembler.py (plan_segment_fit/assemble signature), app/estimate_edits.py (new), app/config.py |
| #15 pron+speakers+consent | tts_text sites :2834 + :7315 (→ seam calls), `_resolve_casting` (consent), reuse check :2846-2864 (speaker equality), `_serialize_segments` :1957 (+non_speech), glossary :9990-10130 (say), /speakers/edit + /consent (new, append), voice_preview `text` :8618 | pipeline/pronounce.py (new), app/config.py, app/admin.py + admin.html copy |
| #16 activity+keys | lifespan (attach_store call), nothing else | app/activity.py, index.html ApiKeys badges |
| #17 frontend | — | index.html only (+ creator.html untouched beyond #13's status list) |
| #18 parity | — | tools/*.py only |

**Ordering verdict:**

1. **#13 first, alone.** Everything else references the evaluator (flagged_only
   hooks), the two new statuses, or the `/qc` shape. It rewrites the driver's
   pause seam — the one region nobody else may touch concurrently.
2. **#16 in parallel with #13** — disjoint files except one lifespan line
   (coordinate that single line or land it with #13).
3. **After #13: #14 ∥ #15.** Their server.py regions are disjoint (assemble
   stage + subtitle routes vs TTS prep + glossary + speaker routes). Two known
   touchpoints to coordinate: (a) both append new routes — assign each a
   marked section; (b) app/config.py FIELD_SPECS — additive lines, trivial.
   Both touch `_serialize_segments`? No — only #15 (non_speech); #14's
   display_text deliberately stays off checkpoints (§4.2).
4. **#17 after #13 lands** (statuses + gates panel scaffolding can start
   immediately; Translation tab needs only existing routes), finishing after
   #14/#15 route shapes settle. #17 owns index.html exclusively — no backend
   task edits index.html except #13's three constant maps and #16's ApiKeys
   badges; hand both to #17 as a note instead if timing collides.
5. **#18 last** — thin wrappers over final route shapes.

Explicit #13/#14 collision flag (asked for): both touch pipeline/assembler.py
(`assemble_dubbed_audio` signature: #13 adds loudness params, #14 adds
fit_overrides) and `_stage_assemble`. Resolution: #13 lands its loudness
threading first; #14 rebases and extends the same signature. Neither touches
`_stage_merge` except #13.

---

## 14. Test strategy

**Pure suites (new, no server):**
- `tests/test_review_gates.py` — resolve_gates precedence (explicit >
  wizard_mode > cfg; "auto" suppresses defaults), wizard_mode mapping,
  first_pending ordering at the translate boundary, flagged_only zero-findings
  sail-through, None-findings pauses, cleared-gate reset on stage rerun.
- `tests/test_subtitles.py` — validator kinds against crafted cues (21 CPS,
  43-char line, 3 lines, 50 ms gap), autofix never-worsens property, vtt/srt
  golden output, build_cues placed-time preference + non_speech skip +
  override precedence.
- `tests/test_estimate_edits.py` — the §7 counting rules incl. idx-union and
  pre-TTS free path.
- `tests/test_assembler.py` (extend) — `plan_segment_fit` with per-seg
  max_stretch, pad_slack bounded by previous end, never-overlap property.
- `tests/test_pronounce.py` — compose_tts_text: respelling word-boundary +
  case, prefix composition order, None when nothing applies, scoping.

**Hermetic route suites** — extend the `tests/test_creator_routes.py` pattern
exactly (TestClient WITHOUT context manager — no lifespan, so the real
gochidubb.db is never opened; patch `server._spawn_background` (server.py:3889)
to capture coroutines and drive them with `asyncio.run()`; monkeypatch
`server._load_checkpoint`/`_save_checkpoint`, glossary path, `server.jobs`):
`/qc`, `/subtitles` + edit + autofix, `/sync_plan`, `/speakers/edit`
(checkpoint write-through asserted via captured saves), `/estimate_edits`,
`/continue` one-gate semantics (job fixture parked at a gate; assert
`gates_cleared` + `now_awaiting` without running a stage), glossary `say`
validation, `/consent` + require-policy downgrade via a stubbed
`_resolve_casting` input, activity store on a tmp db, voice_preview `text`
validation (409/400 paths only — no engine).

Note: `/continue` uses `asyncio.create_task` directly (server.py:9204) — have
#13 route it through `_spawn_background` so the gate tests don't race the
1 ms task-cancellation window (CLAUDE.md's measured hazard).

**Cannot be tested hermetically — the real-dub e2e (task #20, allowed on this
machine, <2 min source, delete jobs after):**
1. Submit with `review_gates: translation=on, subtitles=on, final_qc=on` →
   assert three pauses, `job.awaiting_review` per pause (local webhook
   catcher), `/continue` advancing exactly one gate each time.
2. At subtitle gate: introduce a CPS violation via display edit, autofix,
   export .vtt, load in a browser `<track>`.
3. At QC: flip `loudness_target` −16 → −14, `retry_stage/assemble` with
   re-armed final_qc, assert measured `output_i` moved ~2 LU.
4. Sync fix: one segment `max_stretch` override + one shorten-text
   regenerate; re-assemble; `/quality` inside ±5% for both; untouched
   segments' audio files byte-identical (mtime/hash).
5. Speaker merge on a 2-speaker clip pre-TTS; assert merged segments speak in
   the target cast voice and ghost is gone from `/speakers`.
6. Respelling spike (§8) + pitch/pace investigation (§6.2).
7. Browser walkthrough of all five tabs + consolidated nav (claude-in-chrome).

wizard_mode regression: one CLI dub with `--wizard translation` equivalent —
pauses once, `continue`, runs to complete (proves legacy mapping + that cfg
defaults don't leak).

---

## 15. Top risks, mitigations baked in

1. **/continue semantic change breaks CLI/MCP/creator.** Mitigated by:
   wizard_mode mapping produces exactly one armed gate → one pause → one
   continue (behavior-identical); cfg defaults suppressed whenever
   wizard_mode was sent (§1.2); regression test in §14.
2. **Double or missing job.completed at the final_qc gate.** Mitigated by
   moving the complete transition into the driver (single firing site, §1.5)
   and an e2e webhook-catcher assertion.
3. **server.py merge chaos.** Region partition table (§13), #13 serialized
   first, index.html owned by #17.
4. **Assembler regressions from pad slack/overrides.** Pure-function property
   tests (never-overlap, bounded early start); behavior byte-identical when no
   overrides present (default args); overrides only reachable via explicit
   /sync_plan.
5. **Checkpoint contamination.** Policy table enforced by review (task #19):
   `tts_text` never whitelisted (recomputed via the §4.5 seam);
   `display_text` lives on the job, not segments; only `non_speech` joins the
   whitelist. Any new segment field must argue its way past
   `_serialize_segments`' docstring.

(6th, accepted: activity writes add one queue push per event; the batching
writer thread and drop-on-overflow bound the cost — measured assertion not
required.)

---

## 16. Contradictions found vs the issue texts (for Linear updates, task #21)

1. **CLD-275**: `environment` + `expires` already fully implemented in
   apikeys.py/routes/UI (commit 7dbe8a1). Remaining: expired-vs-revoked
   display, expiry presets check, defaults for legacy records. The explorer
   survey's item 10 is stale on the same point.
2. **CLD-264 "what exists"**: `_WEBHOOK_EVENT_FOR_STATUS` (server.py:1732)
   does NOT map `awaiting_transcript_review` — the transcript gate silently
   fires no webhook today. Fixed by §1.7.
3. **CLD-266**: claims `/voice_preview` "can synthesize arbitrary text" — it
   cannot (it previews mid-length real lines, server.py:8664-8681). §2 adds
   the `text` param. Also its fear that edge-tts/F5 "would speak whatever
   tts_text holds" is structurally impossible — only VoxCPM reads tts_text
   (synthesizer.py:545-550).
4. **CLD-263** epic cites `ReviewView` as only-continue — correct, but note it
   already commits the voice cast before continuing (index.html:3948-3960);
   the workbench must keep that commit-before-continue ordering.
