# Linear drafts — workbench follow-ups (2026-08-25)

The Siteloom workspace is at its free-plan issue cap (`save_issue` create →
400). File these under team **Claude**, project **GoChiDUBB**, once
capacity exists. Same pattern as `linear-drafts-pr38-42.md`.

---

## 1. Auto bed-balance: set the background gain from measurement, not a constant

**State: Backlog · Feature · relates CLD-263, the ×10-default commit**

The ×10 default background gain is an empirical constant that happens to
restore the source's voice:music balance *at the current dub-voice
loudness* (−16 LUFS). Both halves of that equation can move (loudness
presets are configurable now). The honest successor: measure it per job.

`pipeline/quality.py::bed_balance` already computes the dub-vs-source
loudness delta over the longest speech-free window (the /qc "Background
bed" row). Auto-balance closes the loop: after assemble, measure the
delta at gain 1.0 once, derive the gain that puts the bed at the source's
voice:music ratio (voice delta ≈ loudnorm target − source program
loudness), apply it in the merge. `bg_volume` becomes an *offset* on the
computed balance rather than an absolute. Needs: one extra measurement
pass per job (~2 s), a `bg_gain_mode: auto|manual` setting, and the QC
row's PASS band tightening to ±3 LU of computed balance.

Acceptance: on three sources with different music levels, the bed lands
within ±3 LU of computed balance with no per-job knob touching; manual
mode reproduces today's behavior exactly.

---

## 2. Flags/QC glossary checks treat case inflections of a correct rendering as misses

**State: Backlog · Bug · relates CLD-230, CLD-263 (noted in its as-built), CLD-270**

Measured live: glossary stores `Erathia → Эрафия`; the translation uses
the genitive `Эрафии`; both `pipeline/flags.py`'s `glossary_miss`
detector and server.py's `_qc_glossary_row` count it as a miss (QC showed
"0/1 terms honored" on a translation that honored the term). Same family
as the calibration's Кевин/Кевина finding — a creator "fixing" the
inflection produces ungrammatical Russian.

Fix shape: the containment check should accept a shared stem — e.g. match
on the rendering minus its last 1–2 chars for Cyrillic targets (bounded,
word-anchored), or any token whose prefix ≥ (len−2) matches the
rendering. Must NOT regress the true-positive case (Киото vs Кёто) — the
calibration corpus in the CLD-230 history is the test set.

---

## 3. Admin's hash convention still differs from Pro/Creator

**State: Backlog · Improvement · small**

Pro and Creator now share `#/route/param` routing; `static/admin.html`
still uses flat `#overview`-style hashes (admin.html:169, :814). Align it
(`#/overview`) with back-compat parsing of the old shape, or explicitly
document the difference. Descoped from the URL-scheme fix to keep that
change surface small. Also: `/static/*.html` remains an unrouted second
entry point to every page (accepted — the mount serves css/js assets).

---

## 4. serverctl restart raced an in-flight job during the flagship E2E

**State: Backlog · Spike/Bug — needs a repro**

During the first execution of `docs/runbooks/gated-dub-review-approve.md`
an unexplained SIGTERM hit the server ~33 s after `serverctl restart`,
interrupting job a3441eb8 mid-extract ("Interrupted by server shutdown";
two "Server started" feed rows 37 s apart). The crash-resume path
recovered the job cleanly from `download_done` — but the second SIGTERM's
origin was never identified (not in the executor's command stream;
possibly a second human/agent restart, possibly a serverctl stop/start
race). Worth a look at serverctl's pidfile handling under overlapping
restarts before trusting the E2E harness in unattended runs.
