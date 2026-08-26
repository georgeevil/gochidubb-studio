# SaaS Redesign — Design & Implementation Plan

Status: **all seven phases implemented.** See §6; §7 records what was and
was not verified.

Design source: Claude Design project `SaaS Redesign Options.dc.html`
([canvas](https://claude.ai/design/p/64a60fdc-6a9e-4868-a1ab-c87966f4696d?file=SaaS+Redesign+Options.dc.html)),
read via the design MCP. It offers three concepts: **1a agent-first** (hi-fi,
5 full screens), **1b task-first** and **1c developer-first** (wireframes).
`support.js` is the generated dc-runtime that renders the canvas — it carries no
design spec and needs nothing from us.

## Decisions locked

| Question | Decision |
|---|---|
| Concept | **1a agent-first** — the feed is home; rail groups Work / Develop / Workspace |
| Scope of this effort | **UI-first.** Build the shell + screens against the existing local backend. Billing and Members render from a stub layer with clearly-labelled placeholder data; API keys and Webhooks are built for real |
| Local offline mode | **Stays first-class.** `GOCHIDUBB_MODE=local` (default) is byte-for-byte today's experience: no auth, no billing surfaces. `hosted` reveals the Workspace group |

Consequence: no accounts, no payments, no multi-tenancy, and **no new
dependencies** in this effort. `httpx` (already required) covers webhook
delivery. The README promise — local, offline, no per-minute fees — stays true
for the default mode.

## 1. What the design actually specifies

Five screens, each 1360×900, dark, sidebar + top bar + optional right rail:

1. **Agent Feed** (home) — a chronological activity stream. Agent runs appear as
   cards: the natural-language prompt that started them, the MCP tool calls
   underneath (`gochidubb.dub(...) ⇒ job_9f2a4c`), per-stage chips
   (`download ✓ … tts ● … merge ○`), a progress bar, per-language status, and
   cost so far. Below the cards, one-line webhook / audit / billing events.
   Filters: All · Runs · Tool calls · System. Right rail: Live (queue depth, per-GPU
   load, minutes today), Spend MTD, MCP clients connected.
2. **MCP Server** — numbered onboarding (endpoint, `claude mcp add …`, "say what
   you want"), the exposed-tool table, a metering warning, recent tool calls.
3. **Members & Roles** — member table with roles/2FA/last-active, a 7-role ×
   8-permission matrix, pending invites. The `agent` role is explicitly "what an
   MCP key can do — never more than its creator."
4. **API Keys** — create-key form (name, live/test, expiry, scope checkboxes),
   key table with masked tokens and revoke, curl quick-start, rate limits.
5. **Billing & Usage** — minutes/invoice/rate/budget-cap tiles, the metered rate
   table, per-project breakdown, invoices, a warning that agent-initiated jobs
   hit the same meter.

Everything is driven by a ⌘K command bar ("Dub the last upload into fr, es, ja
and stitch a showcase…") present in both the rail and the top bar.

## 2. Visual system — token delta

The design keeps the current family (dark, lime accent, Geist + JetBrains Mono)
but retunes every value and **drops Instrument Serif entirely**:

| Role | Current (`index.html`) | Design 1a |
|---|---|---|
| ground | `#0a0a0d` | `#060607` (darker) |
| panel | `#101014` | `#0e0e11` |
| panel raised | `#16161c` | `#14141a` |
| hairline | `#24242e` | `#1e1e25` |
| border | `#2f2f3c` | `#2a2a33` |
| ink | `#ece9e0` | `#e9e7de` |
| ink-2 | `#a3a096` | `#9d9c92` |
| ink-3 | `#5a5a63` | `#5e5e67` |
| ink-4 | `#35353d` | `#3a3a42` |
| accent | `oklch(0.88 0.18 125)` ≈ `#bdeb58` | `#c8f542` (yellower, more saturated) |
| warn | `oklch(0.78 0.15 60)` | `#e0a34e` |
| display font | Instrument Serif | **dropped** — Geist 600 / JetBrains Mono 600 |

Also: radii 4–12px (currently mostly 6), accent glow shadows on the logo and
live dots (`0 0 16px rgba(200,245,66,.35)`), a 1.4–1.5s `pulse` animation on
anything live, and heavy `font-variant-numeric: tabular-nums` mono for all
numbers. The existing film-grain overlay is absent from the design — drop it.

## 3. Gap analysis

### Already real — the design just re-presents it

| Design element | Backing code |
|---|---|
| Stage chips, retry-on-stage | `GET /api/dub/{id}/stages`, `POST /api/dub/{id}/retry_stage/{stage}` |
| Job cards, per-language status | `GET /api/jobs`, `/api/job/{id}`, `/api/dub/batch/{id}` |
| Showcase / quick-test / redub | `POST /api/showcase`, `/api/quick_test`, `/api/job/{id}/redub` |
| MCP tool table | 23 tools in `tools/gochidubb_mcp.py` |
| Live GPU / queue tiles | `GET /api/system` |
| Minutes source-of-truth | `job["duration"]` (seconds) is already recorded per job |
| Storage line on Billing | `GET /api/storage/stats` |
| System event lines | `GET /api/logs`, `app/logbuf.py` |
| Per-job audit | `GET /api/dub/{id}/audit` |

### Built for real in this effort

| Feature | Approach |
|---|---|
| **API keys** | New `app/apikeys.py` following the proven `app/secrets.py` pattern — keys stored **hashed, outside `config-user.json`** (that file is served by an unauthenticated `GET /api/config` with CORS `*`; the VK-token incident already established this rule). Scope enum matches the design: `dub:write`, `jobs:read`, `outputs:read`, `mcp:invoke`, `webhooks:manage`, `voices:write`. Shown once on create. |
| **Webhooks** | `app/webhooks.py`; fire-and-forget `httpx` POST on `job.completed`, `job.failed`, `job.awaiting_review` from the job runner, with delivery log + manual re-send. |
| **Agent feed** | New `GET /api/activity` merging job transitions, MCP tool calls, webhook deliveries and log events into one reverse-chronological stream. Requires recording MCP tool invocations — currently nothing logs them. |
| **⌘K command bar** | Client-side parser mapping plain language onto existing endpoints. Scoped deliberately narrow: source + target languages + showcase/quick-test flags. Anything it cannot parse confidently pre-fills the New Dub form instead of guessing. |

### Stubbed, clearly labelled

| Feature | Stub behaviour |
|---|---|
| **Billing & usage** | Computes **real** minutes from `job["duration"]` × target languages, prices them with the design's published tiers ($0.080 first 500 min → $0.065 → $0.050 at 2,000+, showcase +$0.020, priority ×1.5), and shows the result as a **cost estimate**, not an invoice. Invoice list and payment method are placeholders behind an "example data" marker. |
| **Members & roles** | Single local `owner` plus the MCP service account. The permission matrix ships as a static reference table — accurate about intent, enforcing nothing in `local` mode. |
| **Audit log** | Real entries for key create/revoke, webhook config, job actions. No cross-user attribution (there are no other users yet). |

### Shipped features the design has no home for — must be placed

The design omits three things that exist and work. Losing them in a redesign
would be a regression, so the plan places them explicitly:

| Feature | Repo surface | Placement |
|---|---|---|
| Translation review (human-in-the-loop) | `awaiting_translation_review`, `/edit_translations`, `/continue` | Job detail state, plus a persistent "N awaiting review" badge on Jobs — it currently has one in `TopBar` and must keep it |
| Publish / VK approval | `/api/publish/*`, `publishPending` inbox | **Library → Publish inbox** (Library is otherwise underspecified in the design) |
| Discover / scout (trending) | `/api/scout/*`, `DiscoverView` | New **Work → Discover** rail item |

Also unplaced by the design but present: voice presets, glossary, quality
panel, beta stage-reuse page. Voices + Glossary go under **Library**; the
quality panel folds into job detail; `beta.html` is re-tokenised in the last phase.

## 4. Mode model

One new `UserConfig` field, `mode: str = "local"` (env `GOCHIDUBB_MODE`), read
through `app.config.cfg` like every other tunable — no scattered `os.getenv`.

- `local` (default): rail shows **Work + Develop**. No auth on any route. No
  Workspace group, no billing surfaces. Identical to today.
- `hosted`: adds the **Workspace** group (Members, Billing, Audit). API-key
  scope checks are enforced on `/api/*`.

`GET /api/system` grows a `mode` field so the UI can branch without a second
round trip. Key enforcement is written from the start but gated on
`mode == "hosted"`, so the local path can never be locked out by a bug in it.

## 5. Information architecture

```
WORK        Agent feed   → NEW (feed of runs/jobs/events)
            New dub      → HomeView (source, langs, voice, mode tabs)
            Jobs         → HistoryView + BatchView + ProcessingView + ReviewView
            Discover     → DiscoverView                    [design-omitted, kept]
            Library      → ResultView + Publish inbox + Voices + Glossary
DEVELOP     API keys     → NEW (real)
            MCP server   → NEW page over the existing 23 MCP tools
            Webhooks     → NEW (real)
WORKSPACE   Members      → NEW (stub)          [hosted mode only]
            Billing      → NEW (metered estimate from real durations)
            Audit log    → NEW (real entries, no multi-user attribution)
            Settings     → SystemView (setup + logs tabs)
```

The ten current views map onto ten rail destinations; no view's logic is
discarded.

## 6. Implementation phases

Each phase ends with a working app and a screenshot pass.

**Phase 1 — extract the style layer. ✅ done.** Moved the `<style>` block
(index.html:10–197) into `static/theme.css` behind a `<link>`. Mechanical,
zero visual change (extracted rules verified byte-identical). One correction
to the original claim: FastAPI served only `/` and `/beta` as individual
routes — `static/` had no URL — so `server.py` gained a one-line
`StaticFiles` mount at `/static`.

**Phase 2 — retune tokens. ✅ done.** Applied the §2 table, dropped Instrument
Serif (the `.serif` class stays, redefined as Geist 600 display so its ~18
call sites need no edits) and the grain overlay, added glow/pulse/tabular-nums
primitives and radius tokens. Swept the hard-coded `#0a0a0d` bypasses in
`.btn-primary` and three components onto `var(--bg)`.

**Phase 3 — shell. ✅ done.** `LeftRail` rebuilt into the grouped
Work/Develop/Workspace rail (Workspace revealed only in hosted mode; Settings
always visible); `TopBar` gained the ⌘K command field (submits to New dub —
the parser is Phase 7's) and the mode pill. Jobs and Library are now tabbed
wrappers over the untouched History/Processing/Review/Batch and
Result/Voices/Glossary views, so every jump target still works; the
awaiting-review badge and running-job indicator moved onto Jobs. Rail items
whose screens land later (feed, Develop, Workspace) render honest
"in development · phase N" placeholders. Backend: `UserConfig.mode`
(`GOCHIDUBB_MODE`) and `mode` in `GET /api/system`.

**Phase 4 — Agent feed + activity API. ✅ done.** `app/activity.py` (bounded,
redacted ring buffer), agent attribution via an `X-GoChiDUBB-Client` header
that `GoChiDUBBClient` now sends, job-transition recording, and
`GET /api/activity` paging on a monotonic event id. The feed renders run
cards, job cards with stage chips, and system one-liners, with the design's
filter tabs and live side tiles; it is now the landing view.

Two elements of the design were deliberately **not** built, because the data
does not exist: the natural-language prompt behind a run (agents send tool
calls, not the sentence that produced them — it would have to be passed
explicitly by the client) and per-job cost, which waits for the phase-6 meter.
The right rail therefore carries Live and MCP tiles but no Spend tile yet.

Note also that phases 1–4 were implemented twice, in two parallel sessions.
The reconciliation kept this branch's phases 1–3 and re-applied phase 4 on
top; the alternative shell survives only on the local branch
`claude/saas-phase4-activity-feed`.

**Phase 5 — Develop group (real). ✅ done.** `app/apikeys.py`, `app/webhooks.py`, their
routes, the API Keys and MCP Server and Webhooks screens. Scope enforcement
written, gated to hosted.

**Phase 6 — Workspace group. ✅ done.** Billing from real durations + design
tiers; Members matrix; Audit log. Every placeholder visibly marked.

**Phase 7 — re-home the omitted features and polish. ✅ done.** Discover into Work;
Publish inbox / Voices / Glossary into Library; review flow into job detail;
`beta.html` re-tokenised; ⌘K parser; narrow-width and focus-state pass across
all screens; delete dead CSS.

## 7. Verification

Automated gates give us nothing here, and it is worth being precise about why:
`.github/workflows/lint.yml` triggers on `main`, but the default branch is
`master`, so **it never runs** — only `claude-review` fires on a PR. And
`ruff check .` reports **105 pre-existing findings** on master (ruff 0.15.21),
so a clean full run is not a reachable baseline; compare per-file before/after
instead. Real verification is therefore manual:

- `python tools/gochidubb_serverctl.py foreground --reload`, then drive the UI in
  Chrome and screenshot every rail destination at desktop **and** narrow widths.
  Every screen renders from polling data, so no dub needs to run to review layout.
- New pure functions get pytest coverage where the suite already has traction:
  the metering tier maths, the ⌘K parser, and activity-stream merge/ordering.
- Both modes checked each phase: `GOCHIDUBB_MODE=local` must look and behave
  exactly as it does today.

## 8. Risks

- **One 3,710-line file compiled by in-browser Babel.** A syntax slip white-screens
  the app with only a console error. Mitigation: phase-sized commits, Phase 1
  strictly mechanical, screenshots after each.
- **Feed needs data nothing currently records.** MCP tool calls are invisible to
  the server today; if that recording proves invasive, the feed degrades to
  job-transition events only and still reads correctly.
- **Metering is an estimate, and could be mistaken for a bill.** Every money
  figure carries an "estimate · local mode" marker; nothing implies a charge.
- **Auth gating could lock out the local path.** Enforcement is gated on
  `mode == "hosted"` and the local path is re-verified every phase.
- **Design omissions are the real regression risk** — see §3; they are placed
  deliberately rather than discovered missing at the end.

## 9. Out of scope (and what it would take)

Real accounts + sessions + 2FA, workspace tenancy on every job and output,
enforced RBAC, Stripe metered billing, hosted streamable-HTTP MCP at
`mcp.gochidubb.com`, a public `api.gochidubb.com/v1` with rate limits, and a
multi-GPU worker pool. That is the "Full hosted SaaS" path: weeks of backend
work, several new dependencies (payments, auth), and infrastructure decisions —
and it would need `CONTRIBUTING.md`'s no-new-deps rule discussed first.

Open, not blocking: the design's header note says "All creator references
removed — brand is gochidubb only", but `README.md` currently credits
@smolekoma and @smolemaru. Whether to strip those is a call for you, not a
side effect of a UI redesign — flagged, untouched.


## 10. What shipped, and what is still open

Phases 1–7 are implemented and verified in Chrome against the running server.
Highlights beyond the phase notes above:

* **Billing is honest by construction.** Minutes are measured from real jobs;
  the money is an estimate at the design's published rates, labelled as such on
  screen and in a `disclaimer` field on the endpoint. This server bills nobody.
* **The audit log is not the activity feed.** Activity is a ring buffer that may
  drop entries; the audit trail is append-only JSONL, fsync'd per entry.
* **⌘K never starts a job.** It parses a source, target languages and a run
  mode, then pre-fills New dub for a human to press Start — a dub costs real
  GPU minutes. It reports what it failed to understand, and leaves the form
  untouched when it understood nothing.
* **Local mode is unchanged.** No auth, no billing surfaces, no Workspace
  group — re-verified after every phase.

The §10 leftovers closed on 2026-08-25 (CLD-249):

* **Scope enforcement is now exercised.** `_enforce_api_scopes` in server.py
  rejects bad keys in hosted mode over a closed route→scope table
  (`_scope_for`), with route-level tests for valid / wrong-scope / expired /
  revoked keys in `tests/test_scope_enforcement.py`. Local mode short-circuits
  before any check, and loopback callers are exempt even in hosted mode so the
  operator (and the session-less browser UI) can never be locked out. A
  genuinely hosted deployment still needs everything in §9 — this closes the
  key surface, not the account story.
* **The natural-language prompt behind a run is real.** The MCP submission
  tools take an optional `prompt`, `GoChiDUBBClient` sends it percent-encoded
  in `X-GoChiDUBB-Prompt`, the agent-attribution middleware records it
  (truncated to 300 chars, redacted like everything else), and the feed's run
  card quotes it. Runs without one simply have no quote — nothing is invented.
* **Per-job cost is in the feed.** `/api/activity` carries a month-to-date
  `spend` block (summarize over the jobs dict — pure arithmetic, safe on the
  feed's 2s poll, unlike /api/billing/usage's storage walk); the right rail
  shows the Spend·MTD tile and job cards price their minutes at the current
  rate. Every figure keeps the "est. / bills nobody" honesty markers.
* **Narrow width had its pass** on a real ~700px viewport via devtools-driven
  resize; what broke was fixed in the same change.

The alternative phase 1–3 shell's branch (`claude/saas-phase4-activity-feed`)
is already deleted — nothing from this plan remains open.
