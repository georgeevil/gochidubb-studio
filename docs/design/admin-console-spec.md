# Vendor admin console — design spec

Source: Claude Design project `64a60fdc-6a9e-4868-a1ab-c87966f4696d`,
artboard file `SaaS Admin Console.dc.html` (screens labelled `3a …`).

Companion to [`creator-mode-spec.md`](creator-mode-spec.md), which imported the
`2a` consumer artboards from the same project. Where creator mode is the front
door a customer walks through, this is the back office: *the pages you use to
run GoChiDUBB as a business.*

## What the design encodes

**A separate surface, not a tab.** The design puts the console on its own host
(`admin.gochidubb.com`, staff SSO only) rather than inside a customer
workspace. Four screens, same terminal-agentic visual language as the `1a`
hi-fi:

1. `3a Business overview` — five KPI tiles, revenue by day, a "needs
   attention" list, top accounts by spend.
2. `3a Customer accounts` — searchable/filterable account list with a detail
   pane: lifetime spend, owner, keys, rate tier, budget cap, payment method,
   a dunning timeline, and staff actions.
3. `3a Revenue ops` — live rate card, dunning pipeline, invoice list, credits
   and adjustments, `Export CSV ↓`.
4. `3a Fleet and abuse queue` — queue depth / running / fleet util / failure
   tiles, `Pause new jobs ⏸`, a GPU-pool table showing stage p95s against a
   7-day baseline, and an abuse queue (voice-clone consent, DMCA, usage
   anomaly).

Rail, on every screen: Overview · Accounts · Revenue · Fleet & jobs · Abuse
queue · Feature flags · Staff access.

## Palette and type (dark surface)

Values transcribed from the artboard's inline styles. Token names are the ones
used in `static/admin.css`, so the two files stay greppable against each other.

    --ground   #0b0b0e   page ground
    --frame    #060607   screen frame — recessed below the page
    --card     #0e0e11   card surface
    --raise    #14141a   selected row, inset panel
    --line     #1e1e25   hairline
    --line-2   #16161c   softer hairline, between table rows
    --line-3   #2a2a33   button border
    --ink      #e9e7de   primary text
    --ink-2    #9d9c92   secondary text
    --ink-3    #5e5e67   tertiary text, table headers
    --ink-4    #3a3a42   axis labels, disabled
    --accent   #c8f542   primary action — #0a0a0c text ON accent, never white
    --accent-dim #3a3f22 the bar before the highlighted one
    --warn     #e0a34e   attention
    --bad      #e26a5a   failure, and the `gd` logo ground

Fonts: Geist 400/500/600/700 + JetBrains Mono 400/500/600 — already the app's
faces. Every number is mono and tabular. Radii: 4–6px chips, 8–10px cards,
12px screen frame. Rail is 222px. Artboards are 1360×900.

The design links Google Fonts. `static/admin.css` does not, for the same
reason `creator.css` does not: a product whose pitch is that everything runs on
your machine cannot have its operations console depend on a CDN to paint. Both
faces are used when the machine has them and fall back to the system stacks
when it does not.

## The honesty boundary — read this before changing a number

The design assumes a hosted, multi-tenant deployment. **This repo is a
single-tenant server that runs on one machine and bills nobody.** Every figure
on the console is therefore in exactly one of three categories, and
`app/admin.py` is written so that nothing can drift between them:

* **Real.** Minutes dubbed, job counts and success rate, per-day and per-month
  usage, rate-card band positions, per-stage p95 latency against a 7-day
  baseline, queue depth, GPU utilisation, storage, API keys and their scopes,
  and the audit trail.
* **Estimated, and labelled everywhere it appears.** Every dollar figure. It
  is `app/billing.py` arithmetic — the design's *published* rates applied to
  real usage — and that module's boundary carries over verbatim: the minutes
  are real, the money is an estimate, this server bills nobody. Every admin
  payload carries `estimate: true` and the same `disclaimer` string, and
  `tests/test_admin_routes.py` asserts both.
* **Absent.** Workspaces, invoices, payments, dunning, credits, DMCA intake,
  staff SSO, multi-node GPU pools. Nothing fabricates these. Where the design
  has a panel for one, the console renders the nearest real thing under an
  honest title, or says plainly that the subject needs a system this build
  does not have.

Hard-coding the design's mock figures (`$38.4k MRR`, `412 workspaces`,
`3 pools`) would have produced a screen that matches the artboard pixel for
pixel and tells the operator nothing true. It was not done, and should not be.

## Screen → implementation map

Every route below is new in this change unless marked otherwise. All
aggregation is in `app/admin.py`, which is pure: it takes jobs, key records, a
storage figure, a GPU snapshot and per-stage durations as arguments, including
its clock, so the whole console is testable without a running pipeline.

### 1. `3a Business overview` → `GET /api/admin/overview?window_days=`

| Design element | What it shows here |
|---|---|
| `30d / 90d / 12m` selector | Real. Over 120 days the series buckets by month rather than drawing 365 three-pixel bars. |
| Tile: MRR equiv. | **Revenue (est.)** — `billing.price_minutes` over the window, Δ vs the preceding equal-length window. |
| Tile: Minutes dubbed | Real. `billing.job_minutes` — source length × target languages. |
| Tile: Active workspaces | **Jobs run**, split done/failed. There is one workspace; see §2. |
| Tile: Gross margin | **Realtime factor** — `estimate.realtime_factor`, median wall-clock per second of source. Margin needs a GPU bill this deployment does not receive; throughput is the nearest real efficiency number, and it is the one the wizard already quotes ETAs from. Blank until `estimate.MIN_SAMPLES` jobs have finished. |
| Tile: Job success | Real. `complete / (complete + failed)`. **Cancelled is excluded from both sides** — a user stopping their own job is not the service failing. `None`, not 100%, before anything has finished. |
| Revenue by day | Real, and **priced marginally per bucket**. Tiers are cumulative, so pricing each day from zero would charge every day the first-band rate and the bars would out-total the headline. Each bucket is the difference of the running total before and after it, so they sum exactly. |
| Needs attention | Real detectors only, worst first: failures in 7d (with the most common error), jobs paused for review, jobs active over a day, degraded pipeline stages, storage past the 100 GB allowance, and credential hygiene. Every entry names something a human can go and do. |
| Top accounts by spend | **Top sources by spend**, grouped by `batch_id` — one submitted source per row. Cost is the window total **allocated pro-rata by minutes**: pricing each group standalone would give every one of them the first, most expensive tier and the column would out-total the workspace. |

### 2. `3a Customer accounts` → `GET /api/admin/accounts`, `/api/admin/account/{key_id}`

**This is the screen that maps least directly, and the page says so.** There is
no tenancy anywhere in the codebase — `grep -rn "workspace\|tenant\|owner"`
over `app/`, `server.py` and `pipeline/` finds only `billing.py`'s own
docstring. So:

- One workspace — this machine — gets a row of its own with the window's real
  minutes, spend and job count.
- Everything below it is an **API key** (`app/apikeys.py`), the only other
  caller identity the server can tell apart. Keys are what a vendor actually
  administers here: scopes, environment, age, expiry, revocation.
- Status column is real key lifecycle — `active / idle / expiring / expired /
  revoked` — in place of the design's `active / unpaid / anomaly / suspended`.
- The detail pane's actions are the ones that exist: **Revoke key**
  (`POST /api/apikeys/{id}/revoke`, existing) and the audit trail. The design's
  footnote *"every action here lands in the customer-visible audit log"* is
  literally true — `app/audit.py` records key creation and revocation, and
  `POST /api/admin/intake` records itself.
- **No per-key spend column.** Jobs do not record which credential started
  them, so splitting the workspace's usage between keys would be a guess
  dressed up as a number. `spend_attributable` is `false` and stays false
  until job records carry a key id.

Not implemented, and not faked: budget caps, payment methods, dunning
timelines, impersonation, suspend-writes.

### 3. `3a Revenue ops` → `GET /api/admin/revenue`, `GET /api/admin/revenue.csv`

| Design element | What it shows here |
|---|---|
| Rate card · live | Real, and read-only. `billing.TIERS` with this window's minutes and cost in each band. **Deliberately not user-editable** — the rates are the design's published schedule, and making them settable would let a local install invent a price while the payload still says `estimate: true`. |
| Dunning pipeline | **Unbilled work** — three buckets of minutes that will not become money: in flight, paused for review, failed. Failed jobs are priced at what they *would* have cost, which is the only interesting thing about them; `billing.job_minutes` correctly returns 0 for a failure, so that bucket reads the source duration directly. |
| Invoices · Aug | **By month** — per-month minutes, jobs and estimated cost. Not invoices; this server issues none. |
| Credits & adjustments | Rendered as unavailable, with the reason. A credit needs an invoice to adjust. Drawing the design's amount field would be drawing an input that writes nowhere. |
| `Export CSV ↓` | Real, and the one design action implemented exactly as drawn. One row per job — the grain someone exporting wants to pivot on — with the same pro-rata cost allocation the top-sources table uses. |

### 4. `3a Fleet and abuse queue` → `GET /api/admin/fleet`, `POST /api/admin/intake`

| Design element | What it shows here |
|---|---|
| Queue depth / Running now | Real. `_job_queue.qsize()` and the count of jobs in a running status. |
| Fleet util | Real GPU utilisation from `gpu_snapshot()`, or blank where there is no telemetry. |
| Failures · 24h | Real, with the most common error as the subtitle. |
| `all systems nominal` chip | Derived: pause beats a badly-degraded stage beats a raised failure rate. |
| **`Pause new jobs ⏸`** | **Really pauses them.** An `asyncio.Event` admission gate in `_job_queue_worker`. Checked *before* the dequeue, not after: holding a job the worker has already taken would drop it out of `qsize()` while it was plainly still waiting, so the console's own queue depth would under-report for as long as the pause lasted. A running job is left alone — cancel is the endpoint for that. In-process and non-persistent, so a restart always comes back admitting work. |
| GPU pools table | **Pipeline stages** — per-stage p50/p95 over 24h against the 7 days before, from each job's `metrics.json`. This is the design's "stage p95s vs 7d baseline" with real numbers behind it. A stage with too few runs on either side reports its p95 with no comparison rather than a degradation built on two samples. The walk is TTL-cached because this screen polls. |
| Three pools | **One node**, said out loud. GoChiDUBB runs the pipeline in-process on the machine serving the page: no scheduler, no autoscaler, no second pool to route around. |
| Abuse queue | **Review queue** — three real detector families (below). |

### The review queue's detectors

* **Usage anomaly** — today's metered minutes against the trailing 13-day
  median, at ≥3× and ≥30 minutes so a quiet install's two-job day is not an
  incident. On a hosted deployment this is the shape of a leaked key; locally
  it is still the shape of a script that got away.
* **Voice clones from an uploaded reference** (`voice_mode == "upload"`, 30d).
  Nothing in this build stores a consent attestation alongside an uploaded
  sample, so the item says exactly that: a list to look at, not an accusation.
* **Credential hygiene** — expired-but-live keys, keys expiring within a week,
  and keys older than 180 days with no expiry.

**Not detected, and not reported as clear: DMCA and copyright.** A claim queue
needs somewhere for a claim to arrive, and this server has no such intake.

### Rail entries with no screen

`Feature flags` and `Staff access` are rendered disabled rather than dropped —
the rail is the design's map of the console, and deleting two of seven entries
would hide what is missing instead of showing it. Neither has a system behind
it: there is no rollout mechanism, and there is no authentication.

(Note the collision: "feature flags" in the design means product rollout
toggles. `pipeline/flags.py` is unrelated — it flags uncertain *translations*
for creator mode's review screen.)

## Security

`GET /admin` is served unconditionally, like `/pro` and `/creator`. That adds
no new exposure class: this server has no authentication of any kind and binds
loopback by default for exactly that reason (see the bind block at the bottom
of `server.py`). But this is the surface that most wants staff auth first — it
reads the revenue estimate, every API key record and the audit trail — so when
`mode == "hosted"` the page renders a banner saying it is unauthenticated and
must be firewalled. Give it real staff auth before giving it a public address.

## Files

| Path | Role |
|---|---|
| `app/admin.py` | All aggregation. Pure — no I/O, no server import, clock injectable. |
| `static/admin.html` | Five views, hash-routed. Vanilla JS, no build step, no CDN. |
| `static/admin.css` | Dark palette above. |
| `server.py` | `GET /admin`; `/api/admin/{overview,accounts,account/{id},revenue,revenue.csv,fleet}`; `POST /api/admin/intake`; the `_intake_gate` admission gate in `_job_queue_worker`. |
| `tests/test_admin.py` | Pure-function tests with a pinned clock. |
| `tests/test_admin_routes.py` | Route tests, hermetic in the same way as `test_creator_routes.py`. |

Only the fleet screen polls (10s, and only when the tab is visible). The others
refetch on request: `/api/admin/overview` and `/api/admin/revenue` walk every
output directory to size storage, and putting that on a timer would spin the
disk for numbers that move once a job finishes.
