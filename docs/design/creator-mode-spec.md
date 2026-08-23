# Creator mode — design spec

Source: Claude Design project `64a60fdc-6a9e-4868-a1ab-c87966f4696d`,
artboard file `Non technical Consumer SaaS option.dc.html` (screens labelled `2a …`).

## The recommendation the design encodes

**One product, two front doors — not two apps.** Creator mode and the existing
expert UI (`static/index.html`, "Pro mode") are two surfaces onto the *same*
account, the same jobs table and the same meter. A separate consumer app would
mean two auth stacks, two billing paths and two roadmaps.

- Mode is chosen once ("I make videos" vs "I'm integrating gochidubb") and is
  switchable in Settings. Never a paywall, never a second login.
- Creator mode **hides, it does not dumb down**. Same pipeline underneath; it
  collapses the 8 stages into one progress story, picks sane defaults, and
  prices per video instead of per minute.
- Creator mode is built on the same public `/api/*` endpoints the CLI and MCP
  use. That keeps the API honest.

## Palette and type (light surface — inverse of theme.css)

    --paper       #faf9f5   page ground
    --paper-2     #f2f0e8   recessed ground
    --card        #ffffff   card surface
    --line        #e2e0d6   hairline
    --line-2      #e8e6db   softer hairline
    --line-3      #d9d7cb   button border
    --fill        #f0eee4   neutral chip fill
    --ink         #1a1a1f   primary text
    --ink-2       #77756c   secondary text
    --ink-3       #a3a199   tertiary text
    --accent      #c8f542   primary action (ink text ON accent, never white)
    --accent-ink  #5c7a0f   accent-coloured text/links
    --ok-fill     #e9f5c0   success chip fill
    --ok-ink      #3f5208   success chip text
    --warn-fill   #fdf0d8   attention chip fill
    --warn-ink    #7a4c0f   attention chip text
    --ink-dark    #1a1a1f   dark cards (spend tile, logo mark)

Fonts: Geist 400/500/600/700 + JetBrains Mono 500 — already the app's fonts.
Radii: 6–8px chips, 9–12px cards/buttons, 14px screen frame.
Desktop artboards are 1240×820. Phone artboards are 330×680 inside an 8px frame.

## Screens

### 1. `2a Creator home`

Top bar (62px, white, hairline bottom): `gd` logo mark (28px, #1a1a1f ground,
accent glyph) · wordmark · tabs **My videos / Voices / Plan & usage** (active tab
is a `--fill` pill) · right side: "128 min left this month", primary
**＋ Dub a video** button (accent), avatar circle.

Body:
- **Active job card** (flex 1) — 112×64 dark thumbnail with a play glyph and an
  accent progress sliver; pulsing dot + "Making your Japanese voice…"; subtitle
  naming the video and which languages are already done; 5px progress bar;
  "About 6 minutes left · you can close this page, we'll email you"; a
  **Watch preview** outline button.
- **Spend tile** (250px, dark `#1a1a1f`) — "This month", `$14.80` at 30px,
  "72 minutes dubbed", footer "Pay-as-you-go · no subscription" in accent.
- **Your videos** — 4-up card grid. Each card: 112px thumbnail, title, meta
  ("12:04 · 2 of 3 languages ready"), language chips (done = `--ok-*`,
  pending = `--fill`/`--ink-2`, needs attention = `--warn-*`). Cards needing
  review carry a **Review** accent chip. Last cell is a dashed **＋ Dub a video**
  tile ("paste a link or drop a file"). Filter row: All · Ready · In progress.
- **Auto-dub new uploads** strip pinned to the bottom — "Connect your channel and
  every new video gets your 3 languages automatically. You approve before
  anything publishes." + **Connect channel** outline button.

### 2. `2a Dub a video` (wizard)

Top bar: **✕ Cancel** · centred stepper **1 Video — 2 Languages & voice —
3 Confirm** (completed connectors accent, upcoming grey) · "saved automatically".

Left column (step 2 shown):
- H1 "Which languages?" + "Your voice, speaking each one. Pick as many as you
  like — we'll tell you the price before anything starts."
- "Popular with travel channels" — language chips; selected are accent with ✓,
  unselected are white with a border, plus a `＋ 58 more` chip.
- "Whose voice?" — two radio cards: **My voice** (selected: 2px accent border,
  "ready" chip, "Cloned from your last upload — sounds like you in every
  language.", `▶ Hear it in Spanish`) and **A professional voice** ("Choose from
  40 native-speaker voices…", "Browse voices").
- Toggle row **"Let me check the translation first"** — "We'll pause and show you
  names, brands and jokes before recording. Recommended for your first video."
  Default ON.

Right rail (352px, white): thumbnail, title, "12 min 04 sec · from your channel",
line items (`3 languages × 12 min` → `$2.34`, `Your voice` → included,
`Subtitles (.srt)` → included), **Total today $2.34**, then the reassurance
"Charged when your first language is ready. If something goes wrong, you're not
charged for it.", **Start dubbing →** (accent, full width) and "Ready in about
20 minutes".

### 3. `2a Review before recording`

Top bar: `← Your videos` · job title · **paused for your OK** warn chip · right:
**Skip review** outline + **Looks good — record it →** accent.

Left rail (210px): "LANGUAGES" list — the language under review is a `--fill`
pill with a warn count badge; finished languages show a ✓. Below, a note card:
"Only 2 things to check — We flagged the words we're least sure about.
Everything else is already fine."

Main column: "Two words we'd like you to confirm", then one card per flagged
item:
- kind chip (`a name`, `a joke`) · timestamp · `▶ Hear this bit` on the right
- "You said: …" with the source term highlighted on `#f5f3e8`
- "In Spanish: …" with the translated term highlighted on `--ok-fill`, optionally
  a grey aside ("— we kept the humour, not the words")
- Action row: accent primary (**Keep the name as-is** / **Love it**) plus
  outline alternatives (**Spell it differently**, **Translate literally**,
  **Write my own**) and the note "we'll remember this for future videos"
  (→ writes to the glossary).

Footer card: "**Nothing has been recorded yet.** Once you approve, recording takes
about 40 minutes for all three languages — we'll email you." +
**Looks good — record it →**.

### 4. `2a Mobile web`

Same three verbs, thumb-first. Primary actions pinned to the bottom of the
viewport, ≥48px tall. Home = active-job card, review-needed card, compact list
row, sticky **＋ Dub a video** + "128 min left · $14.80 this month". Wizard = a
3-segment progress bar, wrapped language chips at 12px vertical padding, sticky
price + **Start dubbing →**. Review = one card per flagged term with stacked
full-width choices, sticky "Nothing has been recorded yet" + approve button.

## The eight consumer rules the design applies

1. **No jargon, ever.** "Languages", not `target_langs`. "Your voice", not
   voice-cloning model. No pipeline stage names in the UI.
2. **Price before commitment.** A dollar total on the screen where Start is
   pressed.
3. **One primary action per screen.** Dub → pick languages → confirm. Three
   steps, no branching, resumable.
4. **Review only the uncertain parts.** Not a transcript editor — a handful of
   flagged spans with Keep / Change. Confidence-gating is the whole trick.
5. **Hear it before you trust it.** A "hear it in Spanish" sample on the voice
   card converts better than copy about quality.
6. **Long jobs must be leave-able.** "Close this page, we'll email you."
7. **Failures are free and silent.** "If something goes wrong, you're not
   charged."
8. **Retention lives in automation.** "Auto-dub new uploads" turns a one-off
   purchase into a recurring meter.
