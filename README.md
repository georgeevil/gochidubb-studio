<div align="center">

# 🎙️ GoChiDUBB Studio

**Local, agent-controllable AI video dubbing.**
YouTube link in → voice-cloned dub in 65 languages out. No cloud, no per-minute fees, no upload of your face to anyone's server.

*by [George Chigrichenko](https://x.com/GChigrichenko) ([@georgeevil](https://github.com/georgeevil)) &mdash; built with [Claude Code](https://claude.ai)*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
optional [![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0+-76B900.svg)](https://developer.nvidia.com/cuda-downloads)
[![MCP enabled](https://img.shields.io/badge/MCP-enabled-7B61FF.svg)](https://modelcontextprotocol.io)
[![GitHub stars](https://img.shields.io/github/stars/georgeevil/gochidubb?style=social)](https://github.com/georgeevil/gochidubb/stargazers)

[**Quickstart**](#-30-second-quickstart) ·
[**Demo**](#-demo) ·
[**MCP / Agent use**](#-agent-control-mcp--cli) ·
[**Languages**](#-supported-languages) ·
[**FAQ**](#-faq) ·
[**Troubleshooting**](#-troubleshooting)

![demo](docs/demo.gif)

</div>

---

## ✨ Why GoChiDUBB

| | GoChiDUBB | ElevenLabs Dubbing | Heygen | Rask |
|---|---|---|---|---|
| **Cost** | Free (your GPU) | $0.30/min and up | $0.15+/min | $0.07+/min |
| **Runs offline** | ✅ 100% local | ❌ cloud | ❌ cloud | ❌ cloud |
| **Voice cloning** | ✅ VoxCPM2 | ✅ | ✅ | ✅ |
| **Languages** | 65 | 29 | 40+ | 130+ |
| **Multi-speaker diarization** | ✅ (pyannote) | ✅ | ✅ | ✅ |
| **Background music preservation** | ✅ (Demucs) | ✅ | ✅ | ✅ |
| **YouTube URL → MP4** | ✅ in one step | ❌ | ❌ | ❌ |
| **Stitched multilingual reel** | ✅ built-in | ❌ | ❌ | ❌ |
| **MCP / agent control** | ✅ first-class | ❌ | ❌ | ❌ |
| **Open source** | ✅ MIT | ❌ | ❌ | ❌ |
| **No upload of your data** | ✅ | ❌ | ❌ | ❌ |
| **API key required** | ❌ none | ✅ paid | ✅ paid | ✅ paid |

If you're dubbing a 10-minute video weekly across 5 languages, this saves you about **$1,800/year** vs cloud tools — and the dub never leaves your machine.

---

## 🚀 30-second quickstart

### Windows (one click)

```text
1. Clone or unzip the repo
2. Double-click install.bat   ← installs everything (~5-10 min)
3. Double-click start.bat     ← browser opens at http://localhost:8910
4. Paste YouTube URL → pick language → Start
```

### Linux / macOS

```bash
git clone https://github.com/georgeevil/gochidubb && cd gochidubb
chmod +x install.sh
./install.sh    # installs everything + creates start.sh
./start.sh
```

First dubbing run downloads the VoxCPM2 model (~5 GB) — one time.

### Server management

The server ships with a process manager that prevents lingering processes after agentic code changes:

```bash
# Start in background (detached, survives terminal close)
python tools/gochidubb_serverctl.py start

# Check if running
python tools/gochidubb_serverctl.py status

# Graceful stop (SIGTERM, auto-force-kills after 10s)
python tools/gochidubb_serverctl.py stop

# Restart (stop + start)
python tools/gochidubb_serverctl.py restart

# Tail server logs
python tools/gochidubb_serverctl.py logs --follow

# Development mode with auto-reload on file changes
python tools/gochidubb_serverctl.py foreground --reload

# Auto-start on macOS login
python tools/gochidubb_serverctl.py install-launchd
```

The manager writes a PID file (`.gochidubb.pid`) on startup and cleans it up on shutdown. If a process lingers after a crash or agentic restart, `status` detects orphaned processes and `stop` kills them all.

---

## 🤖 Agent control (MCP + CLI)

This is what makes GoChiDUBB different. You don't have to touch the UI to use it.

### Tell Claude Code (or any MCP-aware agent) what you want

```text
You:    Dub https://youtu.be/abc into French, Spanish and Japanese,
        then stitch them into one 60-second showcase reel.

Claude: [calls gochidubb_showcase(...)]
        [polls gochidubb_get_showcase(...)]
        Done — http://localhost:8910/outputs/showcase_sc_2f1a.../showcase.mp4
```

Add the MCP server in 10 seconds:

```bash
claude mcp add gochidubb python /path/to/gochidubb/tools/gochidubb_mcp.py
```

Or paste into `~/.claude.json`:

```json
{
  "mcpServers": {
    "gochidubb": {
      "command": "/path/to/gochidubb/venv/Scripts/python.exe",
      "args": ["/path/to/gochidubb/tools/gochidubb_mcp.py"],
      "env": { "GOCHIDUBB_URL": "http://localhost:8910" }
    }
  }
}
```

The MCP server exposes **32 tools**: dub / compare / showcase / redub, job status and listing, voice casting (get / set / preview), continue / cancel / rescue / delete, per-stage retry, review flags, translation edits, glossary terms, quality report, artifact audit, publish workflow (stage / approve / cancel / inbox), trending scout, duplicate check, and system / languages / models / voices.

The repo ships a Claude Code skill at [`.claude/skills/gochidubb/SKILL.md`](.claude/skills/gochidubb/SKILL.md). Copy it to `~/.claude/skills/` and Claude knows when and how to drive the pipeline.

### CLI — works from any shell, any OS, any cron

```bash
# Single language, blocking
python tools/gochidubb_cli.py dub https://youtu.be/abc --lang fr --wait

# Compare 5 languages side-by-side
python tools/gochidubb_cli.py compare ./clip.mp4 --langs es,fr,de,ja,pt --trim 60

# Stitched multilingual showcase reel
python tools/gochidubb_cli.py showcase https://youtu.be/abc \
  --langs es,fr,de,ja,pt --trim 60 --wait

# Re-dub an existing job into new languages — skips re-upload
python tools/gochidubb_cli.py redub 5038e404 --langs ja,it --mode showcase --wait

# Health, status, history
python tools/gochidubb_cli.py system
python tools/gochidubb_cli.py jobs --limit 20
python tools/gochidubb_cli.py status <job_id>

# Cast a voice per speaker (with audition), resume a job parked at a
# review gate, or hand a stuck download a manually-fetched file
python tools/gochidubb_cli.py cast <job_id> --set SPEAKER_00=anna --preview
python tools/gochidubb_cli.py continue <job_id>
python tools/gochidubb_cli.py rescue <job_id> ./video.mp4
```

The full subcommand list: `dub, compare, showcase, redub, status, jobs, wait, showcase-status, showcase-rebuild, cast, continue, retry-stage, flags, edit-translations, glossary-term, quality, audit, publish, approve, publish-cancel, publish-inbox, scout, scout-dub, check-dup, cancel, rescue, delete, system, languages, models, voices`.

Drive a remote box: `set GOCHIDUBB_URL=http://192.168.0.10:8910`

See [`examples/`](examples/) for ready-to-run scripts.

---

## 🎛️ Five front doors

One server, five surfaces — pick the one that fits the person in front of it:

- **`/pro`** (also the default `/`) — the review workbench: tabbed review of subtitles, sync fit, cost, pronunciation, speakers and consent, with review gates that park a job before the expensive stages.
- **`/creator`** — Creator mode: a guided, non-technical flow with confidence flags, a glossary, and a cost estimate up front.
- **`/go`** — GoChiDUBB Go, the phone surface. Same jobs, a screen that fits a hand.
- **`/admin`** — the vendor admin console: API keys, audit trail, usage and revenue estimates.
- **`/beta`** — the stage-reuse cache inspector (see [Stage reuse](docs/stage-reuse.md)).

Around them, the pieces that recent releases added:

- **Voice library** — upload, record, edit and delete reference voices, or *Design a voice* from a style description (drawn once, then cloned, so it stays one person for the whole job).
- **Voice casting** — assign a voice per detected speaker (`speaker_voice_map`) and audition it *before* the hours-long synthesis stage.
- **Quality gates** — a suspect transcription or translation parks the job in an awaiting-review state before any GPU time is spent; resume from the UI, `continue` on the CLI, or `gochidubb_continue_job` over MCP, and subscribe to the `job.awaiting_review` webhook to be told when. Disable with `GOCHIDUBB_QUALITY_GATE=0`.
- **Glossary, Transcript tab, model manager** — pin terminology across jobs, read the full transcript in place, and install/remove translation models from the UI.
- **Background bed control** — the separated music bed has a per-job 0–10 gain with a QC row that measures the dub-vs-source balance; "the music is too quiet" is fixable on a finished dub in seconds (the merge stage is a pure re-mix).
- **Subtitle burn-in** — `POST /api/dub/{id}/burn_subs` renders the SRT into the video with styling presets.
- **Publish workflow** — stage an upload, review it in an inbox, approve or cancel; nothing is published without a human saying so.

---

## 🎬 Demo

| What | Length | Languages | Time on RTX 3080 Ti |
|---|---|---|---|
| Single-speaker YouTube short → French | 60 s | 1 | ~2 min |
| Compare 5 languages | 60 s × 5 | 5 | ~10-15 min |
| Showcase reel (stitched) | 60 s | 5 | ~12-18 min |
| Multi-speaker podcast (diarized) | 5 min | 1 | ~8-10 min |

> 📺 [Watch the full demo](docs/demo.mp4) (no audio, ~2 min) — submit a YouTube URL, pick 5 languages, get a stitched showcase reel.

---

## 🏗️ How it works

```
YouTube URL or local file
        │
        ▼
   yt-dlp ───────────────────────► (downloads source)
        │
        ▼
   FFmpeg ───────────────────────► (extracts audio)
        │
        ▼
  faster-whisper ───────────────► (transcript + word timestamps)
        │
        ▼
   pyannote ─────────────────────► (speaker diarization, optional)
        │
        ▼
   LM Studio / Ollama (local LLM) ► (translation, length-matched)
        │
        ▼
   VoxCPM2 ──────────────────────► (voice cloning per speaker, 48 kHz)
        │
        ▼
   FFmpeg ───────────────────────► (time-align, mix bg music, render)
        │
        ▼
   Dubbed MP4 + SRT subtitles
```

Every step is modular, swappable, and runs on your hardware.

### Stages are checkpointed and independently retryable

The pipeline runs as eight discrete stages, and each one snapshots its full
state to `outputs/<job>/checkpoint_<stage>.json` when it finishes:

| Stage | Produces | Checkpoint |
|---|---|---|
| `download` | source video | `download_done` |
| `extract` | 16 kHz audio (denoise, VAD, background split) | `extract_done` |
| `transcribe` | segments + word timings | `transcribe_done` |
| `diarize` | speaker labels + per-speaker voice refs | `transcription_done` |
| `translate` | translated segments + `subtitles.srt` | `translation_done` |
| `tts` | one cloned-voice wav per segment | `tts_done` |
| `assemble` | time-aligned, loudness-normalized dub track | `assemble_done` |
| `merge` | final MP4 | `merge_done` |

Because a stage only needs the *previous* stage's checkpoint, any stage can
be re-run on its own — nothing before it is recomputed:

```bash
# Diarization failed (no HF_TOKEN?) — redo it without speaker detection.
# Transcription is NOT redone.
curl -X POST localhost:8910/api/dub/$JOB/retry_stage/diarize \
     -F 'overrides={"skip_diarization": true}'

# Translation came back half-empty — finish it with a different model,
# keeping the segments that already translated cleanly.
curl -X POST localhost:8910/api/dub/$JOB/retry_stage/translate \
     -F 'overrides={"model": "qwen2.5:14b", "translate_failed_only": true}'

# Re-synthesize voices only, and stop before rendering the video.
curl -X POST localhost:8910/api/dub/$JOB/retry_stage/tts \
     -F 'overrides={"voice_preset": "zhirik"}' -F 'stop_after=tts'
```

The **Pipeline stages** panel in the UI (Processing, Result, and History
views) shows the same thing: per-stage state, the artifacts each produced,
and a Retry control with the settings that stage accepts. Stages whose
output predates a later re-run of an earlier stage are flagged `stale`.

### Where the time went

Every stage is timed and resource-sampled while it runs. Logs carry a
`[perf]` line per stage:

```
[perf] ▶ stage='transcribe' job=1fd8736a started
[perf] ■ stage='transcribe' job=1fd8736a status=ok took=10.66s ·
       cpu 22%avg/76%peak · rss 777MB · gpu 88%avg/99%peak · vram 6100MB/12288MB ·
       segments=13 · whisper_model=tiny · realtime_x=10.55
```

The same data is persisted to `outputs/<job>/metrics.json` (latest run per
stage plus a capped history of every attempt) and served by:

- `GET /api/dub/{id}/stages` — state, artifacts, timings, resource usage
- `GET /api/dub/{id}/metrics` — raw per-attempt history
- `GET /api/system` → `resources` — live CPU/RAM/GPU snapshot

GPU telemetry works through `pynvml` → `torch.cuda` → `nvidia-smi` →
`torch.mps`, whichever is available; with none of them, stages are still
timed and CPU/RAM sampled.

### Auditing a finished job for lost content

`tools/audit_job.py` walks a job's checkpoints and artifacts and reports
anything that went missing between stages:

```bash
python tools/audit_job.py 947ca81d      # one job
python tools/audit_job.py --all          # every job in outputs/
python tools/audit_job.py 947ca81d --json
```

```
stage         segments  unique idx    words
transcribe         190         190     3439
diarize            152         150     3439  <-- DUPLICATE IDX
translate          150         150     3421

✗ LOSS  [diarize] 2 duplicate idx — downstream keys segments by idx, so 2
        segment(s) will be silently overwritten
        idx=188:
            'could you sign this book?'
            'These four players decided to keep my return a secret'
```

Counting segments per stage is not enough on its own: the pipeline
*legitimately* merges sentence fragments and splits over-long ones, so
190 → 152 is healthy while 152 → 150 is two lines of dialogue gone. The
primary check is therefore **word coverage** — every word transcribed from the
source must still be accounted for in the segments that reached the render.
Merges and splits preserve words; dropped segments do not.

It also flags segments that were never translated, never synthesized, whose
wav is missing, or that TTS produced but the assembler never placed — plus
leftover `seg_*.wav` from earlier attempts, which is the most confusing thing
to find when checking a job directory by hand. Exit code is 1 when anything
was lost, so it works as a post-run assertion or a CI check.

### Setup problems come to you

A stage can finish successfully and still not have done what you assume. The
motivating case: `pyannote/speaker-diarization-3.1` fails to download, the
loader falls back to `speaker-diarization-community-1`, the run completes
normally — and your multi-speaker video is diarized by a weaker model with no
sign anything happened. Nothing in the API said so, and the only hint was a
line on a terminal nobody reads.

So stages now emit **notices**: `{code, severity, title, detail, remediation,
url}`, where `code` is a stable slug (`pyannote.fallback_model`,
`ffmpeg.missing`, `tts_qa.device_unavailable`). They surface in four places —
a banner under the top bar, a chip in the top bar and left rail, a **⚠ degraded**
marker on the affected stage row, and **System → Setup**.

```bash
# Passive: no network, rides the poll the UI already makes
curl -s localhost:8910/api/system | jq '.notices, .checks, .accelerator'

# Deep: validates the HF token, asks the Hub whether the gated pyannote repos
# are really accessible, pings the translation backend. On demand only.
curl -X POST localhost:8910/api/diagnostics/run | jq '.notices'

# Server log ring, including third-party stdout/stderr, secrets scrubbed
curl -s 'localhost:8910/api/logs?level=WARNING&limit=100' | jq '.entries'
```

A confirmed deep check retires findings it disproves — if the Hub says your
access is fine, a past download failure was transient and its warning clears
instead of nagging forever.

---

## 🌍 Supported languages

65 target languages out of the box (via VoxCPM2 + edge-tts fallback):

| Code | Language |     | Code | Language |     | Code | Language |     | Code | Language |
|---|---|---|---|---|---|---|---|---|---|---|
| `en` | English | | `ru` | Russian | | `es` | Spanish | | `fr` | French |
| `de` | German | | `it` | Italian | | `pt` | Portuguese | | `pl` | Polish |
| `tr` | Turkish | | `ja` | Japanese | | `ko` | Korean | | `zh` | Chinese |
| `ar` | Arabic | | `hi` | Hindi | | `nl` | Dutch | | `uk` | Ukrainian |
| `sv` | Swedish | | `th` | Thai | | `vi` | Vietnamese | | `cs` | Czech |
| `ro` | Romanian | | `hu` | Hungarian | | `bg` | Bulgarian | | `el` | Greek |
| `fi` | Finnish | | `id` | Indonesian | | `no` | Norwegian | | `da` | Danish |
| `bn` | Bengali | | `ur` | Urdu | | `fa` | Persian | | `he` | Hebrew |
| `sw` | Swahili | | `tl` | Filipino | | `ms` | Malay | | `ta` | Tamil |
| `te` | Telugu | | `mr` | Marathi | | `gu` | Gujarati | | `kn` | Kannada |
| `ml` | Malayalam | | `sk` | Slovak | | `hr` | Croatian | | `sr` | Serbian |
| `sl` | Slovenian | | `lt` | Lithuanian | | `lv` | Latvian | | `et` | Estonian |
| `ca` | Catalan | | `is` | Icelandic | | `af` | Afrikaans | | `mk` | Macedonian |
| `sq` | Albanian | | `bs` | Bosnian | | `cy` | Welsh | | `kk` | Kazakh |
| `az` | Azerbaijani | | `uz` | Uzbek | | `ka` | Georgian | | `mn` | Mongolian |
| `ne` | Nepali | | `si` | Sinhala | | `my` | Burmese | | `km` | Khmer |
| `lo` | Lao | | | | | | | |

**Voice cloning** (VoxCPM2) covers the languages in its training set — most of
the list, including `he`, `sw`, `tl`, `ms`, `my`, `km`, `lo`. Languages outside
that set (`uk`, `cs`, `ro`, `hu`, `bg`, `bn`, `ur`, `fa`, the Indian languages,
the rest of Europe and Central Asia) are synthesized with Microsoft edge-tts
neural voices: no cloning, but clean and intelligible.

Source detection is automatic (Whisper). Translation goes through your local LLM server — LM Studio by default (`openai/gpt-oss-20b` recommended; see [Choosing a translation model](docs/choosing-a-translation-model.md)), or Ollama.

---

## 🖥️ Hardware

| | Minimum | Recommended | Why |
|---|---|---|---|
| **VRAM** | 8 GB | 12 GB+ | VoxCPM2 + Whisper + a translation LLM coexist |
| **RAM** | 16 GB | 32 GB | Vocal separation (Demucs, background preservation) is hungry |
| **Disk** | 20 GB | 40 GB+ | Models + outputs |
| **GPU** | Any CUDA 12.0+ | RTX 30/40 series | CPU fallback works but ~15× slower |
| **Python** | 3.10–3.14 | 3.11–3.12 | The <3.15 ceiling comes from `spaces` |
| **OS** | Win 10+, Linux, macOS | — | Apple Silicon runs on MPS (voxcpm ≥ 2.0.3); Intel Mac is CPU-only |

No GPU? It still runs — just expect long jobs. The pipeline auto-falls back to `edge-tts` (Microsoft cloud TTS) if VoxCPM2 won't load, which sacrifices voice cloning but produces intelligible output fast.

### Disk budget (what gets downloaded)

| Component | Size | When |
|---|---|---|
| Python deps (PyTorch + transformers + faster-whisper + ...) | ~4 GB | At `install.bat` / `./install.sh` |
| FFmpeg + yt-dlp (Windows static build) | ~100 MB | At install |
| VoxCPM2 model weights | ~5 GB | First dubbing run, cached forever |
| Whisper `large-v3` weights | ~3 GB | First dubbing run, cached forever |
| Translation model — LM Studio or Ollama (e.g. `gpt-oss-20b`) | ~5 GB | At install (you pick it) |
| pyannote diarization weights (optional) | ~500 MB | First multi-speaker run |
| Demucs `htdemucs_ft` weights (or audio-separator UVR) (optional) | ~250 MB | First background-preserve run |

**Total for full setup: ~18 GB.** Skinny single-language setup without diarization or BGM preservation: ~12 GB.

---

## 🔑 Tokens & API keys

**Required tokens: NONE.** The default install runs 100% offline once dependencies are downloaded. No OpenAI / ElevenLabs / Anthropic key needed — translation is local (LM Studio or Ollama, both on your machine), TTS is local (VoxCPM2), ASR is local (Whisper).

| Token | Required? | What for | Where to get |
|---|---|---|---|
| Hugging Face token (`HF_TOKEN`) | Only for multi-speaker diarization | Downloading pyannote diarization weights — gated by free terms-of-use acceptance | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — also accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| YouTube cookies (`YT_DLP_COOKIES_FROM_BROWSER`) | Only for age-restricted / member-only YouTube videos | yt-dlp downloads & metadata probes via your existing browser session | Set to a browser name — `firefox` is the most reliable on macOS (Chrome/Safari keychains often block access); `chrome`, `edge`, `safari` also work. Alternative: `YT_DLP_COOKIEFILE=/path/to/cookies.txt` (Netscape format). Both are also editable as `ytdlp_cookies_from_browser` / `ytdlp_cookiefile` in Settings. |
| OpenAI / ElevenLabs / Anthropic keys | **Never.** | — | — |

What "phones home" by default:
- `yt-dlp` reaches YouTube/Vimeo/etc. — only when you submit a URL
- `huggingface.co` for model downloads — first run only, then cached
- `lmstudio.ai` / `ollama.com` for translation model pulls — first install only
- `edge-tts` for the cloud TTS fallback — only triggers if VoxCPM2 fails to load on your GPU

There's no telemetry, no analytics, no phone-home from GoChiDUBB itself. Audit the network calls: search the repo for `httpx.` / `requests.` — only the integrations above.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
# Speaker diarization (multi-speaker videos)
# Also settable from System → Setup, which applies it without a restart.
HF_TOKEN=hf_xxxxx                  # from huggingface.co/settings/tokens

# TTS model selection
VOXCPM_MODEL=openbmb/VoxCPM2       # or openbmb/VoxCPM1.5 (lighter)
VOXCPM_CFG=2.0                     # 1.5-3.0, higher = closer to reference voice
VOXCPM_STEPS=10                    # 5-20, lower = faster

# Translation backend
OLLAMA_URL=http://localhost:11434  # used when USE_LM_STUDIO=0

# LM Studio (default translation backend)
USE_LM_STUDIO=1
LM_STUDIO_URL=http://localhost:1234   # with or without /v1 — both work
LM_STUDIO_MODEL=                      # empty = auto-pick an installed model
LM_STUDIO_TIMEOUT=300                 # seconds per segment (cold model load is slow)
LM_STUDIO_MAX_OUTPUT_TOKENS=4096      # must exceed the model's reasoning budget
LM_STUDIO_REASONING=off               # off|low|medium|high|on — off is much faster
LM_STUDIO_MAX_CONCURRENT=1            # LM Studio serves one model at a time

# Timing
GOCHIDUBB_TTS_MAX_STRETCH=1.4      # hardest time-compression allowed to hold sync

# UI behavior
GOCHIDUBB_OPEN_BROWSER=1           # 0 to disable auto-open
GOCHIDUBB_QA_THRESHOLD=0.4         # stricter (lower) = more re-rolls on bad TTS
```

### Which translation model to use

This matters more than any other setting. A bad translation model does not
fail — it returns fluent, confident, wrong text, and you find out by watching
the finished video in a language you may not speak. One model in our test set
rendered Spanish *gracias* as Bulgarian "please" every time it appeared, and
*Adiós* as a word that does not exist, while every stage reported success.

**[→ Choosing a translation model](docs/choosing-a-translation-model.md)** — the
evidence, the failure modes to watch for, and `tools/gochidubb_benchmark.py` to
re-run the comparison against whatever you have installed.

Short answer: `LM_STUDIO_MODEL=openai/gpt-oss-20b`.

### Reusing work across runs (beta)

Re-dubbing into a second language recomputes the download, audio extraction,
transcript and diarization — 60% of pipeline time — even though none of it
depends on the target language. Stage reuse keys each stage on its inputs
instead of on the job that ran it, so a second job can copy the first's work.

```bash
GOCHIDUBB_REUSE=1                  # off by default
GOCHIDUBB_REUSE_STAGES=download,extract,transcribe,diarize
```

A separate page at `/beta` shows what is cached, what is being reused, and why
anything is being refused. **[→ Stage reuse](docs/stage-reuse.md)**

### Keeping the dub in sync

Translations are rarely the same length as their source, and speech has to fit
the slot it is dubbed into. The assembler time-compresses each segment (pitch
preserved) up to `GOCHIDUBB_TTS_MAX_STRETCH`, budgeting against the next
segment's start so the pause after someone stops talking gets used before
anything is sped up.

Raise it toward `1.6` for fast-talking sources or dense target languages; lower
it toward `1.2` if the dub sounds rushed. Past the ceiling a segment is allowed
to run long, and the segment after it absorbs the delay rather than passing it
on. The job's metrics record `max_drift_sec` so you can see whether the ceiling
is binding.

### Thinking models and translation speed

Reasoning models (`qwen3.x`, `qwq`, `deepseek-r1`, `gpt-oss`) spend most of
their output budget thinking before writing a word of the answer. Measured on
`qwen/qwen3.6-27b`, one 8-word sentence produced **1606 reasoning tokens and 14
tokens of translation**. Two consequences:

- `LM_STUDIO_MAX_OUTPUT_TOKENS` must comfortably exceed the reasoning budget.
  If it doesn't, the response gets truncated mid-thought and comes back with no
  answer at all — which shows up as segments that "failed to translate" with no
  other symptom.
- `LM_STUDIO_REASONING=off` is requested by default. Some models honour it;
  `qwen3.6-27b` accepts the field and reasons anyway, so for Qwen models we also
  append Qwen's in-prompt `/no_think` switch, which does work — the same four
  segments went from **171.7s to 8.6s**. When a model ignores the setting
  entirely you'll get one warning in the log; a non-thinking model
  (`gemma`, `qwen2.5`) is the better choice for bulk translation.

If translation fails wholesale, the log line to look for is LM Studio's
`Unexpected endpoint or method` — it means the request went to the wrong path.
GoChiDUBB normalizes `LM_STUDIO_URL` itself, so this should only appear if
something else is pointed at the server.

### Optional dependencies

| Feature | Install | Notes |
|---|---|---|
| Multi-speaker diarization | `pip install pyannote.audio` + HF token | Auto-detects N speakers, clones each |
| Background music preservation | `pip install demucs` (recommended) or `pip install audio-separator` | Demuxes vocals, keeps original BGM; Demucs uses the `htdemucs_ft` model |
| Lighter voice cloning | `pip install f5-tts`, then set `tts_engine=f5tts` | F5-TTS needs ~3 GB VRAM vs VoxCPM2's 8+; good quality, faster |
| Faster Whisper on GPU | (already in requirements) | If CUDA isn't found, falls back to CPU |

---

## 🧠 The agent skill

If you use Claude Code, copy `.claude/skills/gochidubb/SKILL.md` into your global skills folder (`~/.claude/skills/gochidubb/`). After that, just say:

- *"Dub this YouTube short into French and German"*
- *"Make a showcase reel of this clip in 5 languages"*
- *"Re-dub job 5038e404 into Japanese and Italian"*
- *"What's the status of my dub?"*

The skill teaches Claude which tool to call, what arguments to use, how to poll, how to recover from errors, and when to suggest a comparison vs a showcase. Read [`SKILL.md`](.claude/skills/gochidubb/SKILL.md) for the full trigger map.

Works with any MCP-compatible agent — Cursor, Cline, Continue, custom agents. The MCP tool schema is auto-discovered.

---

## 🛟 Troubleshooting

**Start at System → Setup.** It lists every subsystem with pass/fail and, for
anything broken, the exact link or command that fixes it. **Run checks** does the
network work — validates your Hugging Face token, asks the Hub whether the gated
pyannote repos are actually accessible, and pings your translation backend. It is
the only thing in the app that touches the network on its own, and only when you
press it.

**System → Logs** shows the last 2000 server lines *including third-party output*
that never goes through Python logging (pyannote, for one, prints its "accept user
conditions" banner with a bare `print()`). Credentials are scrubbed before anything
is stored. This is the fastest way to see what actually happened during a run
without a terminal.

> **A pyannote message worth knowing about.** pyannote prints
> *"Could not download … It might be because the repository is private or gated"*
> for **any** HTTP failure — a timeout, a 503 and a rate limit all produce that same
> text (`pyannote/audio/utils/hf_hub.py`). It is a guess, not a diagnosis. If you
> see it, run **System → Setup → Run checks**: that asks the Hub directly and tells
> you whether it is really a permissions problem or just a failed download you can
> retry.

<details>
<summary><b>Ollama shows a red dot in the UI</b></summary>

Run `ollama serve` in a separate terminal, or restart the app — `start.bat` auto-starts Ollama. If you've never installed Ollama, the System panel has an install button.

</details>

<details>
<summary><b>Ollama has no models installed</b></summary>

Open the System tab → Models → click "Install" on `aya-expanse:8b` (best multilingual, ~5 GB) or `qwen3:8b` (good general, ~5 GB). Or from CLI: `ollama pull aya-expanse:8b`.

</details>

<details>
<summary><b>YouTube download fails / SSL error</b></summary>

Update yt-dlp: `venv\Scripts\activate && pip install -U yt-dlp`. If it's an age-restricted or region-blocked video, set `YT_DLP_COOKIES_FROM_BROWSER=firefox` in `.env` (firefox is the most reliable on macOS; `chrome`/`edge`/`safari` also work but their keychains can block access). If browser cookie extraction fails, export a Netscape `cookies.txt` and point `YT_DLP_COOKIEFILE=/path/to/cookies.txt` at it. For SSL errors, check firewall/VPN/corporate proxy.

Failed downloads are also classified and retried with different strategies automatically (`pipeline/rescue.py`), and a job that stays stuck can be handed a manually-downloaded file: `python tools/gochidubb_cli.py rescue <job_id> ./video.mp4`.

</details>

<details>
<summary><b>VoxCPM2 runs out of VRAM</b></summary>

Three knobs, easiest first:

1. System tab → switch Whisper to `small` (frees ~3 GB)
2. `.env` → `VOXCPM_STEPS=6` (faster, less VRAM)
3. `.env` → `VOXCPM_MODEL=openbmb/VoxCPM1.5` (smaller model, slight quality drop)

</details>

<details>
<summary><b>Voice sounds like two different people mid-video</b></summary>

This was a real bug we fixed: in cross-lingual cloning, QA retries were mutating the random seed mid-job, producing different timbres for failed-then-retried segments. Make sure you're on the latest commit — the fix is in `pipeline/tts_worker.py`.

If you still hit it: try `VOXCPM_CFG=2.5` (more reference-anchored) or upload a longer, cleaner reference voice in the speaker tab.

</details>

<details>
<summary><b>First VoxCPM2 call is slow</b></summary>

Normal. The model downloads ~5 GB on first use; progress is in the terminal. Subsequent runs use the cached weights.

</details>

<details>
<summary><b>Hugging Face 401 / "access denied"</b></summary>

You need to (1) create a token at https://huggingface.co/settings/tokens, (2) accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1 (and https://huggingface.co/pyannote/segmentation-3.0), (3) put `HF_TOKEN=hf_…` in `.env`, or paste it into **System → Setup** (it takes effect immediately, no restart).

**System → Setup → Run checks** tells you which of those three is missing instead of making you guess.

</details>

<details>
<summary><b>The dub sounds like one voice even though the video has several speakers</b></summary>

Check the **Diarize** row in the job's stage panel. If it says **⚠ degraded**, diarization ran on the fallback model (`speaker-diarization-community-1`) because the preferred `speaker-diarization-3.1` could not be downloaded — the run still succeeds, just with weaker speaker separation, which is why it used to be invisible. The row names the reason and links to the fix; re-run the Diarize stage afterwards.

If it says **done** with `speaker_turns=0`, the video genuinely has one speaker, or `skip_diarization` was set on a retry.

</details>

<details>
<summary><b>Every segment logs <code>qa=0.00</code></b></summary>

Whisper-roundtrip QA (used on cross-lingual dubs) requests its model on CUDA (`pipeline/tts_qa.py`), so on Apple Silicon or CPU-only machines it never loads and scores everything as perfect — while the log still says "QA: Whisper roundtrip enabled". `qa=0.00` there means *not measured*, not *good*. The Setup tab reports this as `tts_qa.device_unavailable`. Judge the output by listening.

</details>

<details>
<summary><b>No GPU detected even though I have one</b></summary>

Verify CUDA is visible: `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, reinstall PyTorch matching your CUDA — see https://pytorch.org/get-started/locally/. On Windows make sure you're using the venv Python, not the system one.

</details>

<details>
<summary><b>Audio is out of sync with video</b></summary>

Usually a duration-mismatch in translation (target language is much longer/shorter than source). The pipeline time-aligns automatically, but extreme cases (German → Japanese, etc.) can drift. Try:

- Translation prompt is length-aware by default — make sure you didn't disable it in the UI
- Use a higher-quality translation model (`qwen3:14b` if you have the VRAM)
- For very long videos, dub in 2-3 minute chunks

</details>

<details>
<summary><b>FFmpeg not found</b></summary>

Linux/macOS: `sudo apt install ffmpeg` or `brew install ffmpeg`. Windows: the installer downloads a static build into `bin/` automatically — if it failed, re-run `install.bat`.

</details>

<details>
<summary><b>Showcase reel renders all black / no audio</b></summary>

Usually one of the child dubs failed silently. `python tools/gochidubb_cli.py showcase-status <batch_id>` shows which language failed. Rerun with `gochidubb showcase-rebuild <batch_id>` after fixing the failing job — it skips re-dubbing the successful ones.

</details>

<details>
<summary><b>Background-preserve toggle does nothing</b></summary>

Install the optional dep: `pip install demucs` (recommended) or `pip install audio-separator`. The UI shows a yellow warning if both are missing. First demux is slow (~30 s on GPU); subsequent ones are cached.

</details>

<details>
<summary><b>The server restarted mid-job / a job is orphaned</b></summary>

A job interrupted by a server restart is requeued with `python tools/gochidubb_cli.py continue <job_id>` (or the Continue button in the UI, or `gochidubb_continue_job` over MCP). Stages are checkpointed, so it resumes from the last finished stage rather than starting over.

</details>

<details>
<summary><b>Linux ALSA / pulse errors during TTS</b></summary>

We don't play audio — these are warnings from a transitive dep. Ignore unless they actually break the run. `export ALSA_CARD=-1` silences them.

</details>

<details>
<summary><b>The server is on a different machine — how do I point the CLI at it?</b></summary>

`export GOCHIDUBB_URL=http://192.168.0.10:8910` (or set `GOCHIDUBB_URL` in your MCP config `env` block). The CLI and MCP server respect the same variable.

</details>

<details>
<summary><b>How do I run it headless / on a server?</b></summary>

The server binds `127.0.0.1` by default. To reach it from another machine:

```bash
GOCHIDUBB_HOST=0.0.0.0 GOCHIDUBB_PORT=8910 python server.py
```

(`server.py` does not parse `--host`/`--port` flags — these env vars are the supported way.)

**There is no authentication of any kind.** Anyone who can reach the port can start jobs, read every transcript, browse `/api/logs` and delete your work. Only open it beyond loopback on a network you trust, and put it behind Tailscale / a Cloudflare Tunnel / nginx-with-auth if it's reachable from the internet. The server prints a warning at startup when it binds anything other than loopback.

</details>

---

## ❓ FAQ

**Is this really free?**
Yes. MIT licensed. The only "cost" is your electricity and GPU. No telemetry, no phone-home.

**Do I need an NVIDIA GPU?**
For reasonable speeds, yes. CPU works but a 1-minute dub takes ~30 minutes instead of ~2.

**Does it work on Apple Silicon (M1/M2/M3)?**
Yes. VoxCPM2 runs on MPS (audio quality on MPS was fixed in voxcpm 2.0.3 — the requirement floor exists for this reason), and Demucs picks MPS automatically; Whisper runs on CPU (int8). Expect it slower than a discrete NVIDIA GPU. One caveat: the whisper-roundtrip TTS QA check is CUDA-only, so `qa=0.00` on a Mac means *not measured* — judge the output by ear.

**Can I voice-clone a specific person?**
Yes — drop a 5-30 second clean WAV/MP3 into `presets/voices/` and pick it as the reference. Please don't do this without that person's consent. See [SECURITY.md](SECURITY.md).

**What's the quality vs ElevenLabs?**
On clean source audio, VoxCPM2 is genuinely close. On noisy / multi-speaker content, ElevenLabs still wins (their diarization is better). For 95% of one-speaker YouTube content, you won't tell the difference.

**Does it preserve emotion / tone?**
Partially. VoxCPM2 picks up energy and pacing from the reference. It doesn't model fine emotion the way some closed models do. If the source is a calm explainer, the dub is calm; if it's a hype reel, the dub is hype.

**Can I run multiple dubs in parallel?**
The server queues GPU work serially (one VoxCPM2 invocation at a time) to avoid OOM. CPU stages (download, transcribe with CPU Whisper, ffmpeg) overlap automatically.

**Does it work for animated content / games / non-real voices?**
Yes — anything VoxCPM2 can fit as a reference (usually 5+ s of clean speech) clones fine. Singing is not supported.

**Why VoxCPM2 instead of XTTS / OpenVoice / F5-TTS?**
VoxCPM2 has the best cross-lingual cloning quality we tested at the 5 GB weight class, so it's the default. F5-TTS now ships as a supported lighter alternative (`pip install f5-tts`, `tts_engine=f5tts`, ~3 GB VRAM). The architecture is swappable — `pipeline/synthesizer.py` has a base class (a CosyVoice 2 stub is waiting for a contributor); PRs for other backends welcome.

**Can agents trigger this without my approval?**
Each MCP tool call requires user confirmation by default (per the MCP spec). Gochidubb doesn't bypass that.

---

## 🗺️ Roadmap

- [x] MCP server + CLI
- [x] Stitched multilingual showcase reels
- [x] Multi-speaker diarization
- [x] Background music preservation
- [x] Deterministic voice across cross-lingual segments
- [x] Subtitle burn-in toggle (`POST /api/dub/{id}/burn_subs`, with styling presets)
- [x] Speaker labelling UI (voice casting: name each detected speaker, assign a voice, audition it)
- [ ] Browser-only mode (no Ollama dependency, use llama.cpp WASM)
- [ ] Batch processing folder watcher
- [ ] Docker image with everything pre-baked
- [ ] Hardware-accelerated diarization (NVIDIA NeMo)
- [ ] Apple Silicon MLX backend

Vote / suggest features in [Discussions](https://github.com/georgeevil/gochidubb/discussions).

---

## 🛡️ Responsible use

Voice cloning is powerful and easily misused. **GoChiDUBB is built for legitimate creators dubbing their own content or content they have rights to.** Please:

- Don't clone someone's voice without their explicit, informed consent.
- Don't impersonate real people (politicians, celebrities, your boss) for deception, fraud, or harassment.
- Disclose AI-generated speech when publishing — most platforms now require this, and it's the right thing to do.
- Comply with your local laws on synthetic media (EU AI Act, US state laws, etc.).

We refuse to add features that defeat watermarking, anti-cloning safeguards, or platform AI-disclosure requirements. See [SECURITY.md](SECURITY.md) for the threat model and how to report abuse.

---

## 🤝 Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code style, and the modular pipeline design — most contributions are a single drop-in file in `pipeline/`.

Good first issues:
- Add a TTS backend (XTTS, OpenVoice, or finish the CosyVoice 2 stub in `pipeline/synthesizer.py`)
- Add a translation backend (vLLM, mlx_lm)
- New language voices in the edge-tts fallback map
- Improve the duration-matching prompt for hard language pairs

---

## 💖 Credits

Built by **[George Chigrichenko](https://github.com/georgeevil)** &mdash; in collaboration with **[Claude](https://claude.ai)** (Anthropic).

Follow the build on X: [@GChigrichenko](https://x.com/GChigrichenko)

Standing on shoulders:
- [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — voice cloning TTS (Apache-2.0)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — ASR (MIT)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — diarization (MIT)
- [LM Studio](https://lmstudio.ai) — local LLM serving, default translation backend
- [Ollama](https://ollama.com) — local LLM serving (MIT)
- [Demucs](https://github.com/adefossez/demucs) — vocal separation (MIT)
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection (MIT)
- [F5-TTS](https://github.com/SWivid/F5-TTS) — lighter voice-cloning engine (MIT)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — universal downloader (Unlicense)
- [edge-tts](https://github.com/rany2/edge-tts) — cloud TTS fallback (GPL-3.0)
- [audio-separator](https://github.com/karaokenerds/python-audio-separator) — stem separation (MIT)
- [FFmpeg](https://ffmpeg.org) — every audio/video operation
- [Model Context Protocol](https://modelcontextprotocol.io) — agent integration (Anthropic)

## 📜 License

MIT — see [LICENSE](LICENSE). VoxCPM2 is Apache-2.0. edge-tts is GPL-3.0; using it doesn't require this project to be GPL because it's a runtime dependency invoked as a process.

---

<div align="center">

**If GoChiDUBB saved you a Heygen or ElevenLabs subscription, smash that ⭐ — that's how more people find it.**

</div>
