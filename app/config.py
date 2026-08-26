"""Application configuration — paths, env vars, and persisted user settings.

UserConfig is the single source of truth for runtime-changeable settings.
It loads from config-user.json at startup and saves back when changed.
All fields have sensible defaults so it works out of the box without a config file.

Usage:
    from app.config import cfg

    cfg.whisper_model          # e.g. "large-v3"
    cfg.set("whisper_model", "medium")   # persists to disk immediately
"""
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger("gochidubb.config")

# ─── Project paths ──────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.resolve()
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "outputs"
JOBS_DB = BASE / "jobs_db"
STATIC_DIR = BASE / "static"
PRESETS_DIR = BASE / "presets"
VOICE_PRESETS_DIR = PRESETS_DIR / "voices"
USER_GLOSSARY_FILE = PRESETS_DIR / "user_glossary.json"
CONFIG_FILE = BASE / "config-user.json"

for _d in (UPLOAD_DIR, OUTPUT_DIR, JOBS_DB, STATIC_DIR, VOICE_PRESETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Bump when a setting's meaning changes and old files need fixing up; add the
# corresponding step to _migrate().
CONFIG_VERSION = 3

# Validation table for the settings the product lets a user edit. set() and
# update() both consult it, so every write path — the Settings UI,
# PATCH /api/config, a direct cfg.set() — gets identical coercion and bounds
# instead of each caller re-checking.
#
#   numeric: (type, low, high, *sentinels)   sentinels bypass the range check
#   enum:    (str, (choice, ...))
#   flag:    (bool,)
#
# Fields absent from this table are stored as-is, exactly as before.
FIELD_SPECS: dict = {
    "tts_engine":          (str, ("voxcpm", "f5tts", "edge-tts", "auto")),
    "tts_speed":           (str, ("fast", "balanced", "quality")),
    "tts_max_stretch":     (float, 1.0, 2.0),
    "voxcpm_cfg":          (float, 1.0, 3.0),
    # 0 is "follow the tts_speed tier"; a real override is 4–24.
    "voxcpm_steps":        (int, 4, 24, 0),
    "voxcpm_denoise_refs": (bool,),
    "voxcpm_xling_cfg":    (float, 1.0, 3.0),
    "voxcpm_xling_steps":  (int, 4, 24),
    "qa_same_language":    (bool,),
    "output_video_codec":  (str, ("h264", "copy")),
    "output_preset":       (str, ("ultrafast", "superfast", "veryfast", "faster",
                                  "fast", "medium", "slow")),
    "output_crf":          (int, 16, 32),
    "output_audio_bitrate": (str, ("128k", "192k", "256k", "320k")),
    # 0 = no cap (best available); otherwise a real frame height.
    "output_max_height":   (int, 360, 2160, 0),
    # A raw ffmpeg gain on the separated bed. 1.0 = the stem at its original
    # level — which sits ~10 dB under the loudnorm-hot dub voice, so the
    # useful range runs well past unity. Measured on a real job: ×10 restores
    # the source's voice:music balance (an alimiter guards the sum).
    "background_volume":   (float, 0.0, 10.0),
    "bg_ducking":          (bool,),
    "eta_realtime_factor": (float, 0.5, 60.0),
    "mode":                (str, ("local", "hosted")),
    # "" is off, and is the default: the other two make yt-dlp fetch and run
    # a solver component from the network. An enum rather than free text so a
    # typo cannot turn into an arbitrary --remote-components argument.
    "ytdlp_remote_components": (str, ("", "ejs:github", "ejs:npm")),
    "download_rescue_llm":     (bool,),
    # Review gates (see app/review_gates.py). These are the Pro defaults;
    # any request that sends wizard_mode (creator.html, the CLI) suppresses
    # them, so arming one here can never surprise those clients.
    "review_gate_transcript":  (str, ("off", "on", "flagged_only")),
    "review_gate_translation": (str, ("off", "on", "flagged_only")),
    "review_gate_voice_cast":  (str, ("off", "on", "flagged_only")),
    "review_gate_subtitles":   (str, ("off", "on", "flagged_only")),
    "review_gate_final_qc":    (str, ("off", "on", "flagged_only")),
    "voice_consent_policy":    (str, ("off", "warn", "require")),
    # Loudness target for the assembled dub track (EBU R128 loudnorm).
    "loudness_target":         (float, -31.0, -8.0),
    "loudness_true_peak":      (float, -9.0, 0.0),
    # Subtitle cue limits (pipeline/subtitles.py validator).
    "subtitle_max_chars_per_line": (int, 16, 80),
    "subtitle_max_lines":          (int, 1, 4),
    "subtitle_max_cps":            (float, 5.0, 40.0),
    "subtitle_min_gap_ms":         (int, 0, 1000),
}

_TRUTHY = ("1", "true", "yes", "on")


def coerce_field(key: str, value):
    """Coerce and validate one setting against FIELD_SPECS.

    Returns the cleaned value, or raises ValueError describing what was wrong.
    Out-of-range numbers are rejected rather than silently clamped: a user who
    typed 99 wants to know it was refused, not discover 3.0 later.
    """
    spec = FIELD_SPECS.get(key)
    if spec is None:
        return value
    kind = spec[0]

    if kind is bool:
        if isinstance(value, str):
            return value.strip().lower() in _TRUTHY
        return bool(value)

    if kind is str:
        v = str(value).strip()
        choices = spec[1] if len(spec) > 1 else None
        if choices and v not in choices:
            raise ValueError(
                f"{key}: {v!r} is not one of {', '.join(choices)}"
            )
        return v

    try:
        v = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: {value!r} is not a {kind.__name__}") from None
    low, high, sentinels = spec[1], spec[2], spec[3:]
    if v in sentinels:
        return v
    if not (low <= v <= high):
        allowed = f"{low}–{high}"
        if sentinels:
            allowed += " (or " + ", ".join(str(x) for x in sentinels) + ")"
        raise ValueError(f"{key}: {v} is outside {allowed}")
    return v


@dataclass
class UserConfig:
    """Runtime-editable settings, persisted to config-user.json.

    All fields are optional — missing keys in config-user.json fall back to
    the dataclass defaults, so adding new settings never breaks existing installs.
    """

    # ── Deployment mode ───────────────────────────────────────────────
    # "local" is today's experience: no auth, no billing surfaces.
    # "hosted" reveals the Workspace UI group and enforces API-key scopes.
    # Pause a job when a stage's output is not worth building on, rather
    # than spending the next stage's GPU time and reporting afterwards.
    quality_gate: bool = True
    mode: str = "local"                  # "local" | "hosted"

    # ── Whisper ───────────────────────────────────────────────────────
    whisper_model: str = "large-v3"       # large-v3 | medium | small | tiny
    auto_denoise: bool = True             # FFT denoise before transcription

    # ── VAD ───────────────────────────────────────────────────────────
    vad_enabled: bool = True              # strip silence before Whisper
    vad_threshold: float = 0.5           # Silero VAD threshold (0.3–0.8)

    # ── Background separation ──────────────────────────────────────────
    separation_backend: str = "demucs"   # "demucs" | "audio-separator" | "none"
    demucs_model: str = "htdemucs_ft"    # Demucs model name

    # ── Translation ────────────────────────────────────────────────────
    translation_model: str = "qwen3:8b"  # default Ollama model
    ollama_url: str = "http://localhost:11434"

    # ── VoxCPM ────────────────────────────────────────────────────────
    voxcpm_model: str = "openbmb/VoxCPM2"
    voxcpm_cfg: float = 2.0              # guidance strength, 1.0–3.0
    # Inference timesteps. 0 = follow the tts_speed tier (SPEED_TIMESTEPS in
    # pipeline/synthesizer.py); any other value is an explicit override that
    # wins over the tier. This used to default to 10 and never reach batch
    # synthesis at all — see _migrate() for what that means on upgrade.
    voxcpm_steps: int = 0
    # Run VoxCPM's own denoiser over reference clips. The pipeline already
    # denoises, normalizes and trims refs itself (_REF_FILTER in
    # pipeline/synthesizer.py), so this is off by default — turn it on only
    # for unusually noisy references.
    voxcpm_denoise_refs: bool = False
    # Cross-lingual guidance. When source != target VoxCPM drifts toward
    # source-language phonetics, so cfg and timesteps are raised for those
    # jobs; these are the values it is raised *to* (a floor, not a delta).
    voxcpm_xling_cfg: float = 2.5
    voxcpm_xling_steps: int = 14

    # ── TTS ───────────────────────────────────────────────────────────
    tts_engine: str = "voxcpm"           # "voxcpm" | "f5tts" | "edge-tts"
    tts_speed: str = "balanced"          # "fast" | "balanced" | "quality"
    # Hardest time-compression the assembler may apply to keep a segment in
    # sync (atempo, pitch preserved). Up to ~1.15 is inaudible; the headroom
    # above that is what stops one overlong segment from pushing every later
    # one late. Raise it for dense dialogue, lower it if the dub sounds rushed.
    tts_max_stretch: float = 1.4
    warmup_on_start: bool = False        # pre-load VoxCPM at server start

    # ── Stage reuse (beta) ────────────────────────────────────────────
    # Skip a pipeline stage when a previous job already computed the same
    # thing from the same inputs. Download/extract/transcribe/diarize are
    # ~60% of pipeline time and none of it depends on the target language,
    # so a redub into a new language currently redoes all of it.
    #
    # Off by default: this is new, and a stale reuse is a silent failure.
    # See docs/stage-reuse.md.
    reuse_enabled: bool = False
    # Stages allowed to be reused, comma-separated. The source-only stages
    # are the safe, high-value set; translate and tts are opt-in on top.
    reuse_stages: str = "download,extract,transcribe,diarize"
    # Quality gates — an artifact that scored worse than this when it was
    # produced is recomputed rather than reused. 0 disables a gate.
    reuse_transcribe_min_word_conf: float = 0.45
    reuse_transcribe_max_no_speech: float = 0.35
    reuse_diarize_min_ref_sec: float = 1.0
    reuse_translate_max_fallback: float = 0.0
    reuse_translate_min_semantic: float = 0.67
    reuse_tts_max_failed_qa: float = 0.25
    qa_same_language: bool = False       # whisper-roundtrip QA on same-language dubs too

    # ── Review gates ──────────────────────────────────────────────────
    # Default gate mode per review point ("off" | "on" | "flagged_only").
    # These apply only to submissions that did NOT send wizard_mode — see
    # app/review_gates.py::resolve_gates for the precedence contract.
    review_gate_transcript: str = "off"
    review_gate_translation: str = "off"
    review_gate_voice_cast: str = "off"
    review_gate_subtitles: str = "off"
    review_gate_final_qc: str = "off"

    # ── Voice consent (CLD-272) ───────────────────────────────────────
    # What happens when a cast resolves to cloning a real person's voice
    # (the source speaker, or an uploaded file preset) with no consent
    # attestation on record. "off" = today's behaviour; "warn" clones and
    # marks the cast report; "require" downgrades that speaker to a preset.
    voice_consent_policy: str = "off"    # "off" | "warn" | "require"

    # ── Loudness (assembled dub track) ────────────────────────────────
    # EBU R128 loudnorm targets for the final mix. -16 LUFS is the historic
    # hardcoded default; -14 matches YouTube's normalization point, -23 EBU
    # broadcast. Free numeric fields (bounded), not an enum — the UI offers
    # presets over them but power users can type.
    loudness_target: float = -16.0       # integrated loudness, LUFS
    loudness_true_peak: float = -1.5     # true peak ceiling, dBTP

    # ── Subtitles (review workbench validator) ────────────────────────
    # Cue limits the subtitle validator holds cues against. The defaults
    # mirror pipeline/subtitles.py::DEFAULT_LIMITS (Netflix-style).
    subtitle_max_chars_per_line: int = 42
    subtitle_max_lines: int = 2
    subtitle_max_cps: float = 17.0
    subtitle_min_gap_ms: int = 120

    # ── Output / render ───────────────────────────────────────────────
    # How the finished video is encoded. Defaults reproduce the behaviour
    # these values were hardcoded to, so upgrading changes nothing on its own.
    #
    # "h264" re-encodes whenever the source stream isn't already H.264/yuv420p.
    # "copy" keeps the source video stream untouched — faster, but a VP9 or
    # AV1 source then sits inside an .mp4 that QuickTime and WhatsApp refuse
    # to play. See docs and the Output settings tab.
    output_video_codec: str = "h264"     # "h264" | "copy"
    output_preset: str = "veryfast"      # x264 speed/size trade-off
    output_crf: int = 23                 # 18 (larger, better) … 28 (smaller)
    output_audio_bitrate: str = "192k"
    output_max_height: int = 1080        # source download cap, 0 = best available
    # Bed gain (raw ffmpeg volume multiplier on the separated stem). The dub
    # voice is normalized ~10 dB hotter than the original program, so unity
    # leaves the bed buried; ×10 restores the source's voice:music balance
    # (measured — see docs/design/pro-workbench-plan.md changelog). A post-mix
    # alimiter keeps the hot bed from clipping the sum.
    background_volume: float = 10.0
    bg_ducking: bool = False  # duck the bed under speech (off by default:
    # with the bed at balance level, an 11 dB dip under every line reads as
    # the music "pumping"; turn on for speech-dense content)

    # ── FFmpeg (extract / render) ─────────────────────────────────────
    # Soft timeout: while ffmpeg keeps reporting encode progress the deadline
    # is extended, so long videos can render past this. It only becomes a
    # hard kill once no progress is seen for ffmpeg_stall_timeout seconds.
    ffmpeg_timeout: int = 600
    ffmpeg_stall_timeout: int = 120

    # ── Downloader (yt-dlp) ───────────────────────────────────────────
    ytdlp_cookies_from_browser: str = ""   # e.g. "firefox", "chrome", "safari"
    ytdlp_cookiefile: str = ""             # path to Netscape cookies.txt
    # YouTube sometimes gates a video behind a JavaScript challenge that
    # yt-dlp can only solve by downloading a solver component at runtime
    # ("ejs:github" or "ejs:npm") and executing it in deno or node. That is
    # remote code fetched at download time, so it stays OFF by default and is
    # only ever used as a last-resort retry after a challenge failure — see
    # pipeline/downloader.py. Empty = never fetch it.
    ytdlp_remote_components: str = ""      # "" | "ejs:github" | "ejs:npm"
    # Let the local translation model suggest what to try next when a download
    # fails in a way pipeline/rescue.py has no rule for. It picks a key out of
    # a fixed strategy table and can never write a flag or a command, so the
    # worst case is a wasted attempt. Off by default: the rules cover every
    # failure seen so far, and a model round trip competes with translation
    # and TTS for the same GPU.
    download_rescue_llm: bool = False
    max_source_duration_sec: int = 0       # 0 = no pre-download duration gate

    # ── Estimates ─────────────────────────────────────────────────────
    # Fallback wall-clock-seconds per second-of-source, used by
    # GET /api/estimate until this install has finished enough jobs for
    # app/estimate.py to measure its own median. 6x is a full-pipeline
    # run on a 12GB GPU; a CPU-only box is far slower and should raise it.
    # Note the ETA also multiplies by the language count — the job queue is
    # single-consumer, so N languages run serially.
    eta_realtime_factor: float = 6.0

    # ── Publishing ────────────────────────────────────────────────────
    # Attribution appended to published-video descriptions. {source_url}
    # is replaced with the original video URL (see pipeline/publisher.py).
    publish_description_template: str = "Original: {source_url}"

    # ── Internal ──────────────────────────────────────────────────────
    # Schema version of the persisted file, so a settings migration runs
    # exactly once instead of re-firing on every startup. A config-user.json
    # written before this existed has no key and is treated as version 0.
    config_version: int = CONFIG_VERSION

    # ── UI behaviour ──────────────────────────────────────────────────
    open_browser: bool = True
    server_port: int = 8910
    hf_token: str = ""                   # HuggingFace token for pyannote

    def set(self, key: str, value) -> None:
        """Set a config value and immediately persist to disk.

        Raises KeyError for an unknown key, ValueError for one that fails
        FIELD_SPECS validation. Nothing is written when either fires.
        """
        if not hasattr(self, key):
            raise KeyError(f"Unknown config key: {key!r}")
        setattr(self, key, coerce_field(key, value))
        self._save()

    def update(self, **kwargs) -> None:
        """Batch-update keys and persist once.

        Validation happens for every key *before* any is applied, so a bad
        value in the middle of a Settings save cannot leave half of it
        written. Unknown keys are ignored with a warning, as before.
        """
        clean = {}
        for k, v in kwargs.items():
            if hasattr(self, k):
                clean[k] = coerce_field(k, v)   # ValueError propagates
            else:
                log.warning(f"[config] Unknown key ignored: {k!r}")
        for k, v in clean.items():
            setattr(self, k, v)
        self._save()

    def to_dict(self) -> dict:
        return asdict(self)

    def _save(self) -> None:
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"[config] Could not save {CONFIG_FILE}: {e}")


# The value voxcpm_steps defaulted to before it became a real override.
_LEGACY_VOXCPM_STEPS = 10

# The background_volume default before ducking existed (flat mix level).
_LEGACY_BG_VOLUME = 0.15


def _migrate(c: "UserConfig", from_version: int) -> None:
    """Fix up a persisted config whose settings have changed meaning.

    Mutates `c` in place and re-saves when anything changed, so each step
    runs exactly once rather than re-firing on every startup.
    """
    if from_version >= CONFIG_VERSION:
        return
    changed = []

    # v0 → v1: voxcpm_steps became an override that actually reaches
    # synthesis. It previously defaulted to 10 and was ignored on the batch
    # path (SPEED_TIMESTEPS won), so _save() has written 10 into most
    # existing files. Leaving it would silently move "balanced" from 8 to 10
    # steps on upgrade — so a value equal to that dead default becomes 0,
    # "follow the tier", which is what these installs actually experienced.
    if from_version < 1 and c.voxcpm_steps == _LEGACY_VOXCPM_STEPS:
        c.voxcpm_steps = 0
        changed.append("voxcpm_steps 10 -> 0 (follow tts_speed tier)")

    # v1 → v2: background_volume changed meaning from flat mix level to the
    # ducked bed's ceiling, and its default moved 0.15 -> 0.5. _save() wrote
    # 0.15 into every existing config-user.json, so a stored 0.15 means "the
    # old default" and follows the new one; an explicitly chosen other value
    # survives untouched.
    if from_version < 2 and c.background_volume == _LEGACY_BG_VOLUME:
        c.background_volume = 0.5
        changed.append("background_volume 0.15 -> 0.5 (ducked bed ceiling)")

    # v2 → v3: the scale grew past unity once measurement showed the dub
    # voice runs ~10 dB hotter than the original program, burying a
    # unity-gain bed. A stored 0.5 is the old default; a stored 1.0 is the
    # old CEILING — a maxed slider meant "as loud as allowed", not "exactly
    # unity" — so both follow the new default. Any other value was a real
    # choice inside the old range and survives. Ducking also flips to off
    # by default, but only for configs still on the old default (True was
    # the v2 default; an explicit False already survives as-is).
    if from_version < 3:
        if c.background_volume in (0.5, 1.0):
            c.background_volume = 10.0
            changed.append("background_volume -> 10.0 (bed balance vs the "
                           "hot dub voice; alimiter guards the sum)")
        if c.bg_ducking:
            c.bg_ducking = False
            changed.append("bg_ducking on -> off (default; the balanced bed "
                           "made the duck read as pumping)")

    c.config_version = CONFIG_VERSION
    if changed:
        log.info("[config] migrated to v%d: %s", CONFIG_VERSION, "; ".join(changed))
    c._save()


def _load_config() -> UserConfig:
    """Load UserConfig from disk, merging over dataclass defaults."""
    c = UserConfig()

    # Layer 1: environment variables (highest priority)
    env_map = {
        "GOCHIDUBB_MODE": "mode",
        "GOCHIDUBB_QUALITY_GATE": "quality_gate",
        "HF_TOKEN": "hf_token",
        "VOXCPM_MODEL": "voxcpm_model",
        "VOXCPM_CFG": "voxcpm_cfg",
        "VOXCPM_STEPS": "voxcpm_steps",
        "VOXCPM_DENOISE_REFS": "voxcpm_denoise_refs",
        "VOXCPM_XLING_CFG": "voxcpm_xling_cfg",
        "VOXCPM_XLING_STEPS": "voxcpm_xling_steps",
        "GOCHIDUBB_TTS_ENGINE": "tts_engine",
        "GOCHIDUBB_OUTPUT_CODEC": "output_video_codec",
        "GOCHIDUBB_OUTPUT_PRESET": "output_preset",
        "GOCHIDUBB_OUTPUT_CRF": "output_crf",
        "GOCHIDUBB_OUTPUT_MAX_HEIGHT": "output_max_height",
        "GOCHIDUBB_BACKGROUND_VOLUME": "background_volume",
        "GOCHIDUBB_BG_DUCKING": "bg_ducking",
        "OLLAMA_URL": "ollama_url",
        "WHISPER_MODEL": "whisper_model",
        "GOCHIDUBB_OPEN_BROWSER": "open_browser",
        "GOCHIDUBB_WARMUP": "warmup_on_start",
        "GOCHIDUBB_QA_SAME_LANGUAGE": "qa_same_language",
        "GOCHIDUBB_TTS_MAX_STRETCH": "tts_max_stretch",
        "GOCHIDUBB_REUSE": "reuse_enabled",
        "GOCHIDUBB_REUSE_STAGES": "reuse_stages",
        "GOCHIDUBB_FFMPEG_TIMEOUT": "ffmpeg_timeout",
        "GOCHIDUBB_FFMPEG_STALL_TIMEOUT": "ffmpeg_stall_timeout",
        "YT_DLP_COOKIES_FROM_BROWSER": "ytdlp_cookies_from_browser",
        "YT_DLP_COOKIEFILE": "ytdlp_cookiefile",
        "YT_DLP_REMOTE_COMPONENTS": "ytdlp_remote_components",
        "GOCHIDUBB_DOWNLOAD_RESCUE_LLM": "download_rescue_llm",
        "GOCHIDUBB_GATE_TRANSCRIPT": "review_gate_transcript",
        "GOCHIDUBB_GATE_TRANSLATION": "review_gate_translation",
        "GOCHIDUBB_GATE_VOICE_CAST": "review_gate_voice_cast",
        "GOCHIDUBB_GATE_SUBTITLES": "review_gate_subtitles",
        "GOCHIDUBB_GATE_FINAL_QC": "review_gate_final_qc",
        "GOCHIDUBB_CONSENT_POLICY": "voice_consent_policy",
        "GOCHIDUBB_LOUDNESS_TARGET": "loudness_target",
        "GOCHIDUBB_LOUDNESS_TRUE_PEAK": "loudness_true_peak",
        "GOCHIDUBB_SUBTITLE_MAX_CHARS_PER_LINE": "subtitle_max_chars_per_line",
        "GOCHIDUBB_SUBTITLE_MAX_LINES": "subtitle_max_lines",
        "GOCHIDUBB_SUBTITLE_MAX_CPS": "subtitle_max_cps",
        "GOCHIDUBB_SUBTITLE_MIN_GAP_MS": "subtitle_min_gap_ms",
        "MAX_SOURCE_DURATION_SEC": "max_source_duration_sec",
        "PUBLISH_DESCRIPTION_TEMPLATE": "publish_description_template",
    }
    for env_k, field_k in env_map.items():
        v = os.getenv(env_k)
        if v is None:
            continue
        cur = getattr(c, field_k)
        try:
            if isinstance(cur, bool):
                parsed = v.strip().lower() in _TRUTHY
            elif isinstance(cur, int):
                parsed = int(v.strip())
            elif isinstance(cur, float):
                parsed = float(v.strip())
            else:
                parsed = v.strip()
            # Same bounds the UI is held to — a bad env var falls back to the
            # default with a warning rather than reaching a model.
            setattr(c, field_k, coerce_field(field_k, parsed))
        except (ValueError, TypeError) as e:
            log.warning(f"[config] env {env_k}={v!r}: {e}")

    # Layer 2: config-user.json (lower priority than env)
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(c, k):
                    # Don't let the file override env-set values
                    env_key = next(
                        (ek for ek, fk in env_map.items() if fk == k), None
                    )
                    if env_key and os.getenv(env_key) is not None:
                        continue
                    setattr(c, k, v)
            # A file written before config_version existed is version 0.
            _migrate(c, int(data.get("config_version", 0) or 0))
        except Exception as e:
            log.warning(f"[config] Could not read {CONFIG_FILE}: {e}")

    return c


# Singleton — import and use `cfg` everywhere
cfg: UserConfig = _load_config()
