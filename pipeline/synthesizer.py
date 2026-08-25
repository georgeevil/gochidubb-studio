"""VoxCPM2 voice cloning + edge-tts fallback.

NOTE: server.py installs speechbrain/k2 stubs at startup (before any
pipeline import) so that VoxCPM can call reference-audio paths without
triggering speechbrain.integrations.k2_fsa lazy-import failures on
Windows. If you ever run this module standalone, import server.py's
bootstrap block first.
"""
import logging
import os
import random as _random
import subprocess
import sys
import time
import traceback
import types
from typing import Optional

log = logging.getLogger("gochidubb.synthesizer")

# Track whether we've logged the full tier-1 traceback once per process
_TIER1_DIAG_LOGGED = False

# ─── Speed presets ───────────────────────────────────────────────────
# Inference steps per quality tier. VoxCPM's default is 10; 6 is ~40%
# faster with minimal quality loss on short text.
SPEED_TIMESTEPS = {"quality": 10, "balanced": 8, "fast": 6}

# retry_badcase_max_times is a LOOP BOUND inside voxcpm (max attempts), NOT a
# retry count. voxcpm binds `latent_pred` only inside
#     while retry_badcase_times < retry_badcase_max_times:
# and reads it after the loop, so 0 means the loop never runs and EVERY
# segment dies with:
#     UnboundLocalError: cannot access local variable 'latent_pred'
# (voxcpm/model/voxcpm2.py:923-958). Every value here MUST be >= 1 — see
# tests/test_tts_worker.py. "fast" still makes exactly one pass with no
# re-rolls because retry_badcase=False makes the loop break after the first
# attempt; the bound alone is what decides whether it runs at all.
SPEED_RETRIES = {"quality": 2, "balanced": 1, "fast": 1}


def resolve_guidance(base_cfg, base_steps, is_cross_lingual,
                     xling_cfg=2.5, xling_steps=14,
                     cfg_override=None, steps_override=None):
    """Final (cfg, timesteps) for one synthesis run.

    Three sources, most specific first:

    1. A per-job override from the New dub form — an explicit "this source is
       hard, give it more guidance" for one job only.
    2. The global setting (engine ``cfg_value`` / ``inference_timesteps``),
       itself resolved against the tts_speed tier by ``effective_timesteps``.
    3. The cross-lingual floor, which raises both when source and target
       languages differ (VoxCPM otherwise drifts toward source-language
       phonetics on a fifth to a third of segments).

    A per-job override is FINAL: it skips the cross-lingual floor rather than
    being clamped up by it. Almost every dub is cross-lingual, so clamping
    would make a lowered cfg silently inert — a knob that does nothing is
    worse than no knob. Raising works either way.

    0 / None means "not overridden" — the same sentinel ``voxcpm_steps``
    already uses for "follow the tier".
    """
    cfg_set = bool(cfg_override)
    steps_set = bool(steps_override)
    out_cfg = float(cfg_override) if cfg_set else float(base_cfg)
    out_steps = int(steps_override) if steps_set else int(base_steps)

    if is_cross_lingual:
        if not cfg_set:
            out_cfg = max(out_cfg, float(xling_cfg))
        if not steps_set:
            out_steps = max(out_steps, int(xling_steps))

    return out_cfg, out_steps


# ─────────────────────────────────────────────────────────────────────
# Backup speechbrain/k2 stubs — in case this module is imported first
# (e.g. from a test harness) before server.py has run.
# ─────────────────────────────────────────────────────────────────────
def _ensure_sb_stubs():
    for name in (
        "k2",
        "speechbrain.k2_integration",
        "speechbrain.integrations.k2_fsa",
        "speechbrain.integrations.k2_fsa.ctc_loss",
        "speechbrain.integrations.k2_fsa.graph_compiler",
        "speechbrain.integrations.k2_fsa.lattice_decoder",
        "speechbrain.integrations.k2_fsa.lexicon",
        "speechbrain.integrations.k2_fsa.losses",
        "speechbrain.integrations.k2_fsa.prepare_lang",
        "speechbrain.integrations.k2_fsa.utils",
        "speechbrain.wordemb",
        "speechbrain.lobes.models.huggingface_transformers",
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__file__ = f"<gochidubb-stub:{name}>"
            m.__path__ = []
            sys.modules[name] = m


_ensure_sb_stubs()

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message=".*speechbrain\\.pretrained.*was deprecated.*")
warnings.filterwarnings("ignore", message=".*torch\\.nn\\.utils\\.weight_norm.*")


# ═══════════════════════════════════════════════════════════════════════
#  TTS Engine Abstraction
# ═══════════════════════════════════════════════════════════════════════
# We aim to support multiple TTS backends (VoxCPM2, CosyVoice2, GPT-SoVITS,
# Fish-Speech, Edge-TTS fallback). Each has different APIs, dependencies,
# and strengths — but from the pipeline's perspective, they're all just
# "something that takes text + reference audio and produces a wav".
#
# This base class formalizes the contract. Adding a new engine means:
#   1. subclass BaseTTSEngine
#   2. implement synthesize_segments() with the expected signature
#   3. register in server.py's get_tts_engine() factory
#   4. optionally expose the engine name in the UI dropdown
#
# We deliberately keep the interface minimal: each engine owns its own
# model loading, subprocess management, QA, etc. The pipeline doesn't
# care HOW a segment gets synthesized, only that it gets a WAV back at
# the right sample rate.
# ═══════════════════════════════════════════════════════════════════════

class BaseTTSEngine:
    """Abstract base for all TTS engines used by the dubbing pipeline.

    Subclasses must implement:
      - synthesize_segments(): takes segment list, returns list with
        'audio_path' populated on each successful segment
      - load() / unload(): lifecycle hooks (can be no-ops)

    Subclasses MAY override:
      - sample_rate: output WAV rate (defaults to 48000)
      - name: human-readable engine name for UI/logs
    """

    name = "base"  # override in subclass
    default_sample_rate = 48000

    def __init__(self):
        self._sample_rate = None

    @property
    def sample_rate(self):
        return self._sample_rate or self.default_sample_rate

    def load(self):
        """Load model into memory. Idempotent. May be slow."""
        raise NotImplementedError

    def unload(self):
        """Free model memory / kill worker processes. Idempotent."""
        raise NotImplementedError

    def last_run_stats(self) -> dict:
        """Aggregate stats of the most recent synthesize_segments run.

        Engines that track them return keys like tier_stats, qa_regens,
        qa_measured_count, qa_unmeasured_count so the server can persist
        them into the tts stage's perf metrics. Default: nothing."""
        return {}

    def synthesize_segments(self, segments, output_dir,
                           speaker_refs=None, speaker_transcripts=None,
                           progress_callback=None, voice_seed=None,
                           tts_speed="balanced",
                           is_cross_lingual=False, target_lang="ru",
                           cfg_override=None, steps_override=None):
        """Core synthesis call.

        Args:
            segments: list of dicts with keys:
                - idx: int
                - start, end: float (original video timing)
                - text: str (source-lang text, usually not used)
                - translated_text: str (what to speak)
                - speaker: str (which speaker ref to use)
                - audio_path: str (if already synthesized, preserve)
            output_dir: str, where seg_NNNN.wav files go
            speaker_refs: {speaker_id: wav_path}
            speaker_transcripts: {speaker_id: str} — text that matches
                each ref (for Ultimate Cloning mode). Empty string means
                use Controllable Cloning (isolated reference).
            progress_callback: fn(done, total) or None
            voice_seed: int or None, reproducibility seed
            tts_speed: "fast" | "balanced" | "quality"
            is_cross_lingual: bool, tweaks guidance for cross-lingual work
            target_lang: str, used by QA to detect wrong-language output
            cfg_override, steps_override: per-job VoxCPM guidance / step
                count. Falsy means "use the global setting". Engines
                without those concepts ignore them.

        Returns:
            Modified segments list with 'audio_path' set on synthesized
            ones. Segments that already had an audio_path and
            preserve_existing_audio_paths=True are untouched.
        """
        raise NotImplementedError


class VoxCPMSynthesizer(BaseTTSEngine):
    """VoxCPM2 wrapper - load once, synthesize many.

    Engine features:
    - Zero-shot voice cloning (Controllable + Ultimate modes)
    - Cross-lingual dubbing with cfg bump
    - Persistent worker subprocess for retry speed
    - Whisper roundtrip QA for auto-regen of bad outputs
    """

    name = "voxcpm2"
    default_sample_rate = 48000

    def __init__(self, model_id="openbmb/VoxCPM2", load_denoiser=False,
                 cfg_value=2.0, inference_timesteps=0):
        super().__init__()
        self.model_id = model_id
        self.load_denoiser = load_denoiser
        self.cfg_value = cfg_value
        # 0 = "no explicit override, follow the tts_speed tier". Read it
        # through effective_timesteps() rather than directly, so the sentinel
        # never reaches VoxCPM (which needs a real step count).
        self.inference_timesteps = inference_timesteps
        self._model = None
        # Persistent worker subprocess — spawned on first synthesize call,
        # kept alive between calls so we don't pay the 34s VoxCPM reload tax
        # on every retry. Killed when engine is unloaded or process exits.
        self._worker_proc = None
        self._worker_stderr_fh = None
        self._worker_stderr_path = None
        # First per-segment error of the most recent run. The worker reports
        # failures as events rather than raising, so without this the caller
        # only knows "0 succeeded" and not why.
        self._last_segment_error = ""
        # Aggregate stats of the most recent run (from the worker's "done"
        # event) — surfaced to the server via last_run_stats().
        self._last_run_stats = {}

    def last_run_stats(self) -> dict:
        return dict(self._last_run_stats)

    def last_failure_detail(self) -> str:
        """Human-readable cause of the last run's segment failures, if any.

        Includes the worker stderr path because the full traceback is written
        there, not to the server log."""
        if not self._last_segment_error:
            return ""
        detail = self._last_segment_error
        if self._worker_stderr_path:
            detail += f" (worker log: {self._worker_stderr_path})"
        return detail

    def load(self):
        if self._model is not None:
            return
        from voxcpm import VoxCPM
        log.info(f"Loading VoxCPM2: {self.model_id} (first load can take a few minutes)")
        t0 = time.time()
        self._model = VoxCPM.from_pretrained(
            self.model_id,
            load_denoiser=self.load_denoiser,
        )
        self._sample_rate = self._model.tts_model.sample_rate
        log.info(f"VoxCPM2 loaded in {time.time() - t0:.1f}s, SR={self._sample_rate}")

    def unload(self):
        # Graceful shutdown of persistent worker
        if self._worker_proc is not None:
            try:
                log.info("Shutting down persistent TTS worker")
                if self._worker_proc.poll() is None:
                    try:
                        self._worker_proc.stdin.write("SHUTDOWN\n")
                        self._worker_proc.stdin.flush()
                        self._worker_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        self._worker_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._worker_proc.kill()
            except Exception as e:
                log.warning(f"Error stopping TTS worker: {e}")
            finally:
                self._worker_proc = None
        if self._worker_stderr_fh is not None:
            try:
                self._worker_stderr_fh.close()
            except Exception:
                pass
            self._worker_stderr_fh = None
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            log.info("VoxCPM2 unloaded")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate or 48000

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _set_voice_seed(seed: Optional[int]):
        """Set ALL RNG sources so VoxCPM voice design produces the same
        voice across consecutive generate() calls. Must be called right
        before each generate()."""
        if seed is None:
            return
        try:
            import torch
            import numpy as np
            s = int(seed) & 0x7FFFFFFF
            _random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            # Force cuDNN to use deterministic algorithms so every segment
            # starts from the SAME noise sample → consistent voice identity
            # across all segments of the same dub.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            log.debug(f"Could not set voice seed: {e}")

    def _generate(self, kwargs):
        wav = self._model.generate(**kwargs)
        if wav is None or len(wav) == 0:
            raise RuntimeError("empty audio")
        return wav

    def effective_timesteps(self, tts_speed: str = "balanced") -> int:
        """Inference steps to actually use.

        cfg.voxcpm_steps (Settings → Voice & TTS) is an explicit override;
        0 means "follow the speed tier", which is the default and matches
        what installs got before this value reached synthesis.
        """
        return self.inference_timesteps or SPEED_TIMESTEPS.get(tts_speed, 8)

    def synthesize_segment(self, text, output_path,
                          reference_wav_path=None, prompt_wav_path=None, prompt_text=None,
                          voice_seed: Optional[int] = None):
        global _TIER1_DIAG_LOGGED

        if self._model is None:
            self.load()

        if not text or not text.strip():
            return None

        import soundfile as sf

        base_kwargs = {
            "text": text,
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.effective_timesteps(),
            "retry_badcase": True,
            "retry_badcase_max_times": 2,
        }

        # Tier 1 — full cloning with reference + prompt transcript.
        # prompt_wav_path is attached ONLY alongside a non-empty prompt_text:
        # voxcpm rejects the pair unless both are set or both are None. See the
        # matching block in tts_worker.py.
        tier1 = dict(base_kwargs)
        if prompt_text and prompt_wav_path and os.path.exists(prompt_wav_path):
            tier1["prompt_wav_path"] = prompt_wav_path
            tier1["prompt_text"] = prompt_text
            tier1["reference_wav_path"] = prompt_wav_path
        elif reference_wav_path and os.path.exists(reference_wav_path):
            tier1["reference_wav_path"] = reference_wav_path
        elif prompt_wav_path and os.path.exists(prompt_wav_path):
            tier1["reference_wav_path"] = prompt_wav_path

        # Tier 2 — reference only (no prompt_text), simpler path
        tier2 = dict(base_kwargs)
        if reference_wav_path and os.path.exists(reference_wav_path):
            tier2["reference_wav_path"] = reference_wav_path
        elif prompt_wav_path and os.path.exists(prompt_wav_path):
            tier2["reference_wav_path"] = prompt_wav_path

        # Tier 3 — pure voice design (no reference). Always works.
        tier3 = dict(base_kwargs)

        last_err = None
        for idx, attempt in enumerate((tier1, tier2, tier3), start=1):
            if idx == 2 and attempt == tier1:
                continue  # skip duplicate
            try:
                # Re-seed before EACH call so voice stays consistent
                self._set_voice_seed(voice_seed)
                wav = self._generate(attempt)
                sf.write(output_path, wav, self._sample_rate)
                if idx > 1:
                    log.warning(f"Tier {idx} fallback used for: {text[:40]}...")
                return output_path
            except Exception as e:
                last_err = e
                msg = str(e)
                if not _TIER1_DIAG_LOGGED and idx == 1:
                    _TIER1_DIAG_LOGGED = True
                    log.warning(
                        "First Tier-1 failure traceback (will log only once):\n"
                        + traceback.format_exc()
                    )
                if "k2_fsa" not in msg and "LazyModule" not in msg and "speechbrain" not in msg:
                    log.error(f"Synth failed (tier {idx}, non-retryable): {e}")
                    return None
                log.warning(f"Synth tier {idx} hit speechbrain/k2 issue, trying next tier...")

        log.error(f"Synth failed all tiers: {last_err}")
        return None

    def synthesize_segments(self, segments, output_dir,
                           speaker_refs=None, speaker_transcripts=None,
                           progress_callback=None, voice_seed: Optional[int] = None,
                           tts_speed: str = "balanced",
                           is_cross_lingual: bool = False,
                           target_lang: str = "ru",
                           cfg_override: Optional[float] = None,
                           steps_override: Optional[int] = None):
        """Run TTS in a SUBPROCESS.

        tts_speed: "quality" (tier 1→2→3, slow), "balanced" (2→3, ~3x faster,
        default), or "fast" (2→3 with a single generation pass, no re-rolls).
        is_cross_lingual: If True (source_lang != target_lang), bumps cfg_value
        and inference_timesteps for better adherence to target-language text.
        Cross-lingual cloning needs stronger guidance or VoxCPM drifts toward
        source-language phonetics.
        target_lang: Target language code (e.g. "ru"), used by Whisper QA
        to detect when TTS produced wrong-language output (a fatal defect
        that triggers regen with a fresh seed).
        cfg_override / steps_override: per-job values from the New dub form,
        overriding the global Settings → Voice & TTS ones for this run only.
        Falsy = not set. See resolve_guidance().
        """
        import json
        import subprocess
        import tempfile

        speaker_refs = speaker_refs or {}
        speaker_transcripts = speaker_transcripts or {}
        os.makedirs(output_dir, exist_ok=True)
        self._last_segment_error = ""  # fresh diagnosis per run
        self._last_run_stats = {}      # fresh aggregate stats per run

        # ─── REFERENCE AUDIO PREPROCESSING ─────────────────────────────
        # VoxCPM is very sensitive to reference quality. Raw diarization-
        # extracted clips often have: background music, breath noise, room
        # tone, uneven loudness. All of these degrade cloning quality — the
        # model "hears" the noise and tries to reproduce it, or drifts voice
        # identity trying to average noisy frames.
        #
        # FFmpeg chain we apply:
        #   1. afftdn=nr=12       — noise reduction, 12dB (gentle, preserves voice)
        #   2. highpass=f=80      — kill low rumble (mic handling, AC hum)
        #   3. lowpass=f=8000     — cut above 8kHz (VoxCPM samples at 16kHz Nyquist)
        #   4. loudnorm           — EBU R128 normalize to -23 LUFS, typical speech
        #   5. atrim=0:20         — trim to first 20s max (shorter refs work better
        #                           per VoxCPM README, "few seconds of clean speech")
        #   6. aresample=16000    — match VoxCPM's expected input rate
        normalized_refs = {}
        refs_norm_dir = os.path.join(output_dir, "_refs_16k")
        os.makedirs(refs_norm_dir, exist_ok=True)
        _REF_FILTER = (
            "afftdn=nr=12,"
            "highpass=f=80,"
            "lowpass=f=8000,"
            "loudnorm=I=-23:LRA=7:TP=-2,"
            "atrim=0:20,"
            "aresample=16000"
        )
        for spk, ref_path in speaker_refs.items():
            if not ref_path or not os.path.exists(ref_path):
                continue
            norm_path = os.path.join(refs_norm_dir, f"{spk}_16k.wav")
            try:
                import subprocess as _sp
                _sp.run(
                    ["ffmpeg", "-y", "-i", ref_path,
                     "-af", _REF_FILTER,
                     "-ac", "1", "-ar", "16000",
                     "-acodec", "pcm_s16le", norm_path],
                    check=True, capture_output=True, timeout=60,
                )
                normalized_refs[spk] = norm_path
                log.info(f"Preprocessed ref for {spk}: {os.path.basename(ref_path)} "
                         f"-> denoised/normalized/trimmed (16kHz)")
            except Exception as e:
                # If the complex chain fails, fall back to simple 16kHz mono
                log.warning(f"Full preprocessing failed for {spk}, trying simple: {e}")
                try:
                    _sp.run(
                        ["ffmpeg", "-y", "-i", ref_path,
                         "-ac", "1", "-ar", "16000",
                         "-acodec", "pcm_s16le", norm_path],
                        check=True, capture_output=True, timeout=60,
                    )
                    normalized_refs[spk] = norm_path
                except Exception as e2:
                    log.warning(f"Could not normalize ref for {spk}: {e2}; using original")
                    normalized_refs[spk] = ref_path
        speaker_refs = normalized_refs

        # Build the job spec
        seg_specs = []
        # Determine default fallback ref: if there's only one ref available
        # (user uploaded reference, or single-speaker case), use it for ALL
        # segments regardless of speaker ID mismatch. This fixes the bug where
        # speaker_refs had key "" (empty) but segments had speaker="SPEAKER_00"
        # and lookup returned None, falling through to voice design instead of
        # cloning the user's voice.
        _all_refs = [v for v in speaker_refs.values() if v]
        fallback_ref = _all_refs[0] if len(_all_refs) >= 1 else ""
        fallback_ptext = ""
        for _k, _v in speaker_transcripts.items():
            if _v:
                fallback_ptext = _v
                break

        for i, seg in enumerate(segments):
            # `tts_text` carries synth-only decoration (the Voice Design
            # "(style)" prefix). Only this engine understands it — an engine
            # that reads it aloud must keep using `translated_text`.
            text = (seg.get("tts_text")
                    or seg.get("translated_text")
                    or seg["text"])
            speaker = seg.get("speaker", "SPEAKER_00")
            out_file = os.path.join(output_dir, f"seg_{i:04d}.wav")
            # Try exact match first, then fall back to any available ref
            ref = speaker_refs.get(speaker) or fallback_ref
            ptext = (speaker_transcripts.get(speaker) or fallback_ptext or "")[:300]
            seg_specs.append({
                "idx": i,
                "text": text,
                "output_path": out_file,
                "reference_wav_path": ref,
                "prompt_wav_path": ref,
                "prompt_text": ptext,
            })

        if fallback_ref and not any(s.get("speaker") in speaker_refs for s in segments):
            log.info(f"Speaker IDs mismatch refs; using fallback ref for all segments: "
                     f"{os.path.basename(fallback_ref)}")

        # Speed-dependent inference steps: VoxCPM default is 10, but 6 is
        # ~40% faster with minimal quality loss for short text.
        #
        # cfg.voxcpm_steps (surfaced in Settings → Voice & TTS) overrides the
        # tier when set; 0 means "follow the tier", which is the default and
        # what every install got before this value reached synthesis at all.
        #
        # Cross-lingual dubbing (e.g. English video → Russian audio) needs
        # stronger guidance on top of that. Without it, VoxCPM sometimes drifts
        # toward source-language phonetics on ~20-30% of segments = "зажёванные
        # слова". Bumping cfg from 2.0 → 2.5 and steps from 8 → 14 trades ~60%
        # more inference time per segment for a MUCH higher success rate, which
        # is a net win because bad segments trigger retry_badcase anyway
        # (another full retry = same cost as just doing it right).
        #
        # resolve_guidance() arbitrates the three sources; see its docstring
        # for why a per-job override beats the cross-lingual floor.
        xling_cfg, xling_steps = 2.5, 14
        try:
            from app.config import cfg as _user_cfg
            xling_cfg = float(getattr(_user_cfg, "voxcpm_xling_cfg", xling_cfg))
            xling_steps = int(getattr(_user_cfg, "voxcpm_xling_steps", xling_steps))
        except Exception:
            pass  # keep the defaults if config is unavailable (bare worker)

        base_cfg, base_timesteps = resolve_guidance(
            self.cfg_value, self.effective_timesteps(tts_speed),
            is_cross_lingual, xling_cfg, xling_steps,
            cfg_override, steps_override,
        )
        if cfg_override or steps_override:
            log.info(
                f"Per-job VoxCPM override: cfg={base_cfg}, "
                f"timesteps={base_timesteps} "
                f"(cfg{'=job' if cfg_override else '=global'}, "
                f"steps{'=job' if steps_override else '=global'})"
            )
        elif is_cross_lingual:
            log.info(
                f"Cross-lingual mode: cfg={base_cfg}, timesteps={base_timesteps}"
            )

        # QA default: cross-lingual only (same-language cloning very rarely
        # produces gibberish). Users can opt in to same-language QA via
        # cfg.qa_same_language (env GOCHIDUBB_QA_SAME_LANGUAGE=1).
        try:
            from app.config import cfg as _user_cfg
            qa_same_language = bool(getattr(_user_cfg, "qa_same_language", False))
        except Exception:
            qa_same_language = False

        job = {
            "model_id": self.model_id,
            "cfg_value": base_cfg,
            "inference_timesteps": base_timesteps,
            "retry_badcase": tts_speed != "fast",
            "retry_badcase_max_times": SPEED_RETRIES.get(tts_speed, 1),
            "tier_policy": tts_speed,
            "voice_seed": voice_seed,
            "segments": seg_specs,
            "is_cross_lingual": is_cross_lingual,
            "target_lang": target_lang,
            "enable_qa": bool(is_cross_lingual or qa_same_language),
        }

        # Write job.json
        tmpdir = tempfile.mkdtemp(prefix="gochidubb_tts_")
        job_path = os.path.join(tmpdir, "job.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)

        # Locate tts_worker.py
        here = os.path.dirname(os.path.abspath(__file__))
        worker_candidates = [
            os.path.join(here, "tts_worker.py"),
            os.path.join(os.path.dirname(here), "tts_worker.py"),
        ]
        worker = next((p for p in worker_candidates if os.path.exists(p)), None)
        if not worker:
            log.error(
                f"tts_worker.py not found - place it at {worker_candidates[0]} "
                f"or the project root"
            )
            for seg in segments:
                seg["audio_path"] = None
            return segments

        log.info(f"TTS worker path: {worker}")
        total = len(segments)
        done = 0
        results = {}
        qa_unmeasured_warned = False

        # Force UTF-8 and disable noisy output in the child
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["TQDM_DISABLE"] = "1"
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        # Required for deterministic cuBLAS (used by attention in VoxCPM2).
        # Without this, torch.backends.cudnn.deterministic=True still allows
        # non-deterministic matmul → voice timbre shifts between segments.
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Check if we have a live daemon worker already
        is_first_call = (self._worker_proc is None or
                         self._worker_proc.poll() is not None)

        if is_first_call:
            # Redirect stderr to a file (NOT a pipe) so it can't deadlock.
            # tqdm/warnings can write kilobytes per second; a PIPE buffer
            # is only 8KB on Windows and will hang the subprocess once full.
            self._worker_stderr_path = os.path.join(tmpdir, "worker_stderr.log")
            self._worker_stderr_fh = open(
                self._worker_stderr_path, "w", encoding="utf-8", errors="replace"
            )

            # Log the stderr path up front: every real VoxCPM traceback lands
            # there, and without this line nothing in the server log points at
            # the temp dir holding it.
            log.info(
                f"Launching persistent TTS worker (daemon mode) — "
                f"worker stderr: {self._worker_stderr_path}"
            )
            self._worker_proc = subprocess.Popen(
                [sys.executable, "-u", worker, "--daemon", job_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._worker_stderr_fh,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=env,
            )
        else:
            # Reuse existing worker — just submit new job path via stdin.
            # This saves the 34s VoxCPM reload on every retry.
            log.info("Reusing persistent TTS worker (no reload)")
            try:
                self._worker_proc.stdin.write(job_path + "\n")
                self._worker_proc.stdin.flush()
            except Exception as e:
                log.warning(f"Worker stdin write failed, respawning: {e}")
                try:
                    self._worker_proc.kill()
                except Exception:
                    pass
                self._worker_proc = None
                # Recurse once with a fresh daemon
                return self.synthesize_segments(
                    segments, output_dir,
                    speaker_refs=speaker_refs,
                    speaker_transcripts=speaker_transcripts,
                    progress_callback=progress_callback,
                    voice_seed=voice_seed,
                    tts_speed=tts_speed,
                )

        proc = self._worker_proc

        try:
            # Read events until we see {"event": "job_done"} — that marks the
            # end of a single job in daemon mode. The worker then idles on
            # stdin waiting for the next job.
            while True:
                line = proc.stdout.readline()
                if not line:
                    # EOF — daemon died
                    log.error("TTS worker stdout closed unexpectedly")
                    break
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                kind = evt.get("event")
                if kind == "loading":
                    log.info(f"TTS worker: loading {evt.get('model')}")
                elif kind == "loaded":
                    log.info(
                        f"TTS worker: loaded in {evt.get('seconds')}s "
                        f"(SR={evt.get('sample_rate')})"
                    )
                    self._sample_rate = evt.get("sample_rate", 48000)
                elif kind == "segment_start":
                    log.info(
                        f"  [{evt.get('idx')+1}/{total}] synthesizing: "
                        f"{evt.get('text_preview', '')}..."
                    )
                elif kind == "segment":
                    done += 1
                    idx = evt.get("idx")
                    results[idx] = evt
                    tier = evt.get("tier", "?")
                    qa = evt.get("qa_score")
                    qa_str = f" qa={qa:.2f}" if isinstance(qa, (int, float)) else ""
                    if evt.get("ok"):
                        # tier=0 is our convention for "degraded — QA retries
                        # exhausted but accepted best attempt"
                        marker = "⚠ " if tier == 0 else ""
                        log.info(
                            f"[{done}/{total}] tier={tier}{qa_str} {marker}"
                            f"{evt.get('text_preview', '')}..."
                        )
                    else:
                        log.warning(
                            f"[{done}/{total}] FAILED: {evt.get('error')}"
                        )
                        # Keep the first failure so callers can report the real
                        # cause instead of a generic "check model/GPU".
                        if not self._last_segment_error:
                            self._last_segment_error = str(evt.get("error") or "")
                    if progress_callback:
                        progress_callback(done, total)
                elif kind == "qa_enabled":
                    log.info(f"QA: Whisper roundtrip enabled "
                             f"(target_lang={evt.get('target_lang')})")
                elif kind == "qa_check":
                    # Suppress individual qa_check logs; aggregated info already
                    # appears in the "segment" event above. Exception: warn ONCE
                    # per run when QA could not actually measure (whisper failed
                    # to load / ASR crashed) so "no bad scores" is never read as
                    # "all segments verified".
                    if evt.get("measured") is False and not qa_unmeasured_warned:
                        qa_unmeasured_warned = True
                        log.warning(
                            "QA could not measure segment quality (whisper "
                            "unavailable or ASR failed) — segments are accepted "
                            "without verification; qa scores will be null, not 0.0"
                        )
                elif kind == "qa_retry":
                    log.info(
                        f"  [qa] segment {evt.get('idx')+1} bad "
                        f"(score={evt.get('prev_score')}), "
                        f"retry {evt.get('attempt')} seed={evt.get('new_seed')}"
                    )
                elif kind == "qa_fallback":
                    log.warning(
                        f"  [qa] segment {evt.get('idx')+1} couldn't pass QA "
                        f"after all retries (score={evt.get('score')}) — using "
                        f"best attempt"
                    )
                elif kind == "done":
                    log.info(
                        f"TTS worker done: {evt.get('ok')}/{evt.get('total')} "
                        f"tiers={evt.get('tier_stats')} "
                        f"qa_regens={evt.get('qa_regens', 0)} "
                        f"qa_measured={evt.get('qa_measured_count', 0)} "
                        f"qa_unmeasured={evt.get('qa_unmeasured_count', 0)}"
                    )
                    # Keep for the server's perf metrics (see last_run_stats)
                    self._last_run_stats = {
                        "tier_stats": evt.get("tier_stats"),
                        "qa_regens": evt.get("qa_regens", 0),
                        "qa_measured_count": evt.get("qa_measured_count", 0),
                        "qa_unmeasured_count": evt.get("qa_unmeasured_count", 0),
                    }
                elif kind == "job_done":
                    # Single job finished in daemon mode; break out of read loop
                    # but DO NOT kill the worker — it's waiting for the next job.
                    break
                elif kind == "job_error":
                    log.error(f"TTS worker job error: {evt.get('error')}")
                    break
                elif kind == "fatal":
                    log.error(f"TTS worker fatal: {evt.get('error')}")
                    log.error(evt.get("traceback", ""))
                    break
        except Exception as e:
            log.error(f"Error reading TTS worker stdout: {e}")

        # NOTE: deliberately NOT closing stderr/stdout or waiting on proc —
        # the worker stays alive for the next job. See unload() for cleanup.

        # Attach outputs to segments. Also surface QA metadata so the server
        # can save it into tts_done checkpoint — the UI uses qa_score to
        # show quality badges in the per-segment review panel.
        for i, seg in enumerate(segments):
            r = results.get(i)
            if r and r.get("ok"):
                seg["audio_path"] = seg_specs[i]["output_path"]
                if r.get("qa_score") is not None:
                    seg["qa_score"] = r.get("qa_score")
                if r.get("tier") is not None:
                    seg["tts_tier"] = r.get("tier")
                # Full QA diagnostics (score/measured/cer/detected_lang/
                # lang_match/attempts/tier). score=None + measured=False
                # means "not measured", NOT "perfect".
                if r.get("qa") is not None:
                    seg["qa"] = r.get("qa")
            else:
                seg["audio_path"] = None

        if self._sample_rate is None:
            self._sample_rate = 48000

        return segments


# ═══════════════════════════════════════════════════════════════════════
#  CosyVoice 2 — placeholder for future integration
# ═══════════════════════════════════════════════════════════════════════
# CosyVoice 2 from FunAudioLLM offers native cross-lingual zero-shot
# cloning with ~2x the speed of VoxCPM2 on equivalent hardware, plus
# streaming mode. Worth adding as an alternative engine.
#
# Integration blockers (why this is still a stub):
#   1. CosyVoice has heavy deps (espnet, hyperpyyaml, conformer, matcha)
#      that may conflict with our existing torch/transformers versions.
#   2. The model downloads are ~5-8GB and need HF auth for some.
#   3. API is different enough (uses reference_text + reference_audio +
#      instruct_text triple) that we need non-trivial adaptation.
#
# This class acts as a friendly placeholder — if the user tries to
# select it in the UI without installing, we show actionable error
# messages rather than cryptic import failures. The interface matches
# BaseTTSEngine so actual integration is a drop-in later.
#
# To actually integrate:
#   pip install cosyvoice2
#   Add to get_tts_engine() factory in server.py
#   Remove "coming_soon" flag from UI dropdown
# ═══════════════════════════════════════════════════════════════════════

class CosyVoiceEngine(BaseTTSEngine):
    """Stub for CosyVoice 2 integration. Currently not functional —
    raises a helpful error if anyone tries to use it.

    When someone is ready to integrate CosyVoice 2:
      1. pip install cosyvoice2 (+ deps)
      2. Replace the _check_installed() stub with actual model init
      3. Implement synthesize_segments() to call cosyvoice's generate API
      4. Register in server.py get_tts_engine() factory
    """

    name = "cosyvoice2"
    default_sample_rate = 22050  # CosyVoice native SR

    INSTALL_HINT = (
        "CosyVoice 2 is not yet integrated in this build. To enable it:\n"
        "  1. pip install cosyvoice2 funasr modelscope\n"
        "  2. Download model: CosyVoice2-0.5B from ModelScope\n"
        "  3. See https://github.com/FunAudioLLM/CosyVoice for details\n"
        "For now, use VoxCPM2 (default) which offers similar quality."
    )

    def __init__(self, model_id="iic/CosyVoice2-0.5B"):
        super().__init__()
        self.model_id = model_id
        self._model = None

    def _check_installed(self):
        try:
            import cosyvoice2  # noqa: F401
            return True
        except ImportError:
            raise RuntimeError(self.INSTALL_HINT)

    def load(self):
        self._check_installed()
        # Real implementation here
        raise NotImplementedError("CosyVoice 2 integration pending")

    def unload(self):
        pass  # no-op for stub

    def synthesize_segments(self, segments, output_dir, **kwargs):
        self._check_installed()
        raise NotImplementedError("CosyVoice 2 integration pending")


class F5TTSEngine(BaseTTSEngine):
    """F5-TTS — zero-shot voice cloning, lighter than VoxCPM2.

    F5-TTS uses a flow-matching diffusion architecture. Requires ~3 GB VRAM
    (vs VoxCPM's 8 GB), making it usable on 4-6 GB cards. Quality is good
    for cross-lingual cloning; slightly below VoxCPM2 for same-language.

    Install:  pip install f5-tts

    In get_tts_engine() this sits between VoxCPM2 and Edge-TTS:
      VoxCPM2 (best, 8+ GB) → F5-TTS (good, 3+ GB) → Edge-TTS (no cloning)
    """

    name = "f5tts"
    default_sample_rate = 24000

    INSTALL_HINT = (
        "F5-TTS is not installed. To enable it:\n"
        "  pip install f5-tts\n"
        "Then set tts_engine=f5tts in config-user.json or the Settings tab."
    )

    def __init__(self, model_name: str = "F5-TTS"):
        super().__init__()
        self.model_name = model_name
        self._model = None
        self._sample_rate = 24000

    def _check_installed(self):
        try:
            import f5_tts  # noqa: F401
        except ImportError:
            raise RuntimeError(self.INSTALL_HINT)

    def load(self):
        if self._model is not None:
            return
        self._check_installed()
        import time
        t0 = time.time()
        log.info(f"Loading F5-TTS model: {self.model_name}")
        try:
            from f5_tts.api import F5TTS
            self._model = F5TTS(model_type=self.model_name)
            log.info(f"F5-TTS loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            log.error(f"F5-TTS load failed: {e}")
            raise

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            log.info("F5-TTS unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def synthesize_segments(self, segments, output_dir,
                           speaker_refs=None, speaker_transcripts=None,
                           progress_callback=None, voice_seed=None,
                           tts_speed="balanced",
                           is_cross_lingual=False, target_lang="ru",
                           cfg_override=None, steps_override=None):
        # cfg_override / steps_override are VoxCPM concepts; accepted so this
        # engine stays substitutable for it, ignored here.
        import soundfile as sf

        if self._model is None:
            self.load()

        self._check_installed()
        speaker_refs = speaker_refs or {}
        speaker_transcripts = speaker_transcripts or {}
        os.makedirs(output_dir, exist_ok=True)

        _all_refs = [v for v in speaker_refs.values() if v and os.path.exists(v)]
        fallback_ref = _all_refs[0] if _all_refs else None
        fallback_ptext = next(
            (v for v in speaker_transcripts.values() if v), ""
        )

        total = len(segments)
        for i, seg in enumerate(segments):
            text = (seg.get("translated_text") or seg.get("text", "")).strip()
            if not text:
                seg["audio_path"] = None
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            speaker = seg.get("speaker", "SPEAKER_00")
            ref_path = speaker_refs.get(speaker) or fallback_ref
            ref_text = (speaker_transcripts.get(speaker) or fallback_ptext)[:200]
            out_path = os.path.join(output_dir, f"seg_{i:04d}.wav")

            try:
                if ref_path and os.path.exists(ref_path):
                    wav, sr, _ = self._model.infer(
                        ref_file=ref_path,
                        ref_text=ref_text,
                        gen_text=text,
                        target_rms=0.1,
                        cross_fade_duration=0.15,
                        speed=1.0,
                    )
                else:
                    # No reference — use a generic style prompt
                    wav, sr, _ = self._model.infer(
                        ref_file=None,
                        ref_text="",
                        gen_text=text,
                    )
                sf.write(out_path, wav, sr)
                seg["audio_path"] = out_path
                self._sample_rate = sr
                log.info(f"[f5tts] [{i+1}/{total}] {text[:40]}...")
            except Exception as e:
                log.error(f"[f5tts] Segment {i} failed: {e}")
                seg["audio_path"] = None

            if progress_callback:
                progress_callback(i + 1, total)

        return segments


class EdgeTTSFallback(BaseTTSEngine):
    """Microsoft Edge-TTS fallback - no GPU, no cloning, always works.

    Used automatically when VoxCPM2 can't load (e.g. CUDA unavailable,
    model download failed, OOM). Cloning is not supported — output is
    generic Microsoft neural voices. Fast and reliable.
    """

    name = "edge-tts"
    default_sample_rate = 24000

    VOICE_MAP = {
        "en": "en-US-ChristopherNeural", "ru": "ru-RU-DmitryNeural",
        "es": "es-ES-AlvaroNeural", "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural", "zh": "zh-CN-YunxiNeural",
        "ja": "ja-JP-KeitaNeural", "ko": "ko-KR-InJoonNeural",
        "pt": "pt-BR-AntonioNeural", "ar": "ar-SA-HamedNeural",
        "hi": "hi-IN-MadhurNeural", "it": "it-IT-DiegoNeural",
        "tr": "tr-TR-AhmetNeural", "uk": "uk-UA-OstapNeural",
        "pl": "pl-PL-MarekNeural", "nl": "nl-NL-MaartenNeural",
        "sv": "sv-SE-MattiasNeural", "th": "th-TH-NiwatNeural",
        "vi": "vi-VN-NamMinhNeural", "cs": "cs-CZ-AntoninNeural",
        "ro": "ro-RO-EmilNeural", "hu": "hu-HU-TamasNeural",
        "bg": "bg-BG-BorislavNeural", "el": "el-GR-NestorasNeural",
        "fi": "fi-FI-HarriNeural", "id": "id-ID-ArdiNeural",
        "no": "nb-NO-FinnNeural", "da": "da-DK-JeppeNeural",
        # ── Wave 2: 38 more languages ──────────────────────────────────
        # Big-audience targets (bn/ur/fa/ta/… are edge-only — VoxCPM2 was
        # not trained on them; he/sw/tl/ms are in VoxCPM2's coverage)
        "bn": "bn-BD-PradeepNeural", "ur": "ur-PK-AsadNeural",
        "fa": "fa-IR-FaridNeural", "he": "he-IL-AvriNeural",
        "sw": "sw-KE-RafikiNeural", "tl": "fil-PH-AngeloNeural",
        "ms": "ms-MY-OsmanNeural", "ta": "ta-IN-ValluvarNeural",
        "te": "te-IN-MohanNeural", "mr": "mr-IN-ManoharNeural",
        "gu": "gu-IN-NiranjanNeural", "kn": "kn-IN-GaganNeural",
        "ml": "ml-IN-MidhunNeural",
        # European gap-fillers
        "sk": "sk-SK-LukasNeural", "hr": "hr-HR-SreckoNeural",
        "sr": "sr-RS-NicholasNeural", "sl": "sl-SI-RokNeural",
        "lt": "lt-LT-LeonasNeural", "lv": "lv-LV-NilsNeural",
        "et": "et-EE-KertNeural", "ca": "ca-ES-EnricNeural",
        "is": "is-IS-GunnarNeural", "af": "af-ZA-WillemNeural",
        "mk": "mk-MK-AleksandarNeural", "sq": "sq-AL-IlirNeural",
        "bs": "bs-BA-GoranNeural", "cy": "cy-GB-AledNeural",
        # Central Asia / Caucasus / mainland SE Asia
        "kk": "kk-KZ-DauletNeural", "az": "az-AZ-BabekNeural",
        "uz": "uz-UZ-SardorNeural", "ka": "ka-GE-GiorgiNeural",
        "mn": "mn-MN-BataaNeural", "ne": "ne-NP-SagarNeural",
        "si": "si-LK-SameeraNeural",
        "my": "my-MM-ThihaNeural", "km": "km-KH-PisethNeural",
        "lo": "lo-LA-ChanthavongNeural",
        # NOTE: "tl" maps to a fil-PH voice on purpose — Microsoft retired
        # the old tl-PH voices; Whisper's code is still "tl".
    }

    def __init__(self):
        super().__init__()
        self._sample_rate = 24000

    @property
    def is_loaded(self):
        return True

    def load(self): pass
    def unload(self): pass

    async def synthesize_segments_async(self, segments, output_dir, target_lang,
                                        voice="", progress_callback=None):
        import edge_tts
        os.makedirs(output_dir, exist_ok=True)
        lang = target_lang[:2].lower()
        selected = voice or self.VOICE_MAP.get(lang, "en-US-ChristopherNeural")
        total = len(segments)
        log.info(
            f"[edge-tts] synthesizing {total} segments to {target_lang} "
            f"(voice={selected}, out={output_dir})"
        )

        for i, seg in enumerate(segments):
            text = seg.get("translated_text", seg["text"])
            if not text.strip():
                seg["audio_path"] = None
                continue

            out_file = os.path.join(output_dir, f"seg_{i:04d}.mp3")
            try:
                comm = edge_tts.Communicate(text, selected)
                await comm.save(out_file)
                seg["audio_path"] = out_file
                log.debug(f"[edge-tts] [{i+1}/{total}] ok -> {os.path.basename(out_file)}")
            except Exception as e:
                log.error(f"edge-tts failed [{i}]: {type(e).__name__}: {e}")
                seg["audio_path"] = None

            if progress_callback:
                progress_callback(i + 1, total)

        ok = sum(1 for s in segments if s.get("audio_path"))
        log.info(f"[edge-tts] done: {ok}/{total} segments synthesized")
        return segments
