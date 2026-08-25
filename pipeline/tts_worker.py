"""VoxCPM subprocess worker.

Why this exists
───────────────
When VoxCPM's `model.generate()` is invoked with `reference_wav_path` AFTER
WhisperX/pyannote have already loaded `speechbrain` in the same process,
speechbrain's LazyModule for `speechbrain.integrations.k2_fsa` fails on
Windows (k2 has no Windows wheels). We can't fix this in-process — the
standalone `diagnose_voxcpm.py` script proves VoxCPM itself works perfectly
when speechbrain hasn't been pre-loaded.

Solution: run all TTS in a subprocess that never imports WhisperX/pyannote.
The parent process passes a JSON job file; the worker writes WAVs.

Usage (from parent):
    python tts_worker.py <job_json_path>

Where job_json_path points to a file like:
{
  "model_id": "openbmb/VoxCPM2",
  "cfg_value": 2.0,
  "inference_timesteps": 10,
  "voice_seed": 404,
  "segments": [
    {
      "idx": 0,
      "text": "Привет",
      "output_path": "C:/...seg_0000.wav",
      "reference_wav_path": "C:/.../ref.wav",
      "prompt_wav_path": "C:/.../ref.wav",
      "prompt_text": "The reference transcript."
    },
    ...
  ]
}

The worker prints one JSON line per segment to stdout so the parent can
display progress:
    {"event":"segment","idx":0,"ok":true,"tier":1}
    {"event":"segment","idx":1,"ok":true,"tier":1}
    ...
    {"event":"done","ok":7,"total":9}
"""
import json
import os
import random as _random
import shutil
import sys
import time
import traceback

# ═══════════════════════════════════════════════════════════════════
# CRITICAL: suppress everything that writes to stderr BEFORE importing
# VoxCPM. Otherwise tqdm/warnings fill the stderr pipe (8KB on Windows),
# the subprocess blocks on write, parent never drains it, DEADLOCK.
# ═══════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Instead of mocking tqdm (which breaks huggingface_hub's thread_map
# that uses tqdm.get_lock/get_instances), just patch tqdm to default
# to disable=True. The real API stays intact.
try:
    import tqdm
    _orig_tqdm_init = tqdm.tqdm.__init__

    def _silent_init(self, *args, **kwargs):
        kwargs.setdefault("disable", True)
        kwargs.setdefault("leave", False)
        return _orig_tqdm_init(self, *args, **kwargs)

    tqdm.tqdm.__init__ = _silent_init

    # Also patch tqdm.auto.tqdm the same way
    try:
        import tqdm.auto
        if tqdm.auto.tqdm is not tqdm.tqdm:
            _orig_auto_init = tqdm.auto.tqdm.__init__

            def _silent_auto_init(self, *args, **kwargs):
                kwargs.setdefault("disable", True)
                kwargs.setdefault("leave", False)
                return _orig_auto_init(self, *args, **kwargs)

            tqdm.auto.tqdm.__init__ = _silent_auto_init
    except Exception:
        pass
except Exception:
    pass


# The job this worker is currently processing. Stamped onto every event so
# the parent can tell whose output it is reading — see the matching comment
# in synthesizer.py. A daemon that handles job after job over one pipe has no
# other way to say "these lines are not yours".
_JOB_TOKEN = ""


def _log_event(**kw):
    if _JOB_TOKEN:
        kw.setdefault("token", _JOB_TOKEN)
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def _set_seed(seed):
    if seed is None:
        return
    try:
        import numpy as np
        import torch
        s = int(seed) & 0x7FFFFFFF
        _random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        # Force cuDNN to use deterministic algorithms so every segment
        # starts from the SAME noise sample → consistent voice identity
        # across all segments of the same dub. Without this, cuDNN uses
        # non-deterministic atomic ops even when the seed is identical,
        # making each segment sound like a slightly different speaker.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _record_qa(seg, score, transcript, diag, attempts, tier):
    """Attach QA outcome to the segment dict.

    seg['qa_score'] stays for backward compat (None when unmeasured);
    seg['qa'] carries the full diagnostics for checkpoints/UI.
    """
    diag = diag or {}
    seg["qa_score"] = score
    seg["qa_transcript"] = transcript
    seg["qa"] = {
        "score": score,
        "measured": bool(diag.get("measured", score is not None)),
        "cer": diag.get("cer"),
        "detected_lang": diag.get("detected_lang"),
        "lang_match": diag.get("lang_match"),
        "attempts": attempts,
        "tier": tier,
    }


def _synth_one(model, seg, base_kwargs, voice_seed, tier_policy="balanced",
               target_lang: str = "ru", enable_qa: bool = True):
    """Try selected tiers in order with optional post-synth QA.

    If QA is enabled and the generated audio fails quality check (CER too
    high, wrong language detected, etc.) we regenerate with a fresh seed
    up to 2 more times. This catches VoxCPM's occasional gibberish output
    that its own retry_badcase doesn't detect.

    A QA result of score=None means "not measured" (whisper unavailable
    or ASR crashed) — the segment is accepted as-is with NO retries and
    NO degraded fallback, and seg['qa']['measured'] is False so callers
    can tell "passed" apart from "couldn't check".

    Returns (ok, tier_used, err). On success sets seg['qa_score'],
    seg['qa_transcript'] and a seg['qa'] diagnostics dict.
    """
    import soundfile as sf

    ref = seg.get("reference_wav_path") or ""
    prompt = seg.get("prompt_wav_path") or ""
    prompt_text = seg.get("prompt_text") or ""
    text = seg["text"]
    out_path = seg["output_path"]

    # Voice Design mode detection: no reference audio AND text starts
    # with a "(style description)" prefix. In this mode every retry with
    # a different seed produces a DIFFERENT voice timbre (not just a
    # different take of the same voice) — which causes the final dub to
    # sound like multiple different speakers stitched together. So we
    # DISABLE seed-mutation retries for voice_design: take the first
    # attempt as-is, even if QA is marginal. Consistency > single-segment
    # quality when there's no reference pinning the timbre.
    has_ref = bool(ref and os.path.exists(ref))
    voice_design_mode = (not has_ref) and text.lstrip().startswith("(")

    # Tier 1 — full cloning (prompt + ref + prompt_text) -- SLOW + sometimes retries
    #
    # voxcpm requires prompt_wav_path and prompt_text to be BOTH set or BOTH
    # None (core.py: "must both be provided or both be None"), so prompt_wav_path
    # is only attached when we actually have a transcript. Setting the path with
    # an empty transcript raised ValueError before any inference — which happened
    # on every cross-lingual dub, since those deliberately clear
    # speaker_transcripts to force Controllable Cloning.
    tier1 = dict(base_kwargs, text=text)
    if prompt_text and prompt and os.path.exists(prompt):
        tier1["prompt_wav_path"] = prompt
        tier1["prompt_text"] = prompt_text
        tier1["reference_wav_path"] = prompt
    elif ref and os.path.exists(ref):
        tier1["reference_wav_path"] = ref
    elif prompt and os.path.exists(prompt):
        tier1["reference_wav_path"] = prompt

    # Tier 2 — reference ONLY (no prompt_text — that triggers retry_badcase
    # which can blow up to 30+ seconds per segment). Reference alone clones
    # the voice well enough at this speed tier.
    tier2 = dict(base_kwargs, text=text)
    if ref and os.path.exists(ref):
        tier2["reference_wav_path"] = ref
    elif prompt and os.path.exists(prompt):
        tier2["reference_wav_path"] = prompt

    # Tier 3 — pure voice design (fastest, no cloning)
    tier3 = dict(base_kwargs, text=text)

    has_real_prompt = bool(prompt_text and prompt and os.path.exists(prompt))

    # Voice Design mode: no reference at all. Using tier1 here is wasted
    # compute (no prompt to work with) AND sometimes produces subtly
    # different voices than tier3 with the same seed because cfg/timestep
    # values differ between tiers — breaks consistency across segments.
    # Force pure tier3 path so EVERY segment uses identical params.
    if voice_design_mode:
        tiers = [(3, tier3)]
    elif tier_policy == "quality":
        tiers = [(1, tier1), (2, tier2), (3, tier3)]
    elif tier_policy == "balanced":
        if has_real_prompt:
            tiers = [(1, tier1), (2, tier2), (3, tier3)]
        else:
            tiers = [(2, tier2), (3, tier3)]
    elif tier_policy == "fast":
        # Tier 2 (reference only, no prompt_text) IS the cheap cloning path —
        # it already skips the slow prompt-continuation work. Going straight to
        # tier 3 threw the extracted speaker refs away entirely and dubbed in an
        # arbitrary synthetic voice, and left no fallback when a tier errored.
        tiers = [(2, tier2), (3, tier3)]
    else:
        tiers = [(2, tier2), (3, tier3)]

    def _try_generate(kw, seed_for_this):
        """Single generate attempt with given kwargs and seed."""
        # In voice_design mode, we ALWAYS use voice_seed (not seed_for_this)
        # because timbre consistency across segments trumps per-segment
        # variation. Even if seed_for_this differs (from QA retry), we
        # override to keep the voice identical across all segments.
        if voice_design_mode:
            seed_for_this = voice_seed
        _set_seed(seed_for_this)
        dbg = {k: (os.path.basename(v) if isinstance(v, str) and os.path.isfile(v) else v)
               for k, v in kw.items() if k != "text"}
        _log_event(event="tier_attempt", idx=seg["idx"], tier=_current_tier,
                   args=dbg, seed=seed_for_this)
        wav = model.generate(**kw)
        if wav is None or len(wav) == 0:
            raise RuntimeError("empty audio")
        sf.write(out_path, wav, model.tts_model.sample_rate)
        if not first_saved[0]:
            try:
                shutil.copy2(out_path, first_path)
                first_saved[0] = True
            except Exception as e:
                _log_event(event="first_copy_failed", idx=seg["idx"], error=str(e))

    # Max QA-retries: regenerate with fresh seed if quality score too high.
    # Fast mode: no QA (would be slower than just accepting output).
    MAX_QA_RETRIES = 0 if tier_policy == "fast" else 2
    # In Voice Design mode (no reference audio) different seeds yield
    # different voice timbres. To keep ONE consistent speaker across the
    # whole dub, we skip seed-mutating retries here — even if QA score
    # is marginal. User can always per-segment regen a specific bad one
    # from the completion UI if needed.
    if voice_design_mode:
        MAX_QA_RETRIES = 0
    # In cloning mode (reference audio provided), QA retries with mutated
    # seeds produce different voice timbres even with the same reference —
    # especially in cross-lingual dubbing where the model is under higher
    # generative pressure. Concretely: segments that pass QA on attempt 1
    # use voice_seed; segments that fail and get a retry use
    # voice_seed+1001, voice_seed+2001 etc — an audibly different voice.
    # Result: dub sounds like two or more different speakers.
    # Fix: disable seed-mutation retries in cloning mode. If QA fails,
    # fall through to the NEXT TIER (which resets to voice_seed) rather
    # than accepting a different-voiced retry on the same tier.
    if has_ref and not voice_design_mode:
        MAX_QA_RETRIES = 0

    last_err = None
    best_score = None       # best REAL (measured) score seen so far
    best_transcript = ""
    best_diag = None
    best_path = out_path + ".best"   # audio behind best_score, see _run_qa
    # The very first attempt is the only one guaranteed to use the unmutated
    # voice_seed, and therefore the only one guaranteed to share a timbre with
    # every other segment of this speaker. Kept aside for the degraded path.
    first_path = out_path + ".first"
    first_saved = [False]
    attempts = 0            # QA-checked generate attempts for this segment

    def _run_qa(check_fn):
        nonlocal attempts, best_score, best_transcript, best_diag
        score, transcript, diag = check_fn(
            out_path, text, target_lang=target_lang,
        )
        attempts += 1
        measured = bool(diag.get("measured", score is not None))
        # Strictly-better keeps the EARLIEST attempt on a tie, which is the
        # one generated with the unmutated voice_seed — the retries below
        # deliberately change the seed, and with VoxCPM a different seed is a
        # different timbre.
        if measured and (best_score is None or score < best_score):
            best_score = score
            best_transcript = transcript
            best_diag = diag
            # Keep the audio too, not just the score. Every attempt overwrites
            # out_path, so without this the fallback ships the LAST retry while
            # reporting the BEST retry's score — and the last retry is the one
            # furthest from the original voice.
            try:
                shutil.copy2(out_path, best_path)
            except Exception as e:
                _log_event(event="best_copy_failed", idx=seg["idx"], error=str(e))
        _log_event(event="qa_check", idx=seg["idx"], tier=_current_tier,
                   score=round(score, 3) if score is not None else None,
                   measured=measured,
                   cer=diag.get("cer"),
                   detected_lang=diag.get("detected_lang"),
                   lang_match=diag.get("lang_match"),
                   transcript_preview=transcript[:60])
        return score, transcript, diag

    for idx, kw in tiers:
        _current_tier = idx  # closure-visible for _try_generate's log
        try:
            _try_generate(kw, voice_seed)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue

        # Post-synth QA. If acceptable, return. If bad, retry with fresh seed.
        if not enable_qa:
            return (True, idx, None)

        try:
            from pipeline.tts_qa import check_segment_quality, is_acceptable
        except ImportError:
            try:
                from tts_qa import check_segment_quality, is_acceptable
            except ImportError:
                # Module not deployed — skip QA gracefully
                return (True, idx, None)

        score, transcript, diag = _run_qa(check_segment_quality)

        if score is None:
            # QA could not measure this segment (whisper unavailable /
            # ASR crashed). Accept the audio as-is without any claim of
            # quality — retrying on an unmeasurable signal would burn
            # compute and (worse) risk voice drift for nothing.
            _record_qa(seg, None, transcript, diag, attempts, idx)
            return (True, idx, None)

        if is_acceptable(score):
            _record_qa(seg, score, transcript, diag, attempts, idx)
            return (True, idx, None)

        # Bad score — retry with different seeds on same tier before
        # falling through to next tier
        for retry_n in range(MAX_QA_RETRIES):
            new_seed = (voice_seed or 0) + 1000 * (retry_n + 1) + idx
            _log_event(event="qa_retry", idx=seg["idx"], tier=idx,
                       attempt=retry_n + 1, new_seed=new_seed,
                       prev_score=round(score, 3))
            try:
                _try_generate(kw, new_seed)
                score, transcript, diag = _run_qa(check_segment_quality)
                if score is None:
                    # QA went dark mid-retry — accept without claim.
                    _record_qa(seg, None, transcript, diag, attempts, idx)
                    return (True, idx, None)
                if is_acceptable(score):
                    _record_qa(seg, score, transcript, diag, attempts, idx)
                    return (True, idx, None)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue

    # All tiers + retries exhausted; if we produced *something*, accept it
    # with a warning. The alternative would be silence which is worse.
    #
    # Ship the BEST attempt, not the last one. The QA retries above vary the
    # seed on purpose, and with VoxCPM a different seed clones a noticeably
    # different voice — so accepting the final retry means a segment that both
    # failed QA *and* speaks in a stranger's voice, in the middle of an
    # otherwise consistent dub. best_path holds the audio that earned
    # best_score; restoring it keeps the reported score and the shipped audio
    # describing the same thing.
    # Prefer the FIRST take over the best-scoring one. QA scores transcription
    # accuracy — CER and language match — and never scores timbre, so "best
    # score" says nothing about whether the voice still matches the speaker.
    # Measured: a degraded Chinese segment shipped at 225 Hz against a 133 Hz
    # reference while being the better-scoring of its two takes.
    #
    # By the time we are here nothing passed QA, so choosing between failures
    # on CER is splitting hairs — but exactly one of them was generated with
    # the unmutated voice_seed and therefore matches every other segment of
    # this speaker. A stranger's voice mid-sentence is more jarring than a
    # slightly worse transcription of a line that was already going to be
    # wrong. This is the same trade voice_design mode already makes.
    restored = None
    for label, path in (("first", first_path), ("best", best_path)):
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            try:
                shutil.copy2(path, out_path)
                restored = label
                break
            except Exception as e:
                _log_event(event="restore_failed", idx=seg["idx"],
                           which=label, error=str(e))
    for path in (first_path, best_path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        _record_qa(seg, best_score, best_transcript,
                   best_diag or {"measured": best_score is not None},
                   attempts, 0)
        _log_event(event="qa_fallback", idx=seg["idx"],
                   score=round(best_score, 3) if best_score is not None else None,
                   restored=restored,
                   msg=f"all retries failed QA, using {restored}-attempt output"
                       if restored else
                       "all retries failed QA, using last output (nothing kept)")
        # Use the tier from the final attempt; log as tier 0 to mark "degraded"
        return (True, 0, None)

    return (False, -1, last_err)


def _load_model(job):
    """Load VoxCPM once per process (expensive, ~15-35s)."""
    from voxcpm import VoxCPM
    tier_policy = job.get("tier_policy", "balanced")
    use_denoiser = (tier_policy == "quality")
    _log_event(event="loading", model=job.get("model_id", "openbmb/VoxCPM2"),
               denoiser=use_denoiser, tier_policy=tier_policy)
    t0 = time.time()
    model = VoxCPM.from_pretrained(
        job.get("model_id", "openbmb/VoxCPM2"),
        load_denoiser=use_denoiser,
    )
    _log_event(event="loaded", seconds=round(time.time() - t0, 1),
               sample_rate=model.tts_model.sample_rate)
    return model


def _process_job(model, job):
    """Run TTS for all segments in a single job spec (already loaded model)."""
    base_kwargs = {
        "cfg_value": job.get("cfg_value", 2.0),
        "inference_timesteps": job.get("inference_timesteps", 10),
        "retry_badcase": job.get("retry_badcase", True),
        # Clamp to >= 1: voxcpm's `while retry_badcase_times < max` loop never
        # runs at 0, leaving `latent_pred` unbound and failing every segment.
        # Guards against old queued job.json files and any other producer.
        # See SPEED_RETRIES in synthesizer.py.
        "retry_badcase_max_times": max(1, int(job.get("retry_badcase_max_times", 2))),
    }
    voice_seed = job.get("voice_seed")
    tier_policy = job.get("tier_policy", "balanced")
    target_lang = job.get("target_lang", "ru")
    # Enable Whisper QA only for cross-lingual or when explicitly requested.
    # Same-language cloning very rarely produces gibberish, so skipping QA
    # there saves ~1-2s per segment of ASR overhead.
    enable_qa = bool(
        job.get("enable_qa", job.get("is_cross_lingual", False))
    )
    segments = job["segments"]
    total = len(segments)
    ok = 0
    qa_regens = 0  # count of segments that needed QA-triggered regen
    tier_stats = {0: 0, 1: 0, 2: 0, 3: 0}

    if enable_qa:
        _log_event(event="qa_enabled", target_lang=target_lang)

    for seg in segments:
        try:
            _log_event(event="segment_start", idx=seg["idx"],
                       text_preview=seg["text"][:40])
            success, tier, err = _synth_one(
                model, seg, base_kwargs, voice_seed,
                tier_policy=tier_policy,
                target_lang=target_lang,
                enable_qa=enable_qa,
            )
            # _synth_one keeps the best-scoring take beside the output while
            # it retries; drop it here so it is cleaned up on every path,
            # including the ones that raise.
            for _suffix in (".best", ".first"):
                try:
                    _side = seg["output_path"] + _suffix
                    if os.path.exists(_side):
                        os.remove(_side)
                except Exception:
                    pass
            if success:
                ok += 1
                tier_stats[tier] = tier_stats.get(tier, 0) + 1
                if tier == 0:
                    qa_regens += 1  # degraded = QA couldn't recover
                _log_event(event="segment", idx=seg["idx"], ok=True, tier=tier,
                           text_preview=seg["text"][:40],
                           qa_score=seg.get("qa_score"),
                           qa=seg.get("qa"))
            else:
                _log_event(event="segment", idx=seg["idx"], ok=False, error=err,
                           text_preview=seg["text"][:40])
        except Exception as e:
            _log_event(event="segment", idx=seg["idx"], ok=False,
                       error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc())

    # Honest QA accounting: how many segments got a REAL measurement vs.
    # how many were passed without a claim (whisper unavailable etc.).
    qa_measured = sum(
        1 for seg in segments if (seg.get("qa") or {}).get("measured")
    )
    qa_unmeasured = sum(
        1 for seg in segments
        if seg.get("qa") is not None and not seg["qa"].get("measured")
    )
    _log_event(event="done", ok=ok, total=total, tier_stats=tier_stats,
               qa_regens=qa_regens,
               qa_measured_count=qa_measured,
               qa_unmeasured_count=qa_unmeasured)


def main(job_path):
    """Single-shot mode: load model, process one job, exit."""
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    model = _load_model(job)
    _process_job(model, job)


def _run_one(model, job):
    """Process one job, always closing it with a job_done carrying its token.

    A job that ends without job_done leaves the parent blocked or, worse,
    reading this job's events as the next job's — which is exactly what
    happened: a run whose eight segments all synthesized cleanly was reported
    as "All 8 TTS segments failed" because the parent had already stopped
    reading, and the next run then consumed those eight events as its own.
    """
    global _JOB_TOKEN
    _JOB_TOKEN = str(job.get("job_token") or "")
    try:
        _process_job(model, job)
    except Exception as e:
        _log_event(event="job_error", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc())
    finally:
        # In the finally block on purpose: the parent's read loop ends on
        # job_done and nothing else is guaranteed to arrive.
        _log_event(event="job_done")
        _JOB_TOKEN = ""


def main_daemon():
    """Daemon mode: load model ONCE, then wait on stdin for job paths.
    Parent writes a line `<path_to_job.json>\\n` to our stdin; we process
    the job and print `{"event": "job_done"}` when finished. Exits on EOF
    or on the special line `SHUTDOWN`. First job path comes from argv[2]
    (same as single-shot mode, so first job bootstraps the model load)."""
    # First job bootstraps model + processes itself
    first_job_path = sys.argv[2] if len(sys.argv) > 2 else None
    if not first_job_path or not os.path.exists(first_job_path):
        _log_event(event="fatal", error="daemon mode requires first job path as argv[2]")
        return
    with open(first_job_path, "r", encoding="utf-8") as f:
        first_job = json.load(f)
    model = _load_model(first_job)
    _run_one(model, first_job)

    # Subsequent jobs arrive via stdin lines (one path per line)
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line == "SHUTDOWN":
            break
        if not os.path.exists(line):
            # No token to stamp — the parent cannot match this to its job, so
            # it is emitted untokenized and treated as a hard stop.
            _log_event(event="job_error", error=f"job file not found: {line}")
            continue
        try:
            with open(line, "r", encoding="utf-8") as f:
                job = json.load(f)
        except Exception as e:
            _log_event(event="job_error", error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc())
            continue
        _run_one(model, job)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"event": "fatal",
                          "error": "usage: tts_worker.py [--daemon] <job.json>"}))
        sys.exit(2)
    try:
        if sys.argv[1] == "--daemon":
            main_daemon()
        else:
            main(sys.argv[1])
    except Exception as e:
        _log_event(event="fatal", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc())
        sys.exit(1)
