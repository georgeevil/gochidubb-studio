"""Time-aligned audio assembly, per-segment + global loudness normalization."""
import json
import logging
import math
import os
import subprocess

from .ffmpeg_run import run_ffmpeg

log = logging.getLogger("gochidubb.assembler")

# Target peak after per-segment peak-normalization (avoids whisper-vs-shout jumps
# between consecutive TTS outputs from VoxCPM2 when fed different reference clips).
PER_SEGMENT_PEAK = 0.7

# EBU R128 loudnorm targets (broadcast-safe, closer to YouTube spec)
LN_I = -16      # integrated loudness LUFS
LN_TP = -1.5    # true peak dBTP
LN_LRA = 11     # loudness range


def _run(cmd, desc="", timeout=None):
    """Run ffmpeg with a soft timeout: extended while the encode reports
    progress, killed only once progress stalls (see pipeline/ffmpeg_run.py).
    Long renders therefore never die mid-encode at an arbitrary cutoff."""
    log.info(f"[{desc}]")
    return run_ffmpeg(cmd, desc=desc, logger=log, timeout=timeout)


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, output_path):
    """Write subtitles against the DUBBED timeline where we know it.

    A segment that had to be shifted or compressed to fit no longer sits at
    its source timestamp, and subtitles written from `start`/`end` drift out
    of sync with the audio by exactly that amount. `placed_start`/
    `placed_end` are set by assemble_dubbed_audio(); they're absent when the
    SRT is written for a source-language transcript that was never assembled,
    in which case the original timing is the right answer.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            text = seg.get("translated_text", seg["text"])
            start = seg.get("placed_start", seg["start"])
            end = seg.get("placed_end", seg["end"])
            f.write(f"{i+1}\n{format_srt_time(start)} --> "
                    f"{format_srt_time(end)}\n{text}\n\n")


def _parse_loudnorm_stderr(text: str) -> dict:
    """Extract the loudnorm measurement JSON that `print_format=json` writes
    to ffmpeg's stderr.

    ffmpeg prints it as the last `{ ... }` block after a
    `[Parsed_loudnorm_...]` banner, with every value as a *string* (including
    "-inf" for silent input, which is not valid JSON-float territory). Returns
    a dict of finite floats plus `normalization_type` as a string; {} when no
    parseable block is present. Never raises — this feeds a metrics field,
    not the render itself.
    """
    keep = ("input_i", "input_tp", "input_lra", "input_thresh",
            "output_i", "output_tp", "output_lra", "output_thresh",
            "target_offset")
    try:
        # Scan candidate JSON blocks last-to-first; the measurement block is
        # the final one and is the only one containing "input_i".
        end = len(text)
        while True:
            close = text.rfind("}", 0, end)
            if close < 0:
                return {}
            open_ = text.rfind("{", 0, close)
            if open_ < 0:
                return {}
            block = text[open_:close + 1]
            if '"input_i"' in block:
                try:
                    raw = json.loads(block)
                    break
                except ValueError:
                    pass
            end = open_
        out = {}
        for k in keep:
            try:
                v = float(raw[k])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(v):
                out[k] = v
        nt = raw.get("normalization_type")
        if isinstance(nt, str):
            out["normalization_type"] = nt
        return out
    except Exception as e:
        log.debug(f"loudnorm stderr parse failed: {e}")
        return {}


def _normalize_loudness_inplace(wav_path: str, target_i=None, target_tp=None):
    """Apply ffmpeg loudnorm + anti-screech chain to wav_path in place.

    `target_i`/`target_tp` override the integrated-loudness / true-peak
    targets; None falls back to the module constants LN_I/LN_TP, so
    standalone callers keep the historic behaviour unchanged.

    VoxCPM occasionally produces brief inter-segment pops/clicks and spiky
    transients that sound like "screech" or "digital crunch" to the ear.
    We add a 3-stage post-processing chain BEFORE loudnorm:

    1. adeclick    — removes isolated clicks/pops (<50ms transients)
    2. highpass=20 — cuts DC offset and sub-sonic rumble that can amplify
                     after loudnorm gain-up
    3. alimiter    — brick-wall limiter, catches any remaining transient
                     peaks above -1 dBTP that would otherwise clip post-loudnorm

    Then loudnorm itself for broadcast-level consistency. `print_format=json`
    makes loudnorm report its input/output measurements on stderr; that block
    is parsed and returned so the caller can persist real loudness numbers.

    Returns the measurement dict on success ({} when the stderr block could
    not be parsed — non-fatal) or None when normalization itself failed.
    """
    ti = LN_I if target_i is None else float(target_i)
    tp = LN_TP if target_tp is None else float(target_tp)
    ln = f"loudnorm=I={ti}:TP={tp}:LRA={LN_LRA}:print_format=json"
    try:
        tmp = wav_path + ".ln.wav"
        # Filter chain: declick → DC-block/rumble-cut → limit → normalize
        af = (
            "adeclick,"
            "highpass=f=20,"
            "alimiter=limit=0.95:level=disabled:attack=5:release=50,"
            + ln
        )
        proc = _run([
            "ffmpeg", "-y", "-i", wav_path,
            "-af", af,
            "-ar", "48000", tmp,
        ], "loudnorm+declick+limit")
        os.replace(tmp, wav_path)
        return _parse_loudnorm_stderr(proc.stderr or "")
    except Exception as e:
        log.warning(f"loudnorm+anti-screech failed, trying plain loudnorm: {e}")
        # Fallback: just loudnorm without the pre-chain
        try:
            tmp = wav_path + ".ln.wav"
            proc = _run([
                "ffmpeg", "-y", "-i", wav_path,
                "-af", ln,
                "-ar", "48000", tmp,
            ], "loudnorm-fallback")
            os.replace(tmp, wav_path)
            return _parse_loudnorm_stderr(proc.stderr or "")
        except Exception as e2:
            log.warning(f"loudnorm fallback also failed: {e2}")
            return None


def _atempo_stretch(wav_path: str, speed: float) -> str:
    """Time-stretch audio via ffmpeg atempo (preserves pitch, NOT np.interp which
    is pitch-shift = chipmunk effect). Returns path to stretched file."""
    if abs(speed - 1.0) < 0.02:
        return wav_path
    # A single atempo only accepts 0.5–2.0. We stay under 2.0 in practice, but
    # chain the filter rather than silently emitting an invalid one if a
    # configured cap ever goes past it.
    remaining, stages = speed, []
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    stages.append(remaining)
    chain = ",".join(f"atempo={s:.3f}" for s in stages)
    out = wav_path + f".{speed:.2f}x.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-filter:a", chain, out],
            check=True, capture_output=True, timeout=60,
        )
        return out
    except Exception as e:
        log.warning(f"atempo failed: {e}, using original")
        return wav_path


# Smallest silence left between two consecutive dubbed segments. Without it,
# compressing a segment to exactly the next one's start makes speakers run
# into each other with no breath at all.
MIN_SEGMENT_GAP = 0.08

# Stretch that listeners don't register as "sped up". Anything above this is
# still applied when it's the only way to hold sync, but it gets logged.
NATURAL_STRETCH = 1.15
_DEFAULT_MAX_STRETCH = 1.4


def plan_segment_fit(seg_start, next_start, tts_dur, current_end,
                     max_stretch=_DEFAULT_MAX_STRETCH):
    """Where a segment goes and how hard to compress it: (start, speed).

    VoxCPM has no speaking-rate control, so a segment comes out however long
    it comes out — against a fast-talking source that is routinely 1.5-2x its
    slot. Time-stretching after the fact is the only lever, and how hard to
    pull it is a timeline decision, which is why it lives here and not in the
    synthesizer.

    The old policy capped the stretch at 1.15x and let anything longer spill
    forward, with each spill pushing the next segment later. The error
    accumulated: measured on a 51s clip, the dub ran ~10s past the last source
    utterance and every pause between speakers had closed into one continuous
    wall of speech.

    Two things fix that. A segment's budget runs to the NEXT segment's start,
    not merely to its own `end` — the pause after someone stops talking is
    fair game and is usually where an overrun fits for free. And when a
    segment does have to start late its budget shrinks with it, so the
    compression that pays the drift back is applied automatically instead of
    being handed to the segment after it.

    `speed` is 1.0 when no stretching is needed; it is never below 1.0,
    because stretching short audio out to fill a slot only adds dead air.
    """
    start = seg_start
    if current_end > 0 and start < current_end + MIN_SEGMENT_GAP:
        start = current_end + MIN_SEGMENT_GAP

    room = next_start - MIN_SEGMENT_GAP - start
    if room <= 0.2 or tts_dur <= room * 1.02:
        return start, 1.0
    return start, min(tts_dur / room, max(max_stretch, 1.0))


def _max_stretch_setting() -> float:
    """Configured ceiling on time-compression (cfg.tts_max_stretch)."""
    try:
        from app.config import cfg as _user_cfg
        value = float(getattr(_user_cfg, "tts_max_stretch", _DEFAULT_MAX_STRETCH))
    except Exception:
        return _DEFAULT_MAX_STRETCH
    # Below 1.0 would mean "stretch it longer", which this code never does.
    return min(max(value, 1.0), 2.5)


def assemble_dubbed_audio(segments, total_duration, output_path,
                          sample_rate=48000, apply_loudnorm=True,
                          loudness_target=None, loudness_true_peak=None):
    """Place each TTS segment at its original timestamp (numpy-based mix).

    Handling of overlong TTS segments (Russian/Spanish are often 20-30%
    longer than English):
    - Speed up via ffmpeg atempo (preserves pitch) up to 1.15x MAX.
    - Anything that still won't fit is allowed to spill into the next
      segment's slot — slight overlap sounds FAR better than the chipmunk
      effect from aggressive pitch-shift.
    - Total audio may exceed total_duration; caller should NOT use -shortest.

    `loudness_target`/`loudness_true_peak` are handed to
    _normalize_loudness_inplace (None = the module defaults LN_I/LN_TP).

    Returns an info dict: {"output_path", "valid_count", "stretched_count",
    "loudnorm"} — loudnorm is the ffmpeg measurement parsed from
    `print_format=json` (None when loudnorm was skipped or failed).
    """
    import numpy as np
    import soundfile as sf

    # Compute how long the mixed track actually needs to be based on placed segments.
    # We allow extending beyond the source video length (+ a small safety).
    est_extra = 0.0
    for seg in segments:
        ap = seg.get("audio_path")
        if ap and os.path.exists(ap):
            try:
                info = sf.info(ap)
                tts_dur = info.frames / info.samplerate
                slot_dur = seg["end"] - seg["start"]
                if tts_dur > slot_dur:
                    est_extra += (tts_dur - slot_dur)
            except Exception:
                pass
    target_duration = total_duration + min(est_extra, 15.0) + 1.0

    n_samples = int(target_duration * sample_rate)
    mix = np.zeros(n_samples, dtype=np.float32)

    valid_count = 0
    stretched_count = 0
    hard_compressed = 0     # segments that needed more than NATURAL_STRETCH
    current_end = 0.0  # track cumulative end time to push next segments forward if needed
    base_max_stretch = _max_stretch_setting()

    # Where the next segment wants to start speaking. A segment's real budget
    # runs up to that point, not merely to its own `end` — the pause after
    # someone stops talking is fair game, and it is usually where an overrun
    # fits at no audible cost.
    next_starts = [segments[i + 1]["start"] for i in range(len(segments) - 1)]
    next_starts.append(total_duration)

    for seg_i, seg in enumerate(segments):
        audio_path = seg.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            continue

        try:
            # Quick duration read without full load
            info = sf.info(audio_path)
            tts_dur = info.frames / info.samplerate

            # Emotion-tagged segments are intentionally 10-20% longer (excited
            # speech draws out vowels, angry speech has more pauses, etc).
            # This is a *feature* not a bug — but it often pushes segments
            # out of their timing slot. When we detect an emotion tag, allow
            # a slightly more aggressive stretch to compensate.
            text = (seg.get("translated_text") or "").lstrip()
            has_emotion = text.startswith("(") and ")" in text[:30]
            max_stretch = base_max_stretch + (0.07 if has_emotion else 0.0)

            start, speed = plan_segment_fit(
                seg["start"], next_starts[seg_i], tts_dur, current_end,
                max_stretch,
            )

            stretched_path = audio_path
            if speed > 1.02:
                stretched_path = _atempo_stretch(audio_path, speed)
                stretched_count += 1
                if speed > NATURAL_STRETCH:
                    hard_compressed += 1
                    log.debug(
                        f"Segment {seg_i} compressed {speed:.2f}x "
                        f"({tts_dur:.2f}s of audio, placed at {start:.2f}s)"
                    )

            data, sr = sf.read(stretched_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Resample if needed (linear interp OK here since sr should match already)
            if sr != sample_rate:
                ratio = sample_rate / sr
                new_len = int(len(data) * ratio)
                idx = np.linspace(0, len(data) - 1, new_len)
                data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)

            # Per-segment peak normalization -> consistent volume across segments
            peak = float(np.abs(data).max())
            if peak > 0.01:
                data = data * (PER_SEGMENT_PEAK / peak)

            # ─── Anti-click fades ──────────────────────────────────
            # VoxCPM outputs sometimes start/end with a tiny DC-step or an
            # abrupt waveform edge. When we mix dozens of those together the
            # boundaries produce audible clicks. A 5ms cosine fade at each
            # end eliminates those without being perceptible as a volume
            # change.
            fade_samples = int(0.005 * sample_rate)  # 5ms
            if len(data) > fade_samples * 2:
                # Use cosine-shaped fade for smoother transient than linear
                fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
                fade_out = fade_in[::-1]
                data[:fade_samples] *= fade_in
                data[-fade_samples:] *= fade_out

            # `start` was decided above, before stretching — the budget and
            # the placement have to agree, or the compression would be sized
            # for a slot the segment doesn't actually land in.
            offset = int(start * sample_rate)
            end = min(offset + len(data), n_samples)
            length = end - offset
            if length > 0:
                mix[offset:offset + length] += data[:length]
                valid_count += 1
                current_end = start + length / sample_rate
                # Record where this segment actually lives in the dubbed track
                # (post-stretch, post-shift). Showcase reels use these to cut
                # at real word boundaries in each language rather than at the
                # original source timestamps where words have drifted.
                seg["placed_start"] = float(start)
                seg["placed_end"] = float(current_end)

        except Exception as e:
            log.warning(f"Skipped segment: {e}")
            continue

    if stretched_count > 0:
        log.info(f"Time-stretched {stretched_count}/{valid_count} segments (pitch preserved)")
    if hard_compressed > 0:
        log.info(
            f"{hard_compressed} segment(s) needed more than "
            f"{NATURAL_STRETCH:.2f}x to hold sync — the translations for those "
            f"are longer than their slots can carry"
        )

    # How far the dub ended up from the source timing. This was invisible
    # before, which is how a 10-second drift shipped: every individual
    # segment looked fine, only the accumulation was wrong.
    max_drift = max(
        (seg["placed_start"] - seg["start"]
         for seg in segments if "placed_start" in seg),
        default=0.0,
    )
    if max_drift > 0.5:
        log.warning(
            f"Dubbed speech drifts up to {max_drift:.1f}s behind the source "
            f"(raise cfg.tts_max_stretch, currently {base_max_stretch:.2f}, "
            f"or shorten the translations to reduce it)"
        )

    # Trim trailing silence beyond last actual audio (keep small tail)
    if current_end > 0 and current_end + 0.5 < target_duration:
        mix = mix[:int((current_end + 0.5) * sample_rate)]

    # Prevent clipping before loudnorm
    max_val = float(np.abs(mix).max())
    if max_val > 1.0:
        mix = mix / max_val * 0.95

    sf.write(output_path, mix, sample_rate, subtype="PCM_16")
    actual_dur = len(mix) / sample_rate
    log.info(f"Assembled {valid_count} segments -> {output_path} ({actual_dur:.1f}s)")

    # Global loudness normalization so YouTube/TV playback matches broadcast levels
    loudnorm_info = None
    if apply_loudnorm and valid_count > 0:
        loudnorm_info = _normalize_loudness_inplace(
            output_path, target_i=loudness_target, target_tp=loudness_true_peak)

    return {
        "output_path": output_path,
        "valid_count": valid_count,
        "stretched_count": stretched_count,
        "hard_compressed": hard_compressed,
        "max_drift_sec": round(float(max_drift), 2),
        "loudnorm": loudnorm_info or None,
    }


def _probe_video_stream(path):
    """(codec_name, pix_fmt) of the first video stream, or (None, None)."""
    try:
        import subprocess as _sp
        r = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        parts = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
    except Exception as e:
        log.warning(f"[merge] could not probe video stream: {e}")
    return None, None


# QuickTime, WhatsApp, Finder preview and most of the Apple ecosystem decode
# H.264 in 8-bit 4:2:0 and little else inside an .mp4. Anything wider — VP9,
# AV1, 10-bit, 4:4:4 — produces a structurally valid file those players
# silently refuse, while YouTube plays it because YouTube re-encodes on upload.
_SAFE_CODECS = ("h264", "avc1")
_SAFE_PIX_FMTS = ("yuv420p", "yuvj420p")


def _video_codec_args(video_path, must_encode, codec_pref, preset, crf):
    """ffmpeg video args: stream-copy when that is safe, otherwise re-encode.

    `must_encode` is set when a filter is in play (the frame-hold path), where
    copying is impossible regardless of the source.

    Copying is the fast path and stays the default *when the source is already
    playable*. A blanket re-encode would cost minutes on a long video for no
    gain; copying blindly is what shipped a VP9 stream inside an .mp4.
    """
    encode = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
              "-pix_fmt", "yuv420p"]
    if must_encode:
        return encode
    if codec_pref == "copy":
        return ["-c:v", "copy"]        # explicit user choice; speed over reach
    codec, pix_fmt = _probe_video_stream(video_path)
    if codec is None:
        # Probe failed — re-encode rather than gamble on the container.
        log.info("[merge] video stream unreadable; re-encoding to H.264")
        return encode
    if codec in _SAFE_CODECS and pix_fmt in _SAFE_PIX_FMTS:
        log.info(f"[merge] source is {codec}/{pix_fmt} — copying video stream")
        return ["-c:v", "copy"]
    log.info(f"[merge] source is {codec}/{pix_fmt}, which many players cannot "
             f"decode inside .mp4 — re-encoding to H.264/yuv420p")
    return encode


# Ducking curve for the music/SFX bed under the dub. The dub is loudnorm'd
# to -16 LUFS (LN_I), i.e. ~-18 dBFS RMS ~= 0.126 linear; a 0.03 threshold
# (~-30 dBFS) with ratio 8 yields ~11 dB of bed dip while speech is present.
# attack 20 ms catches word onsets without pumping; release 400 ms rides
# through inter-word gaps and brings the bed back smoothly in pauses.
DUCK_THRESHOLD = 0.03
DUCK_RATIO = 8
DUCK_ATTACK_MS = 20
DUCK_RELEASE_MS = 400


def build_audio_mix_filter(bg_volume: float, ducking: bool = True) -> str:
    """filter_complex mixing dub ([1:a]) with the separated bed ([2:a]) into [out].

    sidechaincompress takes [main][sidechain]: the bed is the main input and
    the dub is the key, so the dub must feed both the key and the final mix —
    hence the asplit. [dubm] stays the FIRST amix input, preserving the
    duration=first semantics of the legacy flat mix. bg_volume applies BEFORE
    the compressor so it keeps meaning "bed ceiling" (level when nobody
    speaks) and existing configs keep their meaning. Headroom: the dub
    true-peaks at -1.5 dBTP ~= 0.84, the ducked bed sits ~= 0.14, sum ~= 0.98
    — no limiter needed.
    """
    if not ducking:
        return (f"[1:a]volume=1.0[dub];[2:a]volume={bg_volume}[bg];"
                f"[dub][bg]amix=inputs=2:duration=first:normalize=0[out]")
    return (
        f"[1:a]asplit=2[dubm][dubsc];"
        f"[2:a]volume={bg_volume}[bgin];"
        f"[bgin][dubsc]sidechaincompress="
        f"threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}:"
        f"attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}:makeup=1[bgduck];"
        f"[dubm][bgduck]amix=inputs=2:duration=first:normalize=0[out]"
    )


def merge_audio_video(video_path, dubbed_audio_path, output_path,
                     background_audio_path="", bg_volume=None, bg_ducking=None,
                     codec_pref=None, preset=None, crf=None,
                     audio_bitrate=None):
    """Merge dubbed audio with video. If dubbed audio is longer than source
    video, freeze the last video frame to match (better than cutting off
    the end of the dub).

    Encoding options default to the user's Output settings. The video stream
    is copied when it is already H.264/yuv420p and re-encoded when it is not,
    so the result plays in QuickTime and WhatsApp rather than only in players
    that re-encode for you.
    """
    try:
        from app.config import cfg as _cfg
        if bg_volume is None:
            bg_volume = _cfg.background_volume
        if bg_ducking is None:
            bg_ducking = _cfg.bg_ducking
        codec_pref = codec_pref or _cfg.output_video_codec
        preset = preset or _cfg.output_preset
        crf = _cfg.output_crf if crf is None else crf
        audio_bitrate = audio_bitrate or _cfg.output_audio_bitrate
    except Exception:
        # Config unavailable (bare pipeline use) — fall back to what these
        # values were hardcoded to before they became settings.
        bg_volume = 0.5 if bg_volume is None else bg_volume
        bg_ducking = True if bg_ducking is None else bg_ducking
        codec_pref = codec_pref or "h264"
        preset = preset or "veryfast"
        crf = 23 if crf is None else crf
        audio_bitrate = audio_bitrate or "192k"
    # Check if audio is longer than video and we need to extend video
    try:
        import subprocess as _sp
        def _dur(path):
            r = _sp.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of",
                        "default=noprint_wrappers=1:nokey=1", path],
                       capture_output=True, text=True, timeout=30)
            return float(r.stdout.strip())
        v_dur = _dur(video_path)
        a_dur = _dur(dubbed_audio_path)
        need_extend = a_dur > v_dur + 0.3
    except Exception:
        need_extend = False
        v_dur = a_dur = 0

    # If audio longer -> extend video by holding last frame via tpad.
    # Otherwise use source video as-is (no -shortest; let it play to end of video).
    if need_extend:
        extend_by = a_dur - v_dur + 0.2
        log.info(f"Extending video by {extend_by:.2f}s "
                 f"(audio {a_dur:.1f}s vs video {v_dur:.1f}s)")
        tpad = f"tpad=stop_mode=clone:stop_duration={extend_by:.2f}"
    else:
        tpad = None

    vargs = _video_codec_args(video_path, bool(tpad), codec_pref, preset, crf)
    aargs = ["-c:a", "aac", "-b:a", audio_bitrate]

    if background_audio_path and os.path.exists(background_audio_path):
        if tpad:
            _run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", dubbed_audio_path,
                "-i", background_audio_path,
                "-filter_complex",
                f"[0:v]{tpad}[v];" + build_audio_mix_filter(bg_volume, bg_ducking),
                "-map", "[v]", "-map", "[out]",
                *vargs, *aargs,
                "-movflags", "+faststart",
                output_path,
            ], "merge with bg + extend")
        else:
            _run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", dubbed_audio_path,
                "-i", background_audio_path,
                "-filter_complex",
                build_audio_mix_filter(bg_volume, bg_ducking),
                "-map", "0:v", "-map", "[out]",
                *vargs, *aargs,
                "-movflags", "+faststart",
                output_path,
            ], "merge with bg")
    else:
        if tpad:
            _run([
                "ffmpeg", "-y",
                "-i", video_path, "-i", dubbed_audio_path,
                "-filter_complex", f"[0:v]{tpad}[v]",
                "-map", "[v]", "-map", "1:a",
                *vargs, *aargs,
                "-movflags", "+faststart",
                output_path,
            ], "merge a+v + extend")
        else:
            _run([
                "ffmpeg", "-y",
                "-i", video_path, "-i", dubbed_audio_path,
                "-map", "0:v", "-map", "1:a",
                *vargs, *aargs,
                "-movflags", "+faststart",
                output_path,
            ], "merge a+v")
    return output_path
