"""Bridge between the live pipeline context and the reuse store (beta).

`app/reuse.py` hashes inputs and grades quality; `app/artifact_store.py`
persists. This module knows the shape of `server.py`'s pipeline context — what
`ctx["audio_16k"]` means, which files a stage writes — and turns it into the
fingerprint inputs and the artifact record.

It is kept out of both of those so they stay testable without a running
pipeline, and out of server.py so the reuse feature can be read (and removed)
in one place.

Everything here is a no-op when `cfg.reuse_enabled` is false.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app import artifact_store, reuse

log = logging.getLogger("gochidubb.reuse")

# What each stage produces: the context keys a reusing job must inherit, which
# of those hold paths, and any files not named by a path key.
#
# `files` is derived from `path_keys` at record time rather than hard-coded,
# because a stage's output filename is not fixed. `_stage_extract` sets
# `ctx["audio_16k"]` to audio_16k.wav, audio_16k_clean.wav or
# audio_16k_vad.wav depending on denoise and VAD — listing filenames by hand
# meant a reused job could inherit a path to a file that was never copied, and
# fail on a missing file with the artifact still recorded as good.
#
# `extra_files` is only for outputs nothing in ctx points at.
# `job_fields` are replayed onto the job on a hit: a stage that is skipped
# never runs its own update(), so without this a reused download leaves
# job["duration"] unset and downstream consumers silently lose it.
STAGE_ARTIFACTS: Dict[str, Dict[str, Any]] = {
    "download": {
        "ctx": ["video_path", "duration", "source_title", "source_info"],
        "path_keys": ["video_path"],
        "job_fields": ["duration"],
    },
    "extract": {
        "ctx": ["audio_16k", "audio_hq", "bg_audio_path", "duration"],
        "path_keys": ["audio_16k", "audio_hq", "bg_audio_path"],
        # The un-denoised original is kept even when ctx points past it: the
        # retry-stage path re-reads audio_16k.wav by name.
        "extra_files": ["audio_16k.wav"],
        "job_fields": [],
    },
    "transcribe": {
        "ctx": ["segments", "source_lang_detected", "effective_src", "asr"],
        "path_keys": [],
        "job_fields": ["segment_count", "source_lang_detected"],
    },
    "diarize": {
        "ctx": ["segments", "speaker_refs", "source_speaker_refs",
                "speaker_transcripts", "speakers", "transcript_raw",
                "diarization"],
        "path_keys": [],
        "dict_path_keys": ["speaker_refs", "source_speaker_refs"],
        "extra_files": ["speaker_refs"],
        "job_fields": ["speaker_count", "segment_count", "speakers",
                       "transcript_raw"],
    },
    "translate": {
        "ctx": ["segments", "srt_path"],
        "path_keys": ["srt_path"],
        "extra_files": ["subtitles.srt"],
        "job_fields": [],
    },
    "tts": {
        "ctx": ["segments"],
        "path_keys": [],
        "seg_path_keys": ["audio_path"],
        "extra_files": ["tts_segments"],
        "job_fields": [],
    },
}


def enabled_stages(cfg) -> Tuple[str, ...]:
    """Stages the user has allowed to be reused."""
    if not getattr(cfg, "reuse_enabled", False):
        return ()
    raw = getattr(cfg, "reuse_stages", "") or ""
    wanted = {s.strip().lower() for s in raw.split(",") if s.strip()}
    return tuple(s for s in reuse.CACHEABLE_STAGES if s in wanted)


def job_fields_for(stage: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Job fields a skipped stage would otherwise have published.

    A reused stage never runs its own `update()`, so without replaying these
    the job silently loses them: no `duration` after a reused download (which
    the publisher's duplicate check relies on), no `speaker_count` after a
    reused diarize. Derived from the now-populated context rather than read
    from stored keys, because several are computed rather than stored.
    """
    out: Dict[str, Any] = {}
    segments = ctx.get("segments") or []
    if stage == "download" and ctx.get("duration") is not None:
        out["duration"] = ctx["duration"]
    elif stage == "transcribe":
        out["segment_count"] = len(segments)
        if ctx.get("source_lang_detected"):
            out["source_lang_detected"] = ctx["source_lang_detected"]
    elif stage == "diarize":
        out["segment_count"] = len(segments)
        out["speaker_count"] = len(ctx.get("speaker_refs") or {}) or len(
            {s.get("speaker") for s in segments if s.get("speaker")})
        for key in ("speakers", "transcript_raw"):
            if ctx.get(key) is not None:
                out[key] = ctx[key]
    return out


def _rebase(value: str, old_dir: Path, new_dir: Path) -> str:
    """Point an absolute path at the reusing job's directory."""
    if not isinstance(value, str) or not value:
        return value
    try:
        rel = Path(value).resolve().relative_to(old_dir.resolve())
    except (ValueError, OSError):
        return value
    return str(new_dir / rel)


def _resolved_asr_backend(cfg) -> str:
    """The ASR backend a transcribe run would use right now ("auto" resolved).

    Resolved rather than read raw: with "auto", installing mlx-whisper
    switches engines without any setting changing.
    """
    try:
        from pipeline.transcriber import resolve_asr_backend
        return resolve_asr_backend(getattr(cfg, "asr_backend", "auto"))
    except Exception:
        return str(getattr(cfg, "asr_backend", "auto") or "auto")


def build_fingerprint_inputs(stage: str, ctx: Dict[str, Any],
                             cfg) -> Dict[str, Any]:
    """The determining inputs for one stage, read out of the live context.

    Content hashes are computed here rather than stored in ctx so a caller
    cannot accidentally fingerprint a stale one.
    """
    out: Dict[str, Any] = {}
    if stage in ("download", "extract"):
        src = ctx.get("video_path") or ""
        # Before download runs there is no local file; the URL is the identity.
        if stage == "download" and not src:
            out["source_fingerprint"] = reuse.hash_value(
                {"source": ctx.get("source"), "type": ctx.get("source_type")})
        else:
            out["source_fingerprint"] = reuse.hash_file(src, max_bytes=64 << 20)
        out["auto_denoise"] = bool(ctx.get("auto_denoise", True))
        out["keep_bg"] = bool(ctx.get("keep_bg", False))
        # Read from cfg, not ctx: _stage_extract reads cfg directly, so a
        # settings change has to move the fingerprint.
        out["vad_enabled"] = bool(getattr(cfg, "vad_enabled", True))
        out["vad_threshold"] = float(getattr(cfg, "vad_threshold", 0.5))
    elif stage in ("transcribe", "diarize"):
        out["audio_fingerprint"] = reuse.hash_file(ctx.get("audio_16k") or "")
        out["whisper_model"] = ctx.get("whisper_model") or getattr(
            cfg, "whisper_model", "")
        out["source_lang"] = ctx.get("source_lang") or "auto"
        out["speaker_mode"] = ctx.get("speaker_mode") or ""
        out["diarization_model"] = "pyannote/speaker-diarization-3.1"
        # "none" rather than "" when no voice was uploaded: an empty content
        # hash means "there was a file and we could not read it", which
        # refuses to cache. Having no reference at all is the normal case.
        ref = ctx.get("reference_audio") or ""
        out["reference_audio_fingerprint"] = reuse.hash_file(ref) if ref else "none"
        out["skip_diarization"] = bool(ctx.get("skip_diarization"))
        # _stage_diarize keeps prompt_text for a same-language dub and clears
        # it for a cross-lingual one, so the two are not interchangeable.
        out["same_language"] = (
            (ctx.get("effective_src") or "")[:2].lower()
            == (ctx.get("target_lang") or "")[:2].lower())
        out["initial_prompt"] = (ctx.get("initial_prompt") or "").strip()
        out["asr_backend"] = _resolved_asr_backend(cfg)
        out["min_speakers"] = ctx.get("min_speakers") or None
        out["max_speakers"] = ctx.get("max_speakers") or None
    elif stage == "translate":
        out["segments_fingerprint"] = reuse.hash_segments(ctx.get("segments"))
        out["target_lang"] = ctx.get("target_lang") or ""
        # Mirror _stage_translate's own resolution. Defaulting to "" here made
        # every job that does not override the model share one key, so
        # switching translation models served the old model's output.
        out["model"] = ctx.get("model") or getattr(cfg, "translation_model", "")
        out["context_hint"] = ctx.get("context_hint") or ""
        out["glossary_fingerprint"] = _glossary_fingerprint(
            ctx.get("target_lang") or "")
    elif stage == "tts":
        out["translation_fingerprint"] = reuse.hash_translations(
            ctx.get("segments"))
        refs = ctx.get("speaker_refs") or {}
        out["speaker_refs_fingerprint"] = reuse.hash_value(
            {k: reuse.hash_file(v) for k, v in sorted(refs.items())})
        # The engine is resolved from cfg, never placed in ctx, so reading
        # ctx alone fingerprinted every job as engine=None.
        out["tts_engine"] = (ctx.get("tts_engine")
                             or getattr(cfg, "tts_engine", ""))
        out["tts_speed"] = ctx.get("tts_speed") or getattr(cfg, "tts_speed", "")
        for key in ("voice_preset", "voice_style", "voice_seed", "target_lang"):
            out[key] = ctx.get(key)
    return out


def _glossary_fingerprint(target_lang: str) -> str:
    from pipeline.translator import _glossary_for
    try:
        return reuse.hash_value(_glossary_for(target_lang))
    except Exception:
        return reuse.hash_value({})


def fingerprint_for(stage: str, ctx: Dict[str, Any], cfg) -> Optional[str]:
    """Fingerprint for a stage about to run, or None if it cannot be keyed."""
    if stage not in reuse.CACHEABLE_STAGES:
        return None
    try:
        return reuse.stage_fingerprint(
            stage, build_fingerprint_inputs(stage, ctx, cfg))
    except Exception as e:            # never let caching break a run
        log.warning(f"[reuse] fingerprint failed for {stage}: {e}")
        return None


def try_reuse(stage: str, fingerprint: str, ctx: Dict[str, Any],
              work: Path, output_root: Path, cfg) -> Optional[Dict[str, Any]]:
    """Populate `ctx` from a cached artifact, or return None to run the stage.

    Returns the store entry on a hit, after materializing its files into
    `work` and rewriting any absolute paths to point there.

    Any failure — a locked database, an unreadable row, a full disk — returns
    None and lets the stage run. Reuse is an optimisation, and an optimisation
    that can fail a job is worse than no optimisation.
    """
    try:
        return _try_reuse(stage, fingerprint, ctx, work, output_root, cfg)
    except Exception as e:
        log.warning(f"[reuse] lookup failed for {stage}, running it instead: "
                    f"{type(e).__name__}: {e}")
        return None


def _try_reuse(stage: str, fingerprint: str, ctx: Dict[str, Any],
               work: Path, output_root: Path, cfg) -> Optional[Dict[str, Any]]:
    entry = artifact_store.lookup(fingerprint, stage, output_root)
    if not entry:
        return None

    ok, why = reuse.passes_gate(stage, entry.get("quality"),
                                reuse.gates_from_config(cfg))
    if not ok:
        log.info(f"[reuse] {stage} {fingerprint[:12]} rejected by quality "
                 f"gate: {why}")
        return None

    if not artifact_store.copy_artifacts(entry, work):
        log.warning(f"[reuse] {stage} {fingerprint[:12]} files could not be "
                    f"materialized; running the stage instead")
        return None

    spec = STAGE_ARTIFACTS.get(stage, {})
    old_dir = entry["source_dir"]
    cached_ctx = entry.get("ctx") or {}
    for key, value in cached_ctx.items():
        if key in spec.get("path_keys", []):
            value = _rebase(value, old_dir, work)
        elif key in spec.get("dict_path_keys", []) and isinstance(value, dict):
            value = {k: _rebase(v, old_dir, work) for k, v in value.items()}
        ctx[key] = value

    for seg_key in spec.get("seg_path_keys", []):
        for seg in ctx.get("segments") or []:
            if seg.get(seg_key):
                seg[seg_key] = _rebase(seg[seg_key], old_dir, work)

    entry["job_fields"] = job_fields_for(stage, ctx)
    artifact_store.mark_hit(fingerprint, stage)
    entry["reason"] = why
    log.info(f"[reuse] {stage}: reusing work from job {entry['job_id']} "
             f"({fingerprint[:12]})")
    return entry


def record_stage(stage: str, fingerprint: str, job_id: str,
                 ctx: Dict[str, Any], work: Path) -> Optional[Dict[str, Any]]:
    """Store a freshly computed stage so later jobs can reuse it.

    Quality is measured now, while the evidence is still here. Returns the
    quality dict, or None when nothing was recorded.

    The stage has already succeeded by the time this runs, so a failure to
    record it must not retroactively fail the job — the cost is a future cache
    miss, nothing more.
    """
    try:
        return _record_stage(stage, fingerprint, job_id, ctx, work)
    except Exception as e:
        log.warning(f"[reuse] could not record {stage}: "
                    f"{type(e).__name__}: {e}")
        return None


def _record_stage(stage: str, fingerprint: str, job_id: str,
                  ctx: Dict[str, Any], work: Path) -> Optional[Dict[str, Any]]:
    spec = STAGE_ARTIFACTS.get(stage)
    if not spec or not fingerprint:
        return None
    ctx_keys = {k: ctx[k] for k in spec["ctx"] if k in ctx}

    # Files come from the paths the stage actually resolved, plus any extras
    # nothing in ctx names. Hard-coding filenames meant a reused job could
    # inherit a path to a file that was never copied.
    files: List[str] = []

    def _add(path: Any) -> None:
        if not isinstance(path, str) or not path:
            return
        try:
            rel = Path(path).resolve().relative_to(work.resolve())
        except (ValueError, OSError):
            return                     # outside the job dir; not ours to cache
        if str(rel) not in files and (work / rel).exists():
            files.append(str(rel))

    for key in spec.get("path_keys", []):
        _add(ctx.get(key))
    for key in spec.get("dict_path_keys", []):
        for value in (ctx.get(key) or {}).values():
            _add(value)
    for seg_key in spec.get("seg_path_keys", []):
        for seg in ctx.get("segments") or []:
            _add(seg.get(seg_key))
    for rel in spec.get("extra_files", []):
        if (work / rel).exists() and rel not in files:
            files.append(rel)

    # Stages before `translate` must never carry a translation with them.
    # Their fingerprints do not include the target language — that is the whole
    # point — so a stray `translated_text` on a recorded segment would be
    # handed to a job dubbing into a *different* language, which would then
    # skip translating and speak the wrong one. Nothing would log a thing.
    if stage in ("transcribe", "diarize") and "segments" in ctx_keys:
        ctx_keys["segments"] = [
            {k: v for k, v in seg.items() if k != "translated_text"}
            for seg in ctx_keys["segments"] or []
        ]
    try:
        quality = reuse.evaluate_quality(stage, ctx)
    except Exception as e:
        log.warning(f"[reuse] quality check failed for {stage}: {e}")
        quality = {}

    # A stage that is already known to be bad is recorded anyway, with its
    # score: the gate refuses it at read time, and the row still tells the
    # beta UI why nothing is being reused.
    artifact_store.record(fingerprint, stage, job_id, files=files,
                          ctx_keys=_jsonable(ctx_keys), quality=quality)
    return quality


def _jsonable(data: Dict[str, Any]) -> Dict[str, Any]:
    import json
    out = {}
    for k, v in data.items():
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


def plan(ctx: Dict[str, Any], output_root: Path, cfg) -> List[Dict[str, Any]]:
    """What a job with this context would reuse, without running anything.

    Only the stages whose inputs are already knowable can be answered: a
    transcribe fingerprint needs the extracted audio, which does not exist
    until extract has run. Those report `known: false` rather than guessing.
    """
    rows = []
    allowed = enabled_stages(cfg)
    for stage in reuse.CACHEABLE_STAGES:
        row: Dict[str, Any] = {"stage": stage, "allowed": stage in allowed}
        fp = fingerprint_for(stage, ctx, cfg) if stage in allowed else None
        row["known"] = bool(fp)
        if fp:
            row["fingerprint"] = fp
            entry = artifact_store.lookup(fp, stage, output_root)
            if entry:
                ok, why = reuse.passes_gate(
                    stage, entry.get("quality"), reuse.gates_from_config(cfg))
                row.update(hit=True, would_reuse=ok, job_id=entry["job_id"],
                           quality=entry.get("quality"), blocked_by=why)
            else:
                row.update(hit=False, would_reuse=False)
        rows.append(row)
    return rows
