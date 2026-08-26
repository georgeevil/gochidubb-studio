#!/usr/bin/env python
"""GoChiDUBB Studio — MCP server for Claude Code and other agents.

Exposes the running GoChiDUBB HTTP API as Model Context Protocol tools.
The server must be running separately (start.bat or `python server.py`);
this MCP layer is a thin wrapper that translates tool calls into HTTP.

Setup:
    pip install "mcp[cli]"

Add to Claude Code MCP config (~/.claude.json or via `claude mcp add`):
    {
      "mcpServers": {
        "gochidubb": {
          "command": "python",
          "args": ["/path/to/gochidubb/tools/gochidubb_mcp.py"],
          "env": { "GOCHIDUBB_URL": "http://localhost:8910" }
        }
      }
    }

Then from Claude Code:
    "Dub https://youtu.be/abc into French and show me the result"
    → calls gochidubb_dub(source=..., target_lang="fr", wait=True)
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: mcp SDK not installed. Run: pip install \"mcp[cli]\"",
          file=sys.stderr)
    sys.exit(1)

from gochidubb_client import GoChiDUBBClient, DEFAULT_URL


mcp = FastMCP(
    "gochidubb",
    instructions=(
        "GoChiDUBB Studio — local AI video dubbing. The user has a running "
        "GoChiDUBB server. Use these tools to: submit dubs (one language or "
        "many), check status, build multilingual showcase reels (one video "
        "that cycles through N languages), or re-dub existing jobs without "
        "re-uploading. Always prefer `wait=True` when the user expects a "
        "finished video; otherwise return the job_id so they can poll. "
        "Source can be a YouTube/direct URL or a local file path. "
        "Also: discover trending videos (gochidubb_scout_trending), check "
        "for platform duplicates, inspect per-stage quality reports, and "
        "stage uploads for publishing — but the actual upload only happens "
        "through gochidubb_publish_approve, which requires explicit human "
        "confirmation first."
    ),
)


# A single client instance is reused across tool calls. asyncio handles
# concurrency. Recreated lazily so tools work even if the server starts
# AFTER the MCP process.
_client: Optional[GoChiDUBBClient] = None
_client_lock = asyncio.Lock()


async def _get_client() -> GoChiDUBBClient:
    global _client
    async with _client_lock:
        if _client is None:
            # Name the agent when one identifies itself, so the activity feed
            # can say "claude-code via MCP" rather than just "an agent".
            _agent = os.environ.get("GOCHIDUBB_AGENT") or "agent"
            _client = GoChiDUBBClient(
                base_url=os.environ.get("GOCHIDUBB_URL", DEFAULT_URL),
                client_id=f"mcp/{_agent}",
            )
        return _client


# ─────────────────────────────────────────────────────────────────────
# Submission tools — start work
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def gochidubb_dub(
    source: str,
    target_lang: str,
    source_lang: str = "auto",
    model: Optional[str] = None,
    voice_preset: str = "auto",
    voice_style: str = "",
    tts_speed: str = "balanced",
    keep_bg: bool = True,
    auto_denoise: bool = False,
    context_hint: str = "",
    wizard_mode: str = "auto",
    mode: str = "dub",
    scheduled_at: Optional[float] = None,
    voxcpm_cfg: float = 0.0,
    voxcpm_steps: int = 0,
    review_gates: Optional[dict] = None,
    wait: bool = False,
    wait_timeout: float = 1800.0,
) -> dict:
    """Dub one video into a single target language.

    Args:
        source: YouTube/direct URL or absolute local file path.
        target_lang: ISO code like 'fr', 'es', 'de', 'ja', 'pt', 'ru'.
        source_lang: 'auto' (default) or explicit code.
        model: Ollama translation model name. Default = server's default.
        voice_preset: Voice preset key, or 'auto' for source-cloned voice.
        voice_style: Free-text voice style hint (overrides preset cloning).
        tts_speed: 'fast' | 'balanced' | 'quality'.
        keep_bg: Preserve background music under the new dub (default: on).
        auto_denoise: Denoise the voice reference before cloning.
        context_hint: Free-text hint for translator (e.g. "tech podcast").
        wizard_mode: 'auto', or one of 'review_translation',
            'review_transcript', 'review_voices' — pause the job for human
            review at that checkpoint.
        mode: 'dub' (full pipeline) or 'reupload' (download + remux only —
            for music videos where dubbing makes no sense).
        scheduled_at: unix epoch seconds; a future timestamp parks the job
            as status='scheduled' and the server starts it at that time.
        voxcpm_cfg: Per-job VoxCPM guidance, 1.0-3.0. 0 (default) means
            "use the server's global setting". Raise it when a source is
            hard — heavy accent, noisy reference, awkward language pair.
        voxcpm_steps: Per-job VoxCPM inference steps, 4-24. 0 (default)
            follows the global setting and the tts_speed tier.
        review_gates: Per-stage pause config superseding wizard_mode, e.g.
            {"translation": "on", "subtitles": "flagged_only"} over gates
            transcript / translation / voice_cast / subtitles / final_qc
            (modes off | on | flagged_only). Each armed gate parks the job
            at an awaiting_* status; act on it (gochidubb_get_flags,
            gochidubb_edit_translations) then gochidubb_continue_job.
        wait: If True, block until job finishes and return final status + url.
            Ignored for scheduled jobs (they start later).
        wait_timeout: Max seconds to wait when wait=True.

    Returns dict with `job_id`. If wait=True, also `status` and `url` —
    or `pending_gate` when the job parked at a review gate instead of
    finishing.
    """
    c = await _get_client()
    res = await c.submit_dub(
        source, target_lang,
        source_lang=source_lang, model=model,
        voice_preset=voice_preset, voice_style=voice_style,
        tts_speed=tts_speed,
        keep_bg=keep_bg, auto_denoise=auto_denoise,
        context_hint=context_hint, wizard_mode=wizard_mode,
        mode=mode, scheduled_at=scheduled_at,
        voxcpm_cfg=voxcpm_cfg, voxcpm_steps=voxcpm_steps,
        review_gates=review_gates,
    )
    job_id = res.get("job_id")
    if wait and job_id and not res.get("scheduled_at"):
        final = await c.wait_for_job(job_id, timeout=wait_timeout)
        res["status"] = final.get("status")
        res["error"] = final.get("error")
        if final.get("pending_gate"):
            res["pending_gate"] = final["pending_gate"]
        if final.get("status") == "complete":
            res["url"] = c.output_url(job_id)
    return res


@mcp.tool()
async def gochidubb_compare(
    source: str,
    target_langs: list[str],
    trim_seconds: int = 60,
    source_lang: str = "auto",
    model: Optional[str] = None,
    voice_preset: str = "auto",
    tts_speed: str = "balanced",
    keep_bg: bool = True,
    voxcpm_cfg: float = 0.0,
    voxcpm_steps: int = 0,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> dict:
    """Quick-test mode: produce N separate dubs (one per language) for
    side-by-side comparison. Best for evaluating settings.

    target_langs: 2-6 language codes.
    trim_seconds: 15-120, default 60.
    keep_bg: background music/SFX kept by default.
    voxcpm_cfg / voxcpm_steps: per-job VoxCPM overrides; 0 = global setting.
    """
    c = await _get_client()
    res = await c.submit_compare(
        source, target_langs, trim_seconds=trim_seconds,
        source_lang=source_lang, model=model,
        voice_preset=voice_preset, tts_speed=tts_speed, keep_bg=keep_bg,
        voxcpm_cfg=voxcpm_cfg, voxcpm_steps=voxcpm_steps,
    )
    if wait and res.get("batch_id"):
        jobs = await c.wait_for_batch(res["batch_id"], timeout=wait_timeout)
        res["jobs"] = [{
            "id": j.get("id"), "title": j.get("title"),
            "lang": j.get("target_lang"),
            "status": j.get("status"), "error": j.get("error"),
            "url": c.output_url(j["id"]) if j.get("status") == "complete" else None,
        } for j in jobs]
    return res


@mcp.tool()
async def gochidubb_showcase(
    source: str,
    target_langs: list[str],
    trim_seconds: int = 60,
    source_lang: str = "auto",
    model: Optional[str] = None,
    voice_preset: str = "auto",
    tts_speed: str = "balanced",
    keep_bg: bool = True,
    voxcpm_cfg: float = 0.0,
    voxcpm_steps: int = 0,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> dict:
    """Multilingual showcase reel: N language segments stitched into ONE
    continuous video with a corner badge showing the current language.

    Use this when the user wants a single deliverable that demos voice
    cloning across languages (vs. gochidubb_compare which gives N files).

    target_langs: 2-6 codes. trim_seconds: 15-120.
    keep_bg: background music/SFX kept by default.
    voxcpm_cfg / voxcpm_steps: per-job VoxCPM overrides; 0 = global setting.
    """
    c = await _get_client()
    res = await c.submit_showcase(
        source, target_langs, trim_seconds=trim_seconds,
        source_lang=source_lang, model=model,
        voice_preset=voice_preset, tts_speed=tts_speed, keep_bg=keep_bg,
        voxcpm_cfg=voxcpm_cfg, voxcpm_steps=voxcpm_steps,
    )
    if wait and res.get("batch_id"):
        bid = res["batch_id"]
        await c.wait_for_batch(bid, timeout=wait_timeout)
        info = await c.wait_for_showcase(bid, timeout=wait_timeout)
        res["showcase_status"] = info.get("status")
        res["url"] = c.showcase_url(bid)
        res["manifest"] = info.get("manifest")
    return res


@mcp.tool()
async def gochidubb_redub(
    job_id: str,
    target_langs: list[str],
    mode: str = "compare",
    model: Optional[str] = None,
    voice_preset: Optional[str] = None,
    tts_speed: Optional[str] = None,
    voxcpm_cfg: Optional[float] = None,
    voxcpm_steps: Optional[int] = None,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> dict:
    """Re-dub an existing job's source into new language(s). Reuses the
    source file/URL — no upload. Inherits settings from the original.

    mode: 'single' (1 lang), 'compare' (2-6 separate dubs),
          'showcase' (2-6 stitched into one reel).
    voxcpm_cfg / voxcpm_steps: omit to inherit the original job's values;
          0 drops back to the server's global setting.
    """
    c = await _get_client()
    res = await c.redub(
        job_id, target_langs, mode=mode,
        model=model, voice_preset=voice_preset, tts_speed=tts_speed,
        voxcpm_cfg=voxcpm_cfg, voxcpm_steps=voxcpm_steps,
    )
    if wait:
        if res.get("batch_id"):
            bid = res["batch_id"]
            jobs = await c.wait_for_batch(bid, timeout=wait_timeout)
            res["jobs"] = [{
                "id": j["id"], "title": j.get("title"),
                "lang": j.get("target_lang"),
                "status": j.get("status"),
                "url": c.output_url(j["id"]) if j.get("status") == "complete" else None,
            } for j in jobs]
            if mode == "showcase":
                info = await c.wait_for_showcase(bid, timeout=wait_timeout)
                res["showcase_url"] = c.showcase_url(bid)
                res["showcase_manifest"] = info.get("manifest")
        elif res.get("job_id"):
            final = await c.wait_for_job(res["job_id"], timeout=wait_timeout)
            res["status"] = final.get("status")
            res["url"] = c.output_url(res["job_id"]) if final.get("status") == "complete" else None
    return res


# ─────────────────────────────────────────────────────────────────────
# Inspection — read-only
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def gochidubb_get_job(job_id: str) -> dict:
    """Get one job's full state (status, progress, error, etc.)."""
    c = await _get_client()
    j = await c.get_job(job_id)
    if j.get("status") == "complete":
        j["url"] = c.output_url(job_id)
    return j


@mcp.tool()
async def gochidubb_list_jobs(
    limit: int = 20,
    status: Optional[str] = None,
    batch_id: Optional[str] = None,
    since: float = 0,
) -> list[dict]:
    """List recent jobs, newest first (filtered server-side).

    status: filter by 'complete'|'error'|'queued'|'scheduled'|'running'|...
        Comma-separated sets work too ('queued,running').
    batch_id: filter to a specific batch (quick_test, showcase, redub).
    since: unix epoch — only jobs created at/after this time.
    Each job dict includes `title` (real video title when known).
    """
    c = await _get_client()
    return await c.list_jobs(limit=limit, status=status, batch_id=batch_id,
                             since=since)


@mcp.tool()
async def gochidubb_get_showcase(batch_id: str) -> dict:
    """Get assembly status of a showcase batch. Returns url when ready."""
    c = await _get_client()
    info = await c.get_showcase(batch_id)
    if info.get("status") == "ready":
        info["url"] = c.showcase_url(batch_id)
    return info


@mcp.tool()
async def gochidubb_rebuild_showcase(batch_id: str, wait: bool = False) -> dict:
    """Re-run the stitch step for an existing showcase batch. Useful if the
    auto-assembly failed but all per-language dubs are complete."""
    c = await _get_client()
    res = await c.rebuild_showcase(batch_id)
    if wait:
        info = await c.wait_for_showcase(batch_id, timeout=600.0)
        res["status"] = info.get("status")
        res["url"] = c.showcase_url(batch_id)
    return res


@mcp.tool()
async def gochidubb_get_voice_casting(job_id: str) -> dict:
    """Who speaks in this job, how each speaker is currently cast, and the
    voices available to cast them in.

    Available once the job has been translated. Returns per-speaker line
    counts and share of the dialogue, so you can tell the lead from a
    two-line walk-on before assigning anything.
    """
    c = await _get_client()
    return await c.get_voice_casting(job_id)


@mcp.tool()
async def gochidubb_set_voice_casting(job_id: str, cast: dict) -> dict:
    """Assign one voice per speaker: {"SPEAKER_00": "male_deep"}.

    A voice is a built-in preset id, a library voice ("file:<name>"), a
    free-text design ("design:gravelly old sailor"), or "source" to keep the
    voice cloned from the video. Any speaker left out of the map keeps their
    own voice — omission is not a reset.

    Prefer this over the whole-job `voice_preset` on any video with more than
    one speaker: a preset applies one voice to everybody and discards the
    references diarization extracted.
    """
    c = await _get_client()
    return await c.set_voice_casting(job_id, cast)


@mcp.tool()
async def gochidubb_preview_voice_casting(
    job_id: str,
    cast: Optional[dict] = None,
    per_speaker: int = 1,
) -> dict:
    """Synthesize a real line per speaker in a proposed cast and return the
    audio URLs. Writes nothing to the dub.

    Slow — it drives the TTS engine — but a rounding error next to the
    synthesis stage it lets you avoid re-running (a measured 12.25 hours for
    1602 segments). Worth doing before continuing a job parked at the voice
    review gate.
    """
    c = await _get_client()
    return await c.preview_voice_casting(job_id, cast, per_speaker=per_speaker)


@mcp.tool()
async def gochidubb_continue_job(job_id: str) -> dict:
    """Resume a job parked at a wizard review gate (awaiting_voice_review,
    awaiting_translation_review, awaiting_transcript_review)."""
    c = await _get_client()
    return await c.continue_job(job_id)


@mcp.tool()
async def gochidubb_cancel_job(job_id: str) -> dict:
    """Request cancellation of a queued or running job."""
    c = await _get_client()
    return await c.cancel_job(job_id)


@mcp.tool()
async def gochidubb_rescue_job(job_id: str, file_path: str) -> dict:
    """Attach a manually-downloaded video file to a job whose download
    failed (see the job's download_hint) and resume it from the extract
    stage."""
    c = await _get_client()
    return await c.attach_source(job_id, file_path)


@mcp.tool()
async def gochidubb_delete_job(job_id: str) -> dict:
    """Delete a job and its output files. Cannot be undone."""
    c = await _get_client()
    return await c.delete_job(job_id)


@mcp.tool()
async def gochidubb_system_status() -> dict:
    """Get server health: ffmpeg, GPU, Ollama, models, whisper, voxcpm, etc.

    Use this first if the user reports an unexpected failure — it surfaces
    missing dependencies (e.g. 'voxcpm.ok=False' or 'ollama not reachable')."""
    c = await _get_client()
    return await c.system_status()


@mcp.tool()
async def gochidubb_list_languages() -> list[str]:
    """Supported target language codes."""
    c = await _get_client()
    return await c.list_languages()


@mcp.tool()
async def gochidubb_list_models() -> list[str]:
    """Installed Ollama translation models on the user's machine."""
    c = await _get_client()
    return await c.list_models()


@mcp.tool()
async def gochidubb_list_voices() -> list[dict]:
    """Available voice presets registered in the server."""
    c = await _get_client()
    return await c.list_voices()


# ─────────────────────────────────────────────────────────────────────
# Quality & audit — read-only diagnostics
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def gochidubb_quality_report(job_id: str) -> dict:
    """Per-stage quality report for a job: 0-100 scores (asr, translation,
    tts, timing, loudness) + overall + actionable verdicts.

    Each verdict's `suggested_action` is now a tool you can actually call:
    edit_translations -> gochidubb_edit_translations, retranslate /
    retry_tts -> gochidubb_retry_stage (stage 'translate' or 'tts'), so
    you can act on the report mechanically. Stages with `available: false`
    were not measured — do not treat them as perfect."""
    c = await _get_client()
    return await c.get_quality(job_id)


@mcp.tool()
async def gochidubb_audit_job(job_id: str) -> dict:
    """Artifact-loss audit for a job: word coverage between pipeline stages,
    segment idx integrity, and TTS QA verdicts. Findings carry severity
    'loss' (real content lost), 'warn', or 'info', with the offending
    segments listed. `ok: false` means at least one loss finding."""
    c = await _get_client()
    return await c.audit(job_id)


@mcp.tool()
async def gochidubb_retry_stage(
    job_id: str,
    stage: str,
    stop_after: str = "",
    overrides: Optional[dict] = None,
) -> dict:
    """Re-run one pipeline stage (and everything after it) from the
    previous stage's checkpoint.

    Args:
        job_id: The job to retry.
        stage: One of download, extract, transcribe, diarize, translate,
            tts, assemble, merge. Costs follow the stage: translate is
            cheap, tts re-synthesizes.
        stop_after: Optionally halt after a later stage instead of running
            to the end (e.g. stage='translate', stop_after='translate' to
            inspect a retranslation before paying for TTS).
        overrides: Pipeline settings to change for this run, whitelisted
            server-side. Pass {"review_gates": {...}} to re-arm gates so
            the retried run pauses for review again.
    """
    c = await _get_client()
    return await c.retry_stage(job_id, stage, stop_after=stop_after,
                               overrides=overrides)


@mcp.tool()
async def gochidubb_get_flags(job_id: str, max_flags: int = 5) -> dict:
    """The handful of translated spans worth attention before synthesis —
    inconsistent renderings, untransliterated names, glossary misses.

    Recomputed on every call, so it reflects edits already applied through
    gochidubb_edit_translations. Use while a job is parked at the
    translation gate; fix what matters, then gochidubb_continue_job."""
    c = await _get_client()
    return await c.get_flags(job_id, max_flags=max_flags)


@mcp.tool()
async def gochidubb_edit_translations(job_id: str, edits: dict) -> dict:
    """Rewrite the translated line for one or more segments in the saved
    checkpoint. `edits` maps segment idx to the new text, e.g.
    {"41": "…y luego Marta Nieves vino al programa."}. Subtitles re-export
    automatically; follow with gochidubb_continue_job to proceed."""
    c = await _get_client()
    return await c.edit_translations(job_id, edits)


@mcp.tool()
async def gochidubb_add_glossary_term(
    term: str,
    target_lang: str,
    translation: str = "",
    say: str = "",
    domain: str = "",
) -> dict:
    """Teach the glossary one term for one target language: a rendering
    (`translation`), a pronunciation respelling (`say`, e.g. "goh-chee" —
    spoken by TTS but never shown in subtitles), or both. Applies to every
    later translation for that language, across jobs."""
    c = await _get_client()
    return await c.add_glossary_term(term, translation=translation,
                                     target_lang=target_lang,
                                     domain=domain, say=say)


# ─────────────────────────────────────────────────────────────────────
# Publish — stage/approve/cancel with a human gate
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def gochidubb_publish_stage(
    job_id: str,
    platform: str = "vk",
    export_preset: Optional[str] = None,
) -> dict:
    """Stage a completed dub for publishing. NEVER uploads anything.

    Builds title/description metadata (translated when available), runs a
    warn-only duplicate check against the platform, and optionally renders
    an export preset first. Returns the staged `publish` block including
    `duplicate_warnings` and `duplicate_verdict`
    (likely_duplicate|possible|clear|unchecked) — show these to the human
    before asking them to approve. The upload itself only happens after
    gochidubb_publish_approve."""
    c = await _get_client()
    return await c.publish_stage(job_id, platform=platform,
                                 export_preset=export_preset)


@mcp.tool()
async def gochidubb_publish_approve(
    job_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """HUMAN-APPROVAL GATE: approving a staged publish TRIGGERS the actual
    upload to the platform. Only call this after the human user has
    explicitly confirmed they want the video uploaded — never call it
    autonomously, even if staging looked clean. Optional title/description
    override the staged metadata before the upload is enqueued."""
    c = await _get_client()
    return await c.publish_approve(job_id, title=title, description=description)


@mcp.tool()
async def gochidubb_publish_cancel(job_id: str) -> dict:
    """Withdraw a staged/approved/failed publish so it never uploads.
    Fails with 409 while an upload is already in flight."""
    c = await _get_client()
    return await c.publish_cancel(job_id)


@mcp.tool()
async def gochidubb_publish_pending() -> list[dict]:
    """Review inbox: all publishes still needing (or doing) work — staged
    awaiting human approval, approved/uploading in progress, or failed.
    Each row has job_id, title and the full `publish` state."""
    c = await _get_client()
    return await c.publish_pending()


# ─────────────────────────────────────────────────────────────────────
# Scout — trending discovery + duplicate checking
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def gochidubb_scout_trending(
    category: Optional[str] = None,
    country: Optional[str] = None,
    limit: int = 20,
    include_shorts: bool = False,
) -> dict:
    """Trending YouTube video candidates (mostviewed.today, with a yt-dlp
    fallback — `source` in the response says which one served).

    Candidates carry title, channel, duration_sec, view_count, rank,
    `is_music` (category 10 — dub these with mode='reupload', not a voice
    dub) and `already_processed`/`existing_job_id` (this server has dubbed
    that video before — skip or redub instead)."""
    c = await _get_client()
    return await c.scout_trending(category=category, country=country,
                                  limit=limit, include_shorts=include_shorts)


@mcp.tool()
async def gochidubb_scout_dub(
    url_or_video_id: str,
    target_lang: str,
    mode: str = "auto",
    scheduled_at: Optional[float] = None,
) -> dict:
    """Send a scout candidate straight into the dub pipeline.

    mode 'auto' (default) routes music candidates to 'reupload' (download +
    remux, no voice dub — dubbing a music video makes no sense) and
    everything else to 'dub'. Pass 'dub' or 'reupload' to force.
    scheduled_at (unix epoch) defers the start to that time. Returns
    job_id, resolved mode, and any duplicate warnings from the pre-check."""
    c = await _get_client()
    return await c.scout_dub(url_or_video_id, target_lang, mode=mode,
                             scheduled_at=scheduled_at)


@mcp.tool()
async def gochidubb_check_duplicate(
    title: str,
    duration_sec: Optional[float] = None,
    alt_title: Optional[str] = None,
) -> dict:
    """Does a similar video already exist on the target platform (VK)?

    Warn-only metadata search: title similarity + duration within ±5s.
    Returns {matches, verdict: likely_duplicate|possible|clear}. Pass the
    original-language title as alt_title when checking a translated one —
    platform search recall is much better in the video's own language."""
    c = await _get_client()
    return await c.check_duplicate(title, duration_sec=duration_sec,
                                   alt_title=alt_title)


if __name__ == "__main__":
    # FastMCP defaults to stdio transport — exactly what Claude Code expects.
    mcp.run()
