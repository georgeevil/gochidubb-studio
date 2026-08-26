"""GoChiDUBB Studio — async HTTP client shared by the CLI and MCP server.

Wraps the FastAPI endpoints in `server.py` with a typed Python interface.
Both `gochidubb_cli.py` and `gochidubb_mcp.py` use this so we keep the
HTTP contract in one place.

Server URL is configurable via env var `GOCHIDUBB_URL` (default
http://localhost:8910). The GoChiDUBB server must already be running —
this client does not start it.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import httpx


DEFAULT_URL = os.environ.get("GOCHIDUBB_URL", "http://localhost:8910")

# Active statuses — used by wait_for_completion to know when to stop polling.
_ACTIVE = {
    "queued", "scheduled", "running", "downloading", "extracting",
    "transcribing", "translating", "synthesizing", "assembling", "merging",
}
_TERMINAL = {"complete", "error", "cancelled"}


class GoChiDUBBError(RuntimeError):
    """Raised when the server returns a non-2xx response or a JSON error field."""


class GoChiDUBBClient:
    """Async client. Use as `async with GoChiDUBBClient() as c:` or call
    `await c.aclose()` manually."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 120.0,
                 client_id: str = "cli"):
        self.base_url = base_url.rstrip("/")
        # Who is calling. The server records header-carrying requests in its
        # activity feed as agent tool calls, which is the only way it can tell
        # an MCP-driven dub from someone clicking Start in the browser: both
        # are the same POST to the same route. Purely informational — it grants
        # nothing and is never used for authorization.
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={"X-GoChiDUBB-Client": client_id},
        )

    async def __aenter__(self) -> "GoChiDUBBClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── low-level helpers ────────────────────────────────────────────
    @staticmethod
    def _is_url(s: str) -> bool:
        return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

    async def _request(self, method: str, path: str, **kw) -> dict:
        r = await self._http.request(method, f"{self.base_url}{path}", **kw)
        try:
            data = r.json()
        except Exception:
            r.raise_for_status()
            return {}
        if r.status_code >= 400:
            err = data.get("error") or data.get("detail") or f"HTTP {r.status_code}"
            raise GoChiDUBBError(err)
        return data

    @staticmethod
    def _prompt_headers(prompt: Optional[str]) -> Optional[dict]:
        """Per-request headers carrying the natural-language request behind a
        submission, for the server's activity feed to quote on the run card.

        Percent-encoded because HTTP headers cannot carry raw UTF-8 and
        prompts routinely will (a Russian dub starts with a Russian
        sentence). The server unquotes and truncates on its side; purely
        informational either way, like X-GoChiDUBB-Client.
        """
        p = (prompt or "").strip()
        if not p:
            return None
        return {"X-GoChiDUBB-Prompt": urllib.parse.quote(p[:300], safe="")}

    @staticmethod
    def _source_fields(source: str) -> tuple[Optional[dict], dict]:
        """Build (files, form) tuple from a source path or URL.

        URLs go into the `source` form field (yt-dlp downloads them).
        Local file paths become a multipart `video` upload.
        """
        if GoChiDUBBClient._is_url(source):
            return None, {"source": source}
        p = Path(source).expanduser().resolve()
        if not p.exists():
            raise GoChiDUBBError(f"Source file not found: {p}")
        # httpx will close the file handle when the request finishes
        f = open(p, "rb")
        return {"video": (p.name, f, "application/octet-stream")}, {}

    # ── dub / batch / quick test / showcase / redub ───────────────────
    async def submit_dub(
        self,
        source: str,
        target_lang: str,
        *,
        source_lang: str = "auto",
        model: Optional[str] = None,
        whisper_model: str = "large-v3",
        voice_preset: str = "auto",
        voice_style: str = "",
        tts_speed: str = "balanced",
        speaker_mode: str = "main",
        keep_bg: bool = True,
        auto_denoise: bool = False,
        context_hint: str = "",
        wizard_mode: str = "auto",
        mode: str = "dub",
        scheduled_at: Optional[float] = None,
        voxcpm_cfg: float = 0.0,
        voxcpm_steps: int = 0,
        review_gates: Optional[dict] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """Submit a single-language dub. Returns dict with `job_id`.

        prompt: the natural-language request that led to this call, quoted on
        the server's activity-feed run card. Optional, informational only.

        review_gates: per-stage pause config, e.g. {"translation": "on",
        "subtitles": "flagged_only"} over gates transcript / translation /
        voice_cast / subtitles / final_qc. Sending it supersedes
        wizard_mode. Each armed gate parks the job at an awaiting_* status
        until /continue.

        mode: 'dub' (full pipeline) or 'reupload' (download + remux only —
        used for music videos where dubbing makes no sense).
        scheduled_at: unix epoch seconds; a future timestamp parks the job
        as status='scheduled' and the server starts it at that time.
        voxcpm_cfg / voxcpm_steps: per-job VoxCPM guidance and inference
        steps. 0 (the default) means "use the server's global setting".
        """
        files, form = self._source_fields(source)
        form.update({
            "source_lang": source_lang,
            "target_lang": target_lang,
            "whisper_model": whisper_model,
            "voice_preset": voice_preset,
            "voice_style": voice_style,
            "tts_speed": tts_speed,
            "speaker_mode": speaker_mode,
            "keep_bg": str(bool(keep_bg)).lower(),
            "auto_denoise": str(bool(auto_denoise)).lower(),
            "context_hint": context_hint,
            "wizard_mode": wizard_mode,
            "mode": mode,
            "voxcpm_cfg": str(float(voxcpm_cfg or 0)),
            "voxcpm_steps": str(int(voxcpm_steps or 0)),
        })
        if model:
            form["model"] = model
        if scheduled_at:
            form["scheduled_at"] = str(float(scheduled_at))
        if review_gates:
            form["review_gates"] = json.dumps(review_gates)
        return await self._request("POST", "/api/dub", data=form, files=files,
                                   headers=self._prompt_headers(prompt))

    async def submit_compare(
        self,
        source: str,
        target_langs: list[str] | str,
        *,
        trim_seconds: int = 60,
        source_lang: str = "auto",
        model: Optional[str] = None,
        whisper_model: str = "large-v3",
        voice_preset: str = "auto",
        voice_style: str = "",
        tts_speed: str = "balanced",
        keep_bg: bool = True,
        auto_denoise: bool = False,
        context_hint: str = "",
        voxcpm_cfg: float = 0.0,
        voxcpm_steps: int = 0,
        prompt: Optional[str] = None,
    ) -> dict:
        """Submit N separate dubs (Quick Test mode). 2-6 target_langs."""
        if isinstance(target_langs, (list, tuple)):
            target_langs = ",".join(target_langs)
        files, form = self._source_fields(source)
        form.update({
            "target_langs": target_langs,
            "trim_seconds": str(int(trim_seconds)),
            "source_lang": source_lang,
            "whisper_model": whisper_model,
            "voice_preset": voice_preset,
            "voice_style": voice_style,
            "tts_speed": tts_speed,
            "keep_bg": str(bool(keep_bg)).lower(),
            "auto_denoise": str(bool(auto_denoise)).lower(),
            "context_hint": context_hint,
            "voxcpm_cfg": str(float(voxcpm_cfg or 0)),
            "voxcpm_steps": str(int(voxcpm_steps or 0)),
        })
        if model:
            form["model"] = model
        return await self._request("POST", "/api/quick_test", data=form,
                                   files=files,
                                   headers=self._prompt_headers(prompt))

    async def submit_showcase(
        self,
        source: str,
        target_langs: list[str] | str,
        *,
        trim_seconds: int = 60,
        source_lang: str = "auto",
        model: Optional[str] = None,
        whisper_model: str = "large-v3",
        voice_preset: str = "auto",
        voice_style: str = "",
        tts_speed: str = "balanced",
        keep_bg: bool = True,
        auto_denoise: bool = False,
        context_hint: str = "",
        voxcpm_cfg: float = 0.0,
        voxcpm_steps: int = 0,
        prompt: Optional[str] = None,
    ) -> dict:
        """Submit a multilingual showcase reel. 2-6 target_langs are
        dubbed independently then stitched into one continuous video."""
        if isinstance(target_langs, (list, tuple)):
            target_langs = ",".join(target_langs)
        files, form = self._source_fields(source)
        form.update({
            "target_langs": target_langs,
            "trim_seconds": str(int(trim_seconds)),
            "source_lang": source_lang,
            "whisper_model": whisper_model,
            "voice_preset": voice_preset,
            "voice_style": voice_style,
            "tts_speed": tts_speed,
            "keep_bg": str(bool(keep_bg)).lower(),
            "auto_denoise": str(bool(auto_denoise)).lower(),
            "context_hint": context_hint,
            "voxcpm_cfg": str(float(voxcpm_cfg or 0)),
            "voxcpm_steps": str(int(voxcpm_steps or 0)),
        })
        if model:
            form["model"] = model
        return await self._request("POST", "/api/showcase", data=form,
                                   files=files,
                                   headers=self._prompt_headers(prompt))

    async def redub(
        self,
        job_id: str,
        target_langs: list[str] | str,
        *,
        mode: str = "compare",     # 'single' | 'compare' | 'showcase'
        model: Optional[str] = None,
        voice_preset: Optional[str] = None,
        tts_speed: Optional[str] = None,
        voxcpm_cfg: Optional[float] = None,
        voxcpm_steps: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """Re-dub an existing job's source into new language(s) without
        re-uploading. Inherits settings from the original; overrides allowed."""
        if isinstance(target_langs, (list, tuple)):
            target_langs = ",".join(target_langs)
        form = {"target_langs": target_langs, "mode": mode}
        for k, v in (("model", model), ("voice_preset", voice_preset),
                     ("tts_speed", tts_speed), ("voxcpm_cfg", voxcpm_cfg),
                     ("voxcpm_steps", voxcpm_steps)):
            if v is not None:
                form[k] = v
        return await self._request("POST", f"/api/job/{job_id}/redub",
                                   data=form,
                                   headers=self._prompt_headers(prompt))

    # ── status / inspection ───────────────────────────────────────────
    async def get_job(self, job_id: str) -> dict:
        return await self._request("GET", f"/api/job/{job_id}")

    async def list_jobs(self, *, limit: int = 50,
                        status: Optional[str] = None,
                        batch_id: Optional[str] = None,
                        since: float = 0) -> list[dict]:
        """List jobs newest-first. Filtering happens server-side.

        status may be a single status or comma-separated set
        (e.g. "queued,running"). since = unix epoch; only jobs created
        at/after it are returned.
        """
        params: dict = {}
        if status:
            params["status"] = status
        if batch_id:
            params["batch_id"] = batch_id
        if limit and limit > 0:
            params["limit"] = int(limit)
        if since and since > 0:
            params["since"] = float(since)
        data = await self._request("GET", "/api/jobs", params=params)
        # /api/jobs returns {"jobs": [...]} sorted newest-first
        return data.get("jobs", []) if isinstance(data, dict) else []

    async def get_showcase(self, batch_id: str) -> dict:
        return await self._request("GET", f"/api/showcase/{batch_id}")

    async def rebuild_showcase(self, batch_id: str) -> dict:
        return await self._request("POST", f"/api/showcase/{batch_id}/rebuild")

    async def cancel_job(self, job_id: str) -> dict:
        return await self._request("POST", f"/api/dub/{job_id}/cancel")

    async def attach_source(self, job_id: str, file_path: str) -> dict:
        """Attach a manually-downloaded video to a job whose download
        failed and resume the pipeline from the extract stage."""
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            raise GoChiDUBBError(f"Video file not found: {p}")
        # httpx will close the file handle when the request finishes
        f = open(p, "rb")
        files = {"video": (p.name, f, "application/octet-stream")}
        return await self._request(
            "POST", f"/api/job/{job_id}/attach_source", files=files)

    async def delete_job(self, job_id: str) -> dict:
        return await self._request("DELETE", f"/api/job/{job_id}")

    async def system_status(self) -> dict:
        return await self._request("GET", "/api/system")

    async def list_languages(self) -> list[str]:
        """Supported target-language codes, fetched from GET /api/languages
        (canonical 65-language list — no longer hardcoded here)."""
        data = await self._request("GET", "/api/languages")
        return data.get("languages", []) if isinstance(data, dict) else []

    async def list_models(self) -> list[str]:
        """Installed Ollama models (from /api/system)."""
        s = await self.system_status()
        models = s.get("ollama", {}).get("models", []) or []
        return [m if isinstance(m, str) else m.get("name", "") for m in models]

    async def list_voices(self) -> list[dict]:
        """Voice presets registered in the server."""
        try:
            d = await self._request("GET", "/api/voice_presets")
            return d.get("presets", []) if isinstance(d, dict) else []
        except GoChiDUBBError:
            return []

    # ── quality / audit ──────────────────────────────────────────────
    async def get_quality(self, job_id: str) -> dict:
        """Per-stage quality report (0-100 scores + actionable verdicts).

        Verdicts carry `suggested_action` naming an existing server route
        (retry_tts / edit_translations / regenerate_segment / retranslate)."""
        return await self._request("GET", f"/api/dub/{job_id}/quality")

    async def audit(self, job_id: str) -> dict:
        """Artifact-loss audit: word coverage, idx integrity, QA verdicts.
        Same engine as `python tools/audit_job.py <id>`."""
        return await self._request("GET", f"/api/dub/{job_id}/audit")

    # ── review workbench (CLD-273 parity) ────────────────────────────
    async def retry_stage(self, job_id: str, stage: str, *,
                          stop_after: str = "",
                          overrides: Optional[dict] = None) -> dict:
        """Re-run one pipeline stage (and everything after it) from the
        previous stage's checkpoint. `overrides` is whitelisted server-side
        (STAGE_RETRY_OPTIONS says what each stage accepts); `stop_after`
        halts before a later, more expensive stage."""
        form = {"overrides": json.dumps(overrides or {}),
                "stop_after": stop_after or ""}
        return await self._request(
            "POST", f"/api/dub/{job_id}/retry_stage/{stage}", data=form)

    async def get_flags(self, job_id: str, *, max_flags: int = 5) -> dict:
        """The handful of translated spans worth a human's attention.
        Recomputed per call, so it reflects edits already applied."""
        return await self._request(
            "GET", f"/api/dub/{job_id}/flags",
            params={"max_flags": int(max_flags)})

    async def edit_translations(self, job_id: str, edits: dict) -> dict:
        """Rewrite translated_text for segments in the saved checkpoint.
        `edits` maps segment idx (int or str) to the new line. Follow with
        continue_job() to proceed."""
        return await self._request(
            "POST", f"/api/dub/{job_id}/edit_translations",
            data={"edits": json.dumps({str(k): v for k, v in edits.items()})})

    async def add_glossary_term(self, term: str, *, translation: str = "",
                                target_lang: str = "", domain: str = "",
                                say: str = "") -> dict:
        """Teach the glossary one term: a rendering (`translation`), a
        pronunciation respelling (`say`), or both. Applies to every later
        translation for that language, across jobs."""
        return await self._request(
            "POST", "/api/glossary/term",
            data={"term": term, "translation": translation,
                  "target_lang": target_lang, "domain": domain, "say": say})

    # ── voice casting ────────────────────────────────────────────────
    async def get_voice_casting(self, job_id: str) -> dict:
        """Who speaks in this job, how they are cast, and what else they
        could be cast as. Available once the job has been translated."""
        return await self._request("GET", f"/api/dub/{job_id}/voice_casting")

    async def set_voice_casting(self, job_id: str, cast: dict) -> dict:
        """Assign voices to speakers: {"SPEAKER_00": "male_deep"}.

        A voice is a built-in preset id, a library voice ("file:name"), a
        free-text design ("design:gravelly old sailor"), or "source" to keep
        the voice cloned from the video. Any speaker left out keeps theirs.
        """
        return await self._request(
            "POST", f"/api/dub/{job_id}/voice_casting", json={"map": cast})

    async def preview_voice_casting(self, job_id: str,
                                    cast: Optional[dict] = None,
                                    per_speaker: int = 1) -> dict:
        """Synthesize a real line per speaker in the proposed cast.

        Writes nothing to the dub. Slow — it drives the TTS engine — but a
        rounding error next to the synthesis stage it lets you avoid
        re-running.
        """
        body: dict = {"per_speaker": per_speaker}
        if cast is not None:
            body["map"] = cast
        return await self._request(
            "POST", f"/api/dub/{job_id}/voice_preview", json=body)

    async def continue_job(self, job_id: str) -> dict:
        """Resume a job parked at a wizard review gate."""
        # No body: every field on the route has a default, so an empty POST
        # means "resume as configured", which is what a review gate wants.
        return await self._request("POST", f"/api/dub/{job_id}/continue")

    # ── publish (VK etc.) ────────────────────────────────────────────
    async def publish_stage(self, job_id: str, *, platform: str = "vk",
                            export_preset: Optional[str] = None) -> dict:
        """Stage a finished job for publishing (builds metadata, runs the
        duplicate check, optional platform export). NEVER uploads."""
        body: dict = {"platform": platform}
        if export_preset:
            body["export_preset"] = export_preset
        return await self._request(
            "POST", f"/api/dub/{job_id}/publish/stage", json=body)

    async def publish_approve(self, job_id: str, *,
                              title: Optional[str] = None,
                              description: Optional[str] = None) -> dict:
        """Approve a staged publish — the human gate that TRIGGERS the
        actual upload. Optional title/description edits are applied first."""
        body: dict = {}
        if title:
            body["title"] = title
        if description:
            body["description"] = description
        return await self._request(
            "POST", f"/api/dub/{job_id}/publish/approve", json=body)

    async def publish_cancel(self, job_id: str) -> dict:
        """Withdraw a staged/approved/failed publish (409 while uploading)."""
        return await self._request("POST", f"/api/dub/{job_id}/publish/cancel")

    async def get_publish(self, job_id: str) -> dict:
        """Current publish state for one job ({} keys under 'publish')."""
        return await self._request("GET", f"/api/dub/{job_id}/publish")

    async def publish_pending(self) -> list[dict]:
        """Review inbox: publishes still needing (or doing) work."""
        data = await self._request("GET", "/api/publish/pending")
        return data.get("pending", []) if isinstance(data, dict) else []

    # ── scout (trending discovery) ───────────────────────────────────
    async def scout_trending(self, *, category: Optional[str] = None,
                             country: Optional[str] = None, limit: int = 20,
                             include_shorts: bool = False) -> dict:
        """Trending YouTube candidates. Returns {source, candidates:[...]};
        each candidate carries is_music and already_processed flags."""
        params: dict = {"limit": int(limit)}
        if category:
            params["category"] = category
        if country:
            params["country"] = country
        if include_shorts:
            params["include_shorts"] = "true"
        return await self._request("GET", "/api/scout/trending", params=params)

    async def scout_dub(self, url_or_video_id: str, target_lang: str, *,
                        mode: str = "auto",
                        scheduled_at: Optional[float] = None,
                        **dub_params) -> dict:
        """Submit a scout candidate straight into the dub pipeline.

        mode 'auto' resolves to 'reupload' for music candidates, else 'dub'.
        Extra dub params (source_lang, model, keep_bg, voice_preset, ...)
        pass through to the normal dub submission."""
        body: dict = {"target_lang": target_lang, "mode": mode, **dub_params}
        if self._is_url(url_or_video_id):
            body["url"] = url_or_video_id
        else:
            body["video_id"] = url_or_video_id
        if scheduled_at:
            body["scheduled_at"] = float(scheduled_at)
        return await self._request("POST", "/api/scout/dub", json=body)

    async def check_duplicate(self, title: str, *,
                              duration_sec: Optional[float] = None,
                              alt_title: Optional[str] = None) -> dict:
        """Does a similar video already exist on the target platform?
        Warn-only: returns {matches, verdict} (likely_duplicate|possible|clear)."""
        body: dict = {"title": title}
        if duration_sec is not None:
            body["duration_sec"] = duration_sec
        if alt_title:
            body["alt_title"] = alt_title
        return await self._request("POST", "/api/scout/check_duplicate", json=body)

    # ── result / output URLs ─────────────────────────────────────────
    def output_url(self, job_id: str, filename: str = "dubbed_video.mp4") -> str:
        """Absolute URL of a file in the job's output directory."""
        return f"{self.base_url}/outputs/{job_id}/{filename}"

    def showcase_url(self, batch_id: str) -> str:
        return f"{self.base_url}/outputs/showcase_{batch_id}/showcase.mp4"

    # ── high-level: wait until done ──────────────────────────────────
    async def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float = 1800.0,
        poll: float = 2.0,
    ) -> dict:
        """Poll until job reaches a terminal state. Returns the final job dict.

        Raises GoChiDUBBError on timeout. Status 'error' is NOT raised — the
        caller inspects `result["status"]` and `result["error"]`.

        A job parked at a review gate (any `awaiting_*` status) also
        returns: it is waiting on a HUMAN (or an agent calling /continue),
        and polling it to timeout would just report the wait as a failure.
        Check `result["pending_gate"]` to see which gate wants attention.
        """
        start = time.monotonic()
        last: dict = {}
        while time.monotonic() - start < timeout:
            try:
                last = await self.get_job(job_id)
            except GoChiDUBBError:
                # Job might not be flushed to disk yet; keep trying briefly
                pass
            status = last.get("status") or ""
            if status in _TERMINAL or status.startswith("awaiting_"):
                return last
            await asyncio.sleep(poll)
        raise GoChiDUBBError(
            f"Timeout after {timeout}s waiting for {job_id} (last status={last.get('status')})")

    async def wait_for_batch(
        self,
        batch_id: str,
        *,
        timeout: float = 3600.0,
        poll: float = 3.0,
    ) -> list[dict]:
        """Poll a batch (quick_test or showcase) until all child jobs are
        terminal — or parked at a review gate, which needs a human, not a
        longer wait. Returns list of final job dicts."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            jobs = await self.list_jobs(batch_id=batch_id, limit=10)
            if jobs and all(
                    (j.get("status") or "") in _TERMINAL
                    or (j.get("status") or "").startswith("awaiting_")
                    for j in jobs):
                return jobs
            await asyncio.sleep(poll)
        raise GoChiDUBBError(f"Timeout after {timeout}s waiting for batch {batch_id}")

    async def wait_for_showcase(
        self,
        batch_id: str,
        *,
        timeout: float = 3600.0,
        poll: float = 3.0,
    ) -> dict:
        """Wait for showcase assembly to finish. Returns the final showcase info
        dict with `status` == 'ready' (success) or raises on timeout/error."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            info = await self.get_showcase(batch_id)
            if info.get("status") == "ready":
                return info
            if info.get("status") == "error":
                raise GoChiDUBBError(f"Showcase assembly failed: {info.get('error', '?')}")
            await asyncio.sleep(poll)
        raise GoChiDUBBError(f"Timeout after {timeout}s waiting for showcase {batch_id}")
