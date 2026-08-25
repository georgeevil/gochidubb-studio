"""Download videos from YouTube or validate local paths (Windows-safe)."""
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time

from app.config import cfg
from pipeline import rescue

log = logging.getLogger("gochidubb.downloader")


def _find_ytdlp() -> str:
    """Find yt-dlp executable across platforms."""
    exe_dir = os.path.dirname(sys.executable)
    # On Windows venv: python.exe lives in venv\Scripts\, yt-dlp.exe is a sibling.
    # On Linux venv:   python lives in venv/bin/, yt-dlp is a sibling.
    if sys.platform.startswith("win"):
        candidates = [
            os.path.join(exe_dir, "yt-dlp.exe"),      # inside venv\Scripts
            os.path.join(exe_dir, "..", "Scripts", "yt-dlp.exe"),  # from venv root
            "yt-dlp.exe",                              # PATH
            "yt-dlp",
        ]
    else:
        candidates = [
            os.path.join(exe_dir, "yt-dlp"),
            "yt-dlp",
        ]

    for c in candidates:
        # shutil.which accepts full paths
        found = shutil.which(c)
        if found:
            return found
        # also accept if it's just a direct file
        if os.path.isfile(c):
            return c

    # Last resort: use Python module invocation
    return None  # caller will use `python -m yt_dlp`


def _base_cmd() -> list:
    """Resolve the yt-dlp invocation: binary if found, else `python -m yt_dlp`."""
    ytdlp = _find_ytdlp()
    if ytdlp is None:
        log.info("Using: python -m yt_dlp (no binary found)")
        return [sys.executable, "-m", "yt_dlp"]
    log.info(f"Using yt-dlp: {ytdlp}")
    return [ytdlp]


def _cookie_args() -> list:
    """yt-dlp cookie flags from user config (both may be set)."""
    args = []
    browser = (cfg.ytdlp_cookies_from_browser or "").strip()
    cookiefile = (cfg.ytdlp_cookiefile or "").strip()
    if browser:
        args += ["--cookies-from-browser", browser]
    if cookiefile:
        args += ["--cookies", cookiefile]
    return args


# YouTube periodically serves HTTP 403 to yt-dlp's default player client for
# videos that are perfectly downloadable through another one. The `web_embedded`
# client is the reliable escape hatch — it is what a manual
#     yt-dlp --extractor-args "youtube:player_client=web_embedded"
# does, and it costs nothing to retry with because we only reach for it after a
# 403 has already been seen.
_PLAYER_CLIENT_FALLBACK = ["--extractor-args", "youtube:player_client=web_embedded"]


def _remote_components_value() -> str:
    return (getattr(cfg, "ytdlp_remote_components", "") or "").strip()


def _advisor_enabled() -> bool:
    """Whether the local LLM may be asked what to try next.

    Off by default and consulted only for failures the rules do not recognise
    — see pipeline/rescue.advise. The rules cover everything this install has
    actually hit; the model is for the tail.
    """
    return bool(getattr(cfg, "download_rescue_llm", False))


def _ask_advisor(stderr: str, tried: list):
    """Bridge the async advisor into this synchronous function.

    download_video already runs in a worker thread (server._blocking), so
    there is no loop on this thread to reuse and asyncio.run is safe here.
    Any failure means "no suggestion", never a crash: a download must not
    fail because a language model was busy.
    """
    import asyncio
    try:
        model = (getattr(cfg, "translation_model", "") or "").strip()
        if not model:
            return None
        return asyncio.run(rescue.advise(
            stderr, tried, model,
            remote_components=_remote_components_value()))
    except Exception as e:
        log.info(f"[rescue] advisor skipped: {type(e).__name__}: {e}")
        return None


def _remote_components_args() -> list:
    """`--remote-components` flags, only if the user has opted in.

    YouTube sometimes gates a video behind a JavaScript challenge that yt-dlp
    can only solve by downloading a solver component at request time and
    running it under deno or node. That is remote code executed on the user's
    machine as a side effect of pasting a link, so it is off unless
    `ytdlp_remote_components` says otherwise, and even then it is only reached
    for after a challenge failure — never on the first attempt.
    """
    value = (getattr(cfg, "ytdlp_remote_components", "") or "").strip()
    return ["--remote-components", value] if value else []


def _needs_challenge_solver(*outputs: str) -> bool:
    """True when the failure looks like an unsolved JS challenge.

    Deliberately narrow. yt-dlp prints its "components were skipped" line as a
    WARNING on runs that go on to succeed, so the warning alone must not be
    taken as the cause — the output has to show a challenge that actually
    could not be solved.
    """
    blob = " ".join(o or "" for o in outputs).lower()
    # None of these appear in the advisory "components were skipped" warning,
    # which is why matching on them is enough to tell a real challenge failure
    # from a run that merely mentioned one.
    return any(k in blob for k in (
        "failed to solve", "unable to solve", "could not solve",
        "no challenge solver", "nsig extraction failed",
        "unable to extract nsig",
    ))


def _ytdlp_error(stderr: str, stdout: str = "") -> str:
    """The line that actually explains a failed run.

    Not `stderr[:300]`. yt-dlp prints warnings first and its fatal ERROR line
    last, so slicing the head of stderr reliably reports a warning as though
    it were the cause. That is not hypothetical: a failed job was persisted
    with "Remote components challenge solver script ... were skipped" as its
    error — a warning, on a run whose real failure was cut off past character
    300 — and it sent the diagnosis in entirely the wrong direction.
    """
    for line in reversed((stderr or "").splitlines()):
        line = line.strip()
        if line.startswith("ERROR:"):
            return line[:300]
    tail = (stderr or "").strip() or (stdout or "").strip()
    return tail[-300:] if tail else "yt-dlp failed without writing an error"


def _is_403(*outputs: str) -> bool:
    """True when yt-dlp output looks like an HTTP 403 from the extractor."""
    blob = " ".join(o or "" for o in outputs).lower()
    return (
        "403" in blob
        and ("forbidden" in blob or "http error 403" in blob or "status code 403" in blob)
    ) or "http error 403" in blob


# ── Do-it-yourself recipes ──────────────────────────────────────────────
# When the server cannot fetch a video, the user still can — and then upload
# it, which resumes the job from the extract stage with nothing recomputed.
# What was missing was the instructions: the rescue panel offered bare yt-dlp
# one-liners, which are no help at all to somebody who does not have yt-dlp.
#
# The format selector below is not a generic "best quality" — it asks for
# H.264 video and AAC audio specifically, because that is what the assembler
# can stream-copy. Anything else (VP9, AV1, Opus) still works but costs a full
# re-encode at the end of the job.
#
# Only first-party sources are named. Every one of these is the project's own
# site or a store listing for it; none is a mirror or an APK host.

def _diy_format() -> str:
    """The yt-dlp -f selector to hand a user, matching what the pipeline wants."""
    try:
        from app.config import cfg as _cfg
        cap = int(getattr(_cfg, "output_max_height", 1080) or 0)
    except Exception:
        cap = 1080
    h = f"[height<={cap}]" if cap else ""
    return (f"bv*{h}[vcodec^=avc1]+ba[acodec^=mp4a]/"
            f"bv*{h}[ext=mp4]+ba[ext=m4a]/b{h}[ext=mp4]/b{h}/b")


def download_guides(url: str = "") -> list:
    """Per-platform instructions for fetching a video by hand.

    Returns a list of {platform, steps: [{label, command?, url?}], note}.
    `url` is quoted into the commands when given so they can be pasted as-is.
    """
    q = shlex.quote(url) if url else "<VIDEO URL>"
    fmt = _diy_format()
    dl = f'yt-dlp -f "{fmt}" --merge-output-format mp4 {q}'

    return [
        {
            "platform": "macOS",
            "steps": [
                {"label": "Install once (Homebrew)",
                 "command": "brew install yt-dlp ffmpeg"},
                {"label": "Download", "command": dl},
            ],
            "note": "ffmpeg is what merges the video and audio streams back "
                    "together; without it you get one or the other.",
        },
        {
            "platform": "Windows",
            "steps": [
                {"label": "Install once (PowerShell)",
                 "command": "winget install yt-dlp.yt-dlp Gyan.FFmpeg"},
                {"label": "Download", "command": dl},
            ],
            "note": "Open a new terminal after installing so the PATH change "
                    "takes effect.",
        },
        {
            "platform": "Linux",
            "steps": [
                {"label": "Install once",
                 "command": "pipx install yt-dlp   # plus ffmpeg from your package manager"},
                {"label": "Download", "command": dl},
            ],
            "note": "Distro packages of yt-dlp are often months old, and "
                    "YouTube breaks old versions constantly. pipx tracks "
                    "upstream.",
        },
        {
            "platform": "Android",
            "steps": [
                {"label": "Seal — a yt-dlp app, no terminal needed",
                 "url": "https://github.com/JunkFood02/Seal"},
                {"label": "Or Termux, then yt-dlp inside it",
                 "url": "https://f-droid.org/packages/com.termux/"},
                {"label": "In Termux, install once",
                 "command": "pkg install python ffmpeg && pip install yt-dlp"},
                {"label": "Then download", "command": dl},
            ],
            "note": "Save to your phone, then send the file to this machine "
                    "and drop it below. Install Termux from F-Droid rather "
                    "than the Play Store — the Play Store build is frozen and "
                    "its packages no longer install.",
        },
        {
            "platform": "iPhone / iPad",
            "steps": [
                {"label": "There is no reliable yt-dlp on iOS",
                 "url": "https://github.com/yt-dlp/yt-dlp/wiki/FAQ"},
            ],
            "note": "Use a computer for this one, or AirDrop the file over "
                    "from someone who has.",
        },
        {
            "platform": "Any — official install docs",
            "steps": [
                {"label": "yt-dlp installation guide",
                 "url": "https://github.com/yt-dlp/yt-dlp#installation"},
                {"label": "ffmpeg downloads",
                 "url": "https://ffmpeg.org/download.html"},
            ],
            "note": "Whatever you download, drop the file into the box below "
                    "and the job carries on from where it stopped — the "
                    "transcript and translation are not redone.",
        },
    ]


class DownloadFailed(RuntimeError):
    """A download attempt failed, with a structured rescue hint attached.

    `hint` is the dict from classify_download_failure — the server persists
    it on the job as `download_hint` so the UI can show the user exactly
    which yt-dlp command to run by hand.
    """

    def __init__(self, msg, hint=None):
        super().__init__(msg)
        self.hint = hint


# Network-level failure needles — DNS, TCP, TLS and plain timeouts. All are
# substrings of what urllib/yt-dlp print, lowercased.
_NETWORK_NEEDLES = (
    "getaddrinfo failed",
    "temporary failure in name resolution",
    "name or service not known",
    "connection refused",
    "connection reset",
    "network is unreachable",
    "urlopen error",
    "timed out",
    "timeout",
    "ssl:",
)


def classify_download_failure(stderr: str, stdout: str = "",
                              source: str = "") -> dict:
    """Classify a failed yt-dlp run into an actionable rescue hint.

    Returns {"failure_class", "summary", "detail", "commands"} where
    `commands` is a list of {"label", "command"} shell one-liners the user
    can run locally to fetch the video themselves. Never raises — this only
    ever runs on an already-failed download, and a classifier crash would
    replace a useful error with a useless one.
    """
    try:
        stderr = stderr if isinstance(stderr, str) else str(stderr or "")
        stdout = stdout if isinstance(stdout, str) else str(stdout or "")
        source = source if isinstance(source, str) else str(source or "")

        # yt-dlp echoes YouTube's own message text, which uses a curly
        # apostrophe ("you’re not a bot") — normalize before matching.
        blob = (stderr + " " + stdout).lower().replace("’", "'")

        detail = ""
        for line in stderr.splitlines():
            line = line.strip()
            if line.startswith("ERROR:"):
                detail = line[:200]
                break
        if not detail:
            detail = stderr.strip()[:200]

        q = shlex.quote(source) if source else "<URL>"
        emb = {"label": "Retry via embedded player (what the server tried)",
               "command": ("yt-dlp --extractor-args "
                           f'"youtube:player_client=web_embedded" {q}')}
        cook = {"label": "Retry with your browser's YouTube cookies",
                "command": f"yt-dlp --cookies-from-browser firefox {q}"}
        plain = {"label": "Plain yt-dlp download",
                 "command": f"yt-dlp {q}"}
        upd = {"label": "Update yt-dlp first",
               "command": "yt-dlp -U"}
        fmt = {"label": "List available formats",
               "command": f"yt-dlp -F {q}"}

        # First match wins — specific before generic, because a bot-check
        # message frequently arrives alongside a 403.
        if ("sign in to confirm you're not a bot" in blob
                or "use --cookies-from-browser or --cookies" in blob):
            failure_class = "bot-check"
            summary = ("YouTube flagged this download as bot traffic and "
                       "wants cookies from a signed-in browser. Set the "
                       "\"Cookies from browser\" option in Settings to "
                       "firefox, or download it yourself with cookies.")
            commands = [cook, emb]
        elif ("sign in to confirm your age" in blob
                or "age-restricted" in blob
                or "inappropriate for some users" in blob):
            failure_class = "age-gate"
            summary = ("The video is age-restricted — YouTube only serves it "
                       "to a signed-in account, so downloading it needs "
                       "cookies from a browser that is logged in.")
            commands = [cook]
        elif ("not available in your country" in blob
                or "not made this video available in your country" in blob
                or "geo restriction" in blob
                or "geo-restricted" in blob
                or "unavailable from your location" in blob
                or "blocked it in your country" in blob):
            failure_class = "geo-block"
            summary = ("YouTube blocks this video in your region. Retry via "
                       "the embedded player, or download it yourself over a "
                       "VPN or from another network.")
            commands = [emb, plain]
        elif _is_403(stderr, stdout):
            # Classification only happens after the automatic web_embedded
            # retry already failed, so this 403 is persistent.
            failure_class = "persistent-403"
            summary = ("YouTube kept answering HTTP 403 even through the "
                       "embedded-player fallback — usually an outdated "
                       "yt-dlp or an IP-level block. Try the commands in "
                       "order.")
            commands = [emb, cook, upd, plain]
        elif ("requested format is not available" in blob
                or "no video formats found" in blob
                or "only images are available" in blob):
            failure_class = "format"
            summary = ("yt-dlp found no downloadable video format — the "
                       "video may be a live stream, images-only, or need a "
                       "newer yt-dlp.")
            commands = [plain, fmt, upd]
        elif any(n in blob for n in _NETWORK_NEEDLES):
            failure_class = "network"
            summary = ("The download failed at the network level — check "
                       "connectivity, DNS and any proxy or VPN, then retry.")
            commands = [plain]
        else:
            failure_class = "unknown"
            summary = (f"yt-dlp failed: {detail}" if detail
                       else "yt-dlp failed without any error output.")
            commands = [upd, plain]

        return {"failure_class": failure_class, "summary": summary,
                "detail": detail, "commands": commands}
    except Exception:
        log.warning("[download] failure classifier crashed", exc_info=True)
        return {"failure_class": "unknown",
                "summary": "yt-dlp failed and the failure could not be "
                           "classified.",
                "detail": "", "commands": [{"label": "Update yt-dlp first",
                                            "command": "yt-dlp -U"}]}


def _parse_probe_json(stdout: str):
    """Parse `--dump-single-json` output. Returns dict or None — never raises."""
    try:
        info = json.loads(stdout)
    except (TypeError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def probe_metadata(source: str):
    """Fetch video metadata for a URL via yt-dlp without downloading.

    Returns the full info dict, or None for non-URL sources and on ANY
    failure (timeout, non-zero exit, unparseable JSON) — never raises.
    """
    if not isinstance(source, str) or not (
        source.startswith("http://") or source.startswith("https://")
    ):
        return None

    probe_flags = [
        "--dump-single-json",
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        "--no-check-certificates",
    ]

    def _run(extra):
        cmd = _base_cmd() + probe_flags + extra + _cookie_args() + [source]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    try:
        result = _run([])
        # A 403 here is why metadata silently went missing on some videos —
        # the probe had no fallback at all, so the job ran with no title.
        if result.returncode != 0 and _is_403(result.stderr, result.stdout):
            log.info(f"[probe] 403 for {source} — retrying with "
                     f"player_client=web_embedded")
            result = _run(_PLAYER_CLIENT_FALLBACK)
    except subprocess.TimeoutExpired:
        log.warning(f"[probe] Metadata probe timed out (60s): {source}")
        return None
    except Exception as e:
        log.warning(f"[probe] Metadata probe failed to run: {e}")
        return None

    if result.returncode != 0:
        stderr = result.stderr or ""
        hint = ""
        if "cookie" in stderr.lower() or "keyring" in stderr.lower():
            hint = (" — check ytdlp_cookies_from_browser setting; "
                    "firefox is most reliable on macOS")
        log.warning(f"[probe] Metadata probe failed for {source}: "
                    f"{stderr[:300]}{hint}")
        return None

    info = _parse_probe_json(result.stdout)
    if info is None:
        log.warning(f"[probe] Could not parse yt-dlp metadata JSON for {source}")
    return info


def curate_metadata(info: dict) -> dict:
    """Distill a full yt-dlp info dict into a small, persistable summary."""
    categories = info.get("categories") or []
    tags = info.get("tags") or []
    title = info.get("title") or ""
    title_low = title.lower()
    is_music = bool(
        ("Music" in categories)
        or info.get("artist")
        or info.get("track")
        or "official video" in title_low
        or "official music video" in title_low
    )
    description = info.get("description") or ""
    # Chapter marks, so a dub can carry the same structure as the source.
    # yt-dlp gives {start_time, end_time, title}; only start + title survive
    # here because that is all a description block needs.
    chapters = []
    for ch in (info.get("chapters") or []):
        if not isinstance(ch, dict):
            continue
        title_ch = (ch.get("title") or "").strip()
        if not title_ch:
            continue
        chapters.append({"start": ch.get("start_time"), "title": title_ch})
    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "upload_date": info.get("upload_date"),
        "thumbnail": info.get("thumbnail"),
        "categories": categories,
        "tags": tags[:20],
        "language": info.get("language"),
        "webpage_url": info.get("webpage_url"),
        # Kept whole (bounded well above YouTube's 5000-char limit) because
        # this is the text a user copies into the re-upload, and a truncated
        # description is worse than none.
        "description": description[:10000],
        "chapters": chapters,
        "is_music": is_music,
    }


def download_video(source: str, output_dir: str, info: dict | None = None) -> str:
    """
    Download from YouTube URL or copy local file.
    Returns path to the local video file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "source_video.mp4")

    # Local file
    if os.path.isfile(source):
        _src_size = os.path.getsize(source)
        _src_ext = os.path.splitext(source)[1].lower() or "(no ext)"
        # Log the real on-disk format — ffmpeg/ffprobe are picky about input
        # containers, and a common silent-failure cause is uploading an image
        # (e.g. .webp/.jpg) or audio-only file that gets copied to
        # source_video.mp4 and then fails downstream with no obvious reason.
        log.info(
            f"[download] Local file source: {source} "
            f"(ext={_src_ext}, size={_src_size/1048576:.2f} MB)"
        )
        if os.path.abspath(source) != os.path.abspath(output_path):
            shutil.copy2(source, output_path)
        log.info(f"Local file ready: {output_path}")
        return output_path

    # YouTube / HTTP URL
    if not (source.startswith("http://") or source.startswith("https://")):
        raise ValueError(f"Not a file or URL: {source}")

    base_cmd = _base_cmd()
    cookie_args = _cookie_args()

    # Prefer H.264 (avc1) + AAC explicitly: those are what QuickTime, WhatsApp
    # and the rest of the Apple ecosystem can decode, so getting them here
    # means the assembler can stream-copy instead of paying for a transcode.
    # The later branches still accept VP9/AV1 rather than failing the download
    # — the assembler re-encodes those (see _video_codec_args).
    try:
        from app.config import cfg as _cfg
        _max_h = int(getattr(_cfg, "output_max_height", 1080) or 0)
    except Exception:
        _max_h = 1080
    _h = f"[height<={_max_h}]" if _max_h else ""
    preferred_fmt = [
        "-f",
        f"bestvideo{_h}[vcodec^=avc1]+bestaudio[acodec^=mp4a]/"
        f"bestvideo{_h}[ext=mp4]+bestaudio[ext=m4a]/"
        f"best{_h}[ext=mp4]/best{_h}/best",
        "--merge-output-format", "mp4",
    ]
    # The fallback used to omit --merge-output-format entirely, which is how a
    # webm could come out of a path that is supposed to produce .mp4.
    simple_fmt = ["-f", "best[ext=mp4]/best", "--merge-output-format", "mp4"]
    common = [
        "--no-playlist",
        "-o", output_path,
        "--no-check-certificates",
        "--retries", "3",
        "--socket-timeout", "30",
    ]

    # "any" exists because a re-encode is cheap and no dub is not: when a
    # selector has already missed twice, take whatever the site will give.
    any_fmt = ["-f", "best", "--merge-output-format", "mp4"]
    FORMATS = {"preferred": preferred_fmt, "simple": simple_fmt, "any": any_fmt}

    def _run(fmt, extra=()):
        cmd = base_cmd + common + fmt + list(extra) + cookie_args + [source]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    log.info(f"Downloading: {source}")
    result = _run(preferred_fmt)

    # Everything after the first attempt is decided by pipeline/rescue.py
    # rather than by a fixed ladder here. The ladder this replaced could only
    # escalate: a transient 403 sent it to the web_embedded player client,
    # that client could not solve YouTube's JS challenge, its format list
    # collapsed to images, and the run died — on a video the default client
    # downloaded at 1080p seconds later. A policy can wait, or go back.
    tried = ["preferred"]
    rescue_log: list = []
    while result.returncode != 0:
        shape = rescue.classify(result.stderr, result.stdout)
        attempt = rescue.plan(shape, tried,
                              remote_components=_remote_components_value())
        if attempt is None and _advisor_enabled() and shape == rescue.UNKNOWN:
            attempt = _ask_advisor(result.stderr, tried)
        if attempt is None:
            break

        st = attempt.strategy
        extra = list(st.args)
        if st.key == "remote_components":
            extra += _remote_components_args()
        log.warning(
            f"[download] {shape} — {attempt.reason}; trying '{st.key}' "
            f"({st.why})" + (f" after {st.sleep:.0f}s" if st.sleep else ""))
        if st.sleep:
            time.sleep(st.sleep)
        tried.append(st.key)
        rescue_log.append({"shape": shape, "strategy": st.key,
                           "reason": attempt.reason})
        result = _run(FORMATS[st.fmt], extra)

    if result.returncode != 0:
        shape = rescue.classify(result.stderr, result.stdout)
        hint = classify_download_failure(result.stderr, result.stdout, source)
        if isinstance(hint, dict):
            # What was attempted, so a failed rescue can be read back rather
            # than re-derived from the log.
            hint["rescue"] = {"shape": shape, "attempts": rescue_log,
                              "summary": rescue.describe(shape)}
        raise DownloadFailed(
            f"YouTube download failed: {_ytdlp_error(result.stderr, result.stdout)}",
            hint=hint)
    if rescue_log:
        log.info(f"[download] recovered after {len(rescue_log)} rescue "
                 f"attempt(s): {' → '.join(a['strategy'] for a in rescue_log)}")

    # yt-dlp sometimes adds extensions; find the actual file
    if not os.path.exists(output_path):
        for f in os.listdir(output_dir):
            if f.startswith("source_video") and f.endswith((".mp4", ".mkv", ".webm")):
                actual = os.path.join(output_dir, f)
                if actual != output_path:
                    os.rename(actual, output_path)
                break

    if not os.path.exists(output_path):
        # No stderr to classify — yt-dlp claimed success but left no file.
        raise DownloadFailed(
            "Download completed but video file not found",
            hint=classify_download_failure("", "", source))

    size_mb = os.path.getsize(output_path) / 1048576
    log.info(f"Downloaded: {output_path} ({size_mb:.1f} MB)")

    # Preserve the full yt-dlp metadata (probed by the caller) next to the
    # video so later stages / users can inspect it. Best-effort only.
    if info:
        info_path = os.path.join(output_dir, "source_info.json")
        try:
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            log.info(f"Wrote metadata: {info_path}")
        except Exception as e:
            log.warning(f"Could not write source_info.json: {e}")

    return output_path
