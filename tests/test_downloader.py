"""Tests for pipeline/downloader.py — yt-dlp discovery, metadata, cookies."""
import shlex
from unittest.mock import patch

import pytest

from app.config import cfg
from pipeline import rescue
from pipeline.downloader import (
    DownloadFailed,
    _cookie_args,
    _find_ytdlp,
    _is_403,
    _parse_probe_json,
    classify_download_failure,
    curate_metadata,
    download_video,
    probe_metadata,
)


class TestFindYtdlp:
    """_find_ytdlp() locates the yt-dlp executable across platforms."""

    @patch("sys.platform", "win32")
    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    def test_win32_not_found(self, mock_isfile, mock_which):
        """On Windows, if nothing is found, return None."""
        result = _find_ytdlp()
        assert result is None

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    def test_linux_not_found(self, mock_isfile, mock_which):
        result = _find_ytdlp()
        assert result is None

    @patch("sys.platform", "darwin")
    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    def test_macos_not_found(self, mock_isfile, mock_which):
        result = _find_ytdlp()
        assert result is None

    @patch("sys.platform", "darwin")
    @patch("shutil.which", side_effect=lambda c: "/usr/local/bin/yt-dlp" if c == "yt-dlp" else None)
    @patch("os.path.isfile", return_value=False)
    def test_macos_found_in_path(self, mock_isfile, mock_which):
        result = _find_ytdlp()
        assert result == "/usr/local/bin/yt-dlp"

    @patch("sys.platform", "win32")
    @patch("shutil.which", side_effect=lambda c: c if c == "yt-dlp.exe" else None)
    @patch("os.path.isfile", return_value=False)
    def test_win32_found_in_path(self, mock_isfile, mock_which):
        result = _find_ytdlp()
        assert result == "yt-dlp.exe"

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=True)
    def test_linux_found_as_file(self, mock_isfile, mock_which):
        """Even if which() fails, isfile check on candidate paths can find it."""
        # All isfile calls return True, so first candidate matches
        result = _find_ytdlp()
        assert result is not None


# ── curate_metadata ──────────────────────────────────────────────────

FULL_INFO = {
    "id": "dQw4w9WgXcQ",
    "title": "Never Gonna Give You Up",
    "channel": "Rick Astley",
    "uploader": "RickAstleyVEVO",
    "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
    "duration": 213,
    "view_count": 1_400_000_000,
    "like_count": 16_000_000,
    "upload_date": "20091025",
    "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxres.jpg",
    "categories": ["Entertainment"],
    "tags": [f"tag{i}" for i in range(30)],
    "language": "en",
    "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "description": "x" * 3000,
}


class TestCurateMetadata:
    def test_full_info(self):
        m = curate_metadata(FULL_INFO)
        assert m["video_id"] == "dQw4w9WgXcQ"
        assert m["title"] == "Never Gonna Give You Up"
        assert m["channel"] == "Rick Astley"          # channel preferred
        assert m["channel_id"] == "UCuAXFkgsw1L7xaCfnd5JJOw"
        assert m["duration"] == 213
        assert m["view_count"] == 1_400_000_000
        assert m["like_count"] == 16_000_000
        assert m["upload_date"] == "20091025"
        assert m["thumbnail"].startswith("https://")
        assert m["categories"] == ["Entertainment"]
        assert len(m["tags"]) == 20                   # capped at 20
        assert m["language"] == "en"
        assert m["webpage_url"].endswith("dQw4w9WgXcQ")
        # Kept whole: the description is what a user copies into a re-upload,
        # and the old 2000-char cap silently truncated it. The 10k bound is
        # only a runaway guard, well above YouTube's own 5000-char limit.
        assert len(m["description"]) == 3000
        assert m["is_music"] is False

    def test_uploader_fallback_for_channel(self):
        info = dict(FULL_INFO)
        del info["channel"]
        assert curate_metadata(info)["channel"] == "RickAstleyVEVO"

    def test_music_via_categories(self):
        info = {**FULL_INFO, "categories": ["Music"]}
        assert curate_metadata(info)["is_music"] is True

    @pytest.mark.parametrize("key", ["artist", "track"])
    def test_music_via_artist_or_track(self, key):
        info = {**FULL_INFO, key: "Rick Astley"}
        assert curate_metadata(info)["is_music"] is True

    def test_empty_artist_is_not_music(self):
        info = {**FULL_INFO, "artist": "", "track": None}
        assert curate_metadata(info)["is_music"] is False

    @pytest.mark.parametrize("title", [
        "Never Gonna Give You Up (Official Video)",
        "Never Gonna Give You Up (Official Music Video)",
        "SONG NAME - OFFICIAL VIDEO",
    ])
    def test_music_via_title_heuristic(self, title):
        info = {**FULL_INFO, "title": title}
        assert curate_metadata(info)["is_music"] is True

    def test_missing_keys(self):
        m = curate_metadata({})
        assert m["video_id"] is None
        assert m["title"] is None
        assert m["channel"] is None
        assert m["duration"] is None
        assert m["categories"] == []
        assert m["tags"] == []
        assert m["description"] == ""
        assert m["is_music"] is False

    def test_null_fields(self):
        """yt-dlp emits explicit nulls for unavailable fields."""
        m = curate_metadata({
            "id": "abc", "title": None, "categories": None,
            "tags": None, "description": None, "duration": None,
        })
        assert m["video_id"] == "abc"
        assert m["tags"] == []
        assert m["description"] == ""
        assert m["is_music"] is False


# ── _cookie_args ─────────────────────────────────────────────────────

class TestCookieArgs:
    def test_neither_set(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", "")
        assert _cookie_args() == []

    def test_browser_only(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "firefox")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", "")
        assert _cookie_args() == ["--cookies-from-browser", "firefox"]

    def test_file_only(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", "/tmp/cookies.txt")
        assert _cookie_args() == ["--cookies", "/tmp/cookies.txt"]

    def test_both(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "chrome")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", "/tmp/cookies.txt")
        assert _cookie_args() == [
            "--cookies-from-browser", "chrome",
            "--cookies", "/tmp/cookies.txt",
        ]

    def test_whitespace_is_empty(self, monkeypatch):
        monkeypatch.setattr(cfg, "ytdlp_cookies_from_browser", "  ")
        monkeypatch.setattr(cfg, "ytdlp_cookiefile", " ")
        assert _cookie_args() == []


# ── probe JSON parsing (no subprocess) ───────────────────────────────

class TestParseProbeJson:
    def test_valid_dict(self):
        assert _parse_probe_json('{"id": "abc", "duration": 10}') == {
            "id": "abc", "duration": 10,
        }

    def test_invalid_json(self):
        assert _parse_probe_json("ERROR: not json") is None

    def test_empty(self):
        assert _parse_probe_json("") is None

    def test_none(self):
        assert _parse_probe_json(None) is None

    def test_non_dict_json(self):
        assert _parse_probe_json("[1, 2, 3]") is None


class TestProbeMetadataNonUrl:
    """Non-URL sources return None without ever invoking a subprocess."""

    @pytest.mark.parametrize("source", [
        "/path/to/local.mp4", "not a url", "", "ftp://weird",
    ])
    def test_non_url_returns_none(self, source):
        with patch("subprocess.run") as mock_run:
            assert probe_metadata(source) is None
        mock_run.assert_not_called()

    def test_none_source(self):
        assert probe_metadata(None) is None


# ── HTTP 403 fallback ────────────────────────────────────────────────
#
# YouTube serves 403 to yt-dlp's default player client for videos that
# download fine through `web_embedded`. These pin down when we reach for that
# escape hatch, and — just as importantly — when we don't.

class TestIs403:
    @pytest.mark.parametrize("text", [
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
        "ERROR: fragment 1 not found, unable to continue (403 Forbidden)",
        "HTTP Error 403",
    ])
    def test_recognises_a_real_403(self, text):
        assert _is_403(text) is True

    @pytest.mark.parametrize("text", [
        "ERROR: Video unavailable",
        "ERROR: HTTP Error 404: Not Found",
        "",
    ])
    def test_ignores_other_failures(self, text):
        assert _is_403(text) is False

    def test_does_not_fire_on_an_unrelated_403(self):
        # A byte count or fragment index containing "403" is not an error.
        assert _is_403("Downloaded 403 fragments successfully") is False

    def test_reads_stdout_as_well_as_stderr(self):
        assert _is_403("", "HTTP Error 403: Forbidden") is True


class _Result:
    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


class TestDownload403Fallback:
    """How download_video() recovers, now that pipeline/rescue.py decides.

    The old ladder went one way: a 403 escalated straight to the web_embedded
    player client. That is what broke a real download — the 403 was transient,
    web_embedded could not solve YouTube's JS challenge, and its format list
    collapsed to images. The escalation is still there; it is no longer the
    first thing tried.
    """

    SIMPLE = "best[ext=mp4]/best"

    @staticmethod
    def _fmt(cmd):
        """The -f selector a command was invoked with."""
        return cmd[cmd.index("-f") + 1]

    def _run(self, tmp_path, responder):
        """Drive download_video with a fake subprocess; return the commands."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # The success path checks the file exists afterwards.
            (tmp_path / "source_video.mp4").write_bytes(b"x")
            return responder(len(calls), cmd)

        # The waits are real seconds and the point of them is not under test
        # here; the policy tests assert that they are asked for.
        with patch("pipeline.downloader.subprocess.run", side_effect=fake_run), \
             patch("pipeline.downloader.time.sleep"):
            download_video("https://youtu.be/abc", str(tmp_path))
        return calls

    def test_no_retry_when_the_first_attempt_works(self, tmp_path):
        calls = self._run(tmp_path, lambda n, cmd: _Result(0))
        assert len(calls) == 1
        assert "web_embedded" not in " ".join(calls[0])

    def test_a_403_is_retried_as_is_before_the_client_is_changed(self, tmp_path):
        """The regression that motivated the rescue module. Most 403s here
        clear on their own, and swapping to a degraded client to dodge one
        trades a failure that would have fixed itself for one that will not."""
        calls = self._run(tmp_path, lambda n, cmd: (
            _Result(0) if n >= 2 else _Result(1, "HTTP Error 403: Forbidden")))
        assert len(calls) == 2
        assert "web_embedded" not in " ".join(calls[1]), (
            "escalated to a different player client on the first 403 instead "
            "of simply trying again")
        assert self._fmt(calls[1]) == self._fmt(calls[0])

    def test_a_persistent_403_does_still_reach_web_embedded(self, tmp_path):
        calls = self._run(tmp_path, lambda n, cmd: (
            _Result(0) if "web_embedded" in " ".join(cmd)
            else _Result(1, "HTTP Error 403: Forbidden")))
        assert "youtube:player_client=web_embedded" in calls[-1]
        assert len(calls) == 3   # preferred, wait+preferred, web_embedded

    def test_403_retries_keep_the_preferred_format(self, tmp_path):
        # A 403 is an access problem, not a format problem — degrading to
        # best[ext=mp4] would hand back a worse rendition of a 1080p video.
        # Asserted as "same selector as attempt 1, and not the degraded one"
        # rather than a literal string, so tuning the selector doesn't break
        # this without changing what it is actually checking.
        calls = self._run(tmp_path, lambda n, cmd: (
            _Result(0) if "web_embedded" in " ".join(cmd)
            else _Result(1, "HTTP Error 403: Forbidden")))
        for c in calls:
            assert self._fmt(c) == self._fmt(calls[0])
            assert self._fmt(c) != self.SIMPLE

    def test_non_403_failure_degrades_format_without_web_embedded(self, tmp_path):
        calls = self._run(tmp_path, lambda n, cmd: (
            _Result(0) if n == 2 else _Result(1, "ERROR: format not available")))
        assert len(calls) == 2
        assert "web_embedded" not in " ".join(calls[1])
        assert self._fmt(calls[1]) == self.SIMPLE   # degraded, as intended

    def test_a_dead_video_is_not_retried_at_all(self, tmp_path):
        """Six attempts at a deleted video is a minute of nothing, and it
        teaches people that the retry count means nothing."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _Result(1, "ERROR: [youtube] abc: Video unavailable")

        with patch("pipeline.downloader.subprocess.run", side_effect=fake_run), \
             patch("pipeline.downloader.time.sleep"), \
             pytest.raises(DownloadFailed):
            download_video("https://youtu.be/abc", str(tmp_path))
        assert len(calls) == 1

    def test_the_ladder_is_bounded(self, tmp_path):
        """Nothing works, everything is tried, and then it stops."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _Result(1, "ERROR: something nobody has a rule for")

        with patch("pipeline.downloader.subprocess.run", side_effect=fake_run), \
             patch("pipeline.downloader.time.sleep"), \
             pytest.raises(DownloadFailed):
            download_video("https://youtu.be/abc", str(tmp_path))
        assert 1 < len(calls) <= rescue.MAX_ATTEMPTS

    def test_what_was_attempted_is_recorded_on_the_failure(self, tmp_path):
        """A rescue that did not work has to be readable afterwards, or the
        only record is a log line nobody kept."""
        def fake_run(cmd, **kwargs):
            return _Result(1, "ERROR: [youtube] Requested format is not available")

        with patch("pipeline.downloader.subprocess.run", side_effect=fake_run), \
             patch("pipeline.downloader.time.sleep"):
            with pytest.raises(DownloadFailed) as ei:
                download_video("https://youtu.be/abc", str(tmp_path))

        rec = (ei.value.hint or {}).get("rescue") or {}
        assert rec.get("shape") == rescue.FORMAT_GONE
        assert [a["strategy"] for a in rec.get("attempts", [])]
        assert "format" in rec.get("summary", "").lower()


class TestProbe403Fallback:
    """probe_metadata() had no fallback at all, so a 403 left the job with no
    title or description."""

    def test_403_retries_with_web_embedded(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "web_embedded" in " ".join(cmd):
                r = _Result(0)
                r.stdout = '{"id": "abc", "title": "Recovered"}'
                return r
            return _Result(1, "ERROR: HTTP Error 403: Forbidden")

        with patch("pipeline.downloader.subprocess.run", side_effect=fake_run):
            info = probe_metadata("https://youtu.be/abc")

        assert len(calls) == 2
        assert "youtube:player_client=web_embedded" in calls[1]
        assert info["title"] == "Recovered"

    def test_no_retry_on_a_non_403_failure(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _Result(1, "ERROR: Video unavailable")

        with patch("pipeline.downloader.subprocess.run", side_effect=fake_run):
            assert probe_metadata("https://youtu.be/abc") is None
        assert len(calls) == 1


class TestChapterCapture:
    """curate_metadata() carries chapter marks so a dub can reproduce the
    source's structure in its own language."""

    def test_chapters_are_captured_with_start_and_title(self):
        out = curate_metadata({
            "id": "x", "title": "T",
            "chapters": [
                {"start_time": 0, "end_time": 60, "title": "Intro"},
                {"start_time": 60, "end_time": 90, "title": "Main"},
            ],
        })
        assert out["chapters"] == [
            {"start": 0, "title": "Intro"},
            {"start": 60, "title": "Main"},
        ]

    def test_untitled_and_malformed_chapters_are_dropped(self):
        out = curate_metadata({
            "id": "x", "title": "T",
            "chapters": [
                {"start_time": 0, "title": "   "},   # blank title
                "not-a-dict",
                {"start_time": 30, "title": "Real"},
            ],
        })
        assert out["chapters"] == [{"start": 30, "title": "Real"}]

    def test_missing_chapters_is_an_empty_list_not_none(self):
        assert curate_metadata({"id": "x", "title": "T"})["chapters"] == []

    def test_description_is_no_longer_clipped_at_2000(self):
        # The description is what a user copies into the re-upload, so a
        # 3000-char one has to survive whole.
        long_desc = "d" * 3000
        out = curate_metadata({"id": "x", "title": "T", "description": long_desc})
        assert len(out["description"]) == 3000


class TestClassifyDownloadFailure:
    """classify_download_failure() turns yt-dlp stderr into a rescue hint."""

    URL = "https://www.youtube.com/watch?v=abc&t=10"

    @pytest.mark.parametrize("stderr,expected", [
        # bot-check — including YouTube's curly-apostrophe phrasing, which
        # yt-dlp echoes verbatim.
        ("ERROR: [youtube] abc: Sign in to confirm you're not a bot. "
         "Use --cookies-from-browser or --cookies for the authentication.",
         "bot-check"),
        ("ERROR: [youtube] abc: Sign in to confirm you’re not a bot. "
         "This helps protect our community.", "bot-check"),
        ("ERROR: [youtube] abc: Use --cookies-from-browser or --cookies "
         "for the authentication.", "bot-check"),
        # age-gate
        ("ERROR: [youtube] abc: Sign in to confirm your age. This video "
         "may be inappropriate for some users.", "age-gate"),
        ("ERROR: [youtube] abc: This video is age-restricted and only "
         "available on YouTube.", "age-gate"),
        # geo-block
        ("ERROR: [youtube] abc: The uploader has not made this video "
         "available in your country", "geo-block"),
        ("ERROR: [youtube] abc: Video unavailable. The uploader has "
         "blocked it in your country", "geo-block"),
        ("ERROR: [generic] abc: This video is unavailable from your "
         "location due to geo restriction", "geo-block"),
        # persistent-403
        ("ERROR: unable to download video data: HTTP Error 403: Forbidden",
         "persistent-403"),
        ("ERROR: fragment 1 not found, unable to continue (403 Forbidden)",
         "persistent-403"),
        # format
        ("ERROR: [youtube] abc: Requested format is not available. "
         "Use --list-formats for a list of available formats", "format"),
        ("ERROR: [youtube] abc: No video formats found!; please report "
         "this issue", "format"),
        ("ERROR: [youtube] abc: Only images are available for download.",
         "format"),
        # network
        ("ERROR: Unable to download webpage: <urlopen error [Errno 8] "
         "nodename nor servname provided>", "network"),
        ("ERROR: Unable to download webpage: The read operation timed out",
         "network"),
        ("ERROR: Unable to download webpage: [Errno 61] Connection refused",
         "network"),
        # unknown — the real reason survives in summary/detail instead
        ("ERROR: [youtube] abc: Video unavailable", "unknown"),
        ("ERROR: [youtube] abc: Private video. Sign in if you've been "
         "granted access to this video", "unknown"),
        ("", "unknown"),
    ])
    def test_classification(self, stderr, expected):
        hint = classify_download_failure(stderr, "", self.URL)
        assert hint["failure_class"] == expected

    def test_bot_check_wins_over_403(self):
        # The two co-occur: YouTube serves the bot interstitial with a 403.
        stderr = ("ERROR: [youtube] abc: Sign in to confirm you're not a "
                  "bot (HTTP Error 403: Forbidden)")
        assert classify_download_failure(
            stderr, "", self.URL)["failure_class"] == "bot-check"

    def test_reads_stdout_as_well_as_stderr(self):
        hint = classify_download_failure(
            "", "Sign in to confirm you're not a bot", self.URL)
        assert hint["failure_class"] == "bot-check"

    def test_detail_is_the_first_error_line(self):
        stderr = ("WARNING: something benign\n"
                  "ERROR: [youtube] abc: Video unavailable\n"
                  "ERROR: a second error line")
        hint = classify_download_failure(stderr, "", self.URL)
        assert hint["detail"] == "ERROR: [youtube] abc: Video unavailable"
        # The unknown class quotes the real reason so it isn't lost.
        assert "Video unavailable" in hint["summary"]

    def test_bot_check_mentions_the_settings_option(self):
        hint = classify_download_failure(
            "Sign in to confirm you're not a bot", "", self.URL)
        assert "Cookies from browser" in hint["summary"]
        assert "firefox" in hint["summary"]

    # ── Command construction ──────────────────────────────────────────

    ALL_CLASS_STDERRS = [
        "Sign in to confirm you're not a bot",
        "Sign in to confirm your age",
        "not available in your country",
        "HTTP Error 403: Forbidden",
        "Requested format is not available",
        "Connection refused",
        "ERROR: something else entirely",
    ]

    @pytest.mark.parametrize("stderr", ALL_CLASS_STDERRS)
    def test_url_is_shell_quoted_in_every_command(self, stderr):
        # The URL contains '&', which unquoted would background the command.
        hint = classify_download_failure(stderr, "", self.URL)
        for cmd in hint["commands"]:
            text = cmd["command"]
            if "youtube.com" in text:
                assert shlex.quote(self.URL) in text
                # The URL must survive shell splitting as ONE argument.
                assert self.URL in shlex.split(text)

    def test_bot_check_leads_with_the_cookies_command(self):
        hint = classify_download_failure(
            "Sign in to confirm you're not a bot", "", self.URL)
        assert "--cookies-from-browser firefox" in hint["commands"][0]["command"]

    @pytest.mark.parametrize("stderr", ALL_CLASS_STDERRS)
    def test_every_class_yields_a_complete_hint(self, stderr):
        hint = classify_download_failure(stderr, "", self.URL)
        assert hint["summary"]
        assert isinstance(hint["commands"], list) and hint["commands"]
        for cmd in hint["commands"]:
            assert isinstance(cmd, dict)
            assert cmd["label"] and cmd["command"]

    @pytest.mark.parametrize("stderr,stdout,source", [
        (None, None, None),
        (b"\xff\xfe garbage bytes", b"", b""),
        (12345, object(), ""),
        ("ERROR: x", "", None),
    ])
    def test_never_raises_on_garbage_input(self, stderr, stdout, source):
        hint = classify_download_failure(stderr, stdout, source)
        assert hint["failure_class"]
        assert hint["summary"]


class TestDownloadFailedCarriesHint:
    def test_hint_reaches_the_exception(self, tmp_path):
        # Every attempt fails with a bot-check message (no 403, so the
        # web_embedded fallback is not taken and no file appears).
        def fake_run(cmd, **kwargs):
            return _Result(1, "ERROR: [youtube] abc: Sign in to confirm "
                              "you're not a bot")

        with patch("pipeline.downloader.subprocess.run", side_effect=fake_run):
            with pytest.raises(DownloadFailed) as exc:
                download_video("https://youtu.be/abc", str(tmp_path))
        assert isinstance(exc.value, RuntimeError)
        assert exc.value.hint["failure_class"] == "bot-check"
        assert exc.value.hint["commands"]
