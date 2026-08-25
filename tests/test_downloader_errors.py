"""What a failed yt-dlp run reports, and when remote code is allowed to run.

Both behaviours come from one incident. A multi-language submit failed and
was persisted with this as its error:

    WARNING: [youtube] [jsc] Remote components challenge solver script (deno)
    and NPM package (deno) were skipped...

That is a warning, not the failure. yt-dlp prints warnings first and its
fatal ERROR line last, so `stderr[:300]` reported the warning as the cause
and cut the real error off. The video downloaded fine on a later attempt with
no flags at all — the diagnosis had been chasing a message that was never the
problem.
"""
import pytest

from pipeline.downloader import (
    _is_403, _needs_challenge_solver, _remote_components_args, _ytdlp_error,
)


# The real stderr from the incident: advisory warning first, error last.
REAL_STDERR = """\
WARNING: [youtube] [jsc] Remote components challenge solver script (deno) and \
NPM package (deno) were skipped. These may be required to solve JS challenges. \
You can enable these downloads with  --remote-components ejs:github \
(recommended) or  --remote-components ejs:npm , respectively. For more info see \
https://github.com/yt-dlp/yt-dlp/wiki/EJS
WARNING: [youtube] Some web client https formats have been skipped.
ERROR: [youtube] K6BYzjwkp24: Sign in to confirm you're not a bot.
"""


class TestWhichLineIsReported:
    def test_the_error_line_wins_over_the_warnings_above_it(self):
        assert _ytdlp_error(REAL_STDERR).startswith("ERROR:")
        assert "not a bot" in _ytdlp_error(REAL_STDERR)

    def test_the_advisory_warning_is_not_reported_as_the_cause(self):
        """The exact regression: this text was shown to the user as the error."""
        assert "Remote components" not in _ytdlp_error(REAL_STDERR)

    def test_the_last_error_wins_when_there_are_several(self):
        """yt-dlp can log a recoverable ERROR for one format and a fatal one
        after it. The final line is the one that ended the run."""
        stderr = "ERROR: [youtube] format 137 unavailable\nERROR: [youtube] fatal\n"
        assert _ytdlp_error(stderr) == "ERROR: [youtube] fatal"

    def test_output_with_no_error_line_falls_back_to_the_tail_not_the_head(self):
        stderr = "warning one\nwarning two\nthe thing that actually broke\n"
        assert "actually broke" in _ytdlp_error(stderr)

    def test_silence_still_says_something(self):
        assert _ytdlp_error("", "") == "yt-dlp failed without writing an error"

    def test_stdout_is_used_when_stderr_is_empty(self):
        assert "boom" in _ytdlp_error("", "boom")


class TestChallengeDetection:
    def test_the_skipped_components_warning_alone_is_not_a_challenge_failure(self):
        """It appears on runs that go on to succeed. Treating it as the
        trigger would fetch and run remote code on healthy downloads."""
        assert not _needs_challenge_solver(REAL_STDERR)

    @pytest.mark.parametrize("text", [
        "ERROR: [youtube] abc: Failed to solve the challenge",
        "ERROR: unable to solve nsig challenge",
        "ERROR: [youtube] nsig extraction failed",
        "ERROR: no challenge solver available",
    ])
    def test_a_genuine_unsolved_challenge_is_detected(self, text):
        assert _needs_challenge_solver(text)

    def test_a_bot_check_is_not_mistaken_for_a_challenge(self):
        assert not _needs_challenge_solver(
            "ERROR: [youtube] Sign in to confirm you're not a bot.")

    def test_a_403_is_not_mistaken_for_a_challenge(self):
        stderr = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
        assert _is_403(stderr)
        assert not _needs_challenge_solver(stderr)


class TestRemoteCodeIsOptIn:
    def test_off_by_default(self, monkeypatch):
        """Solving a challenge means downloading a component and executing it.
        Pasting a link must not be enough to consent to that."""
        import pipeline.downloader as dl
        monkeypatch.setattr(dl.cfg, "ytdlp_remote_components", "", raising=False)
        assert _remote_components_args() == []

    def test_opting_in_produces_the_flag(self, monkeypatch):
        import pipeline.downloader as dl
        monkeypatch.setattr(dl.cfg, "ytdlp_remote_components", "ejs:github",
                            raising=False)
        assert _remote_components_args() == ["--remote-components", "ejs:github"]

    def test_the_setting_is_an_enum_so_a_typo_cannot_become_an_argument(self):
        from app.config import coerce_field
        assert coerce_field("ytdlp_remote_components", "ejs:npm") == "ejs:npm"
        with pytest.raises(ValueError):
            coerce_field("ytdlp_remote_components", "--exec rm")
