"""Tests for pipeline/transcriber.py — thread count logic and edge cases."""
from unittest.mock import patch

from pipeline.transcriber import (
    get_optimal_thread_count,
    word_confidence_stats,
)


class TestWordConfidenceStats:
    """word_confidence_stats() aggregates per-word ASR confidences."""

    def test_mean_and_min(self):
        words = [
            {"word": "a", "probability": 0.9},
            {"word": "b", "probability": 0.5},
            {"word": "c", "probability": 0.7},
        ]
        stats = word_confidence_stats(words)
        assert stats["word_conf_mean"] == 0.7
        assert stats["word_conf_min"] == 0.5

    def test_empty_words_returns_empty(self):
        assert word_confidence_stats([]) == {}
        assert word_confidence_stats(None) == {}

    def test_words_without_probability_returns_empty(self):
        """Never fabricate a confidence that was not reported."""
        assert word_confidence_stats([{"word": "a"}, {"word": "b"}]) == {}

    def test_ignores_non_dict_entries_and_bad_values(self):
        words = [
            "not-a-dict",
            {"word": "a", "probability": "high"},
            {"word": "b", "probability": True},   # bool is not a confidence
            {"word": "c", "probability": 0.8},
        ]
        stats = word_confidence_stats(words)
        assert stats == {"word_conf_mean": 0.8, "word_conf_min": 0.8}

    def test_rounds_to_four_places(self):
        words = [{"probability": 1 / 3}, {"probability": 2 / 3}]
        stats = word_confidence_stats(words)
        assert stats["word_conf_mean"] == 0.5
        assert stats["word_conf_min"] == round(1 / 3, 4)


class TestGetOptimalThreadCount:
    """get_optimal_thread_count() returns appropriate thread counts per CPU size."""

    @patch("multiprocessing.cpu_count", return_value=2)
    def test_small_system(self, mock_cpu):
        """<=4 cores: use all but 1."""
        assert get_optimal_thread_count() == 1

    @patch("multiprocessing.cpu_count", return_value=4)
    def test_four_cores(self, mock_cpu):
        """4 cores → 3 threads."""
        assert get_optimal_thread_count() == 3

    @patch("multiprocessing.cpu_count", return_value=6)
    def test_six_cores(self, mock_cpu):
        """6 cores (Apple M1 base): use 4 threads."""
        assert get_optimal_thread_count() == 4

    @patch("multiprocessing.cpu_count", return_value=8)
    def test_eight_cores(self, mock_cpu):
        """8 cores: use 6 threads."""
        assert get_optimal_thread_count() == 6

    @patch("multiprocessing.cpu_count", return_value=10)
    def test_ten_cores(self, mock_cpu):
        """10 cores (Apple M1 Pro/Max): use 75% → 7."""
        assert get_optimal_thread_count() == 7

    @patch("multiprocessing.cpu_count", return_value=16)
    def test_sixteen_cores(self, mock_cpu):
        """16 cores: use 75% → 12."""
        assert get_optimal_thread_count() == 12

    @patch("multiprocessing.cpu_count", return_value=64)
    def test_epyc_system(self, mock_cpu):
        """64 cores: use 75% → 48."""
        assert get_optimal_thread_count() == 48

    @patch("multiprocessing.cpu_count", return_value=1)
    def test_single_core(self, mock_cpu):
        """Single core: at least 1."""
        assert get_optimal_thread_count() == 1


# ── ASR backend selection + mlx-whisper path ─────────────────────────
# mlx only exists on arm64 macOS, so none of these import it for real: a
# fake `mlx_whisper` module is planted in sys.modules (or removed) instead.

import sys
import types

import pytest

from pipeline import transcriber as tr


def _fake_mlx(monkeypatch, result=None, calls=None):
    mod = types.ModuleType("mlx_whisper")

    def transcribe(audio, path_or_hf_repo=None, language=None,
                   word_timestamps=False, initial_prompt=None,
                   condition_on_previous_text=True, temperature=(0.0,),
                   verbose=None):
        if calls is not None:
            calls.append({"audio": audio, "repo": path_or_hf_repo,
                          "language": language, "prompt": initial_prompt,
                          "word_timestamps": word_timestamps})
        return result or {"language": "es", "segments": []}

    mod.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", mod)
    return mod


def _no_mlx(monkeypatch):
    # A None entry makes `import mlx_whisper` raise ImportError.
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)


def _platform(monkeypatch, plat, machine):
    monkeypatch.setattr(tr.sys, "platform", plat)
    monkeypatch.setattr(tr.platform, "machine", lambda: machine)


class TestResolveAsrBackend:
    def test_auto_picks_mlx_on_apple_silicon_with_package(self, monkeypatch):
        _platform(monkeypatch, "darwin", "arm64")
        _fake_mlx(monkeypatch)
        assert tr.resolve_asr_backend("auto") == "mlx"

    def test_auto_without_package_is_faster_whisper(self, monkeypatch):
        _platform(monkeypatch, "darwin", "arm64")
        _no_mlx(monkeypatch)
        assert tr.resolve_asr_backend("auto") == "faster-whisper"

    def test_auto_on_linux_or_intel_mac_is_faster_whisper(self, monkeypatch):
        _fake_mlx(monkeypatch)
        _platform(monkeypatch, "linux", "x86_64")
        assert tr.resolve_asr_backend("auto") == "faster-whisper"
        _platform(monkeypatch, "darwin", "x86_64")
        assert tr.resolve_asr_backend("auto") == "faster-whisper"

    def test_explicit_choices_are_honoured(self, monkeypatch):
        _platform(monkeypatch, "linux", "x86_64")
        _no_mlx(monkeypatch)
        assert tr.resolve_asr_backend("mlx") == "mlx"
        _platform(monkeypatch, "darwin", "arm64")
        _fake_mlx(monkeypatch)
        assert tr.resolve_asr_backend("faster-whisper") == "faster-whisper"

    def test_none_follows_config(self, monkeypatch):
        from app.config import cfg
        monkeypatch.setattr(cfg, "asr_backend", "faster-whisper")
        _platform(monkeypatch, "darwin", "arm64")
        _fake_mlx(monkeypatch)
        assert tr.resolve_asr_backend(None) == "faster-whisper"


class TestMlxRepoFor:
    def test_known_names(self):
        assert tr.mlx_repo_for("large-v3") == "mlx-community/whisper-large-v3-mlx"
        assert tr.mlx_repo_for("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
        assert tr.mlx_repo_for("medium") == "mlx-community/whisper-medium-mlx"

    def test_repo_id_passes_through(self):
        assert tr.mlx_repo_for("someone/whisper-es-mlx") == "someone/whisper-es-mlx"

    def test_unknown_short_name_follows_the_scheme(self):
        assert tr.mlx_repo_for("small") == "mlx-community/whisper-small-mlx"


class TestMlxTranscribe:
    RESULT = {
        "language": "es",
        "segments": [{
            "id": 0, "start": 0.0, "end": 2.5, "text": " Buenas tardes.",
            "avg_logprob": -0.21, "no_speech_prob": 0.01,
            "words": [{"word": " Buenas", "start": 0.0, "end": 0.6,
                       "probability": 0.95},
                      {"word": " tardes.", "start": 0.6, "end": 1.2,
                       "probability": 0.9}],
        }, {
            "id": 1, "start": 2.5, "end": 4.0, "text": " Se abre la sesión.",
        }],
    }

    def test_segments_have_the_faster_whisper_shape(self, monkeypatch):
        calls = []
        _fake_mlx(monkeypatch, self.RESULT, calls)
        info = {}
        segs, lang = tr.transcribe("a.wav", "es", "large-v3",
                                   initial_prompt="Talamanca",
                                   backend="mlx", info=info)
        assert lang == "es"
        assert info == {"backend": "mlx",
                        "model": "mlx-community/whisper-large-v3-mlx"}
        assert calls[0]["prompt"] == "Talamanca"
        assert calls[0]["repo"] == "mlx-community/whisper-large-v3-mlx"
        assert calls[0]["word_timestamps"] is True
        s0, s1 = segs
        assert set(s0) >= {"start", "end", "text", "language", "avg_logprob",
                           "no_speech_prob", "words"}
        assert s0["avg_logprob"] == -0.21 and s0["no_speech_prob"] == 0.01
        assert s0["words"][1] == {"start": 0.6, "end": 1.2,
                                  "word": " tardes.", "probability": 0.9}
        # Missing confidences stay missing — never fabricated.
        assert s1["avg_logprob"] is None and s1["no_speech_prob"] is None
        assert s1["words"] == []

    def test_auto_language_is_none(self, monkeypatch):
        calls = []
        _fake_mlx(monkeypatch, self.RESULT, calls)
        tr.transcribe("a.wav", "auto", "medium", backend="mlx")
        assert calls[0]["language"] is None

    def test_unsupported_kwargs_are_dropped(self, monkeypatch):
        # The fake's signature lacks e.g. compression_ratio_threshold; a
        # real older mlx-whisper might too. It must not raise TypeError.
        _fake_mlx(monkeypatch, self.RESULT)
        segs, _ = tr.transcribe("a.wav", "es", "large-v3", backend="mlx")
        assert len(segs) == 2

    def test_explicit_mlx_without_package_says_how_to_fix(self, monkeypatch):
        _no_mlx(monkeypatch)
        with pytest.raises(ImportError, match="requirements-mac.txt"):
            tr.transcribe("a.wav", "es", "large-v3", backend="mlx")

    def test_faster_whisper_path_reports_itself(self, monkeypatch):
        monkeypatch.setattr(tr, "FASTER_WHISPER_AVAILABLE", False)
        info = {}
        with pytest.raises(ImportError):
            tr.transcribe("a.wav", "es", "large-v3",
                          backend="faster-whisper", info=info)
        assert info == {"backend": "faster-whisper", "model": "large-v3"}
