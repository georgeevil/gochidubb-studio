"""Tests for the LM Studio client in pipeline/translator.py.

Covers the bug this module was rewritten for: LM_STUDIO_URL already ends in
"/v1", so concatenating "/chat/completions" onto it (or using it verbatim)
produced a POST to "/v1", which LM Studio answers with "Unexpected endpoint
or method ... Returning 200 anyway" — a 200 with no usable body, which the
old code turned into a silent no-op translation.

The HTTP layer is exercised against a real aiohttp server on a loopback
port rather than a mock, so the request shape LM Studio actually receives
is what gets asserted.
"""
import asyncio
import json
import re

import pytest

from aiohttp import web

from pipeline import translator as T


# ── URL normalization ────────────────────────────────────────────────

class TestHostNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("http://localhost:1234", "http://localhost:1234"),
        ("http://localhost:1234/", "http://localhost:1234"),
        ("http://localhost:1234/v1", "http://localhost:1234"),
        ("http://localhost:1234/v1/", "http://localhost:1234"),
        ("http://localhost:1234/api/v1", "http://localhost:1234"),
        ("http://localhost:1234/v1/chat/completions", "http://localhost:1234"),
        ("http://localhost:1234/api/v1/chat", "http://localhost:1234"),
        ("http://192.168.1.9:1234/v1", "http://192.168.1.9:1234"),
        ("", "http://localhost:1234"),
    ])
    def test_any_spelling_reduces_to_the_host_root(self, raw, expected):
        assert T._lm_studio_host(raw) == expected

    def test_endpoints_are_built_from_the_root(self):
        assert T.LM_STUDIO_CHAT_ENDPOINT.endswith("/api/v1/chat")
        assert T.LM_STUDIO_COMPAT_ENDPOINT.endswith("/v1/chat/completions")
        assert T.LM_STUDIO_MODELS_ENDPOINT.endswith("/v1/models")
        # The regression: never post to a bare /v1
        assert not T.LM_STUDIO_CHAT_ENDPOINT.rstrip("/").endswith("/v1")


# ── Response parsing ─────────────────────────────────────────────────

class TestNativeResponseParsing:
    def test_takes_message_content(self):
        assert T._parse_native_response(
            {"output": [{"type": "message", "content": "Привет"}]}
        ) == "Привет"

    def test_ignores_reasoning_items(self):
        """A thinking model returns reasoning as its own output item; only
        the message is the translation."""
        out = T._parse_native_response({"output": [
            {"type": "reasoning", "content": "The user wants Russian. Let me think…"},
            {"type": "message", "content": "Привет"},
        ]})
        assert out == "Привет"

    def test_concatenates_multiple_messages(self):
        assert T._parse_native_response({"output": [
            {"type": "message", "content": "Привет "},
            {"type": "message", "content": "мир"},
        ]}) == "Привет мир"

    def test_reasoning_only_raises_actionable_error(self):
        """Budget exhausted by thinking — the old code returned '' here and
        the segment silently kept its English text."""
        with pytest.raises(T.LMStudioError, match="only reasoning"):
            T._parse_native_response({
                "output": [{"type": "reasoning", "content": "hmm " * 100}],
                "stats": {"reasoning_output_tokens": 500, "total_output_tokens": 500},
            })

    def test_empty_output_raises(self):
        with pytest.raises(T.LMStudioError):
            T._parse_native_response({"output": []})

    def test_tolerates_nested_content_lists(self):
        assert T._parse_native_response({"output": [
            {"type": "message", "content": [{"type": "text", "content": "Привет"}]},
        ]}) == "Привет"


class TestCompatResponseParsing:
    def test_takes_message_content(self):
        assert T._parse_compat_response(
            {"choices": [{"message": {"content": " Привет "}}]}
        ) == "Привет"

    def test_strips_inline_think_block(self):
        out = T._parse_compat_response({"choices": [{"message": {
            "content": "<think>Russian for hello is Привет</think>Привет"}}]})
        assert out == "Привет"

    def test_missing_choices_raises(self):
        """This is literally what a POST to /v1 returned."""
        with pytest.raises(T.LMStudioError, match="no 'choices'"):
            T._parse_compat_response({"error": "Unexpected endpoint or method."})

    def test_reasoning_only_raises(self):
        with pytest.raises(T.LMStudioError, match="only reasoning"):
            T._parse_compat_response({"choices": [{"message": {
                "content": "", "reasoning_content": "thinking..."}}]})


class TestErrorEnvelope:
    def test_structured_error(self):
        err = T._lm_studio_error(json.dumps({"error": {
            "message": "Invalid model identifier", "code": "model_not_found",
            "param": "model"}}))
        assert err["code"] == "model_not_found"

    def test_bare_string_error(self):
        err = T._lm_studio_error('{"error":"Unexpected endpoint or method. (POST /v1)"}')
        assert "Unexpected endpoint" in err["message"]

    def test_non_json_body(self):
        assert T._lm_studio_error("<html>502</html>") == {}


class TestCleanTranslation:
    @pytest.mark.parametrize("raw,expected", [
        ("Translation: Привет", "Привет"),
        ("  Привет  ", "Привет"),
        ('"Привет"', "Привет"),
        ("«Привет»", "Привет"),
        ("<think>reasoning</think>Привет", "Привет"),
        ("Перевод: Привет", "Привет"),
    ])
    def test_strips_scaffolding(self, raw, expected):
        assert T._clean_translation(raw) == expected

    def test_keeps_inner_quotes(self):
        assert T._clean_translation('Он сказал "да" вчера') == 'Он сказал "да" вчера'


# ── HTTP behaviour against a stub LM Studio ──────────────────────────

class StubLMStudio:
    """Minimal stand-in for LM Studio; records the requests it receives."""

    def __init__(self):
        self.requests = []          # (path, body)
        self.native_status = 200
        self.native_body = {"output": [{"type": "message", "content": "Привет"}]}
        self.compat_body = {"choices": [{"message": {"content": "Привет (compat)"}}]}

    async def _native(self, request):
        body = await request.json()
        self.requests.append(("/api/v1/chat", body))
        if callable(self.native_body):
            status, payload = self.native_body(body)
        else:
            status, payload = self.native_status, self.native_body
        return web.json_response(payload, status=status)

    async def _compat(self, request):
        self.requests.append(("/v1/chat/completions", await request.json()))
        return web.json_response(self.compat_body)

    async def _catch_all(self, request):
        # Mirrors LM Studio's real behaviour for an unknown route
        self.requests.append((request.path, None))
        return web.json_response(
            {"error": f"Unexpected endpoint or method. ({request.method} {request.path})"},
            status=404,
        )

    def app(self):
        app = web.Application()
        app.router.add_post("/api/v1/chat", self._native)
        app.router.add_post("/v1/chat/completions", self._compat)
        app.router.add_route("*", "/{tail:.*}", self._catch_all)
        return app


@pytest.fixture
def stub(monkeypatch):
    """Run a stub LM Studio and point the client's endpoints at it."""
    server = StubLMStudio()
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(server.app())
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = runner.addresses[0][1]
    base = f"http://127.0.0.1:{port}"

    monkeypatch.setattr(T, "LM_STUDIO_CHAT_ENDPOINT", f"{base}/api/v1/chat")
    monkeypatch.setattr(T, "LM_STUDIO_COMPAT_ENDPOINT", f"{base}/v1/chat/completions")
    monkeypatch.setattr(T, "_LM_USE_NATIVE", None)
    monkeypatch.setattr(T, "_LM_SEND_REASONING", True)
    monkeypatch.setattr(T, "LM_STUDIO_REASONING", "off")

    server.loop = loop
    server.base = base
    yield server

    loop.run_until_complete(runner.cleanup())
    loop.close()


def _run(stub, coro):
    """Drive a client coroutine on the same loop as the stub server."""
    return stub.loop.run_until_complete(coro)


class TestChatRequest:
    def test_posts_to_the_native_endpoint(self, stub):
        out = _run(stub, T.lm_studio_chat("translate this", model="m"))
        assert out == "Привет"
        assert stub.requests[0][0] == "/api/v1/chat"

    def test_uses_the_native_request_schema(self, stub):
        _run(stub, T.lm_studio_chat("translate this", model="m",
                                    system_prompt="You translate."))
        body = stub.requests[0][1]
        # Native API: `input` + `system_prompt` + `max_output_tokens` — NOT
        # the OpenAI `messages`/`max_tokens` shape.
        assert body["input"] == "translate this"
        assert body["system_prompt"] == "You translate."
        assert "max_output_tokens" in body and "max_tokens" not in body
        assert "messages" not in body
        assert body["stream"] is False
        assert body["reasoning"] == "off"

    def test_output_budget_is_large_enough_for_thinking_models(self, stub):
        _run(stub, T.lm_studio_chat("x", model="m"))
        # The old hard-coded 500 was what a 27B thinking model blew through
        # on reasoning alone.
        assert stub.requests[0][1]["max_output_tokens"] >= 2048

    def test_falls_back_to_compat_when_native_route_is_missing(self, stub, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_CHAT_ENDPOINT", f"{stub.base}/api/v1/nope")
        out = _run(stub, T.lm_studio_chat("x", model="m"))
        assert out == "Привет (compat)"
        assert stub.requests[-1][0] == "/v1/chat/completions"
        body = stub.requests[-1][1]
        assert body["messages"][0]["role"] == "system" or "messages" in body

    def test_unknown_model_does_not_trigger_the_compat_fallback(self, stub):
        """LM Studio returns 404 for a bad model too — downgrading the
        endpoint for the whole run because of a typo would hide the cause."""
        stub.native_status = 404
        stub.native_body = {"error": {
            "message": 'Invalid model identifier "nope".',
            "type": "invalid_request", "param": "model", "code": "model_not_found"}}
        with pytest.raises(T.LMStudioError, match="not available in LM Studio"):
            _run(stub, T.lm_studio_chat("x", model="nope"))
        assert all(p != "/v1/chat/completions" for p, _ in stub.requests)

    def test_retries_without_reasoning_when_the_model_rejects_it(self, stub):
        def responder(body):
            if "reasoning" in body:
                return 400, {"error": {
                    "message": "Invalid enum value", "param": "reasoning",
                    "code": "invalid_enum_value"}}
            return 200, {"output": [{"type": "message", "content": "Привет"}]}
        stub.native_body = responder

        assert _run(stub, T.lm_studio_chat("x", model="m")) == "Привет"
        assert len(stub.requests) == 2
        assert "reasoning" in stub.requests[0][1]
        assert "reasoning" not in stub.requests[1][1]
        # And it stops sending the field afterwards
        assert T._LM_SEND_REASONING is False

    def test_server_error_raises_instead_of_returning_the_prompt(self, stub):
        stub.native_status = 500
        stub.native_body = {"error": {"message": "engine crashed"}}
        with pytest.raises(T.LMStudioError, match="engine crashed"):
            _run(stub, T.lm_studio_chat("x", model="m"))


class TestTranslateTextPropagatesFailures:
    def test_failure_raises_so_the_caller_can_retry(self, stub):
        """translate_segments() retries three times; that only works if
        translate_text stops swallowing the error and returning the source."""
        stub.native_status = 500
        stub.native_body = {"error": {"message": "boom"}}
        with pytest.raises(T.LMStudioError):
            _run(stub, T.translate_text("Hello", "ru", model="m"))

    def test_success_returns_the_cleaned_translation(self, stub):
        stub.native_body = {"output": [
            {"type": "message", "content": "Translation: Привет"}]}
        assert _run(stub, T.translate_text("Hello", "ru", model="m")) == "Привет"

    def test_segments_are_translated_end_to_end(self, stub):
        segs = [{"start": 0, "end": 1, "text": "Hello"},
                {"start": 1, "end": 2, "text": "World"}]
        out = _run(stub, T.translate_segments(segs, "ru", model="m", max_concurrent=1))
        assert [s["translated_text"] for s in out] == ["Привет", "Привет"]

    def test_failed_segments_fall_back_to_source_after_retries(self, stub):
        stub.native_status = 500
        stub.native_body = {"error": {"message": "boom"}}
        segs = [{"start": 0, "end": 1, "text": "Hello"}]
        out = _run(stub, T.translate_segments(segs, "ru", model="m", max_concurrent=1))
        # Still marked untranslated (== source) so the retry-stage UI can
        # offer "only failed segments" with a different model.
        assert out[0]["translated_text"] == "Hello"


# ── Thinking-model handling ──────────────────────────────────────────

class TestNoThinkSwitch:
    """qwen3.6-27b accepts `reasoning: "off"` and then thinks anyway (1606
    reasoning tokens for an 8-word sentence, measured). Qwen's in-prompt
    switch does work, so we append it for Qwen models."""

    @pytest.mark.parametrize("model,expected", [
        ("qwen/qwen3.6-27b", True),
        ("qwen/qwen3-8b", True),
        ("qwq-32b", True),
        ("google/gemma-4-e4b", False),
        ("qwen2.5:14b", False),      # not a thinking model — leave it alone
        ("", False),
    ])
    def test_only_qwen_thinking_models_get_the_suffix(self, model, expected):
        assert T._supports_no_think(model) is expected

    def test_suffix_is_appended_for_qwen(self, stub, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_REASONING", "off")
        _run(stub, T.lm_studio_chat("Translate: hi", model="qwen/qwen3.6-27b"))
        assert stub.requests[0][1]["input"].endswith("/no_think")

    def test_suffix_is_not_appended_for_other_models(self, stub, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_REASONING", "off")
        _run(stub, T.lm_studio_chat("Translate: hi", model="google/gemma-4-e4b"))
        assert "/no_think" not in stub.requests[0][1]["input"]

    def test_suffix_is_not_appended_when_reasoning_is_wanted(self, stub, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_REASONING", "high")
        _run(stub, T.lm_studio_chat("Translate: hi", model="qwen/qwen3.6-27b"))
        assert "/no_think" not in stub.requests[0][1]["input"]

    def test_echoed_control_token_is_stripped_from_output(self):
        assert T._clean_translation("Привет /no_think") == "Привет"

    def test_warns_once_when_a_model_ignores_reasoning_off(self, monkeypatch, caplog):
        monkeypatch.setattr(T, "LM_STUDIO_REASONING", "off")
        monkeypatch.setattr(T, "_LM_WARNED_IGNORED_REASONING", False)
        payload = {
            "output": [{"type": "reasoning", "content": "…"},
                       {"type": "message", "content": "Привет"}],
            "stats": {"reasoning_output_tokens": 1606, "total_output_tokens": 1620},
        }
        with caplog.at_level("WARNING"):
            for _ in range(3):
                assert T._parse_native_response(payload) == "Привет"
        warnings = [r for r in caplog.records if "ignored reasoning" in r.message]
        assert len(warnings) == 1, "should warn once, not once per segment"


# ── Batched translation over HTTP ────────────────────────────────────

def _numbering_responder(prefix="ES:", drop_line=None, unnumbered=False):
    """Stand in for a well-behaved (or misbehaving) translator model.

    It echoes each source line back with a prefix, so the reply is always in
    the Latin script. These tests therefore dub into Spanish: against a
    Cyrillic target the echo would (rightly) be rejected as "the model
    answered in the wrong language" and every alignment assertion here would
    be testing that guard instead of the batching it means to test.
    """
    def respond(body):
        text = body.get("input") or body["messages"][-1]["content"]
        lines = []
        if "Lines to translate:" in text:
            work = text.split("Lines to translate:", 1)[1]
            for raw in work.splitlines():
                m = re.match(r"^(\d+)\.\s+(.*)$", raw.strip())
                if m:
                    lines.append((int(m.group(1)), m.group(2)))
        if not lines:                       # single-line prompt: "Text: foo"
            src = text.split("Text:")[-1].split("\n\n")[0].strip()
            return 200, {"output": [{"type": "message",
                                     "content": f"{prefix} {src}"}]}
        out = []
        for n, src in lines:
            if n == drop_line:
                continue
            out.append(f"{prefix} {src}" if unnumbered else f"{n}. {prefix} {src}")
        return 200, {"output": [{"type": "message", "content": "\n".join(out)}]}
    return respond


@pytest.fixture
def batching(monkeypatch):
    """Deterministic batch sizing, no glossary, no dedupe surprises.

    `LM_STUDIO_MAX_CONCURRENT` is pinned off because it is read from the
    environment at import time, and whether the repo's `.env` has been loaded
    by then depends on whether some *other* test imported `server` first
    (server.py calls load_dotenv before importing this module). That made the
    whole class order-dependent: run alone the setting was 0, run after a
    route test it was 2, and the tests that assert on cross-batch context
    only hold at a concurrency of 1. Tests that care about the ceiling set it
    themselves — see TestConcurrencyCeiling.
    """
    monkeypatch.setattr(T, "TRANSLATE_BATCH_SIZE", 4)
    monkeypatch.setattr(T, "TRANSLATE_BATCH_CHARS", 100_000)
    monkeypatch.setattr(T, "TRANSLATE_CONTEXT_LINES", 2)
    monkeypatch.setattr(T, "LM_STUDIO_MAX_CONCURRENT", 0)
    monkeypatch.setattr(T, "_GLOSSARY_CACHE", {"ru": {}})


def _segs(*texts):
    return [{"start": i, "end": i + 1, "text": t} for i, t in enumerate(texts)]


class TestUntranslatedRepliesAreRejected:
    """A reply that isn't a translation must not reach the TTS engine.

    Observed on a real dub: gemma-4-e4b answered a batch with its whole
    chain-of-thought as plain prose (no <think> tags, so _strip_reasoning
    couldn't see it). The batch parse failed, the ladder split down to single
    lines, and the single-line path — which had no validation at all — handed
    838 characters of English commentary back as the translation of a
    132-character line. All six segments were synthesized that way.
    """

    def test_wrong_script_reply_is_discarded_and_the_source_survives(
            self, stub, batching):
        # Echoes the source (Spanish) back while the target is Russian.
        stub.native_body = _numbering_responder()
        segs = _segs("Hola mamá, te amo mucho", "Gracias por todo, mamá")

        out = _run(stub, T.translate_segments(
            segs, "ru", model="m", max_concurrent=1))

        # Falls back to source text, which server.py reports as untranslated
        # so the UI can offer "retry failed segments only".
        assert [s["translated_text"] for s in out] == [
            "Hola mamá, te amo mucho", "Gracias por todo, mamá"]

    def test_narrated_reasoning_is_discarded(self, stub, batching):
        def respond(body):
            return 200, {"output": [{"type": "message", "content": (
                "The user wants me to translate this text into Spanish. "
                "Source Text: ... Breakdown and Translation: " + "x" * 300)}]}

        stub.native_body = respond
        segs = _segs("Gracias por todo lo que has hecho por mí")

        out = _run(stub, T.translate_segments(
            segs, "es", model="m", max_concurrent=1))

        assert out[0]["translated_text"] == "Gracias por todo lo que has hecho por mí"


class TestConcurrencyCeiling:
    """LM_STUDIO_MAX_CONCURRENT is a ceiling on the caller, never a floor.

    It used to be assigned unconditionally, so a caller asking for 1 got
    whatever the config said. That is not a harmless speed-up: above a
    concurrency of 1 the next batch is built before the previous one has
    returned, so `prev_context` can only offer bare source lines and the
    continuity prompt the caller asked for silently disappears.
    `tools/gochidubb_benchmark.py` passes max_concurrent=1 for exactly that
    prompt, and on any machine using this repo's .env (which sets 2) it had
    been measuring a different one.
    """

    def test_a_config_ceiling_never_raises_the_callers_choice(
            self, stub, batching, monkeypatch):
        monkeypatch.setattr(T, "USE_LM_STUDIO", True)
        monkeypatch.setattr(T, "LM_STUDIO_MAX_CONCURRENT", 4)
        stub.native_body = _numbering_responder()
        segs = _segs(*[f"line {i}" for i in range(8)])

        _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))

        # Still serial, so batch 2 sees batch 1's *translations*. Under the
        # bug the batches overlapped and this was the bare source line.
        second = stub.requests[1][1]["input"]
        assert "ES: line 3" in second

    def test_the_ceiling_still_lowers_a_caller_that_asks_for_more(
            self, stub, batching, monkeypatch, caplog):
        monkeypatch.setattr(T, "USE_LM_STUDIO", True)
        monkeypatch.setattr(T, "LM_STUDIO_MAX_CONCURRENT", 1)
        stub.native_body = _numbering_responder()
        segs = _segs(*[f"line {i}" for i in range(8)])

        with caplog.at_level("INFO", logger="pipeline.translator"):
            _run(stub, T.translate_segments(segs, "es", model="m",
                                            max_concurrent=8))
        assert "Limiting translation concurrency to 1" in caplog.text

    def test_no_ceiling_is_announced_when_none_was_applied(
            self, stub, batching, monkeypatch, caplog):
        """The log line used to fire on a value that changed nothing."""
        monkeypatch.setattr(T, "USE_LM_STUDIO", True)
        monkeypatch.setattr(T, "LM_STUDIO_MAX_CONCURRENT", 4)
        stub.native_body = _numbering_responder()

        with caplog.at_level("INFO", logger="pipeline.translator"):
            _run(stub, T.translate_segments(_segs("line 0"), "es", model="m",
                                            max_concurrent=1))
        assert "Limiting translation concurrency" not in caplog.text


class TestBatchedTranslation:
    def test_many_segments_take_few_requests(self, stub, batching):
        """The point of the whole exercise: 8 subtitles, not 8 round trips."""
        stub.native_body = _numbering_responder()
        segs = _segs(*[f"line {i}" for i in range(8)])

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))

        assert [s["translated_text"] for s in out] == [
            f"ES: line {i}" for i in range(8)]
        assert len(stub.requests) == 2, "8 lines / batch of 4 = 2 requests"

    def test_each_line_keeps_its_own_translation(self, stub, batching):
        """Alignment is the thing that must not break — a shifted mapping
        would put every later subtitle onto the wrong audio segment."""
        stub.native_body = _numbering_responder()
        segs = _segs("first", "second", "third")

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))
        assert [s["translated_text"] for s in out] == [
            "ES: first", "ES: second", "ES: third"]

    def test_a_dropped_line_makes_the_batch_split_instead_of_shifting(
            self, stub, batching):
        stub.native_body = _numbering_responder(drop_line=2)
        segs = _segs("a", "b", "c", "d")

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))

        # Every line still maps to its own source, via the split-and-retry
        # ladder — never "b" translated as "c".
        assert [s["translated_text"] for s in out] == [
            "ES: a", "ES: b", "ES: c", "ES: d"]
        assert len(stub.requests) > 1, "should have split the bad batch"

    def test_an_unnumbered_reply_falls_back_to_single_lines(self, stub, batching):
        stub.native_body = _numbering_responder(unnumbered=True)
        segs = _segs("a", "b")

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))
        assert [s["translated_text"] for s in out] == ["ES: a", "ES: b"]

    def test_empty_segments_cost_no_requests(self, stub, batching):
        stub.native_body = _numbering_responder()
        segs = _segs("", "   ", "")

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))
        assert [s["translated_text"] for s in out] == ["", "", ""]
        assert stub.requests == []

    def test_repeated_short_lines_are_translated_once(self, stub, batching):
        stub.native_body = _numbering_responder()
        segs = _segs("Okay.", "something else entirely", "Okay.", "okay.")

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))

        assert [s["translated_text"] for s in out] == [
            "ES: Okay.", "ES: something else entirely", "ES: Okay.", "ES: Okay."]
        # Only 2 unique lines were sent, so they fit in one batch
        sent = stub.requests[0][1]["input"]
        assert sent.count("Okay.") == 1

    def test_preceding_lines_are_given_as_context(self, stub, batching):
        stub.native_body = _numbering_responder()
        segs = _segs(*[f"line {i}" for i in range(8)])

        _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))

        second = stub.requests[1][1]["input"]
        assert "do NOT translate" in second
        assert "ES: line 3" in second, "should see the previous batch's output"

    def test_progress_is_reported(self, stub, batching):
        stub.native_body = _numbering_responder()
        seen = []
        segs = _segs(*[f"line {i}" for i in range(8)])

        _run(stub, T.translate_segments(
            segs, "es", model="m", max_concurrent=1,
            progress_callback=lambda done, total, eta: seen.append((done, total))))

        assert seen[-1] == (8, 8)
        assert [d for d, _ in seen] == sorted(d for d, _ in seen)

    def test_output_budget_covers_the_whole_batch(self, stub, batching):
        stub.native_body = _numbering_responder()
        segs = _segs(*["a rather long line of source text " * 40 for _ in range(4)])

        _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))
        assert stub.requests[0][1]["max_output_tokens"] > T.LM_STUDIO_MAX_OUTPUT_TOKENS

    def test_dead_server_gives_up_instead_of_retrying_every_line(
            self, stub, batching, monkeypatch):
        """Without the circuit breaker a 40-line job would fire ~200 doomed
        requests before finishing."""
        monkeypatch.setattr(T._Circuit, "LIMIT", 3)
        stub.native_status = 500
        stub.native_body = {"error": {"message": "engine crashed"}}
        segs = _segs(*[f"line {i}" for i in range(40)])

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))

        # Untranslated == source text, which the retry-failed-only UI keys on
        assert [s["translated_text"] for s in out] == [f"line {i}" for i in range(40)]
        assert len(stub.requests) <= 6

    def test_unknown_model_fails_fast(self, stub, batching):
        stub.native_status = 404
        stub.native_body = {"error": {
            "message": 'Invalid model identifier "nope".',
            "type": "invalid_request", "param": "model", "code": "model_not_found"}}
        segs = _segs(*[f"line {i}" for i in range(12)])

        with pytest.raises(T.LMStudioError, match="not available in LM Studio"):
            _run(stub, T.translate_segments(segs, "nope", model="nope",
                                            max_concurrent=1))


# ── Budget starvation vs timeout ─────────────────────────────────────

def _budget_gated_responder(needs_budget, prefix="ES:"):
    """A thinking model: answers only once its output budget is big enough.

    Below the threshold it burns the whole budget reasoning and returns no
    message — exactly what qwen/qwen3.6-27b did to a 16-line batch at 4096
    tokens (4094 reasoning, 1 of answer), and identically at 8 and 4 lines.
    """
    def respond(body):
        budget = body.get("max_output_tokens") or body.get("max_tokens") or 0
        if budget < needs_budget:
            return 200, {"output": [{"type": "reasoning", "content": "hmm " * 50}],
                         "stats": {"reasoning_output_tokens": budget - 2,
                                   "total_output_tokens": budget - 1}}
        text = body.get("input") or body["messages"][-1]["content"]
        lines = []
        if "Lines to translate:" in text:
            for raw in text.split("Lines to translate:", 1)[1].splitlines():
                m = re.match(r"^(\d+)\.\s+(.*)$", raw.strip())
                if m:
                    lines.append(m.group(2))
        if not lines:
            src = text.split("Text:")[-1].split("\n\n")[0].strip()
            return 200, {"output": [{"type": "message", "content": f"{prefix} {src}"}]}
        out = "\n".join(f"{i+1}. {prefix} {s}" for i, s in enumerate(lines))
        return 200, {"output": [{"type": "message", "content": out}]}
    return respond


class TestBudgetStarvation:
    def test_only_reasoning_raises_a_budget_error_not_a_generic_one(self, stub):
        stub.native_body = lambda body: (200, {
            "output": [{"type": "reasoning", "content": "thinking"}],
            "stats": {"reasoning_output_tokens": 4094, "total_output_tokens": 4095}})
        with pytest.raises(T.LMStudioBudgetError):
            _run(stub, T.lm_studio_chat("x", model="m"))

    def test_budget_is_raised_and_the_same_batch_retried(self, stub, batching):
        """The reasoning cost is near-constant per request, so splitting is
        the wrong move — the batch must be re-sent with more room."""
        stub.native_body = _budget_gated_responder(needs_budget=9000)
        segs = _segs(*[f"line {i}" for i in range(4)])

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))

        assert [s["translated_text"] for s in out] == [f"ES: line {i}" for i in range(4)]
        budgets = [b["max_output_tokens"] for _, b in stub.requests]
        assert budgets[0] < budgets[-1], "budget should have been escalated"
        # Same 4 lines every time — never split into smaller batches
        assert all("4. line 3" in b["input"] for _, b in stub.requests)

    def test_the_raised_budget_carries_to_later_batches(self, stub, batching):
        """Otherwise every batch in the job rediscovers this at minutes a go."""
        stub.native_body = _budget_gated_responder(needs_budget=9000)
        segs = _segs(*[f"line {i}" for i in range(8)])   # 2 batches of 4

        _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))

        budgets = [b["max_output_tokens"] for _, b in stub.requests]
        # The last request must not have started back at 1×
        assert budgets[-1] > budgets[0]
        starved = sum(1 for b in budgets if b < 9000)
        assert starved <= 3, f"should stop paying for starved requests: {budgets}"

    def test_budget_starvation_does_not_trip_the_circuit_breaker(self, stub, batching):
        """The server answered; it just answered with only reasoning. Treating
        that as 'backend down' abandoned a job smaller budgets could finish."""
        stub.native_body = _budget_gated_responder(needs_budget=9000)
        segs = _segs(*[f"line {i}" for i in range(12)])

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))
        assert all(s["translated_text"].startswith("ES:") for s in out)

    def test_gives_up_escalating_eventually(self, stub, batching):
        """A model stuck in a reasoning loop must not escalate forever."""
        stub.native_body = _budget_gated_responder(needs_budget=10 ** 9)
        segs = _segs("only line")

        out = _run(stub, T.translate_segments(segs, "es", model="m", max_concurrent=1))
        assert out[0]["translated_text"] == "only line"      # left untranslated
        budgets = [b["max_output_tokens"] for _, b in stub.requests]
        assert max(budgets) <= T._batch_output_budget(["only line"]) * \
            T._BudgetTuner.MAX_MULTIPLIER * 2


class TestBudgetAndTimeoutSizing:
    def test_budget_adds_reasoning_headroom_to_the_answer(self):
        """max() was the bug: a 1852-token answer estimate was clamped back
        down to the 4096 reasoning floor, leaving nothing for the answer."""
        texts = ["a fairly ordinary subtitle line of speech"] * 16
        budget = T._batch_output_budget(texts)
        assert budget > T.LM_STUDIO_MAX_OUTPUT_TOKENS, \
            "must leave room for the answer on top of the reasoning budget"

    def test_timeout_scales_with_the_budget(self):
        small = T._batch_timeout(T.LM_STUDIO_MAX_OUTPUT_TOKENS)
        big = T._batch_timeout(T.LM_STUDIO_MAX_OUTPUT_TOKENS * 3)
        assert small == pytest.approx(T.LM_STUDIO_TIMEOUT)
        assert big > small, "a batch that generates 3x as much needs 3x the clock"

    def test_timeout_never_shrinks_below_the_configured_value(self):
        assert T._batch_timeout(1) == pytest.approx(T.LM_STUDIO_TIMEOUT)


class TestTimeoutIsReportedWithItsType:
    def test_empty_str_timeout_still_produces_a_message(self, stub, monkeypatch):
        """str(TimeoutError()) is "" — the original logged
        "Batch of 16 failed (attempt 1/2):" and named no cause."""
        async def boom(*a, **k):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(T, "lm_studio_chat", boom)
        with pytest.raises(T.LMStudioTimeoutError) as ei:
            _run(stub, T._translate_batch(["a", "b"], "es", "m", "", None, None))
        assert str(ei.value), "must not be an empty message"
        assert "TimeoutError" in str(ei.value)


class TestTimeoutSizing:
    """A 182-segment job timed out on every batch because the clock was
    sized for a solo request while two ran concurrently against one model."""

    def test_scales_with_concurrency(self):
        solo = T._batch_timeout(T.LM_STUDIO_MAX_OUTPUT_TOKENS, concurrency=1)
        pair = T._batch_timeout(T.LM_STUDIO_MAX_OUTPUT_TOKENS, concurrency=2)
        assert pair == pytest.approx(solo * 2), \
            "two in-flight requests each take ~2x as long on one model instance"

    def test_stretch_extends_the_clock(self):
        base = T._batch_timeout(T.LM_STUDIO_MAX_OUTPUT_TOKENS, 1, stretch=1)
        assert T._batch_timeout(T.LM_STUDIO_MAX_OUTPUT_TOKENS, 1, stretch=3) > base

    def test_all_three_multipliers_compose(self):
        t = T._batch_timeout(T.LM_STUDIO_MAX_OUTPUT_TOKENS * 2, concurrency=2, stretch=3)
        assert t == pytest.approx(T.LM_STUDIO_TIMEOUT * 2 * 2 * 3)


class TestTimeoutRetriesBeforeSplitting:
    def test_a_slow_batch_gets_a_longer_clock_not_a_split(self, stub, batching):
        """Splitting is the wrong first move when the model's cost is
        per-request: each half pays the reasoning again."""
        calls = {"n": 0}

        real_chat = T.lm_studio_chat

        async def slow_once(prompt, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise asyncio.TimeoutError()
            return await real_chat(prompt, **kw)

        stub.native_body = _numbering_responder()
        import pipeline.translator as mod
        mod.lm_studio_chat = slow_once
        try:
            segs = _segs("a", "b", "c", "d")
            out = _run(stub, T.translate_segments(segs, "es", model="m",
                                                  max_concurrent=1))
        finally:
            mod.lm_studio_chat = real_chat

        assert [s["translated_text"] for s in out] == \
            ["ES: a", "ES: b", "ES: c", "ES: d"]
        # The retry must be the SAME 4 lines, not two halves
        assert any("4. d" in b["input"] for _, b in stub.requests)
