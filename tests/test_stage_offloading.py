"""Long pipeline calls must not run on the event loop.

`_blocking` exists because a synchronous stage call pins the single asyncio
thread and every HTTP request queues behind it — its docstring says so, and
names the symptom: the UI cannot fetch /api/jobs while a stage runs.

Synthesis was the one call that never got moved, and it is the longest in the
pipeline. Measured against a live dub: /api/system did not answer in 90
seconds. A twelve-hour synthesis stage meant twelve hours with no UI, no CLI
and no MCP — and, once voice casting existed, no way to preview a cast while
anything was rendering.

This is a structural test rather than a behavioural one on purpose. The bug
is not that a call returns the wrong value; it is that a call is written
without `await _blocking`, which no amount of exercising the function will
catch. Reading the source is the only thing that does.
"""
import ast
import pathlib

import pytest

SERVER = pathlib.Path(__file__).resolve().parent.parent / "server.py"
TREE = ast.parse(SERVER.read_text(encoding="utf-8"))

# (function, method) pairs that must never be called bare. Each is a call
# that can run for minutes or hours.
OFFLOAD_REQUIRED = [
    ("_stage_tts", "synthesize_segments"),
    ("_run_tts_and_merge_stage", "synthesize_segments"),
]


def _function(name):
    for node in ast.walk(TREE):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name):
            return node
    raise AssertionError(f"{name} not found in server.py — did it get renamed?")


def _calls_named(node, method):
    """Every Call whose callee is `<something>.method(...)`."""
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == method]


def _offloaded_names(node):
    """Attribute names passed as the first argument to `_blocking(fn, ...)`.

    That is the correct shape — `await _blocking(obj.method, arg, ...)` — and
    it passes the method as a value, so it never appears as a Call node.
    """
    out = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_blocking" and n.args):
            first = n.args[0]
            if isinstance(first, ast.Attribute):
                out.add(first.attr)
            elif isinstance(first, ast.Name):
                out.add(first.id)
    return out


@pytest.mark.parametrize("func_name,method", OFFLOAD_REQUIRED)
def test_the_long_call_is_handed_to_a_worker_thread(func_name, method):
    fn = _function(func_name)
    bare = _calls_named(fn, method)
    assert not bare, (
        f"{func_name} calls {method}() directly on the event loop. Every HTTP "
        f"request queues behind it for as long as it runs — hours, for "
        f"synthesis. Pass it to _blocking instead: "
        f"`await _blocking(engine.{method}, ...)`."
    )
    assert method in _offloaded_names(fn), (
        f"{func_name} no longer hands {method} to _blocking at all — if the "
        f"call moved somewhere else, move this test with it rather than "
        f"deleting it."
    )


def test_blocking_really_uses_a_thread():
    """The guarantee above is only worth anything if _blocking offloads."""
    fn = _function("_blocking")
    src = ast.dump(fn)
    assert "to_thread" in src or "run_in_executor" in src, (
        "_blocking stopped offloading, which silently un-fixes every caller"
    )
