"""Deciding what to try next when a download fails.

Extracted from downloader.py because the retry ladder there had grown into a
staircase that only went one way. It escalated — preferred format, then the
web_embedded player client, then a simpler format — and never came back. That
turned out to matter: measured on a real job, the default client hit a
transient 403, the ladder escaped to web_embedded, web_embedded could not
solve YouTube's JS challenge, its format list collapsed to "only images are
available", and the run died on "Requested format is not available". Seconds
later the default client downloaded the same video at 1080p on the first try.

So the escape hatch turned a failure that would have cleared on its own into
a permanent one, and nothing in the ladder could notice or walk back.

What this module knows:

* `classify` — which of a small set of shapes a failure has. A shape, not a
  message: the same underlying problem reaches us worded differently by
  yt-dlp version and player client.
* `plan` — given what has been tried and how the last attempt failed, what to
  try next. This is a policy, not a fixed sequence, so it can go back to a
  client that was working, wait out something transient, or stop.

Every strategy is an entry in STRATEGIES. That closed table is the whole
action space: nothing else can reach the subprocess. It is what makes the
optional LLM advisor safe — see `advise`, which chooses a key from this table
and can never write an argument.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("gochidubb.rescue")


# ── Failure shapes ──────────────────────────────────────────────────────

FORBIDDEN = "forbidden"                 # HTTP 403 from the extractor
FORMAT_GONE = "format-unavailable"      # the selector matched nothing
CHALLENGE = "challenge-unsolved"        # JS challenge yt-dlp could not solve
BOT_CHECK = "bot-check"                 # wants cookies from a signed-in browser
AGE_GATE = "age-gate"
GEO_BLOCK = "geo-block"
GONE = "gone"                           # deleted, private, terminated
NETWORK = "network"                     # timeout, connection reset, DNS
UNKNOWN = "unknown"

# Shapes no amount of retrying will fix. Trying anyway wastes a minute and
# teaches the user that the retry counter means nothing.
TERMINAL = frozenset({GONE, AGE_GATE, GEO_BLOCK, BOT_CHECK})

_SIGNATURES = (
    # Order matters: the specific before the generic, because YouTube
    # frequently serves a bot-check message alongside a 403.
    (GONE, ("video unavailable", "this video has been removed",
            "private video", "account associated with this video has been "
            "terminated", "video is not available")),
    (BOT_CHECK, ("sign in to confirm you're not a bot",
                 "use --cookies-from-browser or --cookies")),
    (AGE_GATE, ("sign in to confirm your age", "age-restricted",
                "inappropriate for some users")),
    (GEO_BLOCK, ("not available in your country",
                 "not made this video available in your country",
                 "video is geo restricted")),
    # Both of these mean the client we are talking to cannot see real media,
    # which is a different problem from "this video has no 1080p".
    (CHALLENGE, ("challenge solving failed", "failed to solve",
                 "unable to solve", "could not solve", "no challenge solver",
                 "nsig extraction failed", "unable to extract nsig",
                 "only images are available")),
    (FORMAT_GONE, ("requested format is not available",
                   "requested format not available")),
    (NETWORK, ("timed out", "connection reset", "temporary failure in name "
               "resolution", "connection refused", "network is unreachable")),
    (FORBIDDEN, ("http error 403", "status code 403", "403: forbidden")),
)


def classify(stderr: str = "", stdout: str = "") -> str:
    """The shape of a failure, from whatever yt-dlp printed."""
    blob = f"{stderr or ''} {stdout or ''}".lower().replace("’", "'")
    for shape, needles in _SIGNATURES:
        if any(n in blob for n in needles):
            return shape
    return UNKNOWN


# ── The closed set of things we know how to try ─────────────────────────
# `fmt` names a format selector defined by the caller; `args` are extra
# yt-dlp flags. Adding a strategy means adding a row here — which is exactly
# the property that makes the action space auditable.

@dataclass(frozen=True)
class Strategy:
    key: str
    fmt: str                    # "preferred" | "simple" | "any"
    args: tuple = ()
    sleep: float = 0.0          # seconds to wait first
    why: str = ""
    needs_opt_in: bool = False  # gated on user configuration


STRATEGIES: dict = {
    "preferred": Strategy(
        "preferred", "preferred", (), 0.0,
        "the format we actually want: H.264 + AAC, stream-copyable later"),
    "simple": Strategy(
        "simple", "simple", (), 0.0,
        "a plain progressive mp4 — fewer ways for a selector to miss"),
    "any": Strategy(
        "any", "any", (), 0.0,
        "whatever exists; a re-encode beats no dub"),
    # A transient 403 clears on its own. Waiting is the cheapest fix there is
    # and the ladder never tried it, preferring to degrade the client instead.
    "wait_preferred": Strategy(
        "wait_preferred", "preferred", (), 4.0,
        "the same request again after a pause — most 403s here are transient"),
    "wait_any": Strategy(
        "wait_any", "any", (), 8.0,
        "one more pause, loosest format, before giving up"),
    "web_embedded": Strategy(
        "web_embedded", "preferred",
        ("--extractor-args", "youtube:player_client=web_embedded"), 0.0,
        "the embedded player, which is served different formats"),
    "tv_client": Strategy(
        "tv_client", "any",
        ("--extractor-args", "youtube:player_client=tv"), 0.0,
        "the TV client, which needs no JS challenge"),
    "remote_components": Strategy(
        "remote_components", "preferred", (), 0.0,
        "let yt-dlp fetch a solver for the JS challenge", needs_opt_in=True),
}


@dataclass
class Attempt:
    """One planned run, and why it was chosen. `why` is logged and persisted
    so a rescue that did not work can be read back afterwards."""
    strategy: Strategy
    reason: str = ""
    extra_args: tuple = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return self.strategy.key


# ── Policy ──────────────────────────────────────────────────────────────

# A cap, not a target. Each attempt is a real network round trip; a ladder
# that can run forever turns one bad link into a stalled queue.
MAX_ATTEMPTS = 6


def plan(shape: str, tried: list, *, remote_components: str = "") -> Attempt | None:
    """What to try after a failure of `shape`, given the keys already tried.

    Returns None when there is nothing sensible left, which the caller should
    treat as final rather than looping.
    """
    if len(tried) >= MAX_ATTEMPTS:
        return None
    if shape in TERMINAL:
        # Cookies or a different account would fix these, and neither is
        # something a retry can supply.
        return None

    def first(*keys):
        for k in keys:
            if k in tried:
                continue
            s = STRATEGIES[k]
            if s.needs_opt_in and not remote_components:
                continue
            return s
        return None

    if shape == CHALLENGE:
        # The client we are on cannot see media. Going back to the default
        # client is the fix that costs nothing; the solver is the fix that
        # costs a download of remote code, so it comes second and only with
        # consent.
        chosen = first("preferred", "tv_client", "remote_components", "wait_any")
        reason = "this player client cannot see real formats without solving a JS challenge"
    elif shape == FORMAT_GONE:
        # Loosen before switching client: a selector that missed is a much
        # more likely cause than a client that is broken.
        chosen = first("simple", "any", "preferred", "tv_client")
        reason = "the format selector matched nothing on this client"
    elif shape == FORBIDDEN:
        # Wait first. Escalating to a degraded client on a transient 403 is
        # the exact move that turned a recoverable failure into a dead one.
        chosen = first("wait_preferred", "web_embedded", "tv_client", "wait_any")
        reason = "403 from the extractor, which is usually transient here"
    elif shape == NETWORK:
        chosen = first("wait_preferred", "wait_any")
        reason = "the network, not the video"
    else:  # UNKNOWN
        chosen = first("simple", "any", "wait_any", "tv_client")
        reason = "unrecognised failure — widening before giving up"

    if chosen is None:
        return None
    return Attempt(chosen, reason)


def describe(shape: str) -> str:
    """One line for a human, for the job's error panel."""
    return {
        FORBIDDEN: "YouTube refused the request (403). Usually temporary.",
        FORMAT_GONE: "No format matched what we asked for on that client.",
        CHALLENGE: "YouTube served a JavaScript challenge yt-dlp could not solve.",
        BOT_CHECK: "YouTube wants cookies from a signed-in browser.",
        AGE_GATE: "The video is age-restricted and needs a signed-in account.",
        GEO_BLOCK: "The video is not available in this country.",
        GONE: "The video is gone — deleted, private, or the channel was removed.",
        NETWORK: "The network failed, not the video.",
        UNKNOWN: "yt-dlp failed in a way we do not have a rule for yet.",
    }.get(shape, "Download failed.")


# ── Optional LLM advisor ────────────────────────────────────────────────

# Deliberately narrow. The model never writes a command, a flag or an
# argument: it picks a key out of STRATEGIES, and a key that is not in the
# table is discarded. The rules above already cover every failure this
# install has actually seen, so the model is only consulted for UNKNOWN —
# the long tail, where a rule does not exist yet and the alternative is
# giving up.
#
# It is also the wrong tool for the common case even when it works. It costs
# a model round trip, and on a single-GPU box the translation model and
# VoxCPM are already competing for the same VRAM.

_ADVISOR_SYSTEM = (
    "You triage yt-dlp download failures. Answer with exactly one word: the "
    "key of the strategy to try next, chosen from the list you are given. "
    "No explanation, no punctuation, no flags. If none fit, answer: none."
)


async def advise(stderr: str, tried: list, model: str,
                 *, remote_components: str = "") -> Attempt | None:
    """Ask the local LLM which known strategy to try next. Never raises.

    Returns None on any doubt — a bad connection, a slow model, an answer
    that is not one of the keys. Falling back to "stop" is correct: the
    deterministic policy has already had its turn by the time this is called.
    """
    options = [k for k, s in STRATEGIES.items()
               if k not in tried and not (s.needs_opt_in and not remote_components)]
    if not options:
        return None
    menu = "\n".join(f"{k}: {STRATEGIES[k].why}" for k in options)
    prompt = (
        f"yt-dlp failed with:\n\n{(stderr or '')[-1200:]}\n\n"
        f"Already tried, in order: {', '.join(tried) or 'nothing'}\n\n"
        f"Strategies available:\n{menu}\n\n"
        f"Which one key should be tried next?"
    )
    try:
        from pipeline.translator import lm_studio_chat
        answer = await lm_studio_chat(
            prompt, model, system_prompt=_ADVISOR_SYSTEM,
            temperature=0.0, max_output_tokens=16, timeout=20,
        )
    except Exception as e:
        log.info(f"[rescue] advisor unavailable ({type(e).__name__}); "
                 f"stopping instead of guessing")
        return None

    # Take the first bare word that is a real key. A model that returns
    # prose, a flag, or a key it was not offered gets ignored rather than
    # sanitised — there is nothing safe to salvage from a wrong answer.
    for token in (answer or "").replace("`", " ").replace(",", " ").split():
        key = token.strip().strip(".:\"'").lower()
        if key in options:
            log.info(f"[rescue] advisor chose {key!r}")
            return Attempt(STRATEGIES[key], "suggested by the local model")
    log.info(f"[rescue] advisor gave no usable key (said {(answer or '')[:60]!r})")
    return None
