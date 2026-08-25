"""Review gates — the one model behind every human pause in the pipeline.

Pure functions and tables only: no server import, no config import, no I/O.
The server adapts these into its driver (`server._evaluate_gate`) and its
`/continue` route; tests exercise them directly.

A *gate* is a named review point (``GATES``). Each gate arms after one
pipeline stage (``BOUNDARY_GATES``) and, when it fires, parks the job in a
dedicated status at a named checkpoint (``GATE_STATUS``). A gate runs in one
of three ``MODES``:

  off           never pauses
  on            always pauses
  flagged_only  pauses only when the boundary's findings count is non-zero —
                and when that count could not be computed at all
                (findings=None), it pauses too. That fail-safe direction is
                deliberate: today's quality gate is non-fatal (a scorer that
                raises must not fail a dub), but a *requested* flagged_only
                review that cannot be computed must not silently ship
                unreviewed work.
"""
from typing import Dict, Iterable, List, Optional

GATES = ("transcript", "translation", "voice_cast", "subtitles", "final_qc")
MODES = ("off", "on", "flagged_only")

# Mirrors server.STAGE_ORDER — asserted equal by the route test suite so the
# two cannot drift apart without a test failing. Kept here (rather than
# imported) so this module stays importable without pulling in server.py.
PIPELINE_STAGE_ORDER = (
    "download", "extract", "transcribe", "diarize",
    "translate", "tts", "assemble", "merge",
)

# Gate boundaries: which gates arm after which pipeline stage, in order.
# Two gates share the translate boundary — translation is reviewed before
# the cast, matching the previously mutually-exclusive wizard modes.
#
# Voice casting gates after translate, not after diarize, even though the
# speaker references exist by then. Casting is only worth reviewing if you
# can hear it, and a preview has to speak the lines the dub will actually
# speak — at the diarize gate the text is still in the source language, so a
# cross-lingual preview would demonstrate the wrong phonetics. Translation is
# minutes; the stage this gate protects is hours (a measured 12.25h for 1602
# segments), so the gate still sits in front of essentially all of the cost.
BOUNDARY_GATES: Dict[str, List[str]] = {
    "diarize":   ["transcript"],
    "translate": ["translation", "voice_cast"],
    "assemble":  ["subtitles"],
    "merge":     ["final_qc"],
}

# gate -> (the status it parks the job in, the checkpoint it parks at)
GATE_STATUS: Dict[str, tuple] = {
    "transcript":  ("awaiting_transcript_review",  "transcription_done"),
    "translation": ("awaiting_translation_review", "translation_done"),
    "voice_cast":  ("awaiting_voice_review",       "translation_done"),
    "subtitles":   ("awaiting_subtitle_review",    "assemble_done"),
    "final_qc":    ("awaiting_final_qc",           "merge_done"),
}

# Reverse lookups, derived so they cannot disagree with the tables above.
GATE_FOR_STATUS: Dict[str, str] = {v[0]: k for k, v in GATE_STATUS.items()}
BOUNDARY_FOR_GATE: Dict[str, str] = {
    g: stage for stage, names in BOUNDARY_GATES.items() for g in names
}

# Legacy wizard_mode -> the single gate it arms.
_WIZARD_GATE = {
    "review_transcript": "transcript",
    "review_translation": "translation",
    "review_voices": "voice_cast",
}


def all_off() -> Dict[str, str]:
    return {g: "off" for g in GATES}


def defaults_from_cfg(cfg) -> Dict[str, str]:
    """Assemble cfg-default gate modes from `review_gate_*` config fields.

    getattr with a default so an older config object (or a test double)
    degrades to "everything off" rather than raising.
    """
    out = {}
    for g in GATES:
        mode = getattr(cfg, f"review_gate_{g}", "off")
        out[g] = mode if mode in MODES else "off"
    return out


def sanitize_explicit(explicit: dict) -> Dict[str, str]:
    """Validate a caller-supplied {gate: mode} dict.

    Raises ValueError naming the offending key/value; unknown gates and
    unknown modes are refused rather than dropped, so an API caller's typo
    ("translaton": "on") is an error instead of a silently unarmed gate.
    """
    if not isinstance(explicit, dict):
        raise ValueError("review_gates must be a JSON object of {gate: mode}")
    out = {}
    for k, v in explicit.items():
        if k not in GATES:
            raise ValueError(
                f"unknown gate {k!r} — valid gates: {', '.join(GATES)}")
        if v not in MODES:
            raise ValueError(
                f"gate {k!r}: mode {v!r} is not one of {', '.join(MODES)}")
        out[k] = v
    return out


def resolve_gates(explicit: Optional[dict], wizard_mode: Optional[str],
                  cfg_defaults: Dict[str, str]) -> Dict[str, str]:
    """Resolve the effective gate set for a job.

    Precedence: explicit review_gates > wizard_mode mapping > cfg defaults.

    Any wizard_mode that was *sent* (INCLUDING "auto" and "") suppresses cfg
    defaults — this is the backward-compat contract: creator.html and the CLI
    always send wizard_mode, so Pro-configured default gates can never
    surprise them. cfg defaults apply only when wizard_mode is None (absent
    from the request entirely). The wizard mapping arms exactly one gate:
    review_transcript → transcript, review_translation → translation,
    review_voices → voice_cast; "auto"/"" arm none.
    """
    if wizard_mode is None:
        gates = all_off()
        for g, mode in (cfg_defaults or {}).items():
            if g in gates and mode in MODES:
                gates[g] = mode
    else:
        gates = all_off()
        armed = _WIZARD_GATE.get(str(wizard_mode))
        if armed:
            gates[armed] = "on"
    if explicit:
        gates.update(sanitize_explicit(explicit))
    return gates


def first_pending(stage_id: str, gates: Dict[str, str],
                  cleared: Iterable[str],
                  findings: Dict[str, Optional[int]]) -> Optional[str]:
    """First gate at this boundary that should pause the pipeline.

    A gate pauses when it is armed ("on", or "flagged_only" with findings),
    and is not already in `cleared`. findings[gate] is None when the caller
    could not compute the count at all — compute failure ⇒ pause (fail safe;
    see the module docstring for why this inverts the legacy quality gate's
    never-fatal philosophy).
    """
    cleared = set(cleared or ())
    for gate in BOUNDARY_GATES.get(stage_id, ()):
        if gate in cleared:
            continue
        mode = gates.get(gate, "off")
        if mode == "on":
            return gate
        if mode == "flagged_only":
            count = findings.get(gate)
            if count is None or count > 0:
                return gate
    return None


def reset_cleared(stage_id: str, cleared: Iterable[str]) -> List[str]:
    """Cleared-gates list after re-running stage `stage_id`.

    Running a stage re-arms every gate at that stage's boundary and every
    later one — a retranslate re-arms the translation gate (and everything
    after it), a re-assemble re-arms the subtitle gate. Gates at earlier
    boundaries stay cleared: their inputs did not change.
    """
    try:
        pos = PIPELINE_STAGE_ORDER.index(stage_id)
    except ValueError:
        return list(cleared or ())
    rearmed = {
        g
        for boundary, names in BOUNDARY_GATES.items()
        if PIPELINE_STAGE_ORDER.index(boundary) >= pos
        for g in names
    }
    return [g for g in (cleared or ()) if g not in rearmed]
