"""Pure tests for app/review_gates.py — no server import, no I/O.

The precedence contract and the fail-safe rule are the two things a future
refactor is most likely to break quietly; every test here states which one
it defends.
"""
import pytest

from app import review_gates as rg


# ── resolve_gates: precedence ───────────────────────────────────────────

DEFAULTS_ALL_ON = {g: "on" for g in rg.GATES}


def test_absent_wizard_mode_applies_cfg_defaults():
    gates = rg.resolve_gates(None, None, {"translation": "flagged_only"})
    assert gates["translation"] == "flagged_only"
    assert all(gates[g] == "off" for g in rg.GATES if g != "translation")


@pytest.mark.parametrize("mode", ["auto", ""])
def test_a_sent_wizard_mode_suppresses_cfg_defaults(mode):
    """The backward-compat contract: creator.html and the CLI always send
    wizard_mode, so Pro-configured default gates can never surprise them —
    even "auto" (and the degenerate "") means "no pauses, please"."""
    gates = rg.resolve_gates(None, mode, DEFAULTS_ALL_ON)
    assert gates == rg.all_off()


@pytest.mark.parametrize("wizard,gate", [
    ("review_transcript", "transcript"),
    ("review_translation", "translation"),
    ("review_voices", "voice_cast"),
])
def test_wizard_modes_arm_exactly_one_gate(wizard, gate):
    gates = rg.resolve_gates(None, wizard, DEFAULTS_ALL_ON)
    assert gates[gate] == "on"
    assert all(gates[g] == "off" for g in rg.GATES if g != gate)


def test_explicit_gates_beat_both_wizard_mode_and_defaults():
    gates = rg.resolve_gates(
        {"translation": "off", "final_qc": "on"},
        "review_translation", DEFAULTS_ALL_ON)
    assert gates["translation"] == "off"     # explicit beats wizard's "on"
    assert gates["final_qc"] == "on"         # explicit beats "auto suppresses"
    assert gates["subtitles"] == "off"       # wizard still suppressed defaults


def test_explicit_gates_merge_over_defaults_when_wizard_absent():
    gates = rg.resolve_gates({"subtitles": "off"}, None, DEFAULTS_ALL_ON)
    assert gates["subtitles"] == "off"
    assert gates["transcript"] == "on"       # untouched default survives


@pytest.mark.parametrize("bad", [
    {"translaton": "on"},                    # typo'd gate name
    {"translation": "always"},               # unknown mode
    ["translation"],                         # not a dict
])
def test_bad_explicit_gates_are_refused_not_dropped(bad):
    with pytest.raises(ValueError):
        rg.resolve_gates(bad, None, {})


def test_unknown_cfg_defaults_are_ignored_rather_than_fatal():
    gates = rg.resolve_gates(None, None, {"translation": "banana",
                                          "not_a_gate": "on"})
    assert gates == rg.all_off()


# ── defaults_from_cfg ───────────────────────────────────────────────────

def test_defaults_from_cfg_reads_the_five_fields_and_degrades():
    class Cfg:
        review_gate_translation = "flagged_only"
        review_gate_final_qc = "banana"      # invalid → off
        # the other three fields absent entirely → off
    d = rg.defaults_from_cfg(Cfg())
    assert d["translation"] == "flagged_only"
    assert d["final_qc"] == "off"
    assert set(d) == set(rg.GATES)


# ── first_pending ───────────────────────────────────────────────────────

def test_translation_is_reviewed_before_the_cast_at_the_shared_boundary():
    gates = {"translation": "on", "voice_cast": "on"}
    assert rg.first_pending("translate", gates, [], {}) == "translation"


def test_a_cleared_gate_yields_to_the_next_one_at_the_boundary():
    gates = {"translation": "on", "voice_cast": "on"}
    assert rg.first_pending("translate", gates, ["translation"], {}) == "voice_cast"
    assert rg.first_pending(
        "translate", gates, ["translation", "voice_cast"], {}) is None


def test_off_gates_never_pause():
    assert rg.first_pending("merge", {"final_qc": "off"}, [], {}) is None
    assert rg.first_pending("merge", {}, [], {}) is None


def test_flagged_only_sails_through_on_zero_findings():
    gates = {"subtitles": "flagged_only"}
    assert rg.first_pending("assemble", gates, [], {"subtitles": 0}) is None


def test_flagged_only_pauses_on_findings():
    gates = {"subtitles": "flagged_only"}
    assert rg.first_pending("assemble", gates, [], {"subtitles": 3}) == "subtitles"


def test_uncomputable_findings_pause_rather_than_ship_unreviewed():
    """The fail-safe rule: findings=None ⇒ pause. A requested flagged_only
    review whose check crashed must park the job, not wave it through."""
    gates = {"final_qc": "flagged_only"}
    assert rg.first_pending("merge", gates, [], {"final_qc": None}) == "final_qc"
    # ...and a missing entry is the same as None (the caller failed to
    # compute anything at all).
    assert rg.first_pending("merge", gates, [], {}) == "final_qc"


def test_non_boundary_stages_have_no_gates():
    assert rg.first_pending("tts", {g: "on" for g in rg.GATES}, [], {}) is None


# ── reset_cleared (gate re-arm on stage rerun) ──────────────────────────

ALL_CLEARED = list(rg.GATES)


def test_rerunning_translate_rearms_translation_and_everything_after():
    left = rg.reset_cleared("translate", ALL_CLEARED)
    assert left == ["transcript"]


def test_rerunning_diarize_rearms_every_gate():
    assert rg.reset_cleared("diarize", ALL_CLEARED) == []


def test_rerunning_assemble_keeps_earlier_boundaries_cleared():
    left = rg.reset_cleared("assemble", ALL_CLEARED)
    assert left == ["transcript", "translation", "voice_cast"]


def test_rerunning_a_non_boundary_stage_rearms_later_boundaries():
    """tts sits between the translate and assemble boundaries: its rerun
    changes the audio, so subtitle and final-QC approvals are stale — but
    the transcript/translation/cast approvals are not."""
    left = rg.reset_cleared("tts", ALL_CLEARED)
    assert left == ["transcript", "translation", "voice_cast"]


def test_an_unknown_stage_resets_nothing():
    assert rg.reset_cleared("not_a_stage", ALL_CLEARED) == ALL_CLEARED


# ── table consistency ───────────────────────────────────────────────────

def test_every_gate_has_a_boundary_a_status_and_a_reverse_entry():
    assert set(rg.BOUNDARY_FOR_GATE) == set(rg.GATES)
    assert set(rg.GATE_STATUS) == set(rg.GATES)
    for gate, (status, checkpoint) in rg.GATE_STATUS.items():
        assert rg.GATE_FOR_STATUS[status] == gate
        assert rg.BOUNDARY_FOR_GATE[gate] in rg.PIPELINE_STAGE_ORDER


def test_boundaries_are_real_pipeline_stages():
    for stage in rg.BOUNDARY_GATES:
        assert stage in rg.PIPELINE_STAGE_ORDER
