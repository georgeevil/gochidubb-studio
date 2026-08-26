"""Per-speaker voice casting.

The behaviour under test is a correction to one specific failure: picking a
style preset used to delete `speaker_refs` outright, so a three-speaker
interview came out in a single voice — and, because voice design is not
timbre-stable, not even reliably one. Casting replaces that whole-job switch
with an assignment, so the rules that matter are about *not* losing speakers:
an unassigned speaker keeps their own voice, a voice that fails to resolve
degrades for that speaker alone, and a typo is a 400 rather than a silent
no-op discovered after a twelve-hour render.
"""
import asyncio
import json

import pytest

import server


@pytest.fixture
def three_speakers():
    """A transcript shaped like the dub that prompted all this: one dominant
    speaker, one substantial second, one with almost nothing."""
    segs = []
    idx = 0
    for speaker, n in (("SPEAKER_00", 6), ("SPEAKER_02", 3), ("SPEAKER_01", 1)):
        for k in range(n):
            segs.append({
                "idx": idx, "start": idx * 2.0, "end": idx * 2.0 + 1.5,
                "speaker": speaker, "text": f"line {idx}",
                # Lengths deliberately vary so the mid-length pick is testable.
                "translated_text": "слово " * (k + 1),
            })
            idx += 1
    return {"segments": segs, "target_lang": "ru"}


# ── the map itself ──────────────────────────────────────────────────────

def test_unassigned_speakers_are_not_in_the_map_and_that_is_the_point():
    clean, errors = server._validate_cast(
        {"SPEAKER_00": "male_deep"}, {"SPEAKER_00", "SPEAKER_01"})
    assert errors == []
    assert clean == {"SPEAKER_00": "male_deep"}
    assert "SPEAKER_01" not in clean, (
        "an unnamed speaker must stay unnamed — _resolve_casting reads a "
        "missing entry as 'keep their own voice', which is the only default "
        "that cannot surprise anyone"
    )


@pytest.mark.parametrize("value", ["", "source"])
def test_empty_and_source_both_mean_keep_their_own_voice(value):
    clean, errors = server._validate_cast({"S": value}, {"S"})
    assert (clean, errors) == ({"S": "source"}, [])


def test_a_free_text_design_passes_through_intact():
    clean, errors = server._validate_cast(
        {"S": "design:gravelly older man, unhurried"}, {"S"})
    assert errors == []
    assert clean["S"] == "design:gravelly older man, unhurried"


def test_an_unknown_voice_is_rejected_not_dropped():
    """Dropping it would leave the speaker on their source voice and report
    success — the mistake would only surface after synthesis."""
    clean, errors = server._validate_cast({"S": "male_smooth_jazz"}, {"S"})
    assert clean == {}
    assert errors and "male_smooth_jazz" in errors[0]


def test_an_unknown_speaker_is_rejected():
    clean, errors = server._validate_cast({"SPEAKER_09": "male_deep"}, {"S"})
    assert clean == {}
    assert errors and "SPEAKER_09" in errors[0]


def test_every_builtin_preset_except_auto_is_assignable():
    choices = server._voice_choices()
    assert "auto" not in choices, "'auto' is the absence of a choice, not a voice"
    for pid, preset in server.VOICE_PRESETS.items():
        if pid == "auto":
            continue
        assert pid in choices
        assert choices[pid]["kind"] == "design"
        assert choices[pid]["style"] == preset["style"]


# ── the speaker summary ─────────────────────────────────────────────────

def test_speakers_are_summarised_loudest_first(three_speakers, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    rows = server._speaker_rows(three_speakers, "job1")

    assert [r["speaker"] for r in rows] == ["SPEAKER_00", "SPEAKER_02", "SPEAKER_01"]
    assert [r["segments"] for r in rows] == [6, 3, 1]
    assert rows[0]["share"] == 0.6
    assert sum(r["segments"] for r in rows) == 10


def test_a_speaker_without_an_extracted_ref_says_so(three_speakers, tmp_path, monkeypatch):
    """The picker offers "keep their own voice" for everyone; for a speaker
    with no reference cut from the video that option is a lie, and the UI
    needs to know in order to say so."""
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    refs = tmp_path / "job1" / "speaker_refs"
    refs.mkdir(parents=True)
    (refs / "ref_SPEAKER_00.wav").write_bytes(b"\0" * 64)

    rows = {r["speaker"]: r for r in server._speaker_rows(three_speakers, "job1")}
    assert rows["SPEAKER_00"]["has_source_ref"] is True
    assert rows["SPEAKER_00"]["audio_url"].endswith("/SPEAKER_00/audio")
    assert rows["SPEAKER_02"]["has_source_ref"] is False
    assert rows["SPEAKER_02"]["audio_url"] == ""


def test_the_audition_line_is_neither_the_shortest_nor_the_longest(
        three_speakers, tmp_path, monkeypatch):
    """"Да." proves nothing about a voice, and a 40-second paragraph makes the
    preview as slow as the render it exists to save you from."""
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    rows = {r["speaker"]: r for r in server._speaker_rows(three_speakers, "job1")}
    lengths = sorted(len(("слово " * (k + 1)).strip()) for k in range(6))
    picked = len(rows["SPEAKER_00"]["sample_text"])
    assert picked == lengths[len(lengths) // 2]
    assert lengths[0] < picked < lengths[-1]


# ── resolving a cast to actual reference audio ──────────────────────────

class _StubEngine:
    """Stands in for VoxCPM. design_reference is the only call _resolve_casting
    makes on the engine, and it is the expensive one."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def design_reference(self, style, output_path, voice_seed=None,
                         target_lang="en", sample_text=""):
        self.calls.append((style, voice_seed, target_lang))
        if self.fail:
            return None
        with open(output_path, "wb") as f:
            f.write(b"\0" * 2048)
        return output_path


def _resolve(cast, speakers, ctx, engine, tmp_path, monkeypatch, lang="ru",
             cache="designed"):
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "DESIGNED_VOICES_DIR", tmp_path / cache)
    return asyncio.run(server._resolve_casting(
        cast, speakers, ctx, "job1", engine, lang))


def _with_source_refs(tmp_path, *speakers):
    refs = tmp_path / "job1" / "speaker_refs"
    refs.mkdir(parents=True, exist_ok=True)
    out = {}
    for sp in speakers:
        p = refs / f"ref_{sp}.wav"
        p.write_bytes(b"\0" * 64)
        out[sp] = str(p)
    return {"speaker_refs": dict(out), "source_speaker_refs": dict(out)}


def test_an_unassigned_speaker_keeps_the_voice_from_the_video(tmp_path, monkeypatch):
    ctx = _with_source_refs(tmp_path, "SPEAKER_00", "SPEAKER_01")
    engine = _StubEngine()
    refs, transcripts, report = _resolve(
        {"SPEAKER_00": "male_deep"}, ["SPEAKER_00", "SPEAKER_01"],
        ctx, engine, tmp_path, monkeypatch)

    assert refs["SPEAKER_01"] == ctx["source_speaker_refs"]["SPEAKER_01"]
    assert refs["SPEAKER_00"] != refs["SPEAKER_01"]
    assert {r["speaker"]: r["status"] for r in report} == {
        "SPEAKER_00": "ok", "SPEAKER_01": "ok"}


def test_speakers_survive_a_cast_which_is_the_whole_point(tmp_path, monkeypatch):
    """The regression this feature exists to prevent: three speakers going in,
    one voice coming out."""
    ctx = _with_source_refs(tmp_path, "SPEAKER_00", "SPEAKER_01", "SPEAKER_02")
    engine = _StubEngine()
    refs, _, _ = _resolve(
        {"SPEAKER_00": "source", "SPEAKER_01": "male_deep",
         "SPEAKER_02": "female_calm"},
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        ctx, engine, tmp_path, monkeypatch)

    assert len(set(refs.values())) == 3, f"speakers collapsed onto {refs}"


def test_a_designed_voice_is_drawn_once_and_shared_by_everyone_cast_in_it(
        tmp_path, monkeypatch):
    """The fix for the stranger-per-line problem. Two speakers cast in the
    same designed voice must get the same clip — a second draw would be a
    second person."""
    ctx = _with_source_refs(tmp_path, "A", "B")
    engine = _StubEngine()
    refs, _, _ = _resolve({"A": "male_deep", "B": "male_deep"}, ["A", "B"],
                          ctx, engine, tmp_path, monkeypatch)

    assert refs["A"] == refs["B"]
    assert len(engine.calls) == 1, (
        f"design_reference called {len(engine.calls)}x — the cache key must "
        f"not vary between two speakers asking for the same voice")


def test_a_designed_voice_uses_the_presets_fixed_seed(tmp_path, monkeypatch):
    ctx = _with_source_refs(tmp_path, "A")
    engine = _StubEngine()
    _resolve({"A": "male_deep"}, ["A"], ctx, engine, tmp_path, monkeypatch)
    style, seed, lang = engine.calls[0]
    assert seed == server.VOICE_PRESETS["male_deep"]["seed"]
    assert style == server.VOICE_PRESETS["male_deep"]["style"]
    assert lang == "ru", "the draw should be made in the language it will speak"


def test_a_free_text_design_gets_a_stable_seed_of_its_own(tmp_path, monkeypatch):
    """Two jobs describing the same voice must draw the same one. A cold cache
    each time, so this measures the seed and not the cache."""
    ctx = _with_source_refs(tmp_path, "A")
    seeds = []
    for run in ("cache1", "cache2"):
        engine = _StubEngine()
        _resolve({"A": "design:tired night-shift dispatcher"}, ["A"],
                 ctx, engine, tmp_path, monkeypatch, cache=run)
        seeds.append(engine.calls[0][1])
    assert seeds[0] == seeds[1] and seeds[0] is not None


def test_a_second_job_wanting_the_same_voice_reuses_the_first_draw(
        tmp_path, monkeypatch):
    """The cache is what makes a preset sound the same across jobs rather
    than being re-rolled every run — and re-rolling is a different person,
    not a different take."""
    ctx = _with_source_refs(tmp_path, "A")
    first, second = _StubEngine(), _StubEngine()
    refs1, _, _ = _resolve({"A": "male_deep"}, ["A"], ctx, first,
                           tmp_path, monkeypatch)
    refs2, _, _ = _resolve({"A": "male_deep"}, ["A"], ctx, second,
                           tmp_path, monkeypatch)

    assert refs1["A"] == refs2["A"]
    assert len(first.calls) == 1
    assert second.calls == [], "the second job re-drew a voice it already had"


def test_a_failed_render_degrades_that_speaker_alone(tmp_path, monkeypatch):
    """A voice that will not render must not take the dub down, and must not
    quietly recast anybody else."""
    ctx = _with_source_refs(tmp_path, "A", "B")
    refs, _, report = _resolve(
        {"A": "male_deep", "B": "source"}, ["A", "B"],
        ctx, _StubEngine(fail=True), tmp_path, monkeypatch)

    assert refs["A"] == ctx["source_speaker_refs"]["A"]
    assert refs["B"] == ctx["source_speaker_refs"]["B"]
    bad = [r for r in report if r["speaker"] == "A"][0]
    assert bad["status"] == "render_failed" and bad["used"] == "source"


def test_a_missing_library_file_degrades_rather_than_raising(tmp_path, monkeypatch):
    ctx = _with_source_refs(tmp_path, "A")
    monkeypatch.setattr(server, "scan_file_presets", lambda: {
        "file:ghost": {"id": "file:ghost", "name": "Ghost",
                       "reference_file": str(tmp_path / "gone.wav")}})
    refs, _, report = _resolve({"A": "file:ghost"}, ["A"], ctx,
                               _StubEngine(), tmp_path, monkeypatch)
    assert refs["A"] == ctx["source_speaker_refs"]["A"]
    assert report[0]["status"] == "missing_file"


def test_a_speaker_with_no_reference_at_all_is_reported_not_faked(
        tmp_path, monkeypatch):
    refs, _, report = _resolve({"A": "source"}, ["A"], {}, _StubEngine(),
                               tmp_path, monkeypatch)
    assert "A" not in refs
    assert report[0]["status"] == "no_source_ref"


def test_source_means_the_video_even_after_a_run_that_used_a_preset(
        tmp_path, monkeypatch):
    """speaker_refs in a checkpoint may hold preset paths left by an earlier
    run. Those are emphatically not "their own voice", so the resolver only
    trusts that field while it points inside the job's own folder."""
    ctx = _with_source_refs(tmp_path, "A")
    stale_preset = tmp_path / "presets" / "voices" / "someone_else.wav"
    stale_preset.parent.mkdir(parents=True)
    stale_preset.write_bytes(b"\0" * 64)
    ctx["speaker_refs"]["A"] = str(stale_preset)

    refs, _, _ = _resolve({"A": "source"}, ["A"], ctx, _StubEngine(),
                          tmp_path, monkeypatch)
    assert refs["A"] == ctx["source_speaker_refs"]["A"]


def test_a_clip_the_user_uploaded_for_one_speaker_still_counts_as_source(
        tmp_path, monkeypatch):
    """edit_speaker_ref writes into the job's own speaker_refs/, so it should
    win over the diarizer's cut — that is the whole reason it exists."""
    ctx = _with_source_refs(tmp_path, "A")
    better = tmp_path / "job1" / "speaker_refs" / "user_A.wav"
    better.write_bytes(b"\0" * 128)
    ctx["speaker_refs"]["A"] = str(better)

    refs, _, _ = _resolve({"A": "source"}, ["A"], ctx, _StubEngine(),
                          tmp_path, monkeypatch)
    assert refs["A"] == str(better)


def test_cloning_is_controllable_not_ultimate(tmp_path, monkeypatch):
    """Every cast reference gets an empty prompt transcript. A non-empty one
    would have to match the reference audio word for word, and none of these
    references have a transcript at all."""
    ctx = _with_source_refs(tmp_path, "A", "B")
    _, transcripts, _ = _resolve({"A": "male_deep"}, ["A", "B"], ctx,
                                 _StubEngine(), tmp_path, monkeypatch)
    assert set(transcripts.values()) == {""}


# ── the designed-voice cache ────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    (("warm male", 101, "ru"), ("warm male", 101, "en")),
    (("warm male", 101, "ru"), ("warm male", 202, "ru")),
    (("warm male", 101, "ru"), ("deep male", 101, "ru")),
])
def test_the_cache_key_separates_style_seed_and_language(a, b):
    assert server._designed_voice_path(*a) != server._designed_voice_path(*b)


def test_the_cache_key_ignores_case_and_surrounding_space():
    assert (server._designed_voice_path("  Warm Male ", 1, "ru")
            == server._designed_voice_path("warm male", 1, "ru"))


def test_designed_clips_hide_from_the_user_voice_picker():
    """They live in a dotted *directory* under presets/voices/, and
    scan_file_presets iterates that folder for audio files — so a directory
    is skipped and machine-made clips never appear as someone's own voice."""
    assert server.DESIGNED_VOICES_DIR.parent == server.VOICE_PRESETS_DIR
    assert server.DESIGNED_VOICES_DIR.name.startswith(".")
    assert server.DESIGNED_VOICES_DIR.suffix not in server._VOICE_AUDIO_EXTS


# ── the gate ────────────────────────────────────────────────────────────

def test_the_casting_gate_waits_until_there_is_translated_text_to_preview():
    """It sits after translate, not after diarize. The references exist by
    the earlier point, but a preview has to speak the lines the dub will
    speak — at the diarize gate the text is still in the source language.
    (The wizard modes now resolve through app/review_gates.py; the legacy
    review_voices mapping must keep the same boundary.)"""
    assert server._evaluate_gate(
        "diarize", {"wizard_mode": "review_voices"}, {}, "j-cast") is None
    status, _detail, checkpoint = server._evaluate_gate(
        "translate", {"wizard_mode": "review_voices"}, {}, "j-cast")
    assert (status, checkpoint) == ("awaiting_voice_review", "translation_done")


def test_the_gate_still_sits_in_front_of_synthesis():
    """Whatever it is called, a review gate is only worth anything if the
    expensive stage has not run yet."""
    order = server.STAGE_ORDER
    assert order.index("translate") < order.index("tts")


def test_a_paused_cast_job_asks_for_review_like_the_others():
    assert (server._WEBHOOK_EVENT_FOR_STATUS["awaiting_voice_review"]
            == "job.awaiting_review")


# ── persistence ─────────────────────────────────────────────────────────

def test_the_cast_is_written_to_every_resumable_checkpoint(tmp_path, monkeypatch):
    """/continue and each stage retry pick a different checkpoint. A cast
    stored on only one of them reverts to whole-job voice mode on the others.
    """
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    work = tmp_path / "job1"
    work.mkdir()
    stages = ("transcription_done", "translation_done", "tts_done")
    for stage in stages:
        (work / f"checkpoint_{stage}.json").write_text(
            json.dumps({"stage": stage, "segments": []}), encoding="utf-8")

    n = server._persist_cast("job1", {"SPEAKER_00": "male_deep"})

    assert n == len(stages)
    for stage in stages:
        cp = json.loads((work / f"checkpoint_{stage}.json").read_text(encoding="utf-8"))
        assert cp["speaker_voice_map"] == {"SPEAKER_00": "male_deep"}
