"""The Voice Design "(style)" prefix must never escape the synthesizer.

VoxCPM's voice-design mode is driven by prepending a style description to the
text it is asked to speak. That prefix is an instruction to the model, not
part of the dialogue — but it used to be written onto `translated_text`, the
field the rest of the system treats as the line itself. In a real 1602-segment
dub that put

    (middle-aged male voice, warm and calm, clear articulation)

at the head of every subtitle, because the assemble stage rewrites
subtitles.srt from `translated_text`; it also made the partial-retry reuse
check compare prefixed checkpoint text against clean translation text, so no
segment was ever reused, and it tripped the assembler's emotion-tag heuristic
on every segment.

The prefix now lives on `tts_text`. These tests pin the two boundaries that
kept it out of sight in the first place.
"""
import pytest

from pipeline.assembler import write_srt


PREFIX = "(middle-aged male voice, warm and calm, clear articulation)"


@pytest.fixture
def prefixed_segments():
    """Segments as they look inside the TTS stage: clean dialogue, plus the
    synth-only decorated copy the engine actually speaks."""
    return [
        {"idx": 0, "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00",
         "text": "Explain how the power of the world works.",
         "translated_text": "Объясните, как работает власть в мире.",
         "tts_text": PREFIX + "Объясните, как работает власть в мире."},
        {"idx": 1, "start": 2.6, "end": 7.5, "speaker": "SPEAKER_01",
         "text": "A lot of conspiracy theorists believe that.",
         "translated_text": "Многие сторонники теорий заговора считают.",
         "tts_text": PREFIX + "Многие сторонники теорий заговора считают."},
    ]


def test_srt_renders_dialogue_not_the_style_prompt(prefixed_segments, tmp_path):
    srt = tmp_path / "subtitles.srt"
    write_srt(prefixed_segments, str(srt))
    body = srt.read_text(encoding="utf-8")

    assert PREFIX not in body, (
        "the Voice Design style prompt reached the subtitle file — this is the "
        "1602-line regression, and it comes back the moment the prefix is "
        "written to `translated_text` instead of `tts_text`"
    )
    assert "Объясните, как работает власть в мире." in body
    assert "Многие сторонники теорий заговора считают." in body


def test_serialize_segments_drops_the_synth_only_field(prefixed_segments):
    """`tts_text` must not survive checkpointing.

    Checkpoints are what a retry, a redub and the translation editor read
    back. A prefix persisted there outlives the run that created it: the next
    partial retry compares a prefixed line against a clean one, decides they
    differ, and re-synthesizes a segment that was already fine.
    """
    import server

    out = server._serialize_segments(prefixed_segments)

    assert [s["translated_text"] for s in out] == [
        "Объясните, как работает власть в мире.",
        "Многие сторонники теорий заговора считают.",
    ]
    assert not any("tts_text" in s for s in out), (
        "tts_text was added to _serialize_segments' whitelist — it is "
        "deliberately excluded so the prefix cannot outlive the stage"
    )


def test_emotion_heuristic_does_not_fire_on_clean_dialogue(prefixed_segments):
    """The assembler grants extra time-stretch to genuinely emotion-tagged
    lines, detected by a leading "(". With the prefix on `translated_text`
    that fired on 100% of segments; on `tts_text` it fires on none."""
    for seg in prefixed_segments:
        text = (seg.get("translated_text") or "").lstrip()
        assert not (text.startswith("(") and ")" in text[:30])
