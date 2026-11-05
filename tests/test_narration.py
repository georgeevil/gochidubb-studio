"""pipeline/narration.py — the pure half: scripts, sentences, timeline, cues.

A narration has no source video, so nothing here can be checked against an
original timeline the way the assembler's tests are. What can be checked is
that the timeline is self-consistent: every line placed, in order, with no
overlaps; slides contiguous; cues inside their sentence and inside the line
limits. Those are the properties a narrated video depends on.
"""
import json

import numpy as np
import pytest
import soundfile as sf

from pipeline import narration as n
from pipeline.subtitles import DEFAULT_LIMITS, cue_text, srt_text


MD = """# The 188 hectares

## Cover
![cover](slides/cover.png)
Seven years ago the court gave the agency **twelve months**.
This is what happened next.

<!-- reminder to self: check the date -->

## The timeline
![](timeline.png "The case, 2014 to 2026")
The case was filed on 11 Dec. 2014. The ruling came in 2019.

- First, the court ordered a delimitation.
- Then it [archived](https://example.org) the file.

<!-- pause: 2.5 -->

## Thanks
"""


# ═══════════════════════════════════════════════════════════════════════
#  Script parsing
# ═══════════════════════════════════════════════════════════════════════

def test_markdown_title_slides_images_and_pauses():
    doc = n.parse_script(MD)
    assert doc["title"] == "The 188 hectares"
    assert [s["id"] for s in doc["slides"]] == ["cover", "the-timeline", "thanks"]
    cover, timeline, thanks = doc["slides"]
    assert cover["image"] == "cover.png"          # the folder is dropped
    assert timeline["image"] == "timeline.png"    # the title in quotes too
    assert timeline["pause_after"] == 2.5
    assert "pause_after" not in cover
    assert thanks["paragraphs"] == []             # a title card


def test_markdown_markup_is_not_spoken():
    doc = n.parse_script(MD)
    spoken = " ".join(p for s in doc["slides"] for p in s["paragraphs"])
    assert "**" not in spoken and "](" not in spoken and "<!--" not in spoken
    assert "twelve months" in spoken
    assert "reminder to self" not in spoken
    assert "Then it archived the file." in spoken
    assert "- First" not in spoken


def test_comments_are_not_spoken_wherever_they_sit():
    doc = n.parse_script(
        "## One\n"
        "The deadline passed in 2021. <!-- check the year -->\n"
        "Nothing <!-- really? --> was delimited.\n"
        "<!-- a note that\n"
        "\n"
        "runs over several lines --> And then the file was archived.\n"
        "\n"
        "## Two\n"
        "Still here. <!-- an unfinished thought\n"
        "that never closes")
    one, two = doc["slides"]
    # One paragraph: the blank line inside the comment is not a break.
    assert one["paragraphs"] == [
        "The deadline passed in 2021. Nothing was delimited. "
        "And then the file was archived."]
    assert two["paragraphs"] == ["Still here."]


def test_a_held_back_lead_in_is_not_left_on_its_own():
    # "a clear," is held back so it goes with the clause it opens. When that
    # clause does not fit beside it, the two are wrapped together; before,
    # the lead-in came out as a chunk of its own: an 8-character subtitle on
    # screen for 0.4 s, and a synthesis call for two words.
    s = ("The State should carry out what the court already ordered: a "
         "clear, public and final line between protected forest and the "
         "land where people live, respecting rights acquired in good faith.")
    chunks = n.chunk_text(s, 78)
    assert " ".join(chunks) == s
    assert all(15 <= len(c) <= 78 for c in chunks), chunks


def test_a_chunk_whose_words_break_badly_becomes_two_cues():
    # 78 characters under a 2 x 42 box, and still no way to wrap it in two
    # lines: the only breaks are after 34 and after 43 characters.
    s = ("They bar municipal permits on 20.1 hectares of forest inside the "
         "coastal zone.")
    placed = [{"idx": 0, "text": s, "translated_text": s, "slide": "a",
               "placed_start": 1.0, "placed_end": 6.2}]
    cues = n.subtitle_cues(placed)
    assert len(cues) == 2
    assert " ".join(c["text"] for c in cues) == s
    assert n.cue_violations(cues) == []
    assert cues[0]["start"] == 1.0 and cues[-1]["end"] <= 6.2


def test_a_note_before_the_first_slide_is_not_a_slide():
    # Found with a real 15-slide script: a comment under the title became an
    # opening slide with no picture and no words, and the slideshow then had
    # nothing to open on.
    doc = n.parse_script(
        "# Briefing\n\n<!-- one heading per slide -->\n\n"
        "## Cover\n![](cover.png)\nHello.\n")
    assert [s["id"] for s in doc["slides"]] == ["cover"]
    assert doc["slides"][0]["image"] == "cover.png"
    # The same for a note that runs over lines and for a line that is all note.
    doc = n.parse_script(
        "# Briefing\n<!-- a note\nover two lines -->\n<!-- x --> <!-- y -->\n"
        "## Cover\nHello.\n")
    assert [s["id"] for s in doc["slides"]] == ["cover"]


def test_a_pause_comment_still_works_next_to_other_comments():
    doc = n.parse_script(
        "## One\nHello. <!-- note -->\n<!-- pause: 3 -->\n\n## Two\nBye.")
    assert doc["slides"][0]["pause_after"] == 3.0
    assert doc["slides"][0]["paragraphs"] == ["Hello."]


def test_soft_wrapped_lines_are_one_paragraph_and_blank_lines_split():
    cover = n.parse_script(MD)["slides"][0]
    assert cover["paragraphs"] == [
        "Seven years ago the court gave the agency twelve months. "
        "This is what happened next."]
    timeline = n.parse_script(MD)["slides"][1]
    assert len(timeline["paragraphs"]) == 2


def test_text_before_the_first_heading_becomes_an_opening_slide():
    doc = n.parse_script("# Title\n\nHello there.\n\n## Two\nSecond slide.")
    assert [s["title"] for s in doc["slides"]] == ["Title", "Two"]
    assert doc["slides"][0]["paragraphs"] == ["Hello there."]


def test_json_and_dict_forms_agree():
    raw = {"title": "T", "slides": [
        {"id": "a", "image": "C:\\decks\\a.PNG", "narration": "One.\n\nTwo."},
        {"title": "Second slide", "narration": ["Three.", "Four."]},
        "Just text.",
    ]}
    from_dict = n.parse_script(raw)
    assert n.parse_script(json.dumps(raw)) == from_dict
    a, b, c = from_dict["slides"]
    assert a["image"] == "a.PNG" and a["paragraphs"] == ["One.", "Two."]
    assert b["id"] == "second-slide" and b["paragraphs"] == ["Three.", "Four."]
    assert c["id"] == "slide-3" and c["paragraphs"] == ["Just text."]


def test_parsing_a_parsed_script_changes_nothing():
    # The route parses to validate and queues the result; the runner parses
    # what it is handed. A second pass that dropped the text (the output key
    # is `paragraphs`, the input key `narration`) failed every queued job.
    once = n.parse_script(MD)
    assert n.parse_script(once) == once
    assert n.parse_script(json.loads(json.dumps(once))) == once


def test_a_bare_json_list_is_a_list_of_slides():
    doc = n.parse_script('[{"narration": "Hello."}]')
    assert len(doc["slides"]) == 1 and doc["title"] == ""


def test_duplicate_ids_are_made_unique():
    doc = n.parse_script("## Intro\nA.\n\n## Intro\nB.\n\n## Intro\nC.")
    assert [s["id"] for s in doc["slides"]] == ["intro", "intro-2", "intro-3"]


@pytest.mark.parametrize("raw, fragment", [
    ("", "empty"),
    ("   \n ", "empty"),
    ('{"slides": [', "does not parse"),
    ('{"title": "x"}', "no slides"),
    ("## Only a heading", "No slide has any narration"),
    ('{"slides": [{"id": "has space", "narration": "x"}]}', "ids use"),
    ('{"slides": [{"image": "notes.pdf", "narration": "x"}]}', "image"),
    ('{"slides": [{"pause_after": "soon", "narration": "x"}]}', "pause_after"),
    ('{"slides": [42]}', "not an object"),
])
def test_bad_scripts_are_refused_with_a_reason(raw, fragment):
    with pytest.raises(n.ScriptError) as e:
        n.parse_script(raw)
    assert fragment in str(e.value)


def test_script_size_is_bounded():
    with pytest.raises(n.ScriptError) as e:
        n.parse_script({"slides": [{"narration": "word " * 30_000}]})
    assert "limit" in str(e.value)
    with pytest.raises(n.ScriptError):
        n.parse_script({"slides": [{"narration": "x"}] * (n.MAX_SLIDES + 1)})


def test_an_image_path_cannot_leave_the_upload_folder():
    doc = n.parse_script(
        {"slides": [{"image": "../../etc/secret.png", "narration": "x"}]})
    assert doc["slides"][0]["image"] == "secret.png"


# ═══════════════════════════════════════════════════════════════════════
#  Sentences and chunks
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text, expected", [
    ("One. Two! Three?", ["One.", "Two!", "Three?"]),
    ("It was filed on 11 Dec. 2014 under art. 12 of the law. Then it waited.",
     ["It was filed on 11 Dec. 2014 under art. 12 of the law.",
      "Then it waited."]),
    ("The U.S. Supreme Court ruled in No. 34. It took 3.5 years.",
     ["The U.S. Supreme Court ruled in No. 34.", "It took 3.5 years."]),
    ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
    ("It cost 1,672 days. 2021 came and went.",
     ["It cost 1,672 days.", "2021 came and went."]),
    ("wait... and see", ["wait... and see"]),
    ("No full stop at the end", ["No full stop at the end"]),
    ("  spaced \n out.   Twice.  ", ["spaced out.", "Twice."]),
    ("", []),
])
def test_split_sentences(text, expected):
    assert n.split_sentences(text) == expected


def test_chunk_text_respects_the_limit_and_loses_nothing():
    text = ("First, the court ordered a delimitation; second, it set a "
            "deadline of twelve months, which passed in February 2021, and "
            "third, it archived the file three days later without checking "
            "whether anything had actually been done.")
    for limit in (40, 78, 120, 220):
        chunks = n.chunk_text(text, limit)
        assert all(len(c) <= limit for c in chunks), (limit, chunks)
        assert " ".join(chunks) == text


def test_chunk_text_balances_instead_of_orphaning():
    text = "word " * 19 + "word"            # 99 characters
    chunks = n.chunk_text(text, 80)
    assert len(chunks) == 2
    assert min(len(c) for c in chunks) > 30   # not 79 + a scrap


def test_a_lead_in_goes_with_the_clause_it_opens():
    chunks = n.chunk_text(
        "The court ordered a delimitation of the forest; second, it set a "
        "deadline of twelve months for the agency.", 60)
    assert not any(c.endswith("second,") for c in chunks)
    assert any(c.startswith("second,") for c in chunks)


def test_chunk_text_keeps_an_overlong_word_whole():
    assert n.chunk_text("a " + "x" * 50 + " b", 10) == ["a", "x" * 50, "b"]
    assert n.chunk_text("", 10) == []
    assert n.chunk_text("short", 10) == ["short"]


def test_build_segments_carries_what_the_engines_read():
    doc = n.parse_script(MD)
    segs = n.build_segments(doc)
    assert [s["idx"] for s in segs] == list(range(len(segs)))
    assert all(s["speaker"] == n.NARRATOR for s in segs)
    assert all(s["text"] == s["translated_text"] and s["text"] for s in segs)
    assert [s["slide"] for s in segs] == (
        ["cover"] * 2 + ["the-timeline"] * 4)
    # A paragraph's last line is marked; the lines before it are not.
    assert [s["paragraph_end"] for s in segs] == [
        False, True, False, True, False, True]


def test_a_long_sentence_becomes_several_synthesis_calls():
    long = ("This sentence goes on, " * 20).strip().rstrip(",") + "."
    segs = n.build_segments(n.parse_script({"slides": [{"narration": long}]}))
    assert len(segs) > 1
    assert all(len(s["text"]) <= n.MAX_TTS_CHARS for s in segs)


# ═══════════════════════════════════════════════════════════════════════
#  Timeline
# ═══════════════════════════════════════════════════════════════════════

def _timeline(durations=None, timing=None, md=MD):
    doc = n.parse_script(md)
    segs = n.build_segments(doc)
    if durations is None:
        durations = {s["idx"]: 2.0 for s in segs}
    return doc, segs, n.build_timeline(doc, segs, durations, timing)


def test_segments_are_placed_in_order_without_overlap():
    _doc, segs, tl = _timeline()
    placed = tl["segments"]
    assert len(placed) == len(segs)
    for a, b in zip(placed, placed[1:]):
        assert b["placed_start"] > a["placed_end"]
    for s in placed:
        assert s["placed_end"] - s["placed_start"] == pytest.approx(2.0)
        assert (s["start"], s["end"]) == (s["placed_start"], s["placed_end"])


def test_slides_are_contiguous_and_cover_the_whole_duration():
    _doc, _segs, tl = _timeline()
    slides = tl["slides"]
    assert slides[0]["start"] == 0.0
    for a, b in zip(slides, slides[1:]):
        assert b["start"] == a["end"]
    assert slides[-1]["end"] == tl["duration"]


def test_every_line_falls_inside_its_own_slide():
    _doc, _segs, tl = _timeline()
    span = {s["id"]: (s["start"], s["end"]) for s in tl["slides"]}
    for seg in tl["segments"]:
        lo, hi = span[seg["slide"]]
        assert lo <= seg["placed_start"] and seg["placed_end"] <= hi


def test_gaps_follow_the_timing_table():
    t = {"lead_in": 1.0, "sentence_gap": 0.25, "paragraph_gap": 2.0,
         "slide_lead": 0.5, "slide_gap": 3.0, "tail": 4.0}
    _doc, _segs, tl = _timeline(timing=t)
    p = tl["segments"]
    assert p[0]["placed_start"] == 1.0                       # lead_in
    assert p[1]["placed_start"] - p[0]["placed_end"] == pytest.approx(0.25)
    cover_end = tl["slides"][0]["end"]
    assert cover_end - p[1]["placed_end"] == pytest.approx(3.0)   # slide_gap
    assert p[2]["placed_start"] - cover_end == pytest.approx(0.5)  # slide_lead
    # p[3] closes the first paragraph of slide two
    assert p[4]["placed_start"] - p[3]["placed_end"] == pytest.approx(2.0)


def test_a_slide_pause_overrides_the_default_gap():
    _doc, _segs, tl = _timeline()
    timeline_slide = tl["slides"][1]
    last_line = [s for s in tl["segments"] if s["slide"] == "the-timeline"][-1]
    assert timeline_slide["end"] - last_line["placed_end"] == pytest.approx(2.5)


def test_a_slide_with_nothing_said_is_held_as_a_title_card():
    _doc, _segs, tl = _timeline()
    thanks = tl["slides"][2]
    assert thanks["end"] - thanks["start"] == pytest.approx(
        n.DEFAULT_TIMING["silent_slide_hold"])


def test_the_last_narrated_slide_ends_on_the_tail():
    doc, segs, tl = _timeline(md="## Only\nOne line.")
    assert tl["duration"] == pytest.approx(
        n.DEFAULT_TIMING["lead_in"] + 2.0 + n.DEFAULT_TIMING["tail"])
    assert len(doc["slides"]) == 1 and len(segs) == 1


def test_a_failed_line_keeps_its_slot_and_is_marked():
    doc = n.parse_script(MD)
    segs = n.build_segments(doc)
    durations = {s["idx"]: 2.0 for s in segs}
    del durations[2]
    durations[3] = 0.0
    tl = n.build_timeline(doc, segs, durations)
    by_idx = {s["idx"]: s for s in tl["segments"]}
    assert by_idx[2]["missing_audio"] and by_idx[3]["missing_audio"]
    assert "missing_audio" not in by_idx[1]
    # Held open at reading speed, so later lines stay on their own slide.
    held = by_idx[2]["placed_end"] - by_idx[2]["placed_start"]
    assert held == pytest.approx(
        max(1.0, len(by_idx[2]["text"]) / n.READING_CPS), abs=0.002)
    assert by_idx[4]["placed_start"] > by_idx[3]["placed_end"]


def test_timing_overrides_are_clamped_and_unknown_keys_ignored():
    t = n.resolve_timing({"slide_gap": 999, "tail": -5, "lead_in": "soon",
                          "bogus": 3, "sentence_gap": None})
    assert t["slide_gap"] == 30.0 and t["tail"] == 0.0
    assert t["lead_in"] == n.DEFAULT_TIMING["lead_in"]
    assert t["sentence_gap"] == n.DEFAULT_TIMING["sentence_gap"]
    assert "bogus" not in t
    assert n.resolve_timing(None) == n.DEFAULT_TIMING


def test_estimate_duration_is_in_the_right_neighbourhood():
    segs = n.build_segments(n.parse_script(MD))
    est = n.estimate_duration(segs)
    chars = sum(len(s["text"]) for s in segs)
    assert chars / n.READING_CPS < est < chars / n.READING_CPS + len(segs)


# ═══════════════════════════════════════════════════════════════════════
#  Subtitles
# ═══════════════════════════════════════════════════════════════════════

LONG = ("## One\n"
        "First, the court ordered a delimitation; second, it set a deadline "
        "of twelve months, which passed in February 2021, and third, it "
        "archived the file three days later without checking whether "
        "anything had actually been done. Then nothing happened.")


def _cues(md=LONG, seconds_per_char=1 / 15):
    doc = n.parse_script(md)
    segs = n.build_segments(doc)
    tl = n.build_timeline(
        doc, segs, {s["idx"]: len(s["text"]) * seconds_per_char for s in segs})
    return tl, n.subtitle_cues(tl["segments"])


def test_cues_fit_the_line_limits():
    _tl, cues = _cues()
    for c in cues:
        lines = cue_text(c).split("\n")
        assert len(lines) <= DEFAULT_LIMITS["max_lines"], c
        assert all(len(ln) <= DEFAULT_LIMITS["max_chars_per_line"]
                   for ln in lines), c


def test_cues_tile_each_sentence_and_never_leave_it():
    tl, cues = _cues()
    assert [c["idx"] for c in cues] == list(range(len(cues)))
    by_seg = {}
    for c in cues:
        by_seg.setdefault(c["segment"], []).append(c)
    for seg in tl["segments"]:
        mine = by_seg[seg["idx"]]
        assert " ".join(c["text"] for c in mine) == seg["text"]
        assert mine[0]["start"] == pytest.approx(seg["placed_start"], abs=0.002)
        assert mine[-1]["end"] <= seg["placed_end"] + 0.002
    for a, b in zip(cues, cues[1:]):
        assert a["end"] <= b["start"]
        assert a["start"] < a["end"]


def test_time_inside_a_sentence_is_shared_by_length():
    tl, cues = _cues()
    seg = tl["segments"][0]
    mine = [c for c in cues if c["segment"] == seg["idx"]]
    assert len(mine) >= 2
    span = seg["placed_end"] - seg["placed_start"]
    total = sum(len(c["text"]) for c in mine)
    # Starts are exact shares; ends may be shaved to open the minimum gap.
    t = seg["placed_start"]
    for c in mine:
        assert c["start"] == pytest.approx(t, abs=0.002)
        t += span * len(c["text"]) / total


def test_the_minimum_gap_is_opened_and_passes_validation():
    _tl, cues = _cues()
    kinds = {v["kind"] for v in n.cue_violations(cues)}
    assert "gap_too_small" not in kinds
    assert "line_too_long" not in kinds and "too_many_lines" not in kinds


def test_custom_limits_are_honoured():
    limits = {"max_chars_per_line": 20, "max_lines": 1, "min_gap_ms": 0}
    doc = n.parse_script(LONG)
    segs = n.build_segments(doc)
    tl = n.build_timeline(doc, segs, {s["idx"]: 10.0 for s in segs})
    cues = n.subtitle_cues(tl["segments"], limits)
    assert all("\n" not in cue_text(c) and len(cue_text(c)) <= 20 for c in cues)


def test_cues_render_as_srt():
    _tl, cues = _cues(md="## A\nHello there.")
    doc = srt_text(cues)
    assert doc.startswith("1\n00:00:00,500 --> ")
    assert "Hello there." in doc


def test_a_line_with_no_audio_still_gets_its_subtitle():
    doc = n.parse_script("## A\nSaid. Not said. Said again.")
    segs = n.build_segments(doc)
    tl = n.build_timeline(doc, segs, {0: 1.0, 2: 1.0})
    cues = n.subtitle_cues(tl["segments"])
    assert [c["text"] for c in cues] == ["Said.", "Not said.", "Said again."]


# ═══════════════════════════════════════════════════════════════════════
#  Slideshow planning
# ═══════════════════════════════════════════════════════════════════════

def test_a_slide_without_a_picture_keeps_the_previous_one_up():
    slides = [
        {"id": "a", "image": "a.png", "start": 0.0, "end": 4.0},
        {"id": "b", "image": "", "start": 4.0, "end": 6.5},
        {"id": "c", "image": "c.png", "start": 6.5, "end": 9.0},
        {"id": "d", "image": "c.png", "start": 9.0, "end": 10.0},
    ]
    assert n.slideshow_spans(slides) == [("a.png", 6.5), ("c.png", 3.5)]


def test_spans_add_up_to_the_timeline():
    _doc, _segs, tl = _timeline()
    spans = n.slideshow_spans(tl["slides"])
    assert sum(d for _img, d in spans) == pytest.approx(tl["duration"], abs=0.01)


def test_a_slideshow_needs_a_picture_to_open_on():
    with pytest.raises(n.ScriptError) as e:
        n.slideshow_spans([{"id": "a", "image": "", "start": 0, "end": 1}])
    assert "'a'" in str(e.value)


def test_concat_script_repeats_the_last_picture_and_quotes_paths():
    text = n.concat_script([("C:\\decks\\it's.png", 3.25), ("b.png", 4.0)])
    assert text.splitlines() == [
        "ffconcat version 1.0",
        "file 'C:/decks/it'\\''s.png'",
        "duration 3.250",
        "file 'b.png'",
        "duration 4.000",
        "file 'b.png'",
    ]
    with pytest.raises(n.ScriptError):
        n.concat_script([])


def test_slideshow_command_soft_subtitles_by_default():
    cmd = n.slideshow_command("s.ffconcat", "n.wav", "out.mp4", duration=12.5,
                              srt_path="subs.srt")
    assert cmd[0] == "ffmpeg" and cmd[-1] == "out.mp4"
    assert cmd.count("-i") == 3 and "subs.srt" in cmd
    assert cmd[cmd.index("-c:s") + 1] == "mov_text"
    assert "-map" in cmd and "2:0" in cmd
    assert cmd[cmd.index("-t") + 1] == "12.500"
    vf = cmd[cmd.index("-vf") + 1]
    assert "subtitles=" not in vf
    assert "scale=1920:1080:force_original_aspect_ratio=decrease" in vf
    assert vf.endswith("format=yuv420p")


def test_slideshow_command_burns_subtitles_on_request():
    cmd = n.slideshow_command("s.ffconcat", "n.wav", "out.mp4", duration=5,
                              srt_path="C:\\jobs\\a b\\subs.srt",
                              burn_subs=True, size=(1280, 720), fps=24)
    assert cmd.count("-i") == 2 and "-c:s" not in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert "subtitles='C\\:/jobs/a b/subs.srt'" in vf
    assert "pad=1280:720" in vf
    # A cue can only be painted onto a frame that exists when it is due, so
    # the frames are made (fps) before the subtitles are drawn on them.
    assert vf.index("fps=24") < vf.index("subtitles=")


def test_the_frame_rate_is_a_filter_and_not_an_output_option():
    # `-r` leaves the last picture on screen for one frame: the encoder only
    # repeats a still once the next one arrives, and `-t` has already dropped
    # the closing one. See slideshow_command's docstring for the measurement.
    cmd = n.slideshow_command("s.ffconcat", "n.wav", "out.mp4", duration=5)
    vf = cmd[cmd.index("-vf") + 1]
    assert "-r" not in cmd
    assert "fps=30" in vf
    assert vf.index("scale=") < vf.index("fps=30") < vf.index("format=yuv420p")


def test_slideshow_command_without_subtitles():
    cmd = n.slideshow_command("s.ffconcat", "n.wav", "out.mp4", duration=5)
    assert cmd.count("-i") == 2
    assert "subtitles=" not in cmd[cmd.index("-vf") + 1]


# ═══════════════════════════════════════════════════════════════════════
#  Measuring and mixing (WAV in, WAV out — no ffmpeg involved)
# ═══════════════════════════════════════════════════════════════════════

SR = 16000


def _tone(path, seconds, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    sf.write(str(path), (amp * np.sin(2 * np.pi * 220 * t)).astype("float32"),
             SR, subtype="PCM_16")
    return str(path)


def test_clip_duration_and_measure_durations(tmp_path):
    a = _tone(tmp_path / "a.wav", 1.25)
    assert n.clip_duration(a) == pytest.approx(1.25, abs=0.001)
    assert n.clip_duration(str(tmp_path / "missing.wav")) == 0.0
    assert n.clip_duration("") == 0.0
    segs = [{"idx": 0, "audio_path": a}, {"idx": 1, "audio_path": None},
            {"idx": 2, "audio_path": str(tmp_path / "gone.wav")}]
    assert n.measure_durations(segs) == {0: pytest.approx(1.25, abs=0.001)}


def test_mix_places_each_clip_at_its_time(tmp_path):
    placed = [
        {"idx": 0, "placed_start": 0.5, "placed_end": 1.5,
         "audio_path": _tone(tmp_path / "0.wav", 1.0)},
        {"idx": 1, "placed_start": 2.0, "placed_end": 3.0,
         "audio_path": None},                      # failed line: silence
        {"idx": 2, "placed_start": 3.5, "placed_end": 4.0,
         "audio_path": _tone(tmp_path / "2.wav", 0.5)},
    ]
    out = tmp_path / "out" / "narration.wav"
    stats = n.mix_narration(placed, str(out), duration=5.0, sample_rate=SR,
                            loudnorm=False)
    assert stats["placed"] == 2 and stats["missing"] == 1
    data, sr = sf.read(str(out))
    assert sr == SR and len(data) == 5 * SR

    def level(lo, hi):
        return float(np.abs(data[int(lo * SR):int(hi * SR)]).max())

    assert level(0.0, 0.49) == 0.0          # lead-in
    assert level(0.6, 1.4) > 0.4            # first line
    assert level(1.6, 3.4) == 0.0           # gap + the failed line's slot
    assert level(3.6, 3.9) > 0.4            # third line
    assert level(4.1, 5.0) == 0.0           # tail


def test_mix_fades_clip_edges_and_never_clips(tmp_path):
    loud = _tone(tmp_path / "loud.wav", 1.0, amp=0.99)
    placed = [{"idx": 0, "placed_start": 0.0, "placed_end": 1.0,
               "audio_path": loud},
              {"idx": 1, "placed_start": 0.0, "placed_end": 1.0,
               "audio_path": loud}]            # stacked on purpose
    out = tmp_path / "n.wav"
    n.mix_narration(placed, str(out), duration=1.0, sample_rate=SR,
                    loudnorm=False)
    data, _ = sf.read(str(out))
    assert abs(data[0]) < 0.01               # starts from the fade, not a step
    assert np.abs(data).max() <= 1.0


def test_mix_truncates_a_clip_that_runs_past_the_end(tmp_path):
    placed = [{"idx": 0, "placed_start": 0.8, "placed_end": 1.8,
               "audio_path": _tone(tmp_path / "a.wav", 1.0)}]
    out = tmp_path / "n.wav"
    n.mix_narration(placed, str(out), duration=1.0, sample_rate=SR,
                    loudnorm=False)
    data, _ = sf.read(str(out))
    assert len(data) == SR
