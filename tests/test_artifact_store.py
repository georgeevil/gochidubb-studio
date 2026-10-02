"""Tests for app/artifact_store.py and app/reuse_runtime.py.

The store's job is to be safe rather than clever: a miss costs recomputation,
but a bad hit produces a wrong dub with nothing in the log. These tests are
mostly about the ways a hit must be refused.
"""
import sqlite3

import pytest

from app import artifact_store, reuse, reuse_runtime


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    artifact_store.init_store(db)
    monkeypatch.setattr(artifact_store, "_DB_PATH", db)
    return db


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    return root


def _job_with(outputs, job_id, files=("audio_16k.wav",)):
    d = outputs / job_id
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).write_bytes(b"audio")
    return d


class TestRecordAndLookup:
    def test_round_trip(self, store, outputs):
        _job_with(outputs, "job1")
        artifact_store.record("fp1", "extract", "job1",
                              files=["audio_16k.wav"],
                              ctx_keys={"audio_16k": "/x/job1/audio_16k.wav"},
                              quality={"segments": 3})
        hit = artifact_store.lookup("fp1", "extract", outputs)
        assert hit["job_id"] == "job1"
        assert hit["quality"] == {"segments": 3}
        assert hit["ctx"]["audio_16k"].endswith("audio_16k.wav")

    def test_miss_returns_none(self, store, outputs):
        assert artifact_store.lookup("nope", "extract", outputs) is None

    def test_stage_is_part_of_the_identity(self, store, outputs):
        _job_with(outputs, "job1")
        artifact_store.record("fp1", "extract", "job1", files=[], ctx_keys={})
        assert artifact_store.lookup("fp1", "transcribe", outputs) is None

    def test_a_deleted_file_is_a_miss_not_a_crash(self, store, outputs):
        """outputs/ is a directory users delete from. Handing back a row that
        points at deleted audio would fail the run instead of missing."""
        d = _job_with(outputs, "job1")
        artifact_store.record("fp1", "extract", "job1",
                              files=["audio_16k.wav"], ctx_keys={})
        (d / "audio_16k.wav").unlink()
        assert artifact_store.lookup("fp1", "extract", outputs) is None
        # and the dead row is forgotten
        assert not artifact_store.entries(stage="extract")

    def test_a_deleted_job_directory_is_a_miss(self, store, outputs):
        _job_with(outputs, "job1")
        artifact_store.record("fp1", "extract", "job1",
                              files=["audio_16k.wav"], ctx_keys={})
        import shutil
        shutil.rmtree(outputs / "job1")
        assert artifact_store.lookup("fp1", "extract", outputs) is None

    def test_re_recording_points_at_the_newest_producer(self, store, outputs):
        _job_with(outputs, "job1")
        _job_with(outputs, "job2")
        artifact_store.record("fp1", "extract", "job1", files=[], ctx_keys={})
        artifact_store.record("fp1", "extract", "job2", files=[], ctx_keys={})
        assert artifact_store.lookup("fp1", "extract", outputs)["job_id"] == "job2"

    def test_hits_are_counted(self, store, outputs):
        _job_with(outputs, "job1")
        artifact_store.record("fp1", "extract", "job1", files=[], ctx_keys={})
        artifact_store.mark_hit("fp1", "extract")
        artifact_store.mark_hit("fp1", "extract")
        assert artifact_store.lookup("fp1", "extract", outputs)["hits"] == 2

    def test_empty_fingerprint_is_never_stored(self, store, outputs):
        assert not artifact_store.record("", "extract", "job1",
                                         files=[], ctx_keys={})


class TestStatsAndPurge:
    def test_stats_group_by_stage(self, store, outputs):
        _job_with(outputs, "job1")
        artifact_store.record("a", "extract", "job1", files=[], ctx_keys={})
        artifact_store.record("b", "transcribe", "job1", files=[], ctx_keys={})
        st = artifact_store.stats(outputs)
        assert st["total"] == 2
        assert {s["stage"] for s in st["stages"]} == {"extract", "transcribe"}

    def test_purge_by_stage_leaves_others(self, store, outputs):
        _job_with(outputs, "job1")
        artifact_store.record("a", "extract", "job1", files=[], ctx_keys={})
        artifact_store.record("b", "transcribe", "job1", files=[], ctx_keys={})
        assert artifact_store.purge(stage="extract") == 1
        assert {e["stage"] for e in artifact_store.entries()} == {"transcribe"}

    def test_purge_only_touches_the_index(self, store, outputs):
        """Job files belong to their job and are deleted through job deletion."""
        d = _job_with(outputs, "job1")
        artifact_store.record("a", "extract", "job1",
                              files=["audio_16k.wav"], ctx_keys={})
        artifact_store.purge()
        assert (d / "audio_16k.wav").exists()

    def test_store_calls_survive_an_uninitialised_db(self, monkeypatch, outputs):
        monkeypatch.setattr(artifact_store, "_DB_PATH", None)
        assert artifact_store.lookup("a", "extract", outputs) is None
        assert artifact_store.record("a", "extract", "j", files=[], ctx_keys={}) is False
        assert artifact_store.entries() == []
        assert artifact_store.purge() == 0


class TestCopyArtifacts:
    def test_materializes_files_into_the_new_job(self, store, outputs, tmp_path):
        _job_with(outputs, "job1", files=("audio_16k.wav", "audio_hq.wav"))
        artifact_store.record("fp", "extract", "job1",
                              files=["audio_16k.wav", "audio_hq.wav"], ctx_keys={})
        entry = artifact_store.lookup("fp", "extract", outputs)
        dest = outputs / "job2"
        assert artifact_store.copy_artifacts(entry, dest)
        assert (dest / "audio_16k.wav").read_bytes() == b"audio"

    def test_copies_directories_too(self, store, outputs):
        d = _job_with(outputs, "job1", files=())
        (d / "speaker_refs").mkdir()
        (d / "speaker_refs" / "ref_SPEAKER_00.wav").write_bytes(b"ref")
        artifact_store.record("fp", "diarize", "job1",
                              files=["speaker_refs"], ctx_keys={})
        entry = artifact_store.lookup("fp", "diarize", outputs)
        dest = outputs / "job2"
        assert artifact_store.copy_artifacts(entry, dest)
        assert (dest / "speaker_refs" / "ref_SPEAKER_00.wav").exists()

    def test_existing_destination_file_is_left_alone(self, store, outputs):
        _job_with(outputs, "job1")
        artifact_store.record("fp", "extract", "job1",
                              files=["audio_16k.wav"], ctx_keys={})
        entry = artifact_store.lookup("fp", "extract", outputs)
        dest = outputs / "job2"
        dest.mkdir()
        (dest / "audio_16k.wav").write_bytes(b"mine")
        artifact_store.copy_artifacts(entry, dest)
        assert (dest / "audio_16k.wav").read_bytes() == b"mine"


class TestEnabledStages:
    class Cfg:
        reuse_enabled = True
        reuse_stages = "download,extract,transcribe,diarize"

    def test_off_means_nothing_is_reused(self):
        class Off(self.Cfg):
            reuse_enabled = False
        assert reuse_runtime.enabled_stages(Off()) == ()

    def test_only_listed_stages_are_enabled(self):
        allowed = reuse_runtime.enabled_stages(self.Cfg())
        assert "transcribe" in allowed
        assert "translate" not in allowed

    def test_unknown_stage_names_are_ignored(self):
        class Weird(self.Cfg):
            reuse_stages = "transcribe,not_a_stage"
        assert reuse_runtime.enabled_stages(Weird()) == ("transcribe",)

    def test_every_stage_spec_matches_a_cacheable_stage(self):
        assert set(reuse_runtime.STAGE_ARTIFACTS) == set(reuse.CACHEABLE_STAGES)


class TestTryReuse:
    class Cfg:
        reuse_enabled = True
        reuse_stages = "extract,transcribe"

    def test_a_hit_populates_context_and_rebases_paths(self, store, outputs):
        """A reused ctx must point at the new job's copies, not the old job's."""
        _job_with(outputs, "job1")
        artifact_store.record(
            "fp", "extract", "job1", files=["audio_16k.wav"],
            ctx_keys={"audio_16k": str(outputs / "job1" / "audio_16k.wav"),
                      "duration": 51.4})
        ctx = {}
        hit = reuse_runtime.try_reuse("extract", "fp", ctx, outputs / "job2",
                                      outputs, self.Cfg())
        assert hit and hit["job_id"] == "job1"
        assert ctx["duration"] == 51.4
        assert "job2" in ctx["audio_16k"]
        assert (outputs / "job2" / "audio_16k.wav").exists()

    def test_a_failed_gate_refuses_the_hit(self, store, outputs):
        _job_with(outputs, "job1")
        artifact_store.record("fp", "transcribe", "job1", files=[],
                              ctx_keys={"segments": []},
                              quality={"word_conf_mean": 0.05})
        ctx = {}
        assert reuse_runtime.try_reuse("transcribe", "fp", ctx,
                                       outputs / "job2", outputs,
                                       self.Cfg()) is None
        assert ctx == {}, "a refused hit must not leave half a context behind"

    def test_a_miss_returns_none(self, store, outputs):
        assert reuse_runtime.try_reuse("extract", "absent", {},
                                       outputs / "job2", outputs,
                                       self.Cfg()) is None

    def test_speaker_ref_dicts_are_rebased(self, store, outputs):
        d = _job_with(outputs, "job1", files=())
        (d / "speaker_refs").mkdir()
        (d / "speaker_refs" / "ref_SPEAKER_00.wav").write_bytes(b"ref")
        artifact_store.record(
            "fp", "diarize", "job1", files=["speaker_refs"],
            ctx_keys={"speaker_refs": {
                "SPEAKER_00": str(d / "speaker_refs" / "ref_SPEAKER_00.wav")}})
        ctx = {}
        reuse_runtime.try_reuse("diarize", "fp", ctx, outputs / "job2",
                                outputs, self.Cfg())
        assert "job2" in ctx["speaker_refs"]["SPEAKER_00"]

    def test_segment_audio_paths_are_rebased(self, store, outputs):
        d = _job_with(outputs, "job1", files=())
        (d / "tts_segments").mkdir()
        (d / "tts_segments" / "seg_0000.wav").write_bytes(b"x")
        artifact_store.record(
            "fp", "tts", "job1", files=["tts_segments"],
            ctx_keys={"segments": [
                {"idx": 0, "audio_path": str(d / "tts_segments" / "seg_0000.wav")}]})
        ctx = {}
        reuse_runtime.try_reuse("tts", "fp", ctx, outputs / "job2", outputs,
                                self.Cfg())
        assert "job2" in ctx["segments"][0]["audio_path"]


class TestRecordStage:
    def test_records_only_files_that_exist(self, store, outputs):
        work = _job_with(outputs, "job1", files=("audio_16k.wav",))
        reuse_runtime.record_stage("extract", "fp", "job1",
                                   {"audio_16k": "x", "duration": 5.0}, work)
        entry = artifact_store.lookup("fp", "extract", outputs)
        assert entry["files"] == ["audio_16k.wav"]      # audio_hq.wav absent

    def test_stores_measured_quality(self, store, outputs):
        work = _job_with(outputs, "job1", files=())
        ctx = {"segments": [{"word_conf_mean": 0.9, "no_speech_prob": 0.1}]}
        q = reuse_runtime.record_stage("transcribe", "fp", "job1", ctx, work)
        assert q["word_conf_mean"] == pytest.approx(0.9)
        assert artifact_store.lookup("fp", "transcribe", outputs)["quality"] == q

    def test_bad_output_is_recorded_with_its_score(self, store, outputs):
        """The gate refuses it at read time; the row still explains why
        nothing is being reused."""
        work = _job_with(outputs, "job1", files=())
        ctx = {"segments": [{"word_conf_mean": 0.05, "no_speech_prob": 0.9}]}
        reuse_runtime.record_stage("transcribe", "fp", "job1", ctx, work)
        entry = artifact_store.lookup("fp", "transcribe", outputs)
        assert entry is not None
        assert not reuse.passes_gate("transcribe", entry["quality"])[0]

    def test_unserialisable_context_is_dropped_not_fatal(self, store, outputs):
        work = _job_with(outputs, "job1", files=())
        ctx = {"segments": [], "source_lang_detected": "es",
               "effective_src": object()}           # not JSON-able
        assert reuse_runtime.record_stage("transcribe", "fp", "job1", ctx, work) is not None
        stored = artifact_store.lookup("fp", "transcribe", outputs)["ctx"]
        assert stored["source_lang_detected"] == "es"
        assert "effective_src" not in stored

    def test_no_fingerprint_records_nothing(self, store, outputs):
        work = _job_with(outputs, "job1", files=())
        assert reuse_runtime.record_stage("extract", "", "job1", {}, work) is None


class TestFingerprintForContext:
    class Cfg:
        reuse_enabled = True
        reuse_stages = ",".join(reuse.CACHEABLE_STAGES)
        whisper_model = "large-v3"

    def test_transcribe_keyed_on_audio_content(self, tmp_path):
        a = tmp_path / "a.wav"
        a.write_bytes(b"one")
        ctx = {"audio_16k": str(a), "whisper_model": "large-v3",
               "source_lang": "es"}
        first = reuse_runtime.fingerprint_for("transcribe", ctx, self.Cfg())
        a.write_bytes(b"two")
        assert reuse_runtime.fingerprint_for("transcribe", ctx, self.Cfg()) != first

    def test_target_language_does_not_affect_transcribe(self, tmp_path):
        """The whole point: a redub into a new language reuses the transcript."""
        a = tmp_path / "a.wav"
        a.write_bytes(b"audio")
        base = {"audio_16k": str(a), "whisper_model": "m", "source_lang": "es"}
        bg = reuse_runtime.fingerprint_for("transcribe", {**base, "target_lang": "bg"}, self.Cfg())
        ru = reuse_runtime.fingerprint_for("transcribe", {**base, "target_lang": "ru"}, self.Cfg())
        assert bg == ru and bg is not None

    def test_translation_model_does_not_affect_transcribe(self, tmp_path):
        a = tmp_path / "a.wav"
        a.write_bytes(b"audio")
        base = {"audio_16k": str(a), "whisper_model": "m", "source_lang": "es"}
        one = reuse_runtime.fingerprint_for("transcribe", {**base, "model": "gpt-oss"}, self.Cfg())
        two = reuse_runtime.fingerprint_for("transcribe", {**base, "model": "qwen"}, self.Cfg())
        assert one == two and one is not None

    def test_initial_prompt_and_backend_affect_transcribe(self, tmp_path):
        """A vocabulary hint or a different ASR engine changes the words."""
        a = tmp_path / "a.wav"
        a.write_bytes(b"audio")
        base = {"audio_16k": str(a), "whisper_model": "m", "source_lang": "es"}
        plain = reuse_runtime.fingerprint_for("transcribe", base, self.Cfg())
        hinted = reuse_runtime.fingerprint_for(
            "transcribe", {**base, "initial_prompt": "Talamanca"}, self.Cfg())
        assert plain != hinted

        class FW(self.Cfg):
            asr_backend = "faster-whisper"

        class MLX(self.Cfg):
            asr_backend = "mlx"

        assert (reuse_runtime.fingerprint_for("transcribe", base, FW())
                != reuse_runtime.fingerprint_for("transcribe", base, MLX()))

    def test_speaker_hints_affect_diarize(self, tmp_path):
        a = tmp_path / "a.wav"
        a.write_bytes(b"audio")
        base = {"audio_16k": str(a), "speaker_mode": "all"}
        auto = reuse_runtime.fingerprint_for("diarize", base, self.Cfg())
        hinted = reuse_runtime.fingerprint_for(
            "diarize", {**base, "min_speakers": 5, "max_speakers": 9}, self.Cfg())
        assert auto != hinted

    def test_target_language_does_affect_translate(self, tmp_path):
        base = {"segments": [{"start": 0, "end": 1, "text": "Hola"}],
                "model": "m", "context_hint": ""}
        bg = reuse_runtime.fingerprint_for("translate", {**base, "target_lang": "bg"}, self.Cfg())
        ru = reuse_runtime.fingerprint_for("translate", {**base, "target_lang": "ru"}, self.Cfg())
        assert bg != ru

    def test_unreadable_audio_refuses_to_key(self):
        ctx = {"audio_16k": "/nope/missing.wav", "whisper_model": "m",
               "source_lang": "es"}
        assert reuse_runtime.fingerprint_for("transcribe", ctx, self.Cfg()) is None

    def test_uncacheable_stage_returns_none(self):
        assert reuse_runtime.fingerprint_for("assemble", {}, self.Cfg()) is None

    def test_a_broken_context_never_raises(self):
        """Caching must never be able to fail a job. A context the hasher
        cannot handle has to come back as "not cacheable", not as an
        exception out of the pipeline runner."""
        got = reuse_runtime.fingerprint_for(
            "translate", {"segments": object(), "target_lang": "bg"}, self.Cfg())
        assert got is None or isinstance(got, str)


class TestPlan:
    class Cfg:
        reuse_enabled = True
        reuse_stages = "transcribe"

    def test_reports_allowed_and_unknown_stages(self, store, outputs):
        rows = {r["stage"]: r for r in
                reuse_runtime.plan({}, outputs, self.Cfg())}
        assert rows["transcribe"]["allowed"] is True
        assert rows["extract"]["allowed"] is False
        # No audio in the context yet, so transcribe cannot be keyed.
        assert rows["transcribe"]["known"] is False

    def test_reports_a_hit_when_one_exists(self, store, outputs, tmp_path):
        a = tmp_path / "a.wav"
        a.write_bytes(b"audio")
        ctx = {"audio_16k": str(a), "whisper_model": "m", "source_lang": "es"}
        fp = reuse_runtime.fingerprint_for("transcribe", ctx, self.Cfg())
        _job_with(outputs, "job1", files=())
        artifact_store.record(fp, "transcribe", "job1", files=[],
                              ctx_keys={}, quality={"word_conf_mean": 0.9})
        row = {r["stage"]: r for r in
               reuse_runtime.plan(ctx, outputs, self.Cfg())}["transcribe"]
        assert row["hit"] and row["would_reuse"] and row["job_id"] == "job1"

    def test_a_gated_hit_is_reported_as_blocked(self, store, outputs, tmp_path):
        a = tmp_path / "a.wav"
        a.write_bytes(b"audio")
        ctx = {"audio_16k": str(a), "whisper_model": "m", "source_lang": "es"}
        fp = reuse_runtime.fingerprint_for("transcribe", ctx, self.Cfg())
        _job_with(outputs, "job1", files=())
        artifact_store.record(fp, "transcribe", "job1", files=[],
                              ctx_keys={}, quality={"word_conf_mean": 0.05})
        row = {r["stage"]: r for r in
               reuse_runtime.plan(ctx, outputs, self.Cfg())}["transcribe"]
        assert row["hit"] and not row["would_reuse"] and row["blocked_by"]


class TestSchema:
    def test_init_is_idempotent(self, tmp_path):
        db = tmp_path / "x.db"
        artifact_store.init_store(db)
        artifact_store.init_store(db)
        conn = sqlite3.connect(str(db))
        cols = {r[1] for r in conn.execute("PRAGMA table_info(artifacts)")}
        conn.close()
        assert {"fingerprint", "stage", "job_id", "quality", "payload"} <= cols


class TestNoCrossLanguageContamination:
    """Stages before `translate` do not have the target language in their
    fingerprint — that is what makes them reusable across languages. So they
    must never carry a translation with them."""

    def test_transcribe_never_records_a_translation(self, store, outputs):
        work = _job_with(outputs, "job1", files=())
        ctx = {"segments": [
            {"idx": 0, "text": "Hola mamá", "translated_text": "Здравей, мамо",
             "word_conf_mean": 0.9},
        ]}
        reuse_runtime.record_stage("transcribe", "fp", "job1", ctx, work)
        stored = artifact_store.lookup("fp", "transcribe", outputs)["ctx"]
        assert "translated_text" not in stored["segments"][0]
        assert stored["segments"][0]["text"] == "Hola mamá"

    def test_diarize_never_records_a_translation(self, store, outputs):
        work = _job_with(outputs, "job1", files=())
        ctx = {"segments": [{"idx": 0, "text": "Hola",
                             "translated_text": "Здравей"}],
               "speaker_refs": {}}
        reuse_runtime.record_stage("diarize", "fp", "job1", ctx, work)
        stored = artifact_store.lookup("fp", "diarize", outputs)["ctx"]
        assert "translated_text" not in stored["segments"][0]

    def test_tts_does_keep_its_translations(self, store, outputs):
        """tts is keyed on the translated text, so it must carry it."""
        work = _job_with(outputs, "job1", files=())
        ctx = {"segments": [{"idx": 0, "text": "Hola",
                             "translated_text": "Здравей"}]}
        reuse_runtime.record_stage("tts", "fp", "job1", ctx, work)
        stored = artifact_store.lookup("fp", "tts", outputs)["ctx"]
        assert stored["segments"][0]["translated_text"] == "Здравей"
