import json
import tempfile
from pathlib import Path

from misterdev.core.planning.lesson_store import (
    _MAX_LESSONS,
    _MIN_SCORE,
    LessonStore,
)


def _store() -> LessonStore:
    return LessonStore(Path(tempfile.mkdtemp()) / ".orchestrator" / "lessons.json")


def _lessons(store: LessonStore):
    lessons, _ = store._load()
    return lessons


def _by_text(store: LessonStore, needle: str):
    return next((le for le in _lessons(store) if needle in le.text), None)


def test_record_adds_new_lessons():
    s = _store()
    added = s.record(["close DB connections in tests", "pin the toolchain"])
    assert added == 2
    texts = s.retrieve()
    assert any("DB connections" in t for t in texts)


def test_recurrence_reinforces_instead_of_duplicating():
    s = _store()
    s.record(["always run black before committing"])
    s.record(["always run black before committing"])
    le = _by_text(s, "black")
    assert le is not None
    assert le.hits == 2
    assert le.score > 1.0  # reinforced above a fresh lesson
    # Exactly one lesson about black — no duplicate accumulated.
    assert sum("black" in x.text for x in _lessons(s)) == 1


def test_reworded_lesson_merges():
    s = _store()
    s.record(["always run black before committing changes"])
    added = s.record(["run black before commit"])  # reworded restatement
    assert added == 0  # merged, not added
    assert sum("black" in x.text for x in _lessons(s)) == 1
    le = _by_text(s, "black")
    assert le.hits == 2
    assert le.text == "run black before commit"  # refreshed to newest wording


def test_distinct_lessons_do_not_merge():
    s = _store()
    s.record(["close DB connections in tests"])
    added = s.record(["the migration must run before the seed script"])
    assert added == 1
    assert len(_lessons(s)) == 2


def test_value_eviction_keeps_proven_lesson_over_newer_noise():
    # The whole point vs the old recency eviction: a lesson reinforced across
    # runs must survive an influx of newer one-off noise.
    s = _store()
    for _ in range(5):
        s.record(["ALWAYS regenerate the parser after editing the grammar"])
    keystone = _by_text(s, "regenerate the parser")
    assert keystone.hits == 5 and keystone.score > 3.0
    # One run floods far more than the cap with distinct one-offs.
    s.record([f"incidental note number {i}" for i in range(_MAX_LESSONS + 15)])
    kept = _lessons(s)
    assert len(kept) <= _MAX_LESSONS
    assert any("regenerate the parser" in le.text for le in kept)  # not forgotten


def test_unreinforced_one_off_decays_and_is_dropped():
    s = _store()
    s.record(["one-off incidental observation about widget X"])
    # Many runs go by without that lesson recurring.
    for i in range(25):
        s.record([f"unrelated lesson {i}"])
    assert _by_text(s, "widget X") is None  # faded below the floor


def test_continually_reinforced_lesson_never_drops():
    s = _store()
    for i in range(25):
        s.record(["keep the wasm build off the shared cargo target lock"])
    le = _by_text(s, "cargo target lock")
    assert le is not None and le.score >= _MIN_SCORE
    assert le.hits == 25


def test_retrieval_ranks_by_relevance_to_query():
    s = _store()
    s.record(["validate every external input at the boundary"])
    s.record(["cache formatter instances; they are expensive to construct"])
    top = s.retrieve(query="add input validation to the request parser", limit=1)
    assert "external input" in top[0]


def test_retrieval_without_query_ranks_by_value():
    s = _store()
    s.record(["low-value note"])
    for _ in range(4):
        s.record(["high-value repeatedly-learned rule"])
    assert "repeatedly-learned" in s.retrieve(limit=1)[0]


def test_store_stays_capped():
    s = _store()
    for i in range(_MAX_LESSONS + 30):
        s.record([f"distinct lesson {i}"])
    assert len(_lessons(s)) <= _MAX_LESSONS


def test_migrates_legacy_string_list_format():
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    s.path.write_text(json.dumps(["legacy rule A", "legacy rule B"]), encoding="utf-8")
    texts = s.retrieve()
    assert any("legacy rule A" in t for t in texts)
    # A new record migrates the file to the scored format without loss.
    s.record(["legacy rule A"])  # reinforces the migrated one
    le = _by_text(s, "legacy rule A")
    assert le.hits == 2


def test_corrupt_file_degrades_to_empty():
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    s.path.write_text("not json", encoding="utf-8")
    assert s.retrieve() == []
    # And a subsequent record still works (overwrites the garbage).
    s.record(["a fresh rule"])
    assert any("fresh rule" in t for t in s.retrieve())


def test_non_string_rules_do_not_crash():
    s = _store()
    added = s.record([{"rule": "objects sometimes come back from the LLM"}, "plain"])
    assert added == 2
    assert len(_lessons(s)) == 2
