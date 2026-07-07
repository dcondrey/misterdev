import tempfile
from pathlib import Path

from misterdev.core.planning.lesson_store import _MAX_INJECT, _MAX_LESSONS
from misterdev.core.planning.metacognition import (
    SessionAuditor,
    _extract_json_array,
)


# --- JSON array extraction (unchanged parser) -----------------------------


def test_extract_json_array_simple():
    assert _extract_json_array('["rule1", "rule2"]') == ["rule1", "rule2"]


def test_extract_json_array_with_preamble():
    text = 'Here are the rules:\n["always run tests", "check imports"]\nDone.'
    assert _extract_json_array(text) == ["always run tests", "check imports"]


def test_extract_json_array_no_array():
    assert _extract_json_array("no json here") == []


def test_extract_json_array_empty():
    assert _extract_json_array("[]") == []


def test_extract_json_array_malformed():
    assert _extract_json_array("[not valid json}") == []


def test_extract_json_array_nested_brackets():
    assert _extract_json_array('text [ "a" ] more text') == ["a"]


# --- SessionAuditor over the scored lesson store --------------------------


def _auditor():
    return SessionAuditor(Path(tempfile.mkdtemp()), object())


def test_save_lessons_records_and_retrieves():
    a = _auditor()
    a._save_lessons(["always run black", "close DB connections"])
    ctx = a.get_lessons_context()
    assert "Project-Specific Lessons" in ctx
    assert "always run black" in ctx and "close DB connections" in ctx


def test_duplicate_in_one_call_is_deduped():
    a = _auditor()
    added = a._save_lessons(["run the migration first", "run the migration first"])
    assert added == 1


def test_reinforced_lesson_survives_newer_noise():
    a = _auditor()
    for _ in range(5):
        a._save_lessons(["regenerate bindings after touching the FFI header"])
    a._save_lessons([f"incidental {i}" for i in range(_MAX_LESSONS + 15)])
    assert "regenerate bindings" in a.get_lessons_context()


def test_injection_is_bounded():
    a = _auditor()
    for i in range(_MAX_LESSONS + 20):
        a._save_lessons([f"distinct lesson {i}"])
    ctx = a.get_lessons_context()
    assert ctx.count("\n- ") <= _MAX_INJECT


def test_get_lessons_context_biases_to_query():
    a = _auditor()
    a._save_lessons(["validate every external input at the boundary"])
    a._save_lessons(["cache the formatter; construction is expensive"])
    ctx = a.get_lessons_context("add input validation to the parser")
    # The relevant lesson leads the injected block.
    first = ctx.splitlines()[1]
    assert "external input" in first


def test_get_lessons_context_empty():
    assert _auditor().get_lessons_context() == ""


def test_get_lessons_context_handles_corrupt_file():
    a = _auditor()
    a.lessons_file.parent.mkdir(parents=True, exist_ok=True)
    a.lessons_file.write_text("not json", encoding="utf-8")
    assert a.get_lessons_context() == ""
