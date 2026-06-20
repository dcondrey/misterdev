import json
import tempfile
from pathlib import Path

from my_project_orchestrator.core.metacognition import _extract_json_array, SessionAuditor


def test_extract_json_array_simple():
    assert _extract_json_array('["rule1", "rule2"]') == ["rule1", "rule2"]


def test_extract_json_array_with_preamble():
    text = 'Here are the rules:\n["always run tests", "check imports"]\nDone.'
    result = _extract_json_array(text)
    assert result == ["always run tests", "check imports"]


def test_extract_json_array_no_array():
    assert _extract_json_array("no json here") == []


def test_extract_json_array_empty():
    assert _extract_json_array("[]") == []


def test_extract_json_array_malformed():
    assert _extract_json_array("[not valid json}") == []


def test_extract_json_array_nested_brackets():
    assert _extract_json_array('text [ "a" ] more text') == ["a"]


def test_save_lessons_creates_file():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        class FakeLLM:
            pass

        auditor = SessionAuditor(td, FakeLLM())
        auditor._save_lessons(["rule1", "rule2"])
        assert auditor.lessons_file.exists()
        data = json.loads(auditor.lessons_file.read_text())
        assert "rule1" in data
        assert "rule2" in data


def test_save_lessons_appends():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        class FakeLLM:
            pass

        auditor = SessionAuditor(td, FakeLLM())
        auditor._save_lessons(["rule1"])
        auditor._save_lessons(["rule2"])
        data = json.loads(auditor.lessons_file.read_text())
        assert "rule1" in data
        assert "rule2" in data


def test_save_lessons_deduplicates():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        class FakeLLM:
            pass

        auditor = SessionAuditor(td, FakeLLM())
        auditor._save_lessons(["rule1", "rule1"])
        data = json.loads(auditor.lessons_file.read_text())
        assert data.count("rule1") == 1


def test_get_lessons_context_empty():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        class FakeLLM:
            pass

        auditor = SessionAuditor(td, FakeLLM())
        assert auditor.get_lessons_context() == ""


def test_get_lessons_context_with_rules():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        class FakeLLM:
            pass

        auditor = SessionAuditor(td, FakeLLM())
        auditor._save_lessons(["always run black", "close DB connections"])
        ctx = auditor.get_lessons_context()
        assert "Project-Specific Lessons" in ctx
        assert "always run black" in ctx
        assert "close DB connections" in ctx


def test_get_lessons_context_handles_corrupt_file():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        class FakeLLM:
            pass

        auditor = SessionAuditor(td, FakeLLM())
        auditor.lessons_file.parent.mkdir(parents=True, exist_ok=True)
        auditor.lessons_file.write_text("not json")
        assert auditor.get_lessons_context() == ""
