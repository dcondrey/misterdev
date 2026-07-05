import pytest
from misterdev.llm.prompt_manager import PromptManager, _safe_format


def _pm(templates):
    return PromptManager({"prompt_templates": templates})


def test_double_brace_syntax():
    pm = _pm({"t": "Hello {{name}}, do {{task}}"})
    result = pm.format_prompt("t", {"name": "world", "task": "fix"})
    assert result == "Hello world, do fix"


def test_legacy_brace_syntax():
    pm = _pm({"t": "Do {task_desc}"})
    result = pm.format_prompt("t", {"task_desc": "implement auth"})
    assert result == "Do implement auth"


def test_unknown_braces_preserved():
    pm = _pm({"t": 'JSON: {"key": 1} and {task_desc}'})
    result = pm.format_prompt("t", {"task_desc": "test"})
    assert '{"key": 1}' in result
    assert "test" in result


def test_code_with_braces():
    pm = _pm({"t": "Context: {{code}}"})
    result = pm.format_prompt("t", {"code": "if x { y } else { z }"})
    assert "if x { y } else { z }" in result


def test_missing_template_raises():
    pm = _pm({})
    with pytest.raises(ValueError):
        pm.format_prompt("nonexistent", {})


def test_inherited_system_prompt():
    pm = _pm({"system": "Be helpful.", "t": "Do {inherited_system_prompt}"})
    result = pm.format_prompt("t", {})
    assert "Be helpful." in result


def test_safe_format_no_match():
    assert _safe_format("{unknown_var}", {}) == "{unknown_var}"


def test_safe_format_dotted():
    assert _safe_format("{task.name}", {"task.name": "foo"}) == "foo"
