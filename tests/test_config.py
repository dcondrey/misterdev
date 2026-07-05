import tempfile
from pathlib import Path

from misterdev.config import ConfigManager, DEFAULT_CONFIG


def test_config_defaults_not_mutated():
    cm1 = ConfigManager()
    cm1.global_config["llm"]["model"] = "changed-model"
    cm2 = ConfigManager()
    assert cm2.global_config["llm"]["model"] == DEFAULT_CONFIG["llm"]["model"]


def test_load_project_config_merges():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "project.yaml").write_text(
            "name: my-proj\nlanguage: rust\nllm:\n  model: anthropic/claude-opus-4\n"
        )
        cm = ConfigManager()
        config = cm.load_project_config(td)
        assert config["name"] == "my-proj"
        assert config["language"] == "rust"
        assert config["llm"]["model"] == "anthropic/claude-opus-4"
        assert config["llm"]["provider"] == "openrouter"


def test_load_project_config_no_yaml():
    with tempfile.TemporaryDirectory() as td:
        cm = ConfigManager()
        config = cm.load_project_config(td)
        assert config["llm"]["provider"] == "openrouter"
        assert config["tools"] == []


def test_load_project_config_invalid_yaml():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "project.yaml").write_text(": : : invalid yaml [[[")
        cm = ConfigManager()
        config = cm.load_project_config(td)
        assert config["llm"]["provider"] == "openrouter"


def test_load_project_config_non_dict_yaml():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "project.yaml").write_text("just a string\n")
        cm = ConfigManager()
        config = cm.load_project_config(td)
        assert config["llm"]["provider"] == "openrouter"


def test_deep_update():
    cm = ConfigManager()
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    overlay = {"a": {"b": 99}, "e": 4}
    result = cm._deep_update(base, overlay)
    assert result["a"]["b"] == 99
    assert result["a"]["c"] == 2
    assert result["d"] == 3
    assert result["e"] == 4


def test_prompt_templates_present():
    assert "system" in DEFAULT_CONFIG["prompt_templates"]
    assert "task_completion_instruction" in DEFAULT_CONFIG["prompt_templates"]
    assert "error_correction_instruction" in DEFAULT_CONFIG["prompt_templates"]


def test_prompt_templates_reference_context_fields():
    task_tmpl = DEFAULT_CONFIG["prompt_templates"]["task_completion_instruction"]
    assert "{interface_contracts}" in task_tmpl
    assert "{recent_changes}" in task_tmpl
    assert "{scratchpad}" in task_tmpl
    assert "{acceptance_criteria}" in task_tmpl
    assert "{code_context}" in task_tmpl


def test_load_does_not_mutate_defaults():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "project.yaml").write_text(
            "prompt_templates:\n  system: custom system prompt\n"
        )
        cm = ConfigManager()
        config = cm.load_project_config(td)
        assert config["prompt_templates"]["system"] == "custom system prompt"
        assert DEFAULT_CONFIG["prompt_templates"]["system"] != "custom system prompt"
