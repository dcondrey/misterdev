"""Tests for the extracted agent helpers."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from my_project_orchestrator.agent_helpers import (
    _apply_budget_ceiling,
    _budget_exhausted,
    _check_golden_config,
    _combine_commands,
    _warn_if_no_test_gate,
)


def _report():
    return SimpleNamespace(degraded_subsystems=[], key_decisions=[])


# --- Finding 1: a malformed `targets` config must not crash the safety warning ---


def test_warn_if_no_test_gate_tolerates_non_dict_targets():
    # `targets` is an open, unvalidated config list; non-dict entries must not
    # raise AttributeError out of this advisory warning and abort the build.
    with tempfile.TemporaryDirectory() as td:
        project = SimpleNamespace(
            config={"targets": ["core", 123, None], "orchestrator": {}},
            path=Path(td),  # empty dir -> no test files -> early return
        )
        assessment = SimpleNamespace(structure=SimpleNamespace(test_command=None))
        report = _report()
        _warn_if_no_test_gate(assessment, project, report)  # must not raise
        assert report.degraded_subsystems == []


def test_warn_if_no_test_gate_respects_valid_dict_target():
    # A real target carrying a command means the repo IS gated per sub-project,
    # so no warning fires even with an empty top-level test command.
    with tempfile.TemporaryDirectory() as td:
        project = SimpleNamespace(
            config={"targets": [{"path": "core", "test_command": "pytest"}]},
            path=Path(td),
        )
        assessment = SimpleNamespace(structure=SimpleNamespace(test_command=None))
        report = _report()
        _warn_if_no_test_gate(assessment, project, report)
        assert report.degraded_subsystems == []


def test_warn_if_no_test_gate_silent_when_command_present():
    project = SimpleNamespace(config={}, path=Path("/nonexistent"))
    assessment = SimpleNamespace(structure=SimpleNamespace(test_command="pytest"))
    report = _report()
    _warn_if_no_test_gate(assessment, project, report)
    assert report.degraded_subsystems == []


# --- Finding 2: bool must not be treated as a numeric budget ---


def test_budget_exhausted_ignores_bool():
    assert _budget_exhausted(SimpleNamespace(budget_remaining=False)) is False
    assert _budget_exhausted(SimpleNamespace(budget_remaining=True)) is False


def test_budget_exhausted_numeric_paths():
    assert _budget_exhausted(SimpleNamespace(budget_remaining=0)) is True
    assert _budget_exhausted(SimpleNamespace(budget_remaining=-1.0)) is True
    assert _budget_exhausted(SimpleNamespace(budget_remaining=5.0)) is False
    assert _budget_exhausted(SimpleNamespace()) is False  # no attribute


def test_apply_budget_ceiling_treats_bool_as_non_numeric():
    client = SimpleNamespace(_budget=True)
    _apply_budget_ceiling(client, 50.0)
    assert client._budget == 50.0  # bool current -> fall back to flag


def test_apply_budget_ceiling_takes_min_of_numeric():
    client = SimpleNamespace(_budget=10.0)
    _apply_budget_ceiling(client, 50.0)
    assert client._budget == 10.0  # config cap is tighter, wins


def test_apply_budget_ceiling_non_numeric_uses_flag():
    client = SimpleNamespace()  # no _budget
    _apply_budget_ceiling(client, 25.0)
    assert client._budget == 25.0


# --- basic coverage for the remaining pure helpers ---


def test_combine_commands():
    assert _combine_commands(None, "", None) is None
    assert _combine_commands("a") == "(a)"
    assert _combine_commands("a", "b") == "(a) && (b)"
    assert _combine_commands(None, "b") == "(b)"


def test_check_golden_config_does_not_raise_on_partial(caplog):
    # Half-configured golden suite logs a warning but never raises.
    _check_golden_config({"orchestrator": {"golden_command": "make golden"}})
    _check_golden_config({"orchestrator": {"golden_paths": ["a.rs"]}})
    _check_golden_config({})  # no orchestrator section
