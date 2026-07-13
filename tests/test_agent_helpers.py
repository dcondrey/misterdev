"""Tests for the extracted agent helpers."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from misterdev.agent_helpers import (
    _apply_budget_ceiling,
    _budget_exhausted,
    _check_golden_config,
    _combine_commands,
    _warn_if_no_test_gate,
    _warn_if_test_gate_is_noop,
)
from misterdev.core.verification.validator import gate_ran_no_tests


def _report():
    return SimpleNamespace(degraded_subsystems=[], key_decisions=[])


def _assessment(test_command, tests_pass, test_count, test_output=""):
    return SimpleNamespace(
        structure=SimpleNamespace(test_command=test_command),
        health=SimpleNamespace(
            tests_pass=tests_pass, test_count=test_count, test_output=test_output
        ),
    )


def test_no_tests_ran_signal_detects_common_runners():
    assert gate_ran_no_tests("no tests ran in 0.01s")  # pytest
    assert gate_ran_no_tests("collected 0 items")  # pytest
    assert gate_ran_no_tests("No tests found, exiting")  # jest
    assert gate_ran_no_tests("?   pkg   [no test files]")  # go
    assert not gate_ran_no_tests("5 passed in 1.2s")


def test_warn_noop_fires_on_zero_tests_with_signal():
    a = _assessment(
        "pytest -k nope",
        tests_pass=True,
        test_count=0,
        test_output="collected 0 items\n\nno tests ran in 0.01s",
    )
    report = _report()
    _warn_if_test_gate_is_noop(a, report)
    assert any("no-op test gate" in d.lower() for d in report.degraded_subsystems)


def test_warn_noop_silent_when_tests_actually_ran():
    a = _assessment("pytest", tests_pass=True, test_count=42, test_output="42 passed")
    report = _report()
    _warn_if_test_gate_is_noop(a, report)
    assert not report.degraded_subsystems


def test_warn_noop_silent_without_explicit_signal():
    # A 0 count with no zero-test phrase may just be an unparsed format: don't cry wolf.
    a = _assessment(
        "run-tests", tests_pass=True, test_count=0, test_output="OK: all green"
    )
    report = _report()
    _warn_if_test_gate_is_noop(a, report)
    assert not report.degraded_subsystems


def test_warn_noop_silent_on_cargo_workspace_mix():
    # cargo prints "running 0 tests" for empty crates; a nonzero total means the
    # gate is real, so the phrase alone must not trip the warning.
    out = "running 0 tests\n\nrunning 5 tests\ntest result: ok. 5 passed; 0 failed"
    a = _assessment("cargo test", tests_pass=True, test_count=5, test_output=out)
    report = _report()
    _warn_if_test_gate_is_noop(a, report)
    assert not report.degraded_subsystems


def test_warn_noop_silent_when_no_command_or_baseline_red():
    report = _report()
    _warn_if_test_gate_is_noop(_assessment(None, True, 0, "no tests ran"), report)
    _warn_if_test_gate_is_noop(_assessment("pytest", False, 0, "no tests ran"), report)
    assert not report.degraded_subsystems  # missing/red baseline handled elsewhere


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
