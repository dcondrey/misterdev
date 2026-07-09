"""Tests for the failure-triggered single-test execution probe.

``isolate_command`` is pure and is covered across all six runners plus the
unrecognized case. ``run_probe`` is exercised only with a trivial shell command
(``echo``) and a forced timeout — never a real toolchain, so these tests don't
contend with a build/benchmark run.
"""

import sys

import pytest

from misterdev.core.execution.probe import isolate_command, run_probe


# --- isolate_command: per-runner isolate flags ------------------------------


def test_pytest_uses_dash_k():
    cmd = isolate_command("python -m pytest -q", "test_widget_parses", "python")
    assert cmd is not None
    assert "-k" in cmd
    assert "test_widget_parses" in cmd
    # The base command is preserved (we filter, not replace).
    assert cmd.startswith("python -m pytest -q")


def test_cargo_uses_positional_filter():
    cmd = isolate_command("cargo test", "my_module::works", "rust")
    assert cmd is not None
    assert cmd.startswith("cargo test ")
    assert "my_module::works" in cmd


def test_jest_via_npm_uses_dash_t():
    cmd = isolate_command("npm test", "renders the header", "javascript")
    assert cmd is not None
    assert "npm test --" in cmd
    assert '-t "renders the header"' in cmd


def test_vitest_via_npm_uses_dash_t():
    # vitest also runs through `npm test`; language typescript resolves it.
    cmd = isolate_command("npm test", "adds two numbers", "typescript")
    assert cmd is not None
    assert "npm test --" in cmd
    assert '-t "adds two numbers"' in cmd


def test_swift_uses_filter():
    cmd = isolate_command("swift test", "MathTests/testAddition", "swift")
    assert cmd is not None
    assert "swift test --filter" in cmd
    assert "MathTests/testAddition" in cmd


def test_dotnet_uses_fully_qualified_name_filter():
    cmd = isolate_command("dotnet test", "Calc.Tests.AddWorks", "csharp")
    assert cmd is not None
    assert "dotnet test --filter" in cmd
    assert "FullyQualifiedName~Calc.Tests.AddWorks" in cmd


def test_unrecognized_runner_returns_none():
    assert isolate_command("make check", "some_test", "cobol") is None


def test_empty_inputs_return_none():
    assert isolate_command("", "test_x", "python") is None
    assert isolate_command("python -m pytest", "", "python") is None


def test_runner_inferred_from_command_when_language_blank():
    cmd = isolate_command("cargo test --workspace", "it_works", "")
    assert cmd is not None
    assert cmd.startswith("cargo test ")
    assert "it_works" in cmd


def test_test_name_is_shell_safe():
    # A name with shell metacharacters must not break quoting / inject.
    cmd = isolate_command("npm test", 'evil"; rm -rf /', "javascript")
    assert cmd is not None
    # The double-quote is escaped, so the argument stays a single quoted string.
    assert '\\"' in cmd


# --- run_probe: bounded external execution, never raising --------------------


def test_run_probe_captures_output(tmp_path):
    out = run_probe(str(tmp_path), "echo hi")
    assert out is not None
    assert "hi" in out


def test_run_probe_times_out_returns_none(tmp_path):
    # A sleep longer than the timeout must be killed and yield None, not hang.
    out = run_probe(
        str(tmp_path), f'{sys.executable} -c "import time; time.sleep(30)"', timeout=1
    )
    assert out is None


def test_run_probe_empty_command_returns_none(tmp_path):
    assert run_probe(str(tmp_path), "") is None


def test_run_probe_bad_cwd_returns_none():
    # A nonexistent cwd makes Popen raise; the probe must swallow it.
    assert run_probe("/no/such/dir/xyz123", "echo hi") is None
