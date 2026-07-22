"""T1.1 — spec-as-tests is default-on, with the baseline-safety invariant locked.

Reproduction-first is now the default (`spec_as_tests=True`), not opt-in. The
"hardened so the baseline does not go red inside a wave" property already exists —
the generated test is written under `.orchestrator/spec_tests/`, OUTSIDE the project
suite, so it can never be collected by the project's own test run and therefore can
never flip the integration-gate baseline. This locks that invariant so default-on is
safe. (Compiled-language spec tests are deliberately still generated as injected
targets even when unrunnable as a gate — that behavior is covered by
test_spec_tests.py and intentionally unchanged.)
"""

from pathlib import Path
from types import SimpleNamespace

from misterdev.config import DEFAULT_CONFIG
from misterdev.task_executors.markdown_plan_executor.critic_spec_mixin import (
    CriticSpecMixin,
)


class _Ex(CriticSpecMixin):
    pass


class _Client:
    def __init__(self):
        self.calls = 0

    def generate_code(self, prompt, system):
        self.calls += 1
        return "```python\nassert False\n```"


def _project(tmp_path):
    return SimpleNamespace(
        path=tmp_path,
        llm_client=_Client(),
        config={
            "orchestrator": {"spec_as_tests": True},
            "language": "python",
            "test_command": "pytest",
        },
    )


def _task():
    return SimpleNamespace(
        id="T1", acceptance_criteria="The widget must render.", description="render it"
    )


def test_spec_as_tests_defaults_on():
    assert DEFAULT_CONFIG["orchestrator"]["spec_as_tests"] is True


def test_generated_spec_written_outside_project_suite(tmp_path):
    # Hardening invariant: default-on is safe because the generated test lives in
    # the baseline-safe .orchestrator/ lane, never the collected project suite.
    project = _project(tmp_path)
    path, source = _Ex()._maybe_generate_spec_test(project, _task())
    assert path is not None and source
    assert Path(path).parent == tmp_path / ".orchestrator" / "spec_tests"


def test_default_on_generation_actually_runs(tmp_path):
    # With the default flipped on, a supported-language task generates (control).
    project = _project(tmp_path)
    path, _ = _Ex()._maybe_generate_spec_test(project, _task())
    assert path is not None
    assert project.llm_client.calls == 1
