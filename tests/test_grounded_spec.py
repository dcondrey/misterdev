"""COMPLETE-mode spec grounds in hard signals, never fabricates from guesses."""

from types import SimpleNamespace

from misterdev.agent import ProjectOrchestrator
from misterdev.core.planning.assessment import (
    ProjectAssessment,
    HealthCheck,
    FeatureInventory,
    FeatureInfo,
)

_PROJECT = SimpleNamespace(name="demo")


def _spec(health=None, features=None):
    a = ProjectAssessment(
        health=health or HealthCheck(builds=True, tests_pass=True),
        features=features or FeatureInventory(),
    )
    return ProjectOrchestrator()._ground_completion_spec(a, _PROJECT)


def test_failing_tests_ground_the_spec():
    spec = _spec(
        health=HealthCheck(
            builds=True, tests_pass=False, test_output="AssertionError: 1 != 2"
        )
    )
    assert "Must Fix" in spec
    assert "AssertionError: 1 != 2" in spec
    assert "No concrete objective" not in spec


def test_located_todos_become_grounded_work():
    spec = _spec(
        features=FeatureInventory(
            todos=[{"file": "a.py", "line": 10, "text": "FIXME handle None"}]
        )
    )
    assert "Must Fix" in spec
    assert "a.py:10" in spec and "handle None" in spec


def test_documented_missing_is_should_add():
    spec = _spec(
        features=FeatureInventory(
            missing=[FeatureInfo(name="export", description="CSV export")]
        )
    )
    assert "Should Add" in spec
    assert "export" in spec


def test_healthy_repo_with_only_guesses_produces_no_tasks():
    # Green build+tests, only speculative "incomplete" inferences: the goal is
    # ill-posed. The spec must refuse to fabricate and demote guesses to advisory.
    spec = _spec(
        features=FeatureInventory(
            incomplete=[FeatureInfo(name="thing", description="maybe partial")],
            stubs=["helper() looks like a stub"],
        )
    )
    assert "No concrete objective" in spec
    assert "ZERO tasks" in spec
    # The guesses appear only under the advisory, never as Must-Fix work.
    assert "Advisory" in spec
    assert "Must Fix" not in spec
    assert "maybe partial" in spec  # present, but explicitly not a task


def test_guesses_are_demoted_even_when_hard_signals_exist():
    spec = _spec(
        health=HealthCheck(builds=False, tests_pass=True, build_output="error: boom"),
        features=FeatureInventory(
            incomplete=[FeatureInfo(name="thing", description="maybe partial")]
        ),
    )
    # Hard signal drives Must Fix; the guess is still quarantined under Advisory.
    assert "Must Fix" in spec and "error: boom" in spec
    assert "Advisory" in spec
    assert spec.index("Must Fix") < spec.index("Advisory")
