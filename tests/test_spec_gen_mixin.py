"""Unit tests for SpecGenMixin — pure spec synthesis logic."""

import pytest
from unittest.mock import MagicMock, patch

from misterdev.core.execution.spec_gen_mixin import SpecGenMixin
from misterdev.core.modes import BuildMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Orch(SpecGenMixin):
    pass


def _make_assessment(
    builds=True,
    tests_pass=True,
    build_output="",
    test_output="",
    lint_clean=True,
    lint_output="",
    broken=None,
    todos=None,
    missing=None,
    incomplete=None,
    stubs=None,
    existing=None,
    purpose="A test project",
    conventions="PEP8",
    languages=["python"],
    frameworks=[],
):
    health = MagicMock()
    health.builds = builds
    health.tests_pass = tests_pass
    health.build_output = build_output
    health.test_output = test_output
    health.lint_clean = lint_clean
    health.lint_output = lint_output

    features = MagicMock()
    features.broken = broken or []
    features.todos = todos or []
    features.missing = missing or []
    features.incomplete = incomplete or []
    features.stubs = stubs or []
    features.existing = existing or []

    context = MagicMock()
    context.purpose = purpose
    context.conventions = conventions

    structure = MagicMock()
    structure.languages = languages
    structure.frameworks = frameworks

    assessment = MagicMock()
    assessment.health = health
    assessment.features = features
    assessment.context = context
    assessment.structure = structure
    return assessment


def _make_project(name="myproject", path="/tmp/proj"):
    p = MagicMock()
    p.name = name
    p.path = MagicMock()
    p.path.__truediv__ = lambda self, other: MagicMock()
    return p


# ---------------------------------------------------------------------------
# _ground_completion_spec
# ---------------------------------------------------------------------------


def test_ground_completion_spec_failing_build():
    orch = _Orch()
    assessment = _make_assessment(builds=False, build_output="error: missing symbol")
    spec = orch._ground_completion_spec(assessment, _make_project())
    assert "FAILING" in spec
    assert "missing symbol" in spec


def test_ground_completion_spec_failing_tests():
    orch = _Orch()
    assessment = _make_assessment(tests_pass=False, test_output="AssertionError: 1!=2")
    spec = orch._ground_completion_spec(assessment, _make_project())
    assert "Tests are FAILING" in spec
    assert "AssertionError" in spec


def test_ground_completion_spec_missing_features():
    orch = _Orch()
    m = MagicMock()
    m.name = "auth"
    m.description = "OAuth2 login"
    assessment = _make_assessment(missing=[m])
    spec = orch._ground_completion_spec(assessment, _make_project())
    assert "auth" in spec
    assert "OAuth2 login" in spec
    assert "Should Add" in spec


def test_ground_completion_spec_no_objective():
    orch = _Orch()
    assessment = _make_assessment()  # all green, nothing to do
    spec = orch._ground_completion_spec(assessment, _make_project())
    assert "No concrete objective found" in spec
    assert "ZERO tasks" in spec


def test_ground_completion_spec_speculative_items_advisory_only():
    orch = _Orch()
    inc = MagicMock()
    inc.name = "maybe_stub"
    inc.description = "looks incomplete"
    assessment = _make_assessment(incomplete=[inc])
    spec = orch._ground_completion_spec(assessment, _make_project())
    assert "Advisory" in spec
    assert "maybe_stub" in spec
    # no hard failures → also has no-objective section
    assert "No concrete objective" in spec


def test_ground_completion_spec_todos_included():
    orch = _Orch()
    assessment = _make_assessment(
        todos=[{"file": "main.py", "line": 42, "text": "TODO: fix this"}]
    )
    spec = orch._ground_completion_spec(assessment, _make_project())
    assert "main.py" in spec
    assert "42" in spec


# ---------------------------------------------------------------------------
# _generate_spec
# ---------------------------------------------------------------------------


def test_generate_spec_debug_mode_broken_items():
    orch = _Orch()
    assessment = _make_assessment(
        broken=["parser crash"], builds=False, build_output="SyntaxError"
    )
    spec = orch._generate_spec(BuildMode.DEBUG, "", assessment, _make_project())
    assert "parser crash" in spec
    assert "Build Failure" in spec
    assert "SyntaxError" in spec


def test_generate_spec_debug_mode_stubs():
    orch = _Orch()
    assessment = _make_assessment(stubs=["stub_api.py"])
    spec = orch._generate_spec(BuildMode.DEBUG, "", assessment, _make_project())
    assert "stub_api.py" in spec


def test_generate_spec_complete_mode_delegates():
    orch = _Orch()
    assessment = _make_assessment(builds=False, build_output="boom")
    proj = _make_project()
    spec = orch._generate_spec(BuildMode.COMPLETE, "", assessment, proj)
    assert "Completion Spec" in spec
    assert "FAILING" in spec


def test_generate_spec_review_mode():
    orch = _Orch()
    assessment = _make_assessment(broken=["x"], stubs=["y"], todos=[{"t": "z"}] * 5)
    spec = orch._generate_spec(BuildMode.REVIEW, "", assessment, _make_project())
    assert "Review Spec" in spec
    assert "5 items" in spec


def test_generate_spec_create_mode_calls_llm():
    orch = _Orch()
    assessment = _make_assessment()
    proj = _make_project()
    proj.llm_client.generate_code.return_value = "generated spec"
    spec = orch._generate_spec(BuildMode.CREATE, "build a thing", assessment, proj)
    assert spec == "generated spec"
    proj.llm_client.generate_code.assert_called_once()
    call_prompt = proj.llm_client.generate_code.call_args[0][0]
    assert "build a thing" in call_prompt


def test_generate_spec_smart_mode_calls_llm_scoped():
    orch = _Orch()
    assessment = _make_assessment()
    proj = _make_project()
    proj.llm_client.generate_code.return_value = "scoped spec"
    spec = orch._generate_spec(BuildMode.SMART, "add login", assessment, proj)
    assert spec == "scoped spec"
    call_prompt = proj.llm_client.generate_code.call_args[0][0]
    assert "add login" in call_prompt
    assert "Do NOT expand scope" in call_prompt


def test_generate_spec_auto_fallback():
    orch = _Orch()
    assessment = _make_assessment()
    # Use a mode value that doesn't match any branch (simulate unknown)
    spec = orch._generate_spec(BuildMode.AUTO, "my prompt", assessment, _make_project())
    assert "my prompt" in spec
