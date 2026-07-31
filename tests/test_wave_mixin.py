"""Unit tests for WaveMixin — wave helper utilities."""

import pytest
from unittest.mock import MagicMock, patch

from misterdev.core.execution.wave_mixin import WaveMixin
from misterdev.core.planning.assessment import HealthCheck


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Orch(WaveMixin):
    pass


def _make_health(
    builds=True,
    tests_pass=True,
    lint_clean=True,
    build_output="",
    test_output="",
    lint_output="",
):
    h = MagicMock(spec=HealthCheck)
    h.builds = builds
    h.tests_pass = tests_pass
    h.lint_clean = lint_clean
    h.build_output = build_output
    h.test_output = test_output
    h.lint_output = lint_output
    return h


def _make_report(failed=None, deferred=None):
    r = MagicMock()
    r.failed_tasks = failed or []
    r.deferred_tasks = deferred or []
    return r


def _make_task(id_, title):
    t = MagicMock()
    t.id = id_
    t.title = title
    return t


# ---------------------------------------------------------------------------
# _build_fix_spec
# ---------------------------------------------------------------------------


def test_build_fix_spec_includes_issues():
    orch = _Orch()
    health = _make_health(builds=True, tests_pass=True, lint_clean=True)
    report = _make_report()
    spec = orch._build_fix_spec(
        report, ["lint: unused import", "test: assert failed"], health
    )
    assert "lint: unused import" in spec
    assert "test: assert failed" in spec
    assert "Gate Failures" in spec


def test_build_fix_spec_includes_build_output():
    orch = _Orch()
    health = _make_health(builds=False, build_output="error: undefined symbol 'foo'")
    report = _make_report()
    spec = orch._build_fix_spec(report, [], health)
    assert "Build Output" in spec
    assert "undefined symbol" in spec


def test_build_fix_spec_includes_test_output():
    orch = _Orch()
    health = _make_health(
        tests_pass=False, test_output="FAILED test_login: AssertionError"
    )
    report = _make_report()
    spec = orch._build_fix_spec(report, [], health)
    assert "Test Output" in spec
    assert "FAILED test_login" in spec


def test_build_fix_spec_includes_lint_output():
    orch = _Orch()
    health = _make_health(lint_clean=False, lint_output="E501 line too long")
    report = _make_report()
    spec = orch._build_fix_spec(report, [], health)
    assert "Lint Output" in spec
    assert "E501" in spec


def test_build_fix_spec_includes_failed_tasks():
    orch = _Orch()
    health = _make_health()
    report = _make_report(failed=[_make_task("t1", "Fix parser")])
    spec = orch._build_fix_spec(report, [], health)
    assert "Failed Tasks" in spec
    assert "t1" in spec
    assert "Fix parser" in spec


def test_build_fix_spec_includes_deferred_tasks():
    orch = _Orch()
    health = _make_health()
    report = _make_report(deferred=[_make_task("t2", "Add auth")])
    spec = orch._build_fix_spec(report, [], health)
    assert "Deferred Tasks" in spec
    assert "t2" in spec


def test_build_fix_spec_all_green_no_noise():
    orch = _Orch()
    health = _make_health()
    report = _make_report()
    spec = orch._build_fix_spec(report, [], health)
    assert "Convergence Fix Spec" in spec
    assert "Build Output" not in spec
    assert "Test Output" not in spec


# ---------------------------------------------------------------------------
# _wave_infra_count
# ---------------------------------------------------------------------------


def test_wave_infra_count_no_infra_failures():
    with patch("misterdev.core.execution.infra.infra_failure", return_value=False):
        result = WaveMixin._wave_infra_count(
            [
                (
                    MagicMock(),
                    MagicMock(status="failed", logs="", message="normal fail"),
                    None,
                )
            ]
        )
    assert result == 0


def test_wave_infra_count_skips_completed():
    completed = MagicMock(status="completed")
    result = WaveMixin._wave_infra_count([(MagicMock(), completed, None)])
    assert result == 0


def test_wave_infra_count_counts_infra_errors():
    task_result = MagicMock(status="failed", logs="connection timeout", message="")
    with patch("misterdev.core.execution.infra.infra_failure", return_value=True):
        result = WaveMixin._wave_infra_count([(MagicMock(), task_result, None)])
    assert result == 1


def test_wave_infra_count_includes_exception_text():
    task_result = MagicMock(status="failed", logs="", message="")
    with patch(
        "misterdev.core.execution.infra.infra_failure", side_effect=lambda t: "OOM" in t
    ):
        result = WaveMixin._wave_infra_count(
            [(MagicMock(), task_result, RuntimeError("OOM killer"))]
        )
    assert result == 1


def test_wave_infra_count_mixed_results():
    completed = MagicMock(status="completed")
    infra_fail = MagicMock(status="failed", logs="disk full", message="")
    normal_fail = MagicMock(status="failed", logs="assertion error", message="")
    with patch(
        "misterdev.core.execution.infra.infra_failure",
        side_effect=lambda t: "disk full" in t,
    ):
        result = WaveMixin._wave_infra_count(
            [
                (MagicMock(), completed, None),
                (MagicMock(), infra_fail, None),
                (MagicMock(), normal_fail, None),
            ]
        )
    assert result == 1


# ---------------------------------------------------------------------------
# _apply_wave_tuning
# ---------------------------------------------------------------------------


def _make_tuning(max_workers=4, timeout_factor=1.5):
    t = MagicMock()
    t.max_workers = max_workers
    t.timeout_factor = timeout_factor
    return t


def _make_project_config(orch_cfg=None, build_cfg=None):
    p = MagicMock()
    p.config = {}
    if orch_cfg:
        p.config["orchestrator"] = dict(orch_cfg)
    if build_cfg:
        p.config["build"] = dict(build_cfg)
    return p


def test_apply_wave_tuning_sets_max_workers():
    orch = _Orch()
    proj = _make_project_config()
    orch._apply_wave_tuning(
        proj,
        _make_tuning(max_workers=6, timeout_factor=1.0),
        {"setup": 60, "build": 120, "test": 90},
    )
    assert proj.config["orchestrator"]["max_workers"] == 6


def test_apply_wave_tuning_scales_timeouts():
    orch = _Orch()
    proj = _make_project_config()
    base = {"setup": 60, "build": 120, "test": 90}
    orch._apply_wave_tuning(proj, _make_tuning(max_workers=4, timeout_factor=2.0), base)
    assert proj.config["orchestrator"]["worktree_setup_timeout"] == 120
    assert proj.config["build"]["build_timeout"] == 240
    assert proj.config["build"]["test_timeout"] == 180


def test_apply_wave_tuning_from_base_not_previous():
    """Repeated application must not drift — always scale from base."""
    orch = _Orch()
    proj = _make_project_config()
    base = {"setup": 60, "build": 120, "test": 90}
    tuning = _make_tuning(max_workers=4, timeout_factor=2.0)
    orch._apply_wave_tuning(proj, tuning, base)
    orch._apply_wave_tuning(proj, tuning, base)
    assert proj.config["build"]["build_timeout"] == 240  # not 480
