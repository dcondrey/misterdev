"""misterdev doctor preflight: pure check routing, aggregation, and exit codes."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from misterdev.core.execution import doctor as d


def test_clean_tree_routing():
    assert d.check_clean_tree("").status == d.PASS
    assert d.check_clean_tree("3 files, e.g. env.ts").status == d.FAIL


def test_on_base_branch_routing():
    assert d.check_on_base_branch("main", "main").status == d.PASS
    assert d.check_on_base_branch("feature", "main").status == d.WARN
    assert d.check_on_base_branch("task/T-1-abc123", "main").status == d.FAIL
    assert d.check_on_base_branch("doctor/abc123", "main").status == d.FAIL
    assert d.check_on_base_branch(None, "main").status == d.FAIL


def test_models_routing():
    assert d.check_models(True, "good/model").status == d.PASS
    assert d.check_models(False, "404 no endpoints").status == d.FAIL


def test_worktree_prime_routing():
    assert d.check_worktree_prime(None, None).status == d.PASS  # nothing to prime
    assert d.check_worktree_prime(True, True).status == d.PASS
    assert d.check_worktree_prime(True, False).status == d.WARN  # toolchain unresolved
    assert d.check_worktree_prime(False, None).status == d.WARN  # prime failed


def test_git_and_branch_and_worktree_and_requirements_warnings():
    assert d.check_git_repo(True).status == d.PASS
    assert d.check_git_repo(False).status == d.WARN
    assert d.check_leftover_task_branches([]).status == d.PASS
    assert d.check_leftover_task_branches(["task/a", "task/b"]).status == d.WARN
    assert d.check_dangling_worktrees([]).status == d.PASS
    assert d.check_dangling_worktrees(["/x/.orchestrator/worktrees/a"]).status == d.WARN
    assert d.check_requirements([]).status == d.PASS
    assert d.check_requirements(["API_KEY"]).status == d.WARN


def test_aggregate_all_pass_exits_zero():
    checks = [
        d.check_clean_tree(""),
        d.check_models(True, "m"),
        d.check_requirements([]),
    ]
    agg = d.aggregate(checks)
    assert agg["exit_code"] == 0
    assert agg["ready"] is True
    assert agg["passed"] == 3 and agg["warnings"] == 0 and agg["failures"] == 0


def test_aggregate_warnings_only_still_exits_zero():
    """Warnings inform but never block an unattended run."""
    checks = [
        d.check_git_repo(False),
        d.check_leftover_task_branches(["task/a"]),
        d.check_requirements(["X"]),
    ]
    agg = d.aggregate(checks)
    assert agg["warnings"] == 3
    assert agg["failures"] == 0
    assert agg["exit_code"] == 0
    assert agg["ready"] is True


def test_aggregate_any_hard_failure_exits_nonzero():
    checks = [
        d.check_clean_tree(""),  # pass
        d.check_leftover_task_branches(["task/a"]),  # warn
        d.check_on_base_branch(None, "main"),  # fail
    ]
    agg = d.aggregate(checks)
    assert agg["failures"] == 1
    assert agg["exit_code"] == 1
    assert agg["ready"] is False


# --- run_doctor integration on a real temp git repo -------------------------
def _git(root, *args):
    import subprocess

    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _clean_repo(root: Path):
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    (root / "f.txt").write_text("x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")


def _doctor_on(root: Path, health=(True, "good/model")):
    import misterdev.agent as agent_mod

    orch = agent_mod.ProjectOrchestrator()
    project = MagicMock()
    project.path = root
    project.config = {}  # non-node -> worktree probe short-circuits to pass
    project.llm_client.health_check.return_value = health
    orch._get_or_register = lambda _p: project  # type: ignore[assignment]
    return orch.run_doctor(str(root))


def test_run_doctor_ready_on_clean_repo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _clean_repo(root)
        result = _doctor_on(root)
        assert result["exit_code"] == 0
        assert result["ready"] is True
        names = {c.name: c.status for c in result["checks"]}
        assert names["clean working tree"] == "pass"
        assert names["on base branch"] == "pass"
        assert names["models resolve"] == "pass"


def test_run_doctor_fails_on_dirty_tree():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _clean_repo(root)
        (root / "f.txt").write_text("changed\n")  # dirty
        result = _doctor_on(root)
        assert result["exit_code"] == 1
        assert result["ready"] is False
        names = {c.name: c.status for c in result["checks"]}
        assert names["clean working tree"] == "fail"


def test_run_doctor_fails_when_model_unresolvable():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _clean_repo(root)
        result = _doctor_on(root, health=(False, "404 no endpoints"))
        assert result["exit_code"] == 1
        names = {c.name: c.status for c in result["checks"]}
        assert names["models resolve"] == "fail"
