import tempfile
from pathlib import Path

from my_project_orchestrator.core.preflight import PreflightValidator, PreflightIssue
from my_project_orchestrator.core.models import Task


def _task(tid, **kw):
    return Task(id=tid, description=kw.pop("description", "d"), project_ref="p", **kw)


def test_preflight_flags_missing_context_dep_binary_and_title():
    with tempfile.TemporaryDirectory() as td:
        tasks = [
            _task(
                "T-001",
                title="",  # missing title -> warning
                context_files=["nope.py"],  # missing context -> warning
                dependencies=["GHOST"],  # dangling dep -> error
                processor_data={"test_command": "definitely-not-a-binary-xyz -q"},
            ),
        ]
        issues = PreflightValidator().validate(tasks, Path(td))
        msgs = [str(i) for i in issues]
        assert any("Context file" in m for m in msgs)
        assert any("Dependency" in m and "GHOST" in m for m in msgs)
        assert any("binary" in m for m in msgs)
        assert any("no title" in m for m in msgs)
        assert PreflightValidator.has_errors(issues)  # the dangling dep is an error


def test_preflight_warns_on_conflicting_independent_modifiers():
    with tempfile.TemporaryDirectory() as td:
        tasks = [
            _task("T-001", title="a", files_to_modify=["src/x.py"]),
            _task("T-002", title="b", files_to_modify=["src/x.py"]),
        ]
        issues = PreflightValidator().validate(tasks, Path(td))
        assert any("may conflict" in str(i) for i in issues)


def test_preflight_clean_plan_has_no_issues():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "ctx.py").write_text("x = 1\n")
        tasks = [
            _task("T-001", title="a", context_files=["ctx.py"]),
            _task("T-002", title="b", dependencies=["T-001"]),
        ]
        issues = PreflightValidator().validate(tasks, td)
        assert issues == []
        assert not PreflightValidator.has_errors(issues)


def test_preflight_issue_repr():
    assert "ERROR" in repr(PreflightIssue("T-1", "error", "boom"))
