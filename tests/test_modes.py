import tempfile
from pathlib import Path

from my_project_orchestrator.core.modes import (
    BuildMode,
    parse_flags,
    resolve_mode,
)


def test_parse_flags_empty():
    remaining, flags = parse_flags([])
    assert remaining == []
    assert flags.budget == 100.0
    assert not flags.commit
    assert not flags.dry_run


def test_parse_flags_all():
    args = [
        "debug",
        "--budget",
        "50",
        "--commit",
        "--no-verify",
        "--dry-run",
        "--interactive",
        "--parallel",
        "--focus",
        "src/",
    ]
    remaining, flags = parse_flags(args)
    assert remaining == ["debug"]
    assert flags.budget == 50.0
    assert flags.commit
    assert flags.no_verify
    assert flags.dry_run
    assert flags.parallel
    assert flags.interactive
    assert flags.focus == "src/"


def test_parse_flags_max_tasks():
    remaining, flags = parse_flags(["complete", "--max-tasks", "3"])
    assert remaining == ["complete"]
    assert flags.max_tasks == 3


def test_parse_flags_max_tasks_default_none():
    _, flags = parse_flags(["complete"])
    assert flags.max_tasks is None


def test_parse_flags_max_tasks_invalid_is_none():
    _, flags = parse_flags(["--max-tasks", "notanumber"])
    assert flags.max_tasks is None


def test_resolve_mode_keywords():
    p = Path(".")
    assert resolve_mode("debug", p) == BuildMode.DEBUG
    assert resolve_mode("complete", p) == BuildMode.COMPLETE
    assert resolve_mode("review", p) == BuildMode.REVIEW
    assert resolve_mode("new my app", p) == BuildMode.CREATE
    assert resolve_mode("add auth", p) == BuildMode.SMART


def test_resolve_mode_empty_with_code():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "main.py").write_text("x = 1")
        assert resolve_mode("", td) == BuildMode.COMPLETE


def test_resolve_mode_empty_no_code():
    with tempfile.TemporaryDirectory() as td:
        assert resolve_mode("", Path(td)) == BuildMode.CREATE


def test_resolve_mode_spec_file():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "spec.md").write_text("# Spec")
        assert resolve_mode("spec.md", td) == BuildMode.SPEC
