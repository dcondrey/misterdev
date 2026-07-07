import tempfile
from pathlib import Path

import pytest

from misterdev.core.evolution.adapters import (
    BenchResult,
    apply_patch_to_worktree,
    baseline_passed,
    results_from_report,
    score_of,
)
from misterdev.core.evolution.guardrail import ProtectedPathError

_PATCH = (
    "```python:pkg/mod.py\n"
    "<<<<<<< SEARCH\n"
    "def f():\n    return 1\n"
    "=======\n"
    "def f():\n    return 2\n"
    ">>>>>>> REPLACE\n"
    "```\n"
)


def test_apply_patch_writes_the_edit_under_root():
    root = Path(tempfile.mkdtemp())
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    written = apply_patch_to_worktree(str(root), _PATCH)
    assert written == ["pkg/mod.py"]
    assert "return 2" in (root / "pkg" / "mod.py").read_text()


def test_apply_patch_refuses_protected_target():
    root = Path(tempfile.mkdtemp())
    protected = _PATCH.replace("pkg/mod.py", "tests/test_x.py")
    with pytest.raises(ProtectedPathError):
        apply_patch_to_worktree(str(root), protected)


def test_apply_patch_refuses_traversal():
    root = Path(tempfile.mkdtemp())
    escaping = _PATCH.replace("pkg/mod.py", "../outside.py")
    with pytest.raises(ProtectedPathError):
        apply_patch_to_worktree(str(root), escaping)


def test_apply_patch_empty_raises():
    with pytest.raises(ValueError):
        apply_patch_to_worktree(tempfile.mkdtemp(), "no blocks here")


def test_results_from_report_reconstructs_records():
    report = {
        "total": 2,
        "resolved": 1,
        "results": [
            {"name": "a", "language": "rust", "resolved": True, "error": ""},
            {"name": "b", "language": "go", "resolved": False, "error": "boom"},
        ],
    }
    results = results_from_report(report)
    assert [(r.name, r.language, r.resolved) for r in results] == [
        ("a", "rust", True),
        ("b", "go", False),
    ]


def test_score_and_baseline_over_results():
    results = [
        BenchResult("a", "rust", True),
        BenchResult("b", "rust", False),
        BenchResult("c", "go", True),
    ]
    score = score_of(results)
    assert (score.resolved, score.total) == (2, 3)
    assert baseline_passed(results) == {"a", "c"}
