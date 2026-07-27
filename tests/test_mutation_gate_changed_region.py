"""T1.3 — the changed-region mutation gate scores ALL changed source files.

Previously the check scored only the LARGEST edited source file and returned, so a
multi-file fix whose crux is a small file was never mutation-tested (a weak suite
there went unsurfaced). This asserts every changed non-test source file with a
mutable region is scored, while test/whitespace files are still skipped.
"""

from types import SimpleNamespace

from misterdev.task_executors.markdown_plan_executor.edits_mixin import EditsMixin


class _Ex(EditsMixin):
    pass


def _project(tmp_path):
    return SimpleNamespace(
        path=tmp_path,
        config={
            "orchestrator": {"changed_region_mutation": True},
            "mutation": {},
            "test_command": "pytest",
        },
    )


def test_all_changed_source_files_are_scored(tmp_path, monkeypatch):
    (tmp_path / "big.py").write_text("def a():\n    return 1\n" * 5)
    (tmp_path / "small.py").write_text("def b():\n    return 2\n")
    pre_edit = {"big.py": "old big", "small.py": "old small"}

    scored = []

    def _fake_run(project_path, path, old, new, cmd, runner, min_score, max_mutants):
        scored.append(path)
        return SimpleNamespace(status="scored", reason="ok")

    monkeypatch.setattr(
        "misterdev.core.verification.changed_region_mutation.run_changed_region_mutation",
        _fake_run,
    )
    _Ex()._changed_region_mutation_check(
        _project(tmp_path), pre_edit, "pytest", tmp_path
    )
    assert set(scored) == {"big.py", "small.py"}


def test_test_files_are_not_scored(tmp_path, monkeypatch):
    (tmp_path / "src.py").write_text("def a():\n    return 1\n")
    (tmp_path / "test_src.py").write_text("def test_a():\n    assert True\n")
    pre_edit = {"src.py": "old", "test_src.py": "old"}

    scored = []
    monkeypatch.setattr(
        "misterdev.core.verification.changed_region_mutation.run_changed_region_mutation",
        lambda *a, **k: (
            scored.append(a[1]),
            SimpleNamespace(status="scored", reason="ok"),
        )[1],
    )
    _Ex()._changed_region_mutation_check(
        _project(tmp_path), pre_edit, "pytest", tmp_path
    )
    assert scored == ["src.py"]
