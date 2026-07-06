"""Offline end-to-end validation of the polyglot harness.

A synthetic Python exercise (a stub that fails its test until fixed) is run
through the real grader with real pytest, with a stub orchestrator that applies
the fix — so the grader/runner/resolved logic is verified with no network, no
benchmark download, and no model cost.
"""

import json
from pathlib import Path

from evaluation.polyglot import PolyglotInstance
from evaluation.polyglot.instance import load_local_exercise
from evaluation.polyglot.runner import run_instance


def _instance() -> PolyglotInstance:
    return PolyglotInstance(
        name="adder",
        language="python",
        instructions="Implement add(a, b) to return a + b.",
        solution_files=["adder.py"],
        test_files=["adder_test.py"],
    )


def _prepare_stub(instance, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "adder.py").write_text("def add(a, b):\n    return 0\n")
    (dest / "adder_test.py").write_text(
        "from adder import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    return dest


class _FixOrchestrator:
    def build(self, path, args):
        (Path(path) / "adder.py").write_text("def add(a, b):\n    return a + b\n")


class _NoOpOrchestrator:
    def build(self, path, args):
        return None


def test_resolves_a_fixed_exercise(tmp_path):
    res = run_instance(
        _instance(), str(tmp_path), _prepare_stub, orchestrator=_FixOrchestrator()
    )
    assert res.resolved is True
    assert res.language == "python"


def test_unfixed_exercise_is_unresolved(tmp_path):
    res = run_instance(
        _instance(), str(tmp_path), _prepare_stub, orchestrator=_NoOpOrchestrator()
    )
    assert res.resolved is False


def test_default_test_command_by_language():
    assert PolyglotInstance("x", "rust", "", [], []).test_command == "cargo test"
    assert PolyglotInstance("x", "go", "", [], []).test_command == "go test ./..."
    assert "pytest" in PolyglotInstance("x", "python", "", [], []).test_command


def test_load_local_exercise_reads_meta_and_docs(tmp_path):
    ex = tmp_path / "affine-cipher"
    (ex / ".meta").mkdir(parents=True)
    (ex / ".docs").mkdir(parents=True)
    (ex / ".meta" / "config.json").write_text(
        json.dumps({"files": {"solution": ["affine.py"], "test": ["affine_test.py"]}})
    )
    (ex / ".docs" / "instructions.md").write_text("Do the affine cipher.")
    inst = load_local_exercise(str(ex), "python")
    assert inst.solution_files == ["affine.py"]
    assert inst.test_files == ["affine_test.py"]
    assert "affine cipher" in inst.instructions
