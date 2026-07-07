"""Offline end-to-end validation of the polyglot harness.

A synthetic Python exercise (a stub that fails its test until fixed) is run
through the real grader with real pytest, with a stub orchestrator that applies
the fix — so the grader/runner/resolved logic is verified with no network, no
benchmark download, and no model cost.
"""

import json
import subprocess
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


def test_discover_exercises_limit_zero_and_negative(tmp_path):
    # limit=0 means zero exercises (not "all"); a negative limit is clamped to
    # empty rather than silently dropping the last exercise via found[:-1].
    from evaluation.polyglot.harness import discover_exercises

    practice = tmp_path / "python" / "exercises" / "practice"
    for name in ("a", "b", "c"):
        (practice / name).mkdir(parents=True)
    assert len(discover_exercises(str(tmp_path), ["python"])) == 3
    assert discover_exercises(str(tmp_path), ["python"], limit=0) == []
    assert discover_exercises(str(tmp_path), ["python"], limit=-1) == []
    assert len(discover_exercises(str(tmp_path), ["python"], limit=2)) == 2


def test_project_yaml_escapes_scalars_and_disables_reflection(tmp_path):
    # A name/model with a quote must not emit invalid YAML, and a pinned-model
    # ($0) run must turn the reflection loop off.
    import yaml
    from evaluation.polyglot.runner import _write_project_yaml

    inst = PolyglotInstance(
        name='weird"name',
        language="python",
        instructions="",
        solution_files=[],
        test_files=[],
    )
    _write_project_yaml(tmp_path, inst, model="x/y:free")
    cfg = yaml.safe_load((tmp_path / "project.yaml").read_text())
    assert cfg["name"] == 'weird"name'
    assert cfg["llm"]["model"] == "x/y:free"
    assert cfg["orchestrator"]["reflection"] is False


def test_sanitize_goal_drops_build_flag_tokens():
    # A flag lookalike in the free-text goal must not survive to be parsed as a
    # build flag (which would also swallow the following token).
    from evaluation.polyglot.runner import _sanitize_goal

    assert (
        _sanitize_goal("Implement add --budget 99 and return the sum")
        == "Implement add 99 and return the sum"
    )
    assert _sanitize_goal("Implement the adder") == "Implement the adder"


def test_sanitize_goal_preserves_multiline_structure():
    # Removing flag tokens must not flatten a multi-line prompt: newlines, blank
    # lines, and code indentation are all preserved verbatim.
    from evaluation.polyglot.runner import _sanitize_goal

    goal = "Line one.\n\n- bullet a\n- bullet b\n\n    code block"
    assert _sanitize_goal(goal) == goal
    # A flag mid-line still collapses its gap without disturbing the newline.
    assert _sanitize_goal("do it --commit\nnext line") == "do it\nnext line"


def test_stage_base_files_stages_only_known_files(tmp_path):
    # The base commit stages the exercise's own files explicitly (never add -A),
    # so a stray untracked file in the work dir is not swept into the snapshot.
    from evaluation.polyglot.runner import _git, _stage_base_files

    inst = PolyglotInstance(
        name="adder",
        language="python",
        instructions="",
        solution_files=["adder.py"],
        test_files=["adder_test.py"],
    )
    (tmp_path / "adder.py").write_text("x = 1\n")
    (tmp_path / "adder_test.py").write_text("y = 2\n")
    (tmp_path / "project.yaml").write_text("name: adder\n")
    (tmp_path / "stray.txt").write_text("do not stage me\n")
    _git(tmp_path, "init -q")
    _stage_base_files(tmp_path, inst)
    staged = subprocess.run(
        "git diff --cached --name-only",
        shell=True,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    ).stdout.split()
    assert set(staged) == {"adder.py", "adder_test.py", "project.yaml"}
    assert "stray.txt" not in staged
