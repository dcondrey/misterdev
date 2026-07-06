"""Offline end-to-end validation of the SWE-bench harness.

A synthetic instance (a one-function repo with a real bug and a test_patch that
fails until it is fixed) is run through the real grader with real pytest, with an
injected repo setup and a stub orchestrator — so the grader/runner/resolved logic
is verified without network, a dataset download, or any model cost.
"""

import subprocess
from pathlib import Path

from evaluation.swebench import SWEBenchInstance
from evaluation.swebench.instance import _as_list
from evaluation.swebench.runner import run_instance

# Adds a root-level test that asserts add(2, 3) == 5 (fails until the bug is fixed).
_TEST_PATCH = """diff --git a/test_calc.py b/test_calc.py
new file mode 100644
--- /dev/null
+++ b/test_calc.py
@@ -0,0 +1,3 @@
+from calc import add
+def test_add():
+    assert add(2, 3) == 5
"""


def _instance() -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id="synthetic-add",
        repo="local/calc",
        base_commit="",
        problem_statement="add() returns a - b instead of a + b; fix it.",
        test_patch=_TEST_PATCH,
        fail_to_pass=["test_calc.py::test_add"],
        pass_to_pass=[],
        test_command="python -m pytest -rA -p no:cacheprovider",
    )


def _prepare_buggy_repo(instance, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    for cmd in (
        "git init -q",
        "git config user.email t@t.t",
        "git config user.name t",
    ):
        subprocess.run(cmd, shell=True, cwd=str(dest), check=True)
    return dest


class _FixOrchestrator:
    """Stub misterdev: applies and commits the correct fix, like a real build."""

    def build(self, path, args):
        repo = Path(path)
        (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        subprocess.run("git add -A && git commit -q -m fix", shell=True, cwd=str(repo))


class _NoOpOrchestrator:
    """Stub misterdev that changes nothing — the bug remains."""

    def build(self, path, args):
        return None


def test_resolves_a_fixed_instance(tmp_path):
    res = run_instance(
        _instance(),
        str(tmp_path),
        orchestrator=_FixOrchestrator(),
        prepare_repo=_prepare_buggy_repo,
        build_args="",
    )
    assert res.resolved is True
    assert res.grade.fail_to_pass["test_calc.py::test_add"] is True
    assert "calc.py" in res.patch


def test_unfixed_instance_is_unresolved(tmp_path):
    res = run_instance(
        _instance(),
        str(tmp_path),
        orchestrator=_NoOpOrchestrator(),
        prepare_repo=_prepare_buggy_repo,
        build_args="",
    )
    assert res.resolved is False


def test_test_patch_that_does_not_apply_is_reported(tmp_path):
    inst = _instance()
    inst.test_patch = "diff --git a/nope\n@@ garbage @@\n"
    res = run_instance(
        inst,
        str(tmp_path),
        orchestrator=_FixOrchestrator(),
        prepare_repo=_prepare_buggy_repo,
        build_args="",
    )
    assert res.resolved is False
    assert "test_patch" in res.error


def test_as_list_accepts_json_string_and_list():
    assert _as_list('["a::b", "c::d"]') == ["a::b", "c::d"]
    assert _as_list(["x"]) == ["x"]
    assert _as_list(None) == []
