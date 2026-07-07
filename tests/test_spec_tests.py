"""Tests for the spec-as-tests helper and its per-task wiring.

The generation + write + path logic is unit-tested with a monkeypatched
generator (no network). Further tests assert the feature is now WIRED into the
executor: a generated test is written under ``.orchestrator/spec_tests/`` (not
the project suite, so the integration-gate baseline is unaffected) and run scoped
after a task's gates pass — and that the off path leaves the loop unchanged.
"""

from pathlib import Path

from misterdev.core.models import Task
from misterdev.core.verification.spec_tests import (
    extract_code,
    generate_spec_test,
    spec_test_path,
    write_spec_test,
)


def _task(criteria="GET /version returns the version", tid="t1"):
    t = Task(id=tid, description="add /version", project_ref="p")
    t.acceptance_criteria = criteria
    return t


# --- generation -------------------------------------------------------------


def test_generate_returns_none_without_criteria():
    t = _task(criteria="")
    assert generate_spec_test(t, generator=lambda p: "```\ncode\n```") is None


def test_generate_returns_none_without_generator_or_client():
    assert generate_spec_test(_task(), llm_client=None) is None


def test_generate_extracts_fenced_code():
    src = generate_spec_test(
        _task(),
        generator=lambda p: "Here:\n```python\ndef test_x():\n    assert False\n```\n",
    )
    assert src == "def test_x():\n    assert False"


def test_generate_prompt_contains_criteria_and_language():
    seen = {}

    def gen(prompt):
        seen["prompt"] = prompt
        return "```\nok\n```"

    generate_spec_test(
        _task(criteria="must reject empty input"),
        generator=gen,
        language="rust",
    )
    assert "must reject empty input" in seen["prompt"]
    assert "rust" in seen["prompt"]


def test_generate_error_is_none_not_crash():
    def boom(prompt):
        raise RuntimeError("model down")

    assert generate_spec_test(_task(), generator=boom) is None


def test_generate_timeout_is_none():
    import time

    def slow(prompt):
        time.sleep(3600)
        return "```\nx\n```"

    start = time.monotonic()
    res = generate_spec_test(_task(), generator=slow, timeout=0.3)
    assert time.monotonic() - start < 10
    assert res is None


def test_default_generator_uses_client_generate_code():
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def generate_code(self, prompt, system=""):
            self.calls += 1
            return "```python\ndef test_q():\n    assert False\n```"

    client = FakeClient()
    src = generate_spec_test(_task(), llm_client=client)
    assert client.calls == 1
    assert "def test_q" in src


def test_client_without_generate_code_is_none():
    class Bare:
        pass

    assert generate_spec_test(_task(), llm_client=Bare()) is None


# --- code extraction --------------------------------------------------------


def test_extract_code_fenced():
    assert extract_code("```python\nbody\n```") == "body"


def test_extract_code_bare_fallback():
    assert extract_code("def test_x(): pass") == "def test_x(): pass"


def test_extract_code_empty_is_none():
    assert extract_code("") is None
    assert extract_code("   ") is None


# --- path / write -----------------------------------------------------------


def test_spec_test_path_python(tmp_path):
    p = spec_test_path(tmp_path, _task(tid="abc-1"), language="python")
    assert p == tmp_path / "tests" / "spec_abc_1.py"


def test_spec_test_path_rust(tmp_path):
    p = spec_test_path(tmp_path, _task(tid="t2"), language="rust")
    assert p == tmp_path / "tests" / "spec_t2.rs"


def test_write_spec_test_creates_dir_and_file(tmp_path):
    path = write_spec_test(
        tmp_path, _task(tid="t3"), "def test_x():\n    assert False\n"
    )
    assert path.exists()
    assert "assert False" in path.read_text()
    assert path == tmp_path / "tests" / "spec_t3.py"


# --- wiring into the executor (per-task) ------------------------------------


def test_spec_as_tests_flags_readable_and_off_by_default():
    from misterdev.config import get_setting, DEFAULT_CONFIG

    assert DEFAULT_CONFIG["orchestrator"]["spec_as_tests"] is False
    assert DEFAULT_CONFIG["orchestrator"]["spec_as_tests_block"] is False
    assert (
        get_setting(
            {"orchestrator": {"spec_as_tests": True}}, "orchestrator", "spec_as_tests"
        )
        is True
    )


def test_pipeline_no_longer_carries_deferral_notice():
    import inspect
    from misterdev.agent import ProjectOrchestrator

    src = inspect.getsource(ProjectOrchestrator._run_pipeline)
    assert "DEFERRED" not in src


def test_generator_is_wired_into_the_executor():
    import inspect
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    src = inspect.getsource(MarkdownPlanExecutor._maybe_generate_spec_test)
    assert "generate_spec_test" in src


class _SpecClient:
    def generate_code(self, prompt, system=""):
        return "```python\ndef test_spec():\n    assert False\n```"


def test_maybe_generate_writes_outside_project_suite(tmp_path):
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _Proj:
        path = tmp_path
        config = {"orchestrator": {"spec_as_tests": True}, "language": "python"}
        llm_client = _SpecClient()

    path, source = MarkdownPlanExecutor()._maybe_generate_spec_test(
        _Proj(), _task(tid="t9")
    )
    assert path is not None
    # The source is returned too, so it can be injected as the reproduction target.
    assert source and source.strip()
    p = Path(path)
    assert p.exists()
    # Lives under .orchestrator/spec_tests/ — NOT the project's tests/ dir, so it
    # can never be collected by the project suite or flip the integration gate.
    assert ".orchestrator" in p.parts and "spec_tests" in p.parts


def test_maybe_generate_off_by_default(tmp_path):
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _Proj:
        path = tmp_path
        config = {}
        llm_client = _SpecClient()

    assert MarkdownPlanExecutor()._maybe_generate_spec_test(
        _Proj(), _task(tid="t1")
    ) == (None, None)


def test_maybe_generate_uses_language_extension_for_compiled_langs(tmp_path):
    # A compiled-language spec test gets its real suffix (from the shared
    # _LANG_EXT map), not a .txt stub that no runner can execute.
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _RustClient:
        def generate_code(self, prompt, system=""):
            return "```rust\n#[test]\nfn spec() { assert!(false); }\n```"

    class _Proj:
        path = tmp_path
        config = {"orchestrator": {"spec_as_tests": True}, "language": "rust"}
        llm_client = _RustClient()

    path, source = MarkdownPlanExecutor()._maybe_generate_spec_test(
        _Proj(), _task(tid="tr")
    )
    assert path is not None and path.endswith(".rs")


def test_spec_test_source_is_injected_as_edit_target(tmp_path, monkeypatch):
    # Reproduction-first: the generated spec test's SOURCE must appear in the edit
    # prompt as the concrete target, not just be checked after the fact.
    import json
    from misterdev.config import DEFAULT_CONFIG
    from misterdev.core.execution.project import Project
    from misterdev.llm.client import LLMResponse, LLMUsage
    from misterdev.task_executors.markdown_plan_executor import MarkdownPlanExecutor
    from tests.test_llm_client import FakeLLMClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    (tmp_path / "mod.py").write_text("def answer():\n    return 0\n")
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update({"name": "s", "build_command": "true", "language": "python"})
    cfg.pop("test_command", None)
    cfg["orchestrator"].update({"spec_as_tests": True, "max_task_attempts": 1})
    project = Project(tmp_path, cfg)

    spec_src = (
        "def test_answer():\n    from mod import answer\n    assert answer() == 42"
    )

    class _Cap(FakeLLMClient):
        def __init__(self):
            super().__init__(responses=[])
            self.edit_prompts = []

        def generate_code(self, prompt, system=""):
            return f"```python\n{spec_src}\n```"

        def _call(self, prompt, system_prompt):
            self.edit_prompts.append(prompt)
            return LLMResponse(
                content="```python:mod.py\ndef answer():\n    return 42\n```\n",
                usage=LLMUsage(),
            )

    cap = _Cap()
    project.llm_client = cap
    task = Task(
        id="T-1",
        description="make answer() return 42",
        project_ref=str(tmp_path),
    )
    task.acceptance_criteria = "answer() returns 42"
    task.files_to_modify = ["mod.py"]
    task.processor_data["strategy"] = "surgical"
    MarkdownPlanExecutor().execute(task, project, use_git_branch=False)

    joined = "\n".join(cap.edit_prompts)
    assert "Reproduction test" in joined  # the injected header
    assert "answer() == 42" in joined  # the actual generated test body


def test_run_spec_test_skip_without_path(tmp_path):
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _Proj:
        path = tmp_path
        config = {}
        env_manager = None

    assert MarkdownPlanExecutor()._run_spec_test(_Proj(), None, 30)[0] == "skip"


def test_run_spec_test_red_then_green(tmp_path):
    # End-to-end: a failing spec file -> red, a passing one -> green, via real
    # scoped pytest. Proves the red->green TDD signal actually works.
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    spec_dir = tmp_path / ".orchestrator" / "spec_tests"
    spec_dir.mkdir(parents=True)
    failing = spec_dir / "spec_fail.py"
    failing.write_text("def test_x():\n    assert False\n", encoding="utf-8")
    passing = spec_dir / "spec_pass.py"
    passing.write_text("def test_x():\n    assert True\n", encoding="utf-8")

    class _Proj:
        path = tmp_path
        config = {"test_command": "pytest -q"}
        env_manager = None

    e = MarkdownPlanExecutor()
    assert e._run_spec_test(_Proj(), str(failing), 60)[0] == "red"
    assert e._run_spec_test(_Proj(), str(passing), 60)[0] == "green"


def test_run_spec_test_skip_for_non_python_suite(tmp_path):
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    class _Proj:
        path = tmp_path
        config = {"test_command": "cargo test"}
        env_manager = None

    # A .rs spec can't be run as a standalone file -> skip, never a false red.
    status, _ = MarkdownPlanExecutor()._run_spec_test(
        _Proj(), str(tmp_path / "spec.rs"), 30
    )
    assert status == "skip"


def test_run_spec_test_node_test_red_then_green(tmp_path):
    # Node's built-in runner (TS stripped) gives the red->green TDD signal for a
    # typecheck-only frontend target — previously skipped (pytest/jest only).
    import shutil
    from misterdev.task_executors.markdown_plan_executor import (
        MarkdownPlanExecutor,
    )

    if shutil.which("node") is None:
        return  # node unavailable in this environment

    spec_dir = tmp_path / ".orchestrator" / "spec_tests"
    spec_dir.mkdir(parents=True)
    failing = spec_dir / "spec_fail.test.ts"
    failing.write_text(
        'import { test } from "node:test";\n'
        'import assert from "node:assert/strict";\n'
        'test("x", () => assert.equal((1 as number) + 1, 3));\n',
        encoding="utf-8",
    )
    passing = spec_dir / "spec_pass.test.ts"
    passing.write_text(
        'import { test } from "node:test";\n'
        'import assert from "node:assert/strict";\n'
        'test("x", () => assert.equal((1 as number) + 1, 2));\n',
        encoding="utf-8",
    )

    class _Proj:
        path = tmp_path
        config = {"test_command": "npm test"}  # not pytest/jest -> node --test branch
        env_manager = None

    e = MarkdownPlanExecutor()
    # Falls into the node --test branch via the .test.ts suffix.
    assert e._run_spec_test(_Proj(), str(failing), 60)[0] == "red"
    assert e._run_spec_test(_Proj(), str(passing), 60)[0] == "green"
