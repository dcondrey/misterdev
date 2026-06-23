"""Tests for the spec-as-tests helper (standalone; DEFERRED from the loop).

The generation + write + path logic is unit-tested with a monkeypatched
generator (no network). A separate test asserts the helper is NOT wired into the
build loop and that enabling the flag only logs a deferral notice — proving the
default build loop is unchanged.
"""

from my_project_orchestrator.core.models import Task
from my_project_orchestrator.core.spec_tests import (
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


# --- deferral: not wired into the loop --------------------------------------


def test_spec_as_tests_flag_is_readable_and_off_by_default():
    # The flag exists, defaults off, and is read via get_setting (so the
    # config-wiring guard test is satisfied) — but reading it never generates a
    # test (the loop is unchanged; see the source-level guard below).
    from my_project_orchestrator.config import get_setting, DEFAULT_CONFIG

    assert DEFAULT_CONFIG["orchestrator"]["spec_as_tests"] is False
    assert (
        get_setting(
            {"orchestrator": {"spec_as_tests": True}}, "orchestrator", "spec_as_tests"
        )
        is True
    )


def test_spec_tests_helper_is_not_wired_into_the_execute_loop():
    # DEFERRED contract: the generator must NOT be invoked from the executor's
    # task loop. Guard at the source level so a future accidental wiring of
    # generate_spec_test/write_spec_test into _execute_tasks fails this test.
    import inspect
    from my_project_orchestrator.agent import ProjectOrchestrator

    src = inspect.getsource(ProjectOrchestrator._execute_tasks)
    assert "generate_spec_test" not in src
    assert "write_spec_test" not in src
    assert "spec_tests" not in src


def test_enabling_flag_only_logs_deferral_notice():
    # When the flag is on, _run_pipeline emits a DEFERRED warning rather than
    # generating tests. Verify the exact notice the pipeline source carries.
    import inspect
    from my_project_orchestrator.agent import ProjectOrchestrator

    src = inspect.getsource(ProjectOrchestrator._run_pipeline)
    assert "spec_as_tests is enabled but DEFERRED" in src
    # And nothing in the pipeline calls the generator.
    assert "generate_spec_test" not in src
