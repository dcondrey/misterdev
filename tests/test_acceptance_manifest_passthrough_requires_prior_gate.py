"""T0.1 — the MANIFEST acceptance-command pass-through requires a prior gate.

`_verify_acceptance` treats a broken acceptance command that errors with a
MANIFEST/CONFIG classification as "satisfied", on the premise that acceptance runs
only after the build/test gates pass, so the manifest demonstrably exists. That
premise holds on the tests-passed path but is FALSE on the certainty-merge path,
where acceptance runs with no objective gate. There, a malformed acceptance command
must not silently pass — that is a false completion resting on the LLM's word plus a
broken command.
"""

import types

from misterdev.task_executors.markdown_plan_executor.gates_mixin import GatesMixin

# Classifies as ErrorCategory.MANIFEST (verified against error_classifier).
MANIFEST_ERR = "error: could not find `Cargo.toml` in /x or any parent directory"


class _Stub(GatesMixin):
    def __init__(self, output):
        self._output = output

    def _run_command(self, project, command, timeout=None, cwd=None):
        return (False, self._output)


def _task(criteria="Run `cargo test` to verify."):
    return types.SimpleNamespace(acceptance_criteria=criteria, description="t", id="x")


def test_manifest_passthrough_denied_without_prior_gate():
    stub = _Stub(MANIFEST_ERR)
    ok, _out = stub._verify_acceptance(
        None, _task(), True, False, 30, prior_gate_passed=False
    )
    assert ok is False, (
        "a broken (MANIFEST) acceptance command must NOT be treated as satisfied "
        "when no objective gate ran (the certainty-merge path)"
    )


def test_manifest_passthrough_allowed_after_prior_gate():
    # Control: on the tests/build-passed path the premise holds; keep the
    # pass-through so a broken command does not false-fail a genuinely gated task.
    stub = _Stub(MANIFEST_ERR)
    ok, _out = stub._verify_acceptance(
        None, _task(), True, False, 30, prior_gate_passed=True
    )
    assert ok is True


def test_no_criteria_gateless_task_still_passes():
    # Control: a genuinely gate-less task with no acceptance criteria is unaffected
    # (the deliberate certainty-only completion path must not regress).
    stub = _Stub("")
    ok, _ = stub._verify_acceptance(
        None, _task(criteria=""), True, False, 30, prior_gate_passed=False
    )
    assert ok is True
