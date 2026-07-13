"""Flaky-test quarantine: a nondeterministic failure must not revert a correct edit."""

from misterdev.core.verification.flaky import (
    RunOutcome,
    classify,
    confirm_test_failure,
    parse_failing_tests,
)

_PYTEST_FAIL = """\
tests/test_x.py::test_alpha PASSED
tests/test_x.py::test_beta FAILED
=== short test summary ===
FAILED tests/test_x.py::test_beta - AssertionError: boom
"""


def test_parse_pytest_node_ids():
    ids = parse_failing_tests(_PYTEST_FAIL)
    assert ids == frozenset({"tests/test_x.py::test_beta"})


def test_parse_go_and_rust_and_jest():
    assert parse_failing_tests("--- FAIL: TestThing (0.01s)") == frozenset(
        {"TestThing"}
    )
    assert parse_failing_tests("test mod::case ... FAILED") == frozenset({"mod::case"})
    assert parse_failing_tests("  ✕ renders a button") == frozenset(
        {"renders a button"}
    )


def test_parse_unrecognized_yields_empty():
    assert parse_failing_tests("Segmentation fault (core dumped)") == frozenset()


def test_flake_when_rerun_passes_cleanly():
    # Same test fails then the whole suite passes -> nondeterministic.
    outcomes = [RunOutcome.of(False, _PYTEST_FAIL), RunOutcome.of(True, "all passed")]
    verdict = classify(outcomes)
    assert not verdict.is_real_failure
    assert "tests/test_x.py::test_beta" in verdict.quarantined


def test_deterministic_failure_stays_red():
    outcomes = [RunOutcome.of(False, _PYTEST_FAIL)] * 3
    verdict = classify(outcomes)
    assert verdict.is_real_failure
    assert verdict.persistent == frozenset({"tests/test_x.py::test_beta"})


def test_real_failure_coexisting_with_flake_stays_red():
    both = "FAILED tests/a.py::test_real\nFAILED tests/b.py::test_flaky"
    only_real = "FAILED tests/a.py::test_real"
    verdict = classify([RunOutcome.of(False, both), RunOutcome.of(False, only_real)])
    assert verdict.is_real_failure  # the real failure is in every run
    assert verdict.persistent == frozenset({"tests/a.py::test_real"})
    assert verdict.quarantined == frozenset({"tests/b.py::test_flaky"})


def test_opaque_failure_falls_back_to_suite_level():
    # No parseable ids: a clean re-run proves the flake; otherwise stay RED.
    opaque = "Segmentation fault"
    assert not classify(
        [RunOutcome.of(False, opaque), RunOutcome.of(True, "ok")]
    ).is_real_failure
    assert classify(
        [RunOutcome.of(False, opaque), RunOutcome.of(False, opaque)]
    ).is_real_failure


def test_confirm_disabled_keeps_red():
    calls = []

    def _run():
        calls.append(1)
        return True, "would pass"

    verdict = confirm_test_failure(_run, _PYTEST_FAIL, reruns=0)
    assert verdict.is_real_failure
    assert not calls  # reruns=0 never re-invokes the command


def test_confirm_stops_early_on_clean_rerun():
    calls = []

    def _run():
        calls.append(1)
        return True, "all passed"

    verdict = confirm_test_failure(_run, _PYTEST_FAIL, reruns=3)
    assert not verdict.is_real_failure
    assert len(calls) == 1  # decisive clean pass short-circuits remaining reruns


def test_confirm_persistent_failure_uses_full_budget():
    calls = []

    def _run():
        calls.append(1)
        return False, _PYTEST_FAIL

    verdict = confirm_test_failure(_run, _PYTEST_FAIL, reruns=2)
    assert verdict.is_real_failure
    assert len(calls) == 2  # never passed, so all reruns are spent
