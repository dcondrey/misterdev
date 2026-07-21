"""T0.2 — a test gate that collects/executes zero tests is a hard reject.

The acceptance layer (`GatesMixin._gate_accepts`) must not treat a test command
that exits 0 while running ZERO tests as a pass. A zero-test "green" greenlights
any edit while catching no regression — the canonical false-GREEN gate. The
zero-test signal machinery already exists in the validator (`gate_ran_no_tests`
paired with a parsed total of 0); this asserts `_gate_accepts` actually consults
it, and that genuine greens and the baseline-red path are unaffected.
"""

import pytest

from misterdev.task_executors.markdown_plan_executor.gates_mixin import GatesMixin

# Runner-specific "zero tests executed" outputs, each paired with exit 0 (success).
ZERO_TEST_GREENS = [
    "collected 0 items\n\nno tests ran in 0.01s",  # pytest
    "no tests ran in 0.00s",  # pytest (variant)
    "Ran 0 tests in 0.000s\n\nOK",  # stdlib unittest
    "No tests found, exiting with code 0",  # jest / vitest
    "no test files\nok  \tpkg\t0.001s",  # go
    "running 0 tests\n\ntest result: ok. 0 passed; 0 failed",  # cargo
]


@pytest.mark.parametrize("output", ZERO_TEST_GREENS)
def test_zero_test_green_is_rejected(output):
    accepted, _post = GatesMixin._gate_accepts(True, output, baseline_failures=0)
    assert accepted is False, (
        "a test gate that ran zero tests must be rejected, not accepted as GREEN; "
        f"output was {output!r}"
    )


def test_real_green_with_tests_still_accepted():
    accepted, post = GatesMixin._gate_accepts(True, "5 passed in 1.2s", 0)
    assert accepted is True
    assert post == 0


def test_empty_output_green_is_not_falsely_rejected():
    # No zero-test signal present -> a plain success must remain accepted (we do
    # not punish output formats we simply do not parse).
    accepted, _ = GatesMixin._gate_accepts(True, "", 0)
    assert accepted is True


def test_baseline_red_incremental_progress_unaffected():
    # Red run, parsed 2 failures, baseline 3 -> still accepted (no-worse rule).
    accepted, post = GatesMixin._gate_accepts(False, "3 passed, 2 failed", 3)
    assert accepted is True
    assert post == 2


def test_zero_test_green_rejected_even_under_red_baseline():
    # A zero-test green must not sneak through the baseline-red branch either.
    accepted, _ = GatesMixin._gate_accepts(
        True, "collected 0 items", baseline_failures=5
    )
    assert accepted is False
