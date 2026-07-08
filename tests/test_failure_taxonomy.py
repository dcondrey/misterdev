"""The failure taxonomy assigns cause by signature, defaults removable, and
self-corrects a mislabeled saturation verdict."""

from types import SimpleNamespace

from misterdev.core.learning.failure_taxonomy import (
    Cause,
    classify_failure,
    counterfactual_correction,
)


def _rec(error, category=""):
    return SimpleNamespace(error=error, category=category)


def test_guard_rejection_is_artifact():
    c = classify_failure(
        _rec("ERROR: this edit removed or renamed a symbol but left references to it")
    )
    assert c.cause is Cause.ARTIFACT and c.removable


def test_timeout_is_artifact():
    c = classify_failure(_rec("Command timed out after 120s: cargo build"))
    assert c.cause is Cause.ARTIFACT and c.removable


def test_budget_exhaustion_is_search():
    c = classify_failure(_rec("BudgetExceeded: exceeded max attempts"))
    assert c.cause is Cause.SEARCH and c.removable


def test_truncated_output_is_observation():
    c = classify_failure(_rec("AssertionError: None != ['99 bottles[12735 chars]ll.']"))
    assert c.cause is Cause.OBSERVATION and c.removable


def test_assertion_misclassified_as_syntax_is_observation():
    c = classify_failure(_rec("assertion failed: left == right", category="syntax"))
    assert c.cause is Cause.OBSERVATION


def test_high_recurrence_is_convergence():
    c = classify_failure(_rec("AssertionError: 1 != 2"), recurrence=5)
    assert c.cause is Cause.CONVERGENCE and c.removable


def test_genuine_wrong_answer_is_saturation_with_ruled_out():
    # A real wrong answer, seen once, with no removable signature -> the residual.
    c = classify_failure(_rec("AssertionError: 1 != 2"), recurrence=1)
    assert c.cause is Cause.SATURATION
    assert not c.removable
    # I3: saturation must record what it ruled out.
    assert set(c.ruled_out) == {"artifact", "search", "observation", "convergence"}


def test_removable_signatures_take_priority_over_saturation():
    # Even a low-recurrence failure is ARTIFACT if it carries a guard/env signature.
    c = classify_failure(
        _rec("Baseline build is failing. Command timed out after 120s"), recurrence=1
    )
    assert c.cause is Cause.ARTIFACT


def test_counterfactual_downgrades_mislabeled_saturation():
    sat = classify_failure(_rec("AssertionError: 1 != 2"), recurrence=1)
    note = counterfactual_correction(
        sat, reattempt_resolved=True, changed_condition="doubled budget"
    )
    assert note and "MISLABELED" in note and "doubled budget" in note


def test_counterfactual_noop_when_not_saturation_or_still_failing():
    art = classify_failure(_rec("Command timed out after 120s"))
    assert counterfactual_correction(art, True, "x") is None
    sat = classify_failure(_rec("AssertionError: 1 != 2"), recurrence=1)
    assert (
        counterfactual_correction(sat, reattempt_resolved=False, changed_condition="x")
        is None
    )
