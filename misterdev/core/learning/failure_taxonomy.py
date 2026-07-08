"""Classify WHY a failure happened, not just where it clustered.

Attribution (:mod:`misterdev.core.evolution.attribution`) ranks failures by
*niche* — which language/category breaks most. That answers "where to aim" but
not "what kind of problem is this," and the second question is what decides
whether a self-edit should be a structural fix, more search, or an escalation —
and whether "give up" is even honest. This module reads a logged
:class:`~misterdev.core.learning.failure_log.FailureRecord` and assigns a
**cause**:

* ``ARTIFACT`` — the harness blocked a correct edit, or the environment broke
  (a guard rejection, a build/test timeout, a poisoned baseline). Removable.
* ``SEARCH`` — the loop ran out of budget/attempts. Removable (spend/guide more).
* ``OBSERVATION`` — the model was fed a truncated/misclassified view of the
  failure, so it could not see what to fix. Removable (fix the seam).
* ``CONVERGENCE`` — the same failure recurs across attempts; the model is
  thrashing one approach. Removable (force diversity).
* ``SATURATION`` — a genuine wrong answer with none of the above signatures: the
  model may be at its capability edge. The ONLY non-removable cause — and the one
  most often assigned in error, so it is the default of last resort, records what
  it ruled out, and is subject to counterfactual correction (see
  :func:`counterfactual_correction`).

Invariant I3 (see docs/path-to-100.md): default to "removable." Every SATURATION
verdict must rule out the other four with evidence, because assuming a wall where
there was an artifact was wrong every time it was tried. Pure and offline-testable.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Cause(str, Enum):
    ARTIFACT = "artifact"
    SEARCH = "search"
    OBSERVATION = "observation"
    CONVERGENCE = "convergence"
    SATURATION = "saturation"


REMOVABLE = frozenset(
    {Cause.ARTIFACT, Cause.SEARCH, Cause.OBSERVATION, Cause.CONVERGENCE}
)

# A guard rejected a candidate edit (a correct edit can be blocked here) — the
# real strings misterdev's guards emit.
_GUARD = re.compile(
    r"removed or renamed a symbol|left references to it|test files were weakened|"
    r"do not delete tests|no applicable file edit|did not apply cleanly|"
    r"stalling detected",
    re.IGNORECASE,
)
# The environment broke, not the code: a slow build/test or a poisoned baseline.
_ENV = re.compile(
    r"command timed out|timed out after|baseline build is failing|"
    r"no endpoints found matching|connection refused|could not resolve host",
    re.IGNORECASE,
)
# The search ran out.
_SEARCH = re.compile(
    r"budget ?exceeded|budgetexceeded|exceeded max attempts|out of budget|no progress",
    re.IGNORECASE,
)
# The model saw a truncated diff (pytest's `[N chars]` elision) or a middle
# elision — it could not see the exact expected/actual to fix.
_TRUNCATED = re.compile(r"\[\d+\s*chars\]|\.\.\.\s|… ")
# A test-assertion failure misclassified as a syntax error gives the model the
# wrong instruction — an observation defect, not the model's fault.
_ASSERTION = re.compile(
    r"assertion|assertionerror|expected:?\b|received:?\b| != |left ==", re.IGNORECASE
)

_RECURRENCE_CONVERGENCE = 3  # a fingerprint seen this many times is thrashing


@dataclass(frozen=True)
class Classification:
    """A failure's cause, with the evidence and (for saturation) what was ruled out."""

    cause: Cause
    evidence: str
    ruled_out: List[str] = field(default_factory=list)

    @property
    def removable(self) -> bool:
        return self.cause in REMOVABLE


def classify_failure(record, recurrence: int = 0) -> Classification:
    """Assign a cause to one logged failure. ``recurrence`` is how many times this
    error's fingerprint has been seen (from ``FailureLog.recurrence()``).

    Cascade, most-specific signature first; SATURATION only as the evidence-backed
    residual so a removable failure is never mislabeled a capability wall.
    """
    error = getattr(record, "error", "") or ""
    category = (getattr(record, "category", "") or "").lower()

    if _GUARD.search(error):
        return Classification(Cause.ARTIFACT, "a guard rejected the candidate edit")
    if _ENV.search(error):
        return Classification(
            Cause.ARTIFACT, "environment failure (timeout/baseline/network)"
        )
    if _SEARCH.search(error):
        return Classification(Cause.SEARCH, "budget or attempts exhausted")
    if _TRUNCATED.search(error):
        return Classification(Cause.OBSERVATION, "failure output was truncated/elided")
    if category == "syntax" and _ASSERTION.search(error):
        return Classification(
            Cause.OBSERVATION, "test-assertion failure misclassified as syntax"
        )
    if recurrence >= _RECURRENCE_CONVERGENCE:
        return Classification(
            Cause.CONVERGENCE,
            f"same failure recurred {recurrence}x — thrashing one approach",
        )
    # Residual: a genuine wrong answer with no removable signature. Record what was
    # ruled out (I3) so the verdict is auditable and correctable.
    return Classification(
        Cause.SATURATION,
        "wrong answer with no artifact/env/budget/truncation/recurrence signature",
        ruled_out=["artifact", "search", "observation", "convergence"],
    )


def counterfactual_correction(
    prior: Classification, reattempt_resolved: bool, changed_condition: str
) -> Optional[str]:
    """The self-correcting hook: if a task labeled SATURATION later PASSES under a
    changed condition (more budget, a new observation seam, forced diversity), the
    verdict was wrong — the failure was removable after all. Return a correction
    note capturing the counterexample (to move the classifier's boundary over
    time); ``None`` when no correction applies.

    This is what stops the taxonomy from inheriting a fixed give-up bias: a
    saturation call is a hypothesis the loop keeps testing, not a final judgment.
    """
    if prior.cause == Cause.SATURATION and reattempt_resolved:
        return (
            f"MISLABELED saturation: resolved after '{changed_condition}'. The "
            "failure was removable; downgrade the boundary that produced this call."
        )
    return None
