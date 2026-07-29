"""Classify a run's terminal non-successes so failures become signal, not noise.

At the end of a run the interesting question is "WHY did it underperform?" — was
the box thrashing (infra), was a credential missing (blocked), did tasks collide
on merge, did the code just not work, or did tasks park for input? This module
routes each terminal non-success into one taxonomy bucket and aggregates a
one-glance run summary. It reuses the existing pure signal detectors
(``infra_failure``, ``blocked_reason``) so a fault is labelled the same way here
as it was handled during the run.

Pure and side-effect-free: classification is a function of (status, text), and
the summary is a function of the per-task classifications — so both the routing
and the aggregation are directly unit-testable. The orchestrator supplies the
texts and writes the result to the console and ``.orchestrator/run_summary.json``.
"""

import re
from typing import Dict, List, Optional, Tuple

from misterdev.core.execution.blocker import blocked_reason
from misterdev.core.execution.infra import infra_failure

# Ordered most-specific → most-generic; also the tie-break order for the top
# obstacle (an earlier, more-specific category wins an equal count).
CATEGORIES: Tuple[str, ...] = (
    "blocked-external",
    "infra",
    "merge-conflict",
    "acceptance-unmet",
    "genuine-code-failure",
    "deferred-needs-input",
)

_MERGE_CONFLICT = re.compile(
    r"merge conflict|merge failed|CONFLICT \(content\)|post-merge health gate", re.I
)
_ACCEPTANCE = re.compile(r"acceptance crit(?:erion|eria).{0,40}not met", re.I)


def classify_failure(status: str, text: str) -> str:
    """Route one terminal non-success into a taxonomy category.

    Strongest signal wins, independent of ``status``: an environment or blocked
    fault is labelled as such even on a task recorded "failed", because that IS
    why it failed. Only when no signal matches does ``status`` decide — a parked
    task is ``deferred-needs-input``, anything else ``genuine-code-failure``.
    """
    t = text or ""
    if blocked_reason(t):
        return "blocked-external"
    if infra_failure(t):
        return "infra"
    if _MERGE_CONFLICT.search(t):
        return "merge-conflict"
    if _ACCEPTANCE.search(t):
        return "acceptance-unmet"
    if (status or "").strip().lower() == "deferred":
        return "deferred-needs-input"
    return "genuine-code-failure"


def _first_meaningful_line(text: str) -> str:
    """The first informative line of a failure blob — skipping blanks and markdown
    fences/headers that would otherwise make a useless exemplar."""
    for line in (text or "").splitlines():
        s = line.strip()
        if s and not s.startswith("```") and not s.startswith("#"):
            return s
    return ""


def build_run_summary(
    completed: int,
    failed_items: List[Tuple[str, str]],
    deferred_items: List[Tuple[str, str]],
    elapsed_seconds: float,
) -> Dict:
    """Aggregate a run's outcomes into a one-glance summary.

    ``failed_items``/``deferred_items`` are ``(task_id, failure_text)`` pairs.
    Returns counts, the failure breakdown by category (zero categories omitted), a
    short exemplar message per category, wall-clock, and the single top recurring
    obstacle (the category with the most failures; ties break toward the more
    specific one). Deterministic given the same inputs.
    """
    breakdown: Dict[str, int] = {c: 0 for c in CATEGORIES}
    exemplars: Dict[str, str] = {}

    def note(status: str, text: str) -> None:
        cat = classify_failure(status, text)
        breakdown[cat] += 1
        if cat not in exemplars:
            msg = _first_meaningful_line(text)
            if msg:
                exemplars[cat] = msg[:200]

    for _id, text in failed_items:
        note("failed", text)
    for _id, text in deferred_items:
        note("deferred", text)

    nonzero = {c: n for c, n in breakdown.items() if n}
    top_obstacle: Optional[str] = None
    if nonzero:
        # Most failures wins; a tie breaks toward the earlier (more specific)
        # category via its negative index.
        top_obstacle = max(nonzero, key=lambda c: (nonzero[c], -CATEGORIES.index(c)))
    return {
        "completed": completed,
        "failed": len(failed_items),
        "deferred": len(deferred_items),
        "elapsed_seconds": round(float(elapsed_seconds), 1),
        "failure_breakdown": nonzero,
        "exemplars": exemplars,
        "top_obstacle": top_obstacle,
    }
