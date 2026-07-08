"""Aim the code evolver at real, recurring, current failures.

Benchmark attribution (:func:`misterdev.core.evolution.attribution.attribute`)
ranks niches by a raw failure count over one benchmark run. A *stream* of real
failures accumulated across runs carries two signals a single run cannot, and
ignoring them wastes the evolver's budget on the wrong target:

* **recurrence** — a failure whose fingerprint keeps reappearing is a standing
  weakness worth a self-edit; a one-off is likely environmental noise.
* **recency** — a failure last seen many runs ago was probably already fixed;
  chasing it regresses nothing real.

So each failing record is weighted by ``recency × log2(1 + recurrence)`` and the
weights are aggregated per niche. The niche with the greatest *weight* (not the
greatest count) is the target, so one nagging, still-current failure outranks a
pile of stale one-offs. The result is an evolution :class:`Blame` tagged
``source="real-build failures"`` so the proposer's wording stays honest about
where the target came from.
"""

import math
from typing import Dict, List, Optional

from misterdev.core.evolution.attribution import Blame
from misterdev.core.execution.error_classifier import classify_error

_MAX_EXAMPLES = 3
_EXAMPLE_CHARS = 600


def _niche_of(record, by_category: bool) -> str:
    lang = str(getattr(record, "language", None) or "unknown")
    if not by_category:
        return lang
    category = getattr(record, "category", "") or classify_error(
        getattr(record, "error", "") or ""
    )
    return f"{lang}/{category}" if category else lang


def stream_blame(
    records: List,
    *,
    current_run: Optional[int] = None,
    recurrence: Optional[Dict[str, int]] = None,
    by_category: bool = True,
    half_life: float = 5.0,
) -> List[Blame]:
    """Rank niches by recency-decayed, recurrence-amplified failure weight.

    ``records`` are :class:`~misterdev.core.learning.failure_log.FailureRecord`-
    shaped (``.language``, ``.error``, ``.category``, ``.run``, ``.fp``).
    ``recurrence`` maps a fingerprint to its count across the whole stream; when
    omitted it is derived from ``records``. ``current_run`` defaults to the newest
    run seen. Returns :class:`Blame` objects (raw counts preserved for the
    proposer's message) ordered by aggregated weight, worst first.
    """
    if not records:
        return []
    runs = [int(getattr(r, "run", 0)) for r in records]
    current = current_run if current_run is not None else max(runs, default=0)
    if recurrence is None:
        recurrence = {}
        for r in records:
            fp = getattr(r, "fp", "") or ""
            if fp:
                recurrence[fp] = recurrence.get(fp, 0) + 1

    buckets: Dict[str, dict] = {}
    for r in records:
        niche = _niche_of(r, by_category)
        b = buckets.setdefault(niche, {"weight": 0.0, "failures": 0, "examples": []})
        age = max(0, current - int(getattr(r, "run", 0)))
        recency = 0.5 ** (age / half_life) if half_life > 0 else 1.0
        fp = getattr(r, "fp", "") or ""
        rec = recurrence.get(fp, 1) if fp else 1
        b["weight"] += recency * math.log2(1 + rec)
        b["failures"] += 1
        if len(b["examples"]) < _MAX_EXAMPLES:
            snippet = (getattr(r, "error", "") or "").strip()[:_EXAMPLE_CHARS]
            if snippet:
                b["examples"].append(snippet)

    blames = [
        Blame(
            niche=n,
            failures=b["failures"],
            total=b["failures"],
            examples=b["examples"],
            source="real-build failures",
        )
        for n, b in buckets.items()
    ]
    blames.sort(key=lambda bl: buckets[bl.niche]["weight"], reverse=True)
    return blames


def top_stream_target(
    records: List, *, by_category: bool = True, half_life: float = 5.0
) -> Optional[Blame]:
    """The single highest-weight real-failure niche to aim evolution at, or None."""
    blames = stream_blame(records, by_category=by_category, half_life=half_life)
    return blames[0] if blames else None
