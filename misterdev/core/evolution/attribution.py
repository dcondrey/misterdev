"""Failure attribution — the "targeted, not blind" half of the loop.

Random mutation is what separates a slow evolutionary search from AlphaEvolve's
directed one. Before proposing a self-edit, mine the last benchmark run to find
where failures actually concentrate — which language, which error category — and
aim the mutation there. This module turns a set of per-instance results into a
ranked blame map plus example errors; the proposer consumes those to localize and
fix the real weakness, rather than a lookup table guessing which module is at
fault (precise component blame needs richer execution traces than the harness
emits today, so we surface the evidence and let the proposer localize).

Reads are duck-typed on the benchmark result record: ``.resolved`` (bool) and,
optionally, ``.language`` (str) and an error text via ``.output`` / ``.error``.
Imports nothing from ``evaluation`` — the layering stays one-way.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from misterdev.core.execution.error_classifier import classify_error

_MAX_EXAMPLES = 3  # error snippets kept per niche for the proposer's context
_EXAMPLE_CHARS = 600


@dataclass
class Blame:
    """Failure concentration for one behavioral niche (a language, or a
    language×error-category), with sample errors for the proposer."""

    niche: str
    failures: int
    total: int
    examples: List[str] = field(default_factory=list)
    # Where the blame came from, phrased to read in the proposer sentence
    # "misterdev fails N/M of {source} in this niche".
    source: str = "benchmark tasks"

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0


def _error_text(result) -> str:
    text = getattr(result, "output", "") or getattr(result, "error", "") or ""
    return str(text)


def attribute(results, by_category: bool = False) -> List[Blame]:
    """Rank behavioral niches by failure concentration (worst first).

    Bins each result by ``language`` (``"unknown"`` when absent); with
    ``by_category`` it sub-bins failures by the classified error category
    (``rust/wrong_type``) so the target is a specific failure mode, not just a
    language. Niches are ranked by failure COUNT — the biggest lever first — with
    the failure rate as a tie-break so a small-but-total-failure niche still
    surfaces. Passing (resolved) results contribute to totals but not to blame.
    """
    counts: dict = {}
    for r in results:
        lang = str(getattr(r, "language", None) or "unknown")
        resolved = bool(getattr(r, "resolved", False))
        if by_category and not resolved:
            niche = f"{lang}/{classify_error(_error_text(r))}"
        else:
            niche = lang
        bucket = counts.setdefault(niche, {"failures": 0, "total": 0, "examples": []})
        bucket["total"] += 1
        if not resolved:
            bucket["failures"] += 1
            if len(bucket["examples"]) < _MAX_EXAMPLES:
                snippet = _error_text(r).strip()[:_EXAMPLE_CHARS]
                if snippet:
                    bucket["examples"].append(snippet)

    blames = [
        Blame(niche=n, failures=b["failures"], total=b["total"], examples=b["examples"])
        for n, b in counts.items()
        if b["failures"] > 0
    ]
    blames.sort(key=lambda bl: (bl.failures, bl.failure_rate), reverse=True)
    return blames


def top_target(results, by_category: bool = True) -> Optional[Blame]:
    """The single highest-blame niche to mutate next, or None if all passed."""
    blames = attribute(results, by_category=by_category)
    return blames[0] if blames else None
