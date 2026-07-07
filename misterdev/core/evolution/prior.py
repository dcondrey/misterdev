"""The mutation prior — meta-learning which edits pay off (the recursive layer).

The archive records, per niche, the elite self-edit and its outcome. Over many
runs a pattern emerges: certain kinds of edit (tagged by their proposer ``note``)
land as elites far more often than others. This module mines that history into a
prior over mutation kinds, so the proposer spends its budget on the moves that
have historically worked — "smarter the more it runs" (the STOP/ADAS recursive
tier), without touching the fitness rule or the gates.

It is a pure read over the archive elites: no LLM, no benchmark, no source edits.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from .archive import EvolutionArchive

# A prior kind seen fewer than this many times is not yet evidence, only anecdote;
# it is reported with its raw count but callers should not over-weight it.
_MIN_EVIDENCE = 2


@dataclass
class KindWeight:
    """One mutation kind and how strongly the archive favours it."""

    kind: str
    elites: int  # times an edit of this kind became a niche elite
    weight: float  # share of all elite edits, in [0, 1]
    proven: bool  # met the minimum-evidence bar


def _kind_of(note: str) -> str:
    """The mutation kind: the token after a ``tag:`` marker if present, else the
    note's leading word, else ``"unknown"`` — a cheap, stable bucket that tolerates
    both an already-parsed kind and a raw ``tag: <kind>`` line."""
    note = (note or "").strip()
    if not note:
        return "unknown"
    marked = re.match(r"tag:\s*(\S+)", note, re.IGNORECASE)
    if marked:
        return marked.group(1).rstrip(":").lower() or "unknown"
    return note.split(None, 1)[0].rstrip(":").lower() or "unknown"


class MutationPrior:
    """A prior over mutation kinds, derived from archive elites."""

    def __init__(self, archive: EvolutionArchive):
        self.archive = archive

    def weights(self) -> List[KindWeight]:
        """Elite-share of each mutation kind, most-favoured first.

        Weight is the fraction of elite edits of that kind, so it sums to ~1 and
        can bias a weighted choice directly. Empty when the archive holds no
        elites yet (a cold start ranks all kinds equally — the caller falls back
        to uniform exploration).
        """
        kinds = Counter(_kind_of(c.note) for c in self.archive.elites())
        total = sum(kinds.values())
        if not total:
            return []
        out = [
            KindWeight(
                kind=k,
                elites=n,
                weight=n / total,
                proven=n >= _MIN_EVIDENCE,
            )
            for k, n in kinds.items()
        ]
        out.sort(key=lambda kw: kw.elites, reverse=True)
        return out

    def favored_kinds(self, limit: int = 3) -> List[str]:
        """The kinds with the most elite evidence, for steering the proposer.

        Only kinds past the evidence bar are returned, so a single lucky edit does
        not dominate the prior; empty means "no proven prior yet, explore freely".
        """
        return [kw.kind for kw in self.weights() if kw.proven][:limit]
