"""The MAP-Elites archive — the "nothing good is forgotten" property.

A greedy loop keeps only the current best and throws away everything else, which
discards the stepping-stones: a change that is neutral (or slightly worse)
overall but the BEST at some behavioral niche often enables a later win. This
archive keeps the elite of each niche instead — "improves Rust tasks", "reduces
cost on large files", "fixes over-decomposition" — so those stepping-stones
survive (the Darwin-Gödel insight, and the code-level analogue of the scored
lesson store).

Promotion into a niche uses the SAME :meth:`FitnessScore.beats` rule as the live
loop, with the niche's current elite as the incumbent, so a candidate replaces an
elite only when it is a real improvement past the noise band with zero
regressions. Persistence mirrors the lesson store: a single JSON file, best-
effort load that degrades to empty rather than raising.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write_json

from .fitness import FitnessScore

logger = setup_logger(__name__)


@dataclass
class Candidate:
    """One evaluated self-edit and its benchmark outcome.

    ``parent_id`` records lineage so stepping-stone chains (and, later, a prior
    over which mutation kinds pay off) can be mined from the archive. ``patch`` is
    the diff (or a reference to it) that produced the score.
    """

    id: str
    niche: str
    resolved: int
    total: int
    cost: float
    regressions: int = 0
    parent_id: Optional[str] = None
    patch: str = ""
    note: str = ""
    run: int = 0

    def score(self) -> FitnessScore:
        return FitnessScore(self.resolved, self.total, self.cost, self.regressions)


class EvolutionArchive:
    """Best-per-niche archive of scored candidates, persisted to one JSON file."""

    def __init__(self, path, noise_band: float = 0.0):
        self.path = Path(path)
        self.noise_band = noise_band

    def _load(self) -> Tuple[Dict[str, Candidate], int]:
        """Return (elites-by-niche, run_counter). Degrades to empty on a missing
        or corrupt file — an unreadable archive must never crash the loop."""
        if not self.path.exists():
            return {}, 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, 0
        if not isinstance(raw, dict):
            return {}, 0
        run = int(raw.get("run", 0))
        elites: Dict[str, Candidate] = {}
        for item in raw.get("elites", []):
            if not isinstance(item, dict) or not item.get("niche"):
                continue
            try:
                elites[str(item["niche"])] = Candidate(
                    id=str(item.get("id", "")),
                    niche=str(item["niche"]),
                    resolved=int(item.get("resolved", 0)),
                    total=int(item.get("total", 0)),
                    cost=float(item.get("cost", 0.0)),
                    regressions=int(item.get("regressions", 0)),
                    parent_id=item.get("parent_id"),
                    patch=str(item.get("patch", "")),
                    note=str(item.get("note", "")),
                    run=int(item.get("run", 0)),
                )
            except (TypeError, ValueError):
                continue  # skip a malformed record, keep the rest
        return elites, run

    def _save(self, elites: Dict[str, Candidate], run: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": run,
            "elites": [asdict(c) for c in elites.values()],
        }
        atomic_write_json(self.path, payload, indent=2)

    def consider(self, candidate: Candidate) -> bool:
        """Insert ``candidate`` as its niche's elite iff it earns the slot.

        A candidate with any regression is never made an elite. Otherwise it takes
        the niche when it is empty or when it beats the incumbent elite past the
        noise band. Returns True when it became (or replaced) the elite. Every
        call bumps the run counter and persists, so the run count reflects total
        candidates considered even when one is rejected.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            try:
                import fcntl

                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            elites, run = self._load()
            run += 1
            candidate.run = run
            incumbent = elites.get(candidate.niche)
            won = candidate.regressions == 0 and (
                incumbent is None
                or candidate.score().beats(incumbent.score(), self.noise_band)
            )
            if won:
                elites[candidate.niche] = candidate
            self._save(elites, run)
        finally:
            os.close(lock_fd)
        if won:
            logger.info(
                f"Archive: candidate {candidate.id!r} is the new elite for niche "
                f"{candidate.niche!r} ({candidate.resolved}/{candidate.total})."
            )
        return won

    def elite(self, niche: str) -> Optional[Candidate]:
        """The current elite for ``niche``, or None."""
        return self._load()[0].get(niche)

    def elites(self) -> List[Candidate]:
        """All niche elites (the full stepping-stone set)."""
        return list(self._load()[0].values())

    def champion(self) -> Optional[Candidate]:
        """The globally best elite: highest resolved-rate, cheapest on a tie.

        The candidate to ship. Distinct from the per-niche elites, which are kept
        even when globally suboptimal precisely so they can seed future wins.
        """
        best: Optional[Candidate] = None
        for cand in self._load()[0].values():
            if cand.regressions > 0:
                continue
            if best is None or _globally_better(cand, best):
                best = cand
        return best


def _globally_better(a: Candidate, b: Candidate) -> bool:
    """True if ``a`` outranks ``b`` globally: higher resolved-rate, or equal rate
    at strictly lower cost-per-task."""
    sa, sb = a.score(), b.score()
    if sa.resolved_rate != sb.resolved_rate:
        return sa.resolved_rate > sb.resolved_rate
    return sa.cost_per_task < sb.cost_per_task
