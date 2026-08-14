"""Persistent, self-authored tool library — the consolidation half of
two-timescale evolution (see ``docs/two-timescale-evolution.md``).

live-SWE-agent (current #1 open scaffold) invents task-specific Python tools at
runtime and *discards them every task*, reinventing the same edit/reproduce tools
on every instance. This module is the memory it lacks: a tool that PROVES IT
GENERALIZES is kept, best-per-capability-niche, so future runs start from
accumulated capability instead of rebuilding it — the compounding that eventually
exceeds a memoryless agent.

It reuses the scaffold-evolution machinery rather than inventing a second engine:
a tool is a new :class:`~.archive.Candidate` substrate (its SOURCE is the artifact,
its capability class the MAP-Elites niche), scored by the same
:class:`~.fitness.FitnessScore`, and — crucially — admitted only through the same
held-out :func:`~.holdout.decide_promotion` gate that stops the loop becoming a
benchmark-specialist. That gate is what keeps the library a set of genuinely
general capabilities, not a benchmark-overfit grab-bag.

Pure and offline: nothing here authors, runs, or trusts a tool. Runtime invention
and sandboxed execution (untrusted, model-authored code) are a separate phase.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write_json, flock_guarded

from .fitness import FitnessScore
from .holdout import PromotionDecision, decide_promotion

logger = setup_logger(__name__)


@dataclass
class ToolCandidate:
    """A self-authored tool and the DERIVE-pool outcome that earns its slot.

    ``source`` is the tool's code (the artifact, mirroring ``Candidate.patch``);
    ``niche`` is the capability class it serves (the MAP-Elites key, e.g.
    ``"reproduce-and-minimize-pytest-failure"``); ``provenance`` records the task
    it was invented on, for lineage and audit. The four count fields are the tool's
    fitness on the DERIVE pool — tasks solved WITH it available — used for
    best-per-niche once the held-out gate has cleared it as generalizing.
    """

    id: str
    niche: str
    source: str
    resolved: int
    total: int
    cost: float
    regressions: int = 0
    provenance: str = ""
    run: int = 0

    def score(self) -> FitnessScore:
        return FitnessScore(self.resolved, self.total, self.cost, self.regressions)


class ToolLibrary:
    """Best-per-niche library of self-authored tools, persisted to one JSON file.

    Admission is a two-gate ratchet: (1) the held-out anti-overfit gate — a tool
    that lifts only the tasks it was born to help, while dropping tasks it never
    saw, is rejected AS overfit; (2) MAP-Elites best-per-niche — even a
    generalizing tool replaces the niche elite only when it beats it past the noise
    band. Persistence mirrors the scaffold archive: a single JSON file, best-effort
    load that degrades to empty rather than raising.
    """

    def __init__(self, path, noise_band: float = 0.0):
        self.path = Path(path)
        self.noise_band = noise_band

    def _load(self) -> Tuple[Dict[str, ToolCandidate], int]:
        """Return (elites-by-niche, run_counter). Degrades to empty on a missing or
        corrupt file — an unreadable library must never crash a run."""
        if not self.path.exists():
            return {}, 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, 0
        if not isinstance(raw, dict):
            return {}, 0
        run = int(raw.get("run", 0))
        elites: Dict[str, ToolCandidate] = {}
        for item in raw.get("elites", []):
            if not isinstance(item, dict) or not item.get("niche"):
                continue
            try:
                elites[str(item["niche"])] = ToolCandidate(
                    id=str(item.get("id", "")),
                    niche=str(item["niche"]),
                    source=str(item.get("source", "")),
                    resolved=int(item.get("resolved", 0)),
                    total=int(item.get("total", 0)),
                    cost=float(item.get("cost", 0.0)),
                    regressions=int(item.get("regressions", 0)),
                    provenance=str(item.get("provenance", "")),
                    run=int(item.get("run", 0)),
                )
            except (TypeError, ValueError):
                continue  # skip a malformed record, keep the rest
        return elites, run

    def _save(self, elites: Dict[str, ToolCandidate], run: int) -> None:
        payload = {"run": run, "elites": [asdict(t) for t in elites.values()]}
        atomic_write_json(self.path, payload, indent=2)

    def consider(
        self,
        tool: ToolCandidate,
        *,
        derive: FitnessScore,
        derive_base: FitnessScore,
        holdout: FitnessScore,
        holdout_base: FitnessScore,
    ) -> PromotionDecision:
        """Admit ``tool`` to the library iff it GENERALIZES and beats its niche
        incumbent. Returns the final admission verdict (with the reason).

        Gate order matters: the held-out gate runs first so an overfit tool is
        rejected before it can ever contend for a niche. A tool that generalizes but
        does not beat its niche's current elite is also not admitted (the elite
        already covers that capability better). Every call bumps the run counter and
        persists, so the count reflects total tools considered, admitted or not.
        """
        with flock_guarded(self.path):
            elites, run = self._load()
            run += 1
            tool.run = run
            decision = decide_promotion(
                derive, derive_base, holdout, holdout_base, self.noise_band
            )
            admitted = decision
            if decision.promote:
                incumbent = elites.get(tool.niche)
                beats = tool.regressions == 0 and (
                    incumbent is None
                    or tool.score().beats(incumbent.score(), self.noise_band)
                )
                if beats:
                    elites[tool.niche] = tool
                else:
                    admitted = PromotionDecision(
                        False,
                        f"generalizes ({decision.reason}) but does not beat the "
                        f"{tool.niche!r} elite",
                    )
            self._save(elites, run)
        if admitted.promote:
            logger.info(
                f"ToolLibrary: {tool.id!r} admitted as elite for niche "
                f"{tool.niche!r} — {admitted.reason}."
            )
        else:
            logger.info(f"ToolLibrary: {tool.id!r} not admitted — {admitted.reason}.")
        return admitted

    def elite(self, niche: str) -> Optional[ToolCandidate]:
        """The current elite tool for ``niche``, or None."""
        return self._load()[0].get(niche)

    def elites(self) -> List[ToolCandidate]:
        """Every niche's elite tool (the full accumulated capability set)."""
        return list(self._load()[0].values())

    def seed(self, limit: Optional[int] = None) -> List[ToolCandidate]:
        """The promoted tools to load into a run's toolbelt, best global first.

        Ordered by global fitness (highest resolved-rate, cheapest on a tie) so a
        capped run takes the most-proven tools. These are the accumulated capability
        a new run starts from instead of reinventing it — the compounding property
        that a memoryless runtime-only agent (live-SWE-agent) does not have.
        """
        ranked = sorted(
            (t for t in self._load()[0].values() if t.regressions == 0),
            key=lambda t: (t.score().resolved_rate, -t.score().cost_per_task),
            reverse=True,
        )
        return ranked[:limit] if limit else ranked
