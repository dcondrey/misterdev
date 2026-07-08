"""The keep-if-better cycle — composition of fitness + archive + guardrail.

One step of empirical self-improvement: propose a targeted diff to a blamed
module, sandbox-evaluate it against the benchmark, archive it, and promote it only
when it beats the champion past the noise band with zero regressions.

The parts that spend money or mutate source are INJECTED callables, so the loop's
decision logic is pure and fully testable, and the dangerous adapters are
swappable and individually guardrailed (tier-2 activation wires the real ones):

* ``propose(target) -> Mutation`` — the LLM proposer, scoped to the blamed module.
* ``evaluate(mutation) -> FitnessScore | None`` — applies the patch in an isolated
  git worktree, runs the FULL existing gate suite as a hard precondition, then the
  (proxy) benchmark. Returns ``None`` when the gates fail, so a candidate that
  breaks the build is discarded before it can ever score.

Targeting (which module, which niche) is the caller's job — it comes from failure
attribution over the last benchmark run, and is passed into :meth:`step`. The loop
guardrail-checks every proposal, archives every scored candidate (nothing good is
forgotten), and advances the incumbent only on a real win.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from misterdev.logging_setup import setup_logger

from .archive import Candidate, EvolutionArchive
from .fitness import FitnessScore
from .guardrail import ProtectedPathError, assert_mutation_allowed

logger = setup_logger(__name__)


@dataclass
class Mutation:
    """A proposed self-edit: the files it touches plus the diff that changes them."""

    target: str
    paths: List[str]
    patch: str
    note: str = ""


@dataclass
class StepResult:
    """Outcome of one loop step."""

    promoted: bool
    archived: bool
    reason: str
    score: Optional[FitnessScore] = None
    candidate_id: Optional[str] = None


@dataclass
class EvolutionLoop:
    """Drives one keep-if-better step; holds the champion incumbent across steps."""

    archive: EvolutionArchive
    evaluate: Callable[[Mutation], Optional[FitnessScore]]
    propose: Callable[[str], Mutation]
    noise_band: float
    champion: FitnessScore
    # Optional cheap screen (a MicroEvaluator.screen): when set, ``beam`` candidates
    # are proposed and screened, and only the best survivor reaches the expensive
    # ``evaluate`` oracle — so the loop widens its search without widening its cost.
    # Unset (the default), behaviour is unchanged: one candidate, straight to eval.
    screen: Optional[Callable[[Mutation], object]] = None
    beam: int = 1
    _counter: int = field(default=0, init=False)

    def _pick(self, target: str) -> Tuple[Optional[Mutation], str]:
        """Choose the candidate to spend the oracle on: propose (a beam of) one or
        more, guardrail each, and — when a screen is armed — keep only screened
        survivors and return the best. Returns (mutation, "") or (None, reason)."""
        n = self.beam if self.screen is not None else 1
        candidates: List[Mutation] = []
        reason = "no viable proposal"
        for _ in range(max(1, n)):
            try:
                m = self.propose(target)
            except Exception as e:  # one dead candidate, not a crash
                logger.warning(f"Evolution: proposal for {target!r} failed: {e}")
                reason = f"proposal failed: {e}"
                continue
            try:
                assert_mutation_allowed(m.paths)
            except ProtectedPathError as e:
                # Reward-hacking wall: refuse before the candidate is ever scored.
                logger.warning(f"Evolution: refused a candidate — {e}")
                reason = f"guardrail: {e}"
                continue
            candidates.append(m)

        if not candidates:
            return None, reason
        if self.screen is None:
            return candidates[0], ""

        survivors: List[Tuple[tuple, Mutation]] = []
        for m in candidates:
            verdict = self.screen(m)
            if getattr(verdict, "accepted", False):
                survivors.append((verdict.rank_key, m))
        if not survivors:
            return None, "all candidates screened out"
        # Best survivor first: most targets fixed with the fewest guard breaks.
        survivors.sort(key=lambda sv: sv[0], reverse=True)
        return survivors[0][1], ""

    def step(self, target: str, niche: str) -> StepResult:
        """Run one propose → (screen) → sandbox-eval → archive → promote cycle.

        Never raises: a proposer/evaluator failure or a guardrail violation ends
        the step as a non-promotion with a reason, so one bad candidate can never
        halt the search.
        """
        mutation, reason = self._pick(target)
        if mutation is None:
            return StepResult(False, False, reason)

        self._counter += 1
        cand_id = f"cand-{self._counter}"

        try:
            score = self.evaluate(mutation)
        except Exception as e:  # sandbox/benchmark failure — discard, don't crash
            logger.warning(f"Evolution: evaluation of {cand_id} failed: {e}")
            return StepResult(False, False, f"evaluation failed: {e}")

        if score is None:
            # Gates failed inside the sandbox: a build-breaking candidate never
            # reaches the fitness comparison.
            return StepResult(False, False, "gates failed", candidate_id=cand_id)

        candidate = Candidate(
            id=cand_id,
            niche=niche,
            resolved=score.resolved,
            total=score.total,
            cost=score.cost,
            regressions=score.regressions,
            patch=mutation.patch,
            note=mutation.note,
        )
        archived = self.archive.consider(candidate)

        promoted = score.beats(self.champion, self.noise_band)
        if promoted:
            self.champion = (
                score  # advance the incumbent so the next step must beat this
            )
            logger.info(
                f"Evolution: promoted {cand_id} "
                f"({score.resolved}/{score.total}, ${score.cost:.4f})."
            )
        reason = (
            "promoted"
            if promoted
            else ("regression" if score.regressions else "within noise band")
        )
        return StepResult(promoted, archived, reason, score=score, candidate_id=cand_id)
