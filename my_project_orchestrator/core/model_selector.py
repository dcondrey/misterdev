"""Cost-aware, data-driven model selection.

Turns the static complexity->tier->model routing into a policy that picks a
model to maximize quality-per-dollar under an unconditional quality floor: the
validation gates. A cheaper model that writes worse code simply fails the gate
and the policy escalates, so model choice can only change cost/latency, never
shipped quality.

The policy is deterministic (UCB scoring from the ledger, not random draws) and
disabled by default; with it off, ``select`` returns None and the executor
keeps its existing static routing.
"""

from typing import List, Optional

from my_project_orchestrator.config import get_setting
from my_project_orchestrator.core.model_ledger import ModelLedger
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class ModelSelector:
    """Picks a model id per attempt from the escalation ladder and ledger."""

    def __init__(
        self, config: dict, ledger: ModelLedger, free_models: Optional[List[str]] = None
    ):
        self.ledger = ledger
        # dynamic_selection is False (off), True (on), or "auto" (self-
        # activating). A non-"auto" string is treated as off, never truthy.
        raw = get_setting(config, "llm", "dynamic_selection")
        if isinstance(raw, str):
            self.auto = raw.strip().lower() == "auto"
            self.enabled = self.auto
        else:
            self.auto = False
            self.enabled = bool(raw)
        self.escalation: List[str] = list(
            get_setting(config, "llm", "escalation") or []
        )
        self.models = dict(get_setting(config, "llm", "models") or {})
        self.min_obs = get_setting(config, "llm", "min_observations")
        self.first_try_floor = get_setting(config, "llm", "first_try_floor")
        self.posture = get_setting(config, "llm", "selection_posture")
        self.maturity_threshold = get_setting(config, "llm", "maturity_threshold")
        # Cells already announced as graduated to cheap-first (one log per cell).
        self._graduated: set = set()
        free = list(free_models or [])
        if self.escalation and free:
            # Harvested free models join the configured cheapest tier as extra
            # candidates, so they are only ever tried on early (non-final)
            # attempts and under the same proven-first-try gate as any cheap
            # model. They go first so an exploration tie among unseen candidates
            # (equal +inf UCB) breaks toward the zero-cost option.
            cheapest = self.escalation[0]
            existing = self._tier_models(cheapest)
            self.models[cheapest] = [m for m in free if m not in existing] + existing
        elif not self.escalation and free:
            # No ladder configured: self-assemble one from the harvested free
            # models (cheap) and the default model (strong). This is what makes
            # free models work out of the box with no escalation/models config.
            default_model = get_setting(config, "llm", "model")
            self.models = {**self.models, "_free": free}
            if default_model:
                self.models["_default"] = [default_model]
                self.escalation = ["_free", "_default"]

    def _tier_models(self, tier: str) -> List[str]:
        v = self.models.get(tier)
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [m for m in v if isinstance(m, str)]

    def _rungs(self) -> List[str]:
        """Escalation tiers that actually resolve to at least one model."""
        return [t for t in self.escalation if self._tier_models(t)]

    def _cost(self, model: str, category: str, complexity: str) -> float:
        """Mean successful-attempt cost, or a neutral 1.0 when unknown.

        Unknown cost stays neutral so an unseen model is ranked purely on its
        (optimistic) quality score rather than being favored for a $0 estimate.
        """
        avg = self.ledger.stat(model, category, complexity).avg_cost
        return avg if avg > 0 else 1.0

    def _proven(self, model: str, category: str, complexity: str) -> bool:
        """True when a model has earned trust for a first attempt."""
        s = self.ledger.stat(model, category, complexity)
        return (
            s.first_try_attempts >= self.min_obs
            and s.first_try_rate >= self.first_try_floor
        )

    def _pick(
        self, models: List[str], category: str, complexity: str, explore: bool
    ) -> Optional[str]:
        """Best model by quality-per-dollar.

        explore=True uses the optimistic UCB score (unseen models score +inf, so
        they get tried); explore=False uses the conservative Wilson lower bound
        (unseen models score 0, so only proven models win).
        """
        if not models:
            return None
        total = self.ledger.total_observations(category, complexity)

        def value(model: str) -> float:
            if explore:
                score = self.ledger.selection_score(model, category, complexity, total)
            else:
                score = self.ledger.stat(model, category, complexity).quality_score()
            return score / self._cost(model, category, complexity)

        return max(models, key=value)

    def _mature(self, category: str, complexity: str) -> bool:
        """True once a cell has enough data to stop exploring (auto mode)."""
        return (
            self.ledger.total_observations(category, complexity)
            >= self.maturity_threshold
        )

    def _explore_on_first(self, category: str, complexity: str) -> bool:
        if self.auto:
            # Self-regulating: explore cheap/free models on easy tasks while the
            # cell is immature, then settle into conservative cheap-first.
            if self._mature(category, complexity):
                return False
            return complexity in ("trivial", "small", "medium")
        if self.posture == "aggressive":
            return True
        if self.posture == "balanced":
            return complexity in ("trivial", "small", "medium")
        return False  # conservative

    def select(
        self, category: str, complexity: str, attempt: int, max_attempts: int
    ) -> Optional[str]:
        """Choose a model id for this attempt, or None to use the default.

        Returns None when disabled or when no escalation ladder is configured,
        leaving the executor's existing static routing in charge.
        """
        if not self.enabled:
            return None
        rungs = self._rungs()
        if not rungs:
            return None
        last = len(rungs) - 1

        # The final attempt always uses the strongest tier: the quality floor's
        # safety net, so every prior attempt can afford to try something cheaper.
        if attempt >= max_attempts - 1:
            return self._pick(
                self._tier_models(rungs[last]), category, complexity, True
            )

        if attempt == 0 and not self._explore_on_first(category, complexity):
            # Conservative first attempt: cheapest rung with a proven model, else
            # the strongest tier. Never gamble the first impression on an
            # unproven cheap model.
            for tier in rungs:
                proven = [
                    m
                    for m in self._tier_models(tier)
                    if self._proven(m, category, complexity)
                ]
                if proven:
                    chosen = self._pick(proven, category, complexity, False)
                    self._announce_graduation(category, complexity, chosen)
                    return chosen
            return self._pick(
                self._tier_models(rungs[last]), category, complexity, True
            )

        # Climbing path: one rung up per attempt, capped at the strongest tier.
        target = min(attempt, last)
        return self._pick(self._tier_models(rungs[target]), category, complexity, True)

    def _announce_graduation(self, category: str, complexity: str, model: str) -> None:
        """Log the first time a cell settles on a proven cheap model (auto)."""
        if not self.auto:
            return
        cell = (category, complexity)
        if cell not in self._graduated:
            self._graduated.add(cell)
            logger.info(
                f"[auto] {category}/{complexity} now using proven cheap model "
                f"{model!r} on first attempt"
            )

    def is_ready(self, category: str, complexity: str) -> bool:
        """Whether this cell would use a proven cheap model on a first attempt."""
        if not self.enabled:
            return False
        return any(
            self._proven(m, category, complexity)
            for tier in self._rungs()[:-1]
            for m in self._tier_models(tier)
        )
