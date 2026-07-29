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

from misterdev.config import get_setting
from misterdev.core.economics.model_ledger import ModelLedger
from misterdev.logging_setup import setup_logger

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
        self.min_obs = get_setting(config, "llm", "min_observations") or 3
        self.first_try_floor = get_setting(config, "llm", "first_try_floor")
        self.incompetence_floor = get_setting(config, "llm", "incompetence_floor")
        self.max_latency = get_setting(config, "llm", "max_attempt_latency_seconds")
        self.max_edit_fail_rate = get_setting(config, "llm", "max_edit_fail_rate")
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

    # Empirical-Bayes prior weight: how many pseudo-observations of the model's
    # GLOBAL first-try rate to fold into a cold cell. Small, so cell-specific data
    # overrides it within a few real attempts — the prior only warm-starts.
    _PRIOR_WEIGHT = 3.0

    def _proven(self, model: str, category: str, complexity: str) -> bool:
        """True when a model has earned trust for a first attempt on this cell.

        A cold cell borrows strength from the model's GLOBAL first-try rate
        (shrinkage): a model reliable everywhere else clears the floor here with
        fewer local observations, while a globally-weak model does not. As the
        cell accumulates real attempts, the blended rate converges to the cell's
        own — the prior fades. A model with no history anywhere stays unproven.
        """
        s = self.ledger.stat(model, category, complexity)
        g_att, g_rate = self.ledger.global_first_try(model, category, complexity)
        prior = self._PRIOR_WEIGHT if g_att >= self.min_obs else 0.0
        effective_obs = s.first_try_attempts + prior
        if effective_obs < self.min_obs:
            return False
        blended = (s.first_try_successes + prior * g_rate) / effective_obs
        return blended >= self.first_try_floor

    def _incompetent(self, model: str, category: str, complexity: str) -> bool:
        """Proven UNABLE to do this kind of task: enough attempts to judge and a
        success rate below the floor. Distinct from unproven (too few attempts).
        Trying such a model only burns a failed attempt and forces escalation."""
        if not self.incompetence_floor:
            return False
        s = self.ledger.stat(model, category, complexity)
        return s.attempts >= self.min_obs and s.success_rate < self.incompetence_floor

    def _too_slow(self, model: str, category: str, complexity: str) -> bool:
        """Proven too slow for the per-task wall-clock budget (ledger avg latency
        above the configured ceiling). Unseen models (avg 0) are never flagged."""
        if not self.max_latency:
            return False
        return (
            self.ledger.stat(model, category, complexity).avg_latency > self.max_latency
        )

    def _edit_unreliable(self, model: str, category: str, complexity: str) -> bool:
        """Proven to produce unanchorable edits: edit-apply-failure rate above the
        configured ceiling, with enough attempts to judge. Unseen models return False."""
        if not self.max_edit_fail_rate:
            return False
        s = self.ledger.stat(model, category, complexity)
        return s.attempts >= self.min_obs and s.edit_fail_rate > self.max_edit_fail_rate

    # Free (`:free`) endpoints are reserved for the easiest tasks: they are slow
    # (2.5-5 min/call observed) and unreliable, so on anything heavier than a
    # small task they cost more wall-clock and failed attempts than they save.
    _FREE_OK_COMPLEXITY = ("trivial", "small")

    def _pick(
        self,
        models: List[str],
        category: str,
        complexity: str,
        explore: bool,
        final: bool = False,
    ) -> Optional[str]:
        """Best model by quality-per-dollar.

        explore=True uses the optimistic UCB score (unseen models score +inf, so
        they get tried); explore=False uses the conservative Wilson lower bound
        (unseen models score 0, so only proven models win). Free endpoints are
        dropped for non-easy tasks (see ``_FREE_OK_COMPLEXITY``), and models the
        ledger has proven incompetent (or, on non-final attempts, too slow) are
        dropped; if that empties the tier, returns None so the caller climbs to
        the next (paid) rung. Incompetence still filters on the final attempt — a
        model that cannot do the task is never the right last resort.
        """
        if complexity not in self._FREE_OK_COMPLEXITY:
            models = [m for m in models if ":free" not in m]
        models = [m for m in models if not self._incompetent(m, category, complexity)]
        if not final:
            models = [m for m in models if not self._too_slow(m, category, complexity)]
            models = [
                m for m in models if not self._edit_unreliable(m, category, complexity)
            ]
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
            # Only trivial/small tasks are explored on the FIRST attempt: on a
            # large repo, unproven free models returned no usable edits on medium
            # refactor tasks (the emathy run burned first attempts on null
            # responses). A cheap model can still earn medium first-attempt use
            # once PROVEN (the conservative branch picks a proven cheap model).
            if self._mature(category, complexity):
                return False
            return complexity in ("trivial", "small")
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
        # final=True lifts the latency guard here (a slow-but-capable last resort
        # beats giving up), but incompetent models are still excluded.
        if attempt >= max_attempts - 1:
            pick = self._pick(
                self._tier_models(rungs[last]), category, complexity, True, final=True
            )
            if pick is not None:
                return pick
            # The strongest tier is empty for this cell (every model in it is proven
            # incompetent). Do NOT silently fall back to the client default on the
            # last resort — widen to the best usable model across ALL tiers and warn.
            wider = [m for r in rungs for m in self._tier_models(r)]
            pick = self._pick(wider, category, complexity, True, final=True)
            if pick is not None:
                logger.warning(
                    "Model selector: strongest tier has no usable model for (%s, %s) "
                    "on the final attempt; using %s from a wider set.",
                    category,
                    complexity,
                    pick,
                )
            else:
                logger.warning(
                    "Model selector: every configured model is excluded for (%s, %s) "
                    "on the final attempt; falling back to the client default.",
                    category,
                    complexity,
                )
            return pick

        if attempt == 0 and not self._explore_on_first(category, complexity):
            # Conservative first attempt: cheapest rung with a proven model, else
            # the strongest NORMAL-work tier. Never gamble the first impression on
            # an unproven cheap model — but don't spend the final-attempt safety
            # net on a first try either. When the ladder has a dedicated top rung
            # (>= 3 rungs, e.g. cheap -> mid -> frontier), that top rung is
            # reserved for the actual final attempt, so cold-start falls back to
            # the second-strongest (the mid ceiling). With <= 2 rungs the top rung
            # IS the normal default, so cold-start uses it (unchanged behavior).
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
            cold = last if len(rungs) <= 2 else last - 1
            return self._pick(
                self._tier_models(rungs[cold]), category, complexity, True
            )

        # Climbing path: one rung up per attempt, capped at the strongest tier.
        # If the target rung is emptied by the free/incompetence/latency filters,
        # climb to the next (stronger) rung rather than fall back to the client
        # default — a filtered tier should escalate, not disable selection.
        target = min(attempt, last)
        return self._pick_climbing(rungs, target, category, complexity)

    def _pick_climbing(
        self, rungs: List[str], start: int, category: str, complexity: str
    ) -> Optional[str]:
        """Pick from rung ``start``, climbing to each stronger rung in turn until
        one yields a model the filters didn't drop; None only when every rung
        from ``start`` up is empty."""
        for i in range(start, len(rungs)):
            chosen = self._pick(self._tier_models(rungs[i]), category, complexity, True)
            if chosen:
                return chosen
        return None

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
