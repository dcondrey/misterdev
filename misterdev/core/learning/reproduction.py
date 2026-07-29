"""The reproduction corpus — a growing record of unfakeable behavioral cases.

A self-improving system evolves only as far as its supply of ground truth it
cannot cheat. The benchmark is that ground truth (real exercises, real graders),
but it was used only as an all-or-nothing aggregate: one whole-suite run per
candidate, minutes and dollars each, most single-edit mutations landing inside
the noise band and registering as *nothing*. That coarse, expensive signal is the
true ceiling on the evolutionary search — selection is starved, so the loop takes
almost no effective steps.

This corpus turns that same ground truth into a DENSE, CHEAP, TARGETED signal. It
records, per case, the outcome history across runs, so at any moment it can name:

* the **currently-failing** cases in a niche — the exact fitness targets a
  mutation must flip red→green, and
* a **guard sample** of currently-passing cases — the regression backstop.

A micro-evaluator (see :mod:`misterdev.core.evolution.screen`) then runs only
those few cases instead of the whole suite, so hundreds of mutations can be
screened for the cost of one old evaluation. The corpus also *accumulates*: every
run folds its per-case outcomes in, so the ground truth grows with use rather than
being re-derived, and a case that keeps failing (high ``fail_streak``) is a
standing target worth more search than a flaky one-off.

Persistence and failure handling mirror the rest of the learning substrate: one
JSON file, atomic write, best-effort load that degrades to empty. The corpus is a
search accelerator, never a correctness authority — the full benchmark remains the
oracle that confirms a screened survivor.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from misterdev.core.execution.error_classifier import classify_error
from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write

logger = setup_logger(__name__)

_MAX_CASES = (
    2000  # bound the corpus; a benchmark has far fewer, this is a runaway guard
)


@dataclass
class Case:
    """One behavioral case (a benchmark instance) and its outcome history.

    ``fail_streak`` / ``pass_streak`` are consecutive counts up to the latest run,
    so a persistently-failing case (high fail_streak) can be prioritized as a
    target over a case that failed once and recovered.
    """

    id: str
    language: str = "unknown"
    category: str = ""  # classified error category when failing (e.g. "wrong_type")
    resolved: bool = True  # outcome in the most recent run that included this case
    runs: int = 0  # times this case has been observed
    fail_streak: int = 0  # consecutive failing runs up to the latest
    pass_streak: int = 0  # consecutive passing runs up to the latest
    first_run: int = 0
    last_run: int = 0

    @property
    def niche(self) -> str:
        """The case's behavioral niche: ``language`` or ``language/category``."""
        return f"{self.language}/{self.category}" if self.category else self.language


class ReproductionCorpus:
    """Persistent per-case outcome history backing cheap, targeted micro-eval."""

    def __init__(self, path: Path):
        self.path = Path(path)

    # -- persistence -------------------------------------------------------
    def _load(self) -> Dict[str, Case]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        cases: Dict[str, Case] = {}
        for item in raw.get("cases", []) if isinstance(raw, dict) else []:
            if isinstance(item, dict) and item.get("id"):
                cases[str(item["id"])] = Case(
                    id=str(item["id"]),
                    language=str(item.get("language", "unknown")),
                    category=str(item.get("category", "")),
                    resolved=bool(item.get("resolved", True)),
                    runs=int(item.get("runs", 0)),
                    fail_streak=int(item.get("fail_streak", 0)),
                    pass_streak=int(item.get("pass_streak", 0)),
                    first_run=int(item.get("first_run", 0)),
                    last_run=int(item.get("last_run", 0)),
                )
        return cases

    def _run_counter(self) -> int:
        if not self.path.exists():
            return 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return int(raw.get("run", 0)) if isinstance(raw, dict) else 0
        except (json.JSONDecodeError, OSError):
            return 0

    def _load_with_run(self):
        if not self.path.exists():
            return {}, 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, 0
        run = int(raw.get("run", 0)) if isinstance(raw, dict) else 0
        cases: Dict[str, Case] = {}
        for item in raw.get("cases", []) if isinstance(raw, dict) else []:
            if isinstance(item, dict) and item.get("id"):
                cases[str(item["id"])] = Case(
                    id=str(item["id"]),
                    language=str(item.get("language", "unknown")),
                    category=str(item.get("category", "")),
                    resolved=bool(item.get("resolved", True)),
                    runs=int(item.get("runs", 0)),
                    fail_streak=int(item.get("fail_streak", 0)),
                    pass_streak=int(item.get("pass_streak", 0)),
                    first_run=int(item.get("first_run", 0)),
                    last_run=int(item.get("last_run", 0)),
                )
        return cases, run

    def _save(self, cases: Dict[str, Case], run: int) -> None:
        # Evict low-value cases first: a persistently-failing case (high
        # fail_streak) is worth more corpus slots than a stale passing one.
        # Primary key: fail_streak (desc) so high-streak targets survive longest.
        # Secondary key: last_run (desc) breaks streak ties by recency.
        ordered = sorted(
            cases.values(), key=lambda c: (c.fail_streak, c.last_run), reverse=True
        )
        kept = ordered[:_MAX_CASES]
        payload = {
            "run": run,
            "cases": [asdict(c) for c in kept],
        }
        atomic_write(self.path, json.dumps(payload, indent=2))

    # -- accumulation ------------------------------------------------------
    def update(self, results) -> int:
        """Fold a benchmark run's per-instance results into the corpus.

        ``results`` are duck-typed on ``.name``/``.language``/``.resolved`` and,
        when failing, ``.error``/``.output`` (to classify the niche). Returns the
        run number recorded. Best-effort: any error is logged and the corpus is
        left unchanged rather than corrupted.
        """
        try:
            cases, _prior_run = self._load_with_run()
            run = _prior_run + 1
            for r in results:
                cid = str(getattr(r, "name", "") or getattr(r, "id", "") or "")
                if not cid:
                    continue
                resolved = bool(getattr(r, "resolved", False))
                lang = str(getattr(r, "language", None) or "unknown")
                case = cases.get(cid)
                if case is None:
                    case = Case(id=cid, language=lang, first_run=run)
                    cases[cid] = case
                case.language = lang
                case.resolved = resolved
                case.runs += 1
                case.last_run = run
                if resolved:
                    case.pass_streak += 1
                    case.fail_streak = 0
                    case.category = ""
                else:
                    case.fail_streak += 1
                    case.pass_streak = 0
                    err = str(getattr(r, "error", "") or getattr(r, "output", "") or "")
                    case.category = classify_error(err) if err else case.category
            self._save(cases, run)
            return run
        except (OSError, ValueError) as e:
            logger.warning(f"Reproduction corpus update failed (non-fatal): {e}")
            return self._run_counter()

    # -- selection ---------------------------------------------------------
    def failing(
        self, niche: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Case]:
        """Currently-failing cases, most-persistently-failing first.

        With ``niche`` (a language or ``language/category``), restricts to cases
        whose niche matches or is prefixed by it, so ``"rust"`` selects every rust
        failure and ``"rust/wrong_type"`` narrows to that category. These are the
        fitness targets a mutation must flip.
        """
        out = [c for c in self._load().values() if not c.resolved]
        if niche:
            out = [
                c
                for c in out
                if c.niche == niche
                or c.niche.startswith(niche + "/")
                or c.language == niche
            ]
        out.sort(key=lambda c: (c.fail_streak, c.runs), reverse=True)
        return out[:limit] if limit else out

    def guard_sample(
        self, k: int, exclude: Optional[Set[str]] = None, seed: int = 0
    ) -> List[Case]:
        """A deterministic sample of ``k`` currently-passing cases — the regression
        guard a mutation must not break. Sampling is stable for a given ``seed`` and
        corpus (no RNG: a deterministic stride over id-sorted passing cases), so a
        screen is reproducible. ``exclude`` drops ids already targeted."""
        exclude = exclude or set()
        passing = sorted(
            (c for c in self._load().values() if c.resolved and c.id not in exclude),
            key=lambda c: c.id,
        )
        if k <= 0 or not passing:
            return []
        if k >= len(passing):
            return passing
        # Deterministic evenly-spaced stride, offset by seed, so the guard covers
        # the id space rather than clustering, and is identical across screen runs.
        n = len(passing)
        step = n / k
        return [passing[int((i * step + seed) % n)] for i in range(k)]

    def known_case_ids(self) -> Set[str]:
        return set(self._load().keys())

    def stats(self) -> Dict[str, int]:
        cases = list(self._load().values())
        failing = sum(1 for c in cases if not c.resolved)
        return {"total": len(cases), "failing": failing}

    def resolve_rate(self, niche: Optional[str] = None) -> Optional[float]:
        """The WITHOUT-tool baseline: the fraction of cases currently resolved,
        optionally scoped to a niche (a language, or ``language/category``).

        Returns None when no case matches, so a caller falls back to a prior
        rather than trusting an empty baseline. This is exactly the counterfactual
        a tool's with-tool resolve-rate must beat to be promoted (see
        ``core.evolution.tool_corpus.promote_from_corpus``)."""
        cases = list(self._load().values())
        if niche:
            cases = [
                c
                for c in cases
                if c.language == niche
                or c.niche == niche
                or c.niche.startswith(niche + "/")
            ]
        if not cases:
            return None
        return sum(1 for c in cases if c.resolved) / len(cases)
