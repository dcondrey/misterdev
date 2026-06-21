"""Persistent model-performance ledger.

Records per-(model, category, complexity) outcomes across runs so model
selection can be data-driven instead of static or random. Stored as JSON under
``.orchestrator/model_stats.json`` alongside the other run artifacts.

The ledger only stores and scores; the selection policy lives in
``core.model_selector``. Quality is scored by the Wilson lower confidence bound
on the gate-pass rate (a conservative estimate that does not let a 1/1 model
outrank a 95/100 one), and exploration of under-sampled models is handled by an
optimistic UCB term so new or freshly-rotated models still get tried.
"""

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# z for a ~95% confidence interval, used by both the Wilson lower bound
# (conservative quality) and the UCB exploration bonus (optimistic).
_Z = 1.96

# Field separator for the composite stat key. Unit-separator char: never
# appears in a model id, category, or complexity.
_SEP = "␟"


@dataclass
class ModelStat:
    """Aggregated outcomes for one (model, category, complexity) cell."""

    model: str
    category: str = ""
    complexity: str = ""
    attempts: int = 0
    successes: int = 0  # attempts that passed the validation gates
    first_try_attempts: int = 0
    first_try_successes: int = 0
    aborts: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    last_seen: float = 0.0  # epoch seconds of the most recent record

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def first_try_rate(self) -> float:
        return (
            self.first_try_successes / self.first_try_attempts
            if self.first_try_attempts
            else 0.0
        )

    @property
    def abort_rate(self) -> float:
        return self.aborts / self.attempts if self.attempts else 0.0

    @property
    def avg_cost(self) -> float:
        """Mean dollar cost of a successful attempt (0.0 with no successes)."""
        return self.total_cost / self.successes if self.successes else 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.attempts if self.attempts else 0.0

    def quality_score(self) -> float:
        """Wilson lower bound on the gate-pass rate.

        A conservative point estimate of quality: under-sampling widens the
        interval and pulls the score down, so a model must earn its ranking
        with volume, not a lucky single success.
        """
        n = self.attempts
        if n == 0:
            return 0.0
        p = self.successes / n
        z2 = _Z * _Z
        denom = 1 + z2 / n
        center = p + z2 / (2 * n)
        margin = _Z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
        return max(0.0, (center - margin) / denom)


class ModelLedger:
    """Thread-safe, file-backed store of per-model outcome statistics."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._stats: Dict[str, ModelStat] = {}
        self.load()

    @staticmethod
    def _key(model: str, category: str, complexity: str) -> str:
        return f"{model}{_SEP}{category}{_SEP}{complexity}"

    def load(self) -> None:
        """Load stats from disk, tolerating a missing or corrupt file."""
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Model ledger at {self.path} unreadable, starting fresh: {e}"
            )
            return
        valid = {f.name for f in fields(ModelStat)}
        with self._lock:
            self._stats = {}
            for key, data in (raw or {}).items():
                if not isinstance(data, dict):
                    continue
                # Drop unknown keys so an older/newer schema still loads.
                clean = {k: v for k, v in data.items() if k in valid}
                try:
                    self._stats[key] = ModelStat(**clean)
                except TypeError:
                    continue

    def save(self) -> None:
        """Persist stats atomically (write-temp-then-rename)."""
        with self._lock:
            snapshot = {k: asdict(v) for k, v in self._stats.items()}
        from my_project_orchestrator.utils.file_utils import ensure_artifact_dir

        ensure_artifact_dir(self.path.parent)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def stat(self, model: str, category: str = "", complexity: str = "") -> ModelStat:
        """Return the stat cell for a key, creating an empty one if absent."""
        key = self._key(model, category, complexity)
        with self._lock:
            if key not in self._stats:
                self._stats[key] = ModelStat(
                    model=model, category=category, complexity=complexity
                )
            return self._stats[key]

    def record(
        self,
        model: str,
        category: str = "",
        complexity: str = "",
        *,
        success: bool,
        first_try: bool = False,
        aborted: bool = False,
        cost: float = 0.0,
        latency: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> ModelStat:
        """Record one attempt's outcome and persist the ledger.

        ``success`` means the attempt passed the validation gates. ``first_try``
        marks attempt 0 so first-try success can be tracked separately (it is the
        signal that decides whether a cheap model is worth trying first).
        """
        ts = time.time() if timestamp is None else timestamp
        key = self._key(model, category, complexity)
        with self._lock:
            s = self._stats.get(key)
            if s is None:
                s = ModelStat(model=model, category=category, complexity=complexity)
                self._stats[key] = s
            s.attempts += 1
            if success:
                s.successes += 1
                s.total_cost += cost
            if first_try:
                s.first_try_attempts += 1
                if success:
                    s.first_try_successes += 1
            if aborted:
                s.aborts += 1
            s.total_latency += latency
            s.last_seen = ts
        self.save()
        return s

    def selection_score(
        self, model: str, category: str, complexity: str, total_observations: int
    ) -> float:
        """Optimistic quality estimate for selection (UCB).

        Wilson-lower-bound quality plus an exploration bonus that shrinks as a
        model accrues attempts. Under-sampled or freshly-rotated models score
        high enough to be tried without being chosen blindly forever. Returns
        +inf for a never-seen model (textbook UCB) so it is always explored at
        least once before the policy can settle on a proven model.
        """
        s = self.stat(model, category, complexity)
        if s.attempts == 0:
            return float("inf")
        bonus = _Z * math.sqrt(math.log(max(total_observations, 2)) / s.attempts)
        return s.quality_score() + bonus

    def total_observations(self, category: str = "", complexity: str = "") -> int:
        """Total recorded attempts, optionally scoped to a cell's context."""
        with self._lock:
            return sum(
                s.attempts
                for s in self._stats.values()
                if (not category or s.category == category)
                and (not complexity or s.complexity == complexity)
            )

    def known_models(self) -> List[str]:
        with self._lock:
            return sorted({s.model for s in self._stats.values()})
