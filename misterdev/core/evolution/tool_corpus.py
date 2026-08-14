"""The tool corpus — passive accumulation of self-authored tools and the task
outcomes they rode along with, closing the two-timescale loop
(``docs/two-timescale-evolution.md``).

The runtime layer (:mod:`.tool_invention`) lets the model author a tool mid-task;
live-SWE-agent throws that tool away when the task ends. This corpus keeps it:
every invented tool is recorded with the task it ran on and whether that task
resolved. The signal is a FREE byproduct of normal operation — no dedicated,
paid A/B runs — which is the only economics a self-improvement loop can afford.
As runs accumulate, each tool's per-task outcome history becomes dense enough for
:func:`promote_from_corpus` to decide, through the SAME held-out generalization
gate as scaffold evolution, which tools become permanent
:class:`~.tool_library.ToolLibrary` entries that seed future runs.

Honesty about the signal: a corpus outcome is CORRELATIONAL — the tool was present
when the task resolved, not proven to have caused it — and the without-tool
baseline is supplied by the caller (from the reproduction corpus or a prior), not
fabricated here. So promotion is a defensible *ranking* of tools by held-out
association, sharpening as data grows; it is not a controlled causal A/B. Per-task
regressions are not attributed (no per-task counterfactual), so the promotion is
association-positive, regression-agnostic — documented, not hidden.

Persistence mirrors the reproduction corpus: one JSON file, atomic write,
best-effort load that degrades to empty.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write, flock_guarded

logger = setup_logger(__name__)

_MAX_TOOLS = 1000  # runaway guard; a real corpus is far smaller


def _tool_id(source: str) -> str:
    """A stable short id for a tool, keyed on its normalized source (so the same
    tool re-authored with trivial whitespace differences collapses to one)."""
    normalized = "\n".join(line.rstrip() for line in source.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass
class ToolRecord:
    """One self-authored tool and its per-task outcome history."""

    tool_id: str
    niche: str
    source: str
    outcomes: Dict[str, bool] = field(default_factory=dict)  # task_id -> resolved
    first_run: int = 0
    last_run: int = 0

    def score_on(self, task_ids) -> Tuple[int, int]:
        """(resolved_count, total) over the given task ids present in this record."""
        rel = [self.outcomes[t] for t in task_ids if t in self.outcomes]
        return sum(1 for x in rel if x), len(rel)


class ToolCorpus:
    """Persistent per-tool outcome history backing held-out tool promotion."""

    def __init__(self, path):
        self.path = Path(path)

    def _load(self) -> Tuple[Dict[str, ToolRecord], int]:
        if not self.path.exists():
            return {}, 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, 0
        if not isinstance(raw, dict):
            return {}, 0
        run = int(raw.get("run", 0))
        recs: Dict[str, ToolRecord] = {}
        for item in raw.get("tools", []):
            if not isinstance(item, dict) or not item.get("tool_id"):
                continue
            try:
                recs[str(item["tool_id"])] = ToolRecord(
                    tool_id=str(item["tool_id"]),
                    niche=str(item.get("niche", "")),
                    source=str(item.get("source", "")),
                    outcomes={
                        str(k): bool(v) for k, v in (item.get("outcomes") or {}).items()
                    },
                    first_run=int(item.get("first_run", 0)),
                    last_run=int(item.get("last_run", 0)),
                )
            except (TypeError, ValueError):
                continue
        return recs, run

    def _save(self, recs: Dict[str, ToolRecord], run: int) -> None:
        ordered = sorted(recs.values(), key=lambda r: r.last_run, reverse=True)
        kept = ordered[:_MAX_TOOLS]
        payload = {"run": run, "tools": [asdict(r) for r in kept]}
        try:
            atomic_write(self.path, json.dumps(payload, indent=2))
        except OSError as e:  # a learning stream never fails a build
            logger.warning(f"Tool corpus save failed (non-fatal): {e}")

    def record(self, source: str, niche: str, task_id: str, resolved: bool) -> str:
        """Fold one (tool, task, outcome) observation in. Returns the tool id.

        Best-effort: a persistence failure is logged, never raised. Last outcome
        wins for a repeated (tool, task) pair, so a re-run updates rather than
        double-counts.
        """
        if not source or not source.strip() or not task_id:
            return ""
        tid = _tool_id(source)
        with flock_guarded(self.path):
            recs, run = self._load()
            run += 1
            rec = recs.get(tid)
            if rec is None:
                rec = ToolRecord(tool_id=tid, niche=niche, source=source, first_run=run)
                recs[tid] = rec
            rec.niche = niche or rec.niche
            rec.outcomes[task_id] = resolved
            rec.last_run = run
            self._save(recs, run)
        return tid

    def records(self) -> List[ToolRecord]:
        return list(self._load()[0].values())

    def stats(self) -> Dict[str, int]:
        recs = self.records()
        obs = sum(len(r.outcomes) for r in recs)
        return {"tools": len(recs), "observations": obs}


def promote_from_corpus(
    corpus: ToolCorpus,
    library,
    *,
    baseline_rate,
    min_observations: int = 5,
    holdout_fraction: float = 0.3,
    noise_band: float = 0.0,
) -> List[str]:
    """Promote corpus tools that GENERALIZE into the library; return promoted ids.

    For each tool with at least ``min_observations`` task outcomes, split its tasks
    into disjoint DERIVE/HOLDOUT pools (stable per-task hash), score the tool's
    with-tool resolve-rate on each, and admit it through the library's held-out
    gate against the WITHOUT-tool baseline. A tool is promoted only if its
    association with success holds on the tasks it was NOT selected on — the same
    anti-overfit ratchet scaffold evolution uses.

    ``baseline_rate`` is the without-tool resolve-rate: a float applied to every
    tool, or a callable ``baseline(niche) -> float`` so the baseline can come
    per-niche from the reproduction corpus (see ``tool_promotion``). Pure: no I/O
    beyond the corpus/library the caller passes.
    """
    from .fitness import FitnessScore
    from .holdout import split_tasks
    from .tool_library import ToolCandidate

    baseline_of = (
        baseline_rate if callable(baseline_rate) else (lambda _n: baseline_rate)
    )
    promoted: List[str] = []
    for rec in corpus.records():
        base = max(0.0, min(1.0, float(baseline_of(rec.niche))))
        task_ids = sorted(rec.outcomes)
        if len(task_ids) < min_observations:
            continue
        derive_ids, holdout_ids = split_tasks(task_ids, holdout_fraction)
        dr, dt = rec.score_on(derive_ids)
        hr, ht = rec.score_on(holdout_ids)
        if dt == 0 or ht == 0:
            continue
        candidate = ToolCandidate(
            id=rec.tool_id,
            niche=rec.niche,
            source=rec.source,
            resolved=dr,
            total=dt,
            cost=0.0,
            provenance=f"corpus:{len(task_ids)} tasks",
        )
        decision = library.consider(
            candidate,
            derive=FitnessScore(dr, dt, 0.0),
            derive_base=FitnessScore(round(base * dt), dt, 0.0),
            holdout=FitnessScore(hr, ht, 0.0),
            holdout_base=FitnessScore(round(base * ht), ht, 0.0),
        )
        if decision.promote:
            promoted.append(rec.tool_id)
    return promoted
