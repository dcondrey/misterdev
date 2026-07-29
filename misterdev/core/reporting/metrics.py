"""Append-only structured metrics log for build outcomes.

Writes one JSON line per completed build to ``.orchestrator/metrics.jsonl``,
creating a queryable time-series of cost, task counts, and success rates
that outlives individual build reports. Suitable for piping into ``jq``,
importing into a dashboard, or running ``misterdev metrics`` trends.

Write errors are swallowed so a disk issue never aborts a build.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_METRICS_FILENAME = "metrics.jsonl"


def append_build_metrics(
    project_path: Path,
    *,
    goal: str = "",
    tasks_completed: int = 0,
    tasks_failed: int = 0,
    tasks_deferred: int = 0,
    llm_cost: float = 0.0,
    llm_calls: int = 0,
    duration_seconds: float = 0.0,
    validation_passed: Optional[bool] = None,
) -> None:
    """Append one build-outcome record to the metrics log (best-effort)."""
    record: Dict[str, Any] = {
        "ts": time.time(),
        "goal": goal[:200],
        "tasks_completed": tasks_completed,
        "tasks_failed": tasks_failed,
        "tasks_deferred": tasks_deferred,
        "llm_cost": round(llm_cost, 6),
        "llm_calls": llm_calls,
        "duration_seconds": round(duration_seconds, 2),
        "success": validation_passed is True,
    }
    dest = Path(project_path) / ".orchestrator" / _METRICS_FILENAME
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
    except OSError:
        pass


def read_build_metrics(project_path: Path) -> List[Dict[str, Any]]:
    """Return all build-outcome records from the metrics log, oldest first."""
    src = Path(project_path) / ".orchestrator" / _METRICS_FILENAME
    records: List[Dict[str, Any]] = []
    try:
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    except OSError:
        pass
    return records


def summarize_metrics(project_path: Path) -> Dict[str, Any]:
    """Aggregate build metrics into a summary dict for the ``report`` command."""
    records = read_build_metrics(project_path)
    if not records:
        return {"builds": 0, "last_build_success": None}
    n = len(records)
    successes = sum(1 for r in records if r.get("success"))
    total_cost = sum(r.get("llm_cost", 0.0) for r in records)
    total_tasks = sum(r.get("tasks_completed", 0) for r in records)
    last = records[-1]
    return {
        "builds": n,
        "success_rate": round(successes / n, 3),
        "total_cost": round(total_cost, 4),
        "avg_cost_per_build": round(total_cost / n, 4) if n else 0.0,
        "total_tasks_completed": total_tasks,
        "last_build_ts": last.get("ts"),
        "last_build_success": last.get("success"),
    }
