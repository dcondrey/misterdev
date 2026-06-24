"""Aggregated, read-only view over a project's ``.orchestrator/`` artifacts.

The orchestrator writes three streams under ``.orchestrator/``: an append-only
audit trail (``audit.jsonl``), the persistent model-performance ledger
(``model_stats.json``), and per-build reports (``reports/report_*.json``). Each
is observability that, until now, had no consolidated reader. This module
summarizes all three into plain dicts so a ``report`` command (or any caller) can
show what happened, what each model actually cost, and how the last build went —
without re-running anything. Pure and defensive: a missing or malformed file
yields an empty/None summary, never an error.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_ARTIFACT_DIR = ".orchestrator"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of objects, skipping unreadable lines."""
    events: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    events.append(obj)
    except OSError:
        return []
    return events


def summarize_audit(audit_path: Path) -> Dict[str, Any]:
    """Aggregate the audit trail: event counts, command pass/fail, edits, gates.

    Returns zeros/empties when the file is absent or empty, so the caller can
    render a consistent shape regardless.
    """
    events = _read_jsonl(Path(audit_path))
    by_type: Dict[str, int] = {}
    cmd_ok = cmd_failed = 0
    edits: Dict[str, int] = {}
    gov_escalated = gov_blocked = 0
    for e in events:
        etype = str(e.get("type", "?"))
        by_type[etype] = by_type.get(etype, 0) + 1
        if etype == "command":
            if e.get("ok"):
                cmd_ok += 1
            else:
                cmd_failed += 1
        elif etype == "edit":
            path = str(e.get("path", "?"))
            edits[path] = edits.get(path, 0) + 1
        elif etype == "gate":
            if e.get("escalated"):
                gov_escalated += 1
            if e.get("allowed") is False:
                gov_blocked += 1
    return {
        "total_events": len(events),
        "by_type": by_type,
        "commands": {"ok": cmd_ok, "failed": cmd_failed},
        "edits": {"total": sum(edits.values()), "by_file": edits},
        "governance": {"escalated": gov_escalated, "blocked": gov_blocked},
    }


def summarize_models(ledger_path: Path) -> List[Dict[str, Any]]:
    """Per-model performance from the ledger, aggregated across category/complexity.

    Surfaces the data that drives selection — attempts, gate-pass rate, first-try
    rate, and mean cost of a success — so a previously-invisible model choice
    becomes legible. Sorted by attempts (most-exercised first). Empty when the
    ledger is absent.
    """
    path = Path(ledger_path)
    if not path.exists():
        return []
    from my_project_orchestrator.core.model_ledger import ModelLedger

    ledger = ModelLedger(path)
    agg: Dict[str, Dict[str, float]] = {}
    for s in ledger.all_stats():
        a = agg.setdefault(
            s.model,
            {
                "attempts": 0.0,
                "successes": 0.0,
                "first_try_attempts": 0.0,
                "first_try_successes": 0.0,
                "total_cost": 0.0,
            },
        )
        a["attempts"] += s.attempts
        a["successes"] += s.successes
        a["first_try_attempts"] += s.first_try_attempts
        a["first_try_successes"] += s.first_try_successes
        a["total_cost"] += s.total_cost
    rows: List[Dict[str, Any]] = []
    for model, a in agg.items():
        att = a["attempts"]
        fta = a["first_try_attempts"]
        succ = a["successes"]
        rows.append(
            {
                "model": model,
                "attempts": round(att, 1),
                "success_rate": (succ / att) if att else 0.0,
                "first_try_rate": (a["first_try_successes"] / fta) if fta else 0.0,
                "avg_cost": (a["total_cost"] / succ) if succ else 0.0,
            }
        )
    rows.sort(key=lambda r: r["attempts"], reverse=True)
    return rows


def latest_report(reports_dir: Path) -> Optional[Dict[str, Any]]:
    """The most recent saved build report (JSON), or None.

    Report filenames are timestamp-stamped (``report_YYYYMMDD_HHMMSS.json``), so
    a lexical sort is chronological and the last entry is the newest.
    """
    d = Path(reports_dir)
    if not d.is_dir():
        return None
    candidates = sorted(d.glob("report_*.json"))
    if not candidates:
        return None
    try:
        obj = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def collect(project_path: Path) -> Dict[str, Any]:
    """Collect the audit, model, and latest-report summaries for a project."""
    root = Path(project_path) / _ARTIFACT_DIR
    return {
        "audit": summarize_audit(root / "audit.jsonl"),
        "models": summarize_models(root / "model_stats.json"),
        "latest_report": latest_report(root / "reports"),
    }
