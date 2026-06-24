import json
import tempfile
from pathlib import Path

from my_project_orchestrator.core.report_view import (
    collect,
    summarize_audit,
    summarize_models,
    latest_report,
)
from my_project_orchestrator.core.model_ledger import ModelLedger


def _orch(tmp):
    d = Path(tmp) / ".orchestrator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_summarize_audit_absent_file_is_empty():
    with tempfile.TemporaryDirectory() as td:
        s = summarize_audit(Path(td) / "audit.jsonl")
        assert s["total_events"] == 0
        assert s["commands"] == {"ok": 0, "failed": 0}


def test_summarize_audit_counts_events():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "audit.jsonl"
        lines = [
            {"type": "command", "command": "pytest", "ok": True},
            {"type": "command", "command": "ruff", "ok": False},
            {"type": "edit", "path": "a.py"},
            {"type": "edit", "path": "a.py"},
            {"type": "edit", "path": "b.py"},
            {"type": "gate", "action": "rm", "allowed": False, "escalated": True},
        ]
        p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
        s = summarize_audit(p)
        assert s["total_events"] == 6
        assert s["commands"] == {"ok": 1, "failed": 1}
        assert s["edits"]["total"] == 3
        assert s["edits"]["by_file"]["a.py"] == 2
        assert s["governance"] == {"escalated": 1, "blocked": 1}


def test_summarize_audit_skips_malformed_lines():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "audit.jsonl"
        p.write_text(
            '{"type": "command", "ok": true}\n{not json}\n\n["wrong shape"]\n',
            encoding="utf-8",
        )
        s = summarize_audit(p)
        # Only the one valid object dict counts.
        assert s["total_events"] == 1


def test_summarize_models_aggregates_across_contexts():
    with tempfile.TemporaryDirectory() as td:
        ledger_path = Path(td) / "model_stats.json"
        led = ModelLedger(ledger_path)
        led.record(
            "cheap/m",
            "feature",
            "small",
            success=True,
            first_try=True,
            cost=0.01,
            timestamp=1.0,
        )
        led.record("cheap/m", "test", "large", success=False, timestamp=2.0)
        led.record(
            "strong/m", "feature", "large", success=True, cost=0.10, timestamp=3.0
        )
        rows = summarize_models(ledger_path)
        by_model = {r["model"]: r for r in rows}
        assert by_model["cheap/m"]["attempts"] == 2
        assert by_model["cheap/m"]["success_rate"] == 0.5
        assert abs(by_model["strong/m"]["avg_cost"] - 0.10) < 1e-9
        # Sorted by attempts descending.
        assert rows[0]["attempts"] >= rows[-1]["attempts"]


def test_summarize_models_absent_ledger_is_empty():
    with tempfile.TemporaryDirectory() as td:
        assert summarize_models(Path(td) / "model_stats.json") == []


def test_latest_report_picks_newest_by_timestamp_name():
    with tempfile.TemporaryDirectory() as td:
        reports = Path(td) / "reports"
        reports.mkdir()
        (reports / "report_20260101_000000.json").write_text(
            json.dumps({"llm_cost": 1.0}), encoding="utf-8"
        )
        (reports / "report_20260623_120000.json").write_text(
            json.dumps({"llm_cost": 2.0}), encoding="utf-8"
        )
        assert latest_report(reports)["llm_cost"] == 2.0


def test_latest_report_none_when_absent():
    with tempfile.TemporaryDirectory() as td:
        assert latest_report(Path(td) / "reports") is None


def test_collect_combines_all_sources():
    with tempfile.TemporaryDirectory() as td:
        d = _orch(td)
        (d / "audit.jsonl").write_text(
            json.dumps({"type": "command", "ok": True}), encoding="utf-8"
        )
        out = collect(td)
        assert set(out) == {"audit", "models", "latest_report"}
        assert out["audit"]["total_events"] == 1
        assert out["models"] == []
        assert out["latest_report"] is None
