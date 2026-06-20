---
category: feat
complexity: small
depends_on: []
files_to_modify:
- my_project_orchestrator/core/report.py
- my_project_orchestrator/agent.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Persist build reports to disk with run history
---

Build reports are generated but only printed to stdout/returned as a string. There's no persistent history of past runs. If the terminal scrolls or the session ends, the report is lost.

1. In `report.py`, add `save()` method:
   ```python
   def save(self, project_path: Path):
       reports_dir = project_path / ".orchestrator" / "reports"
       reports_dir.mkdir(parents=True, exist_ok=True)
       
       timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
       filename = f"report_{timestamp}.md"
       report_path = reports_dir / filename
       report_path.write_text(self.to_markdown(), encoding="utf-8")
       
       # Also save structured data as JSON for programmatic access
       json_path = reports_dir / f"report_{timestamp}.json"
       json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
       
       logger.info(f"Report saved to {report_path}")
       return report_path
   ```

2. Add `to_dict()` method that returns the report as a structured dict:
   ```python
   def to_dict(self) -> dict:
       return {
           "mode": self.mode.value,
           "project": self.project_name,
           "start_time": self.start_time.isoformat(),
           "end_time": self.end_time.isoformat() if self.end_time else None,
           "completed": [t.id for t in self.completed_tasks],
           "failed": [t.id for t in self.failed_tasks],
           "deferred": [t.id for t in self.deferred_tasks],
           "llm_calls": self.llm_calls,
           "llm_tokens": self.llm_tokens,
           "llm_cost": self.llm_cost,
       }
   ```

3. In `agent.py` `build()`, call `report.save(project.path)` after `report.finalize()`.

4. In `agent.py` `run_project()`, create a lightweight report and save it after all tasks complete.

5. Add a CLI command `project-orchestrator history .` that lists past reports:
   ```
   2026-06-19 17:15 | SMART  | 28/30 done, 2 failed | $8.42 | report_20260619_171500.md
   2026-06-19 19:30 | RUN    | 20/20 done, 0 failed | $3.10 | report_20260619_193000.md
   ```