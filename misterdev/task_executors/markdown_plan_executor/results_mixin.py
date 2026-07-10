"""Terminal task-status transitions (complete/fail)."""

from misterdev.core.execution.project import Project
from misterdev.core.models import Task, ExecutionResult

from .helpers import logger


class ResultsMixin:
    def _record_invented_tools(
        self, project: Project, task: Task, resolved: bool
    ) -> None:
        """Fold any runtime-invented tools (two-timescale P2c) into the tool
        corpus with this task's outcome. Best-effort: a learning stream never
        fails a build, so any error is swallowed. No-op when tooling is off (no
        tools were captured)."""
        tools = (getattr(task, "processor_data", None) or {}).get("invented_tools")
        if not tools:
            return
        try:
            from misterdev.core.evolution.tool_corpus import ToolCorpus

            corpus = ToolCorpus(
                project.path / ".orchestrator" / "evolution" / "tool_corpus.json"
            )
            niche = str(project.config.get("language") or "unknown")
            for source in tools:
                corpus.record(source, niche, task.id, resolved)
        except Exception as e:  # a learning stream must never sink the build
            logger.debug(f"Tool-corpus record skipped: {e}")

    def _complete_task(
        self, project: Project, task: Task, msg: str, logs: str
    ) -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "completed")
        self._record_invented_tools(project, task, resolved=True)
        # The task changed the tree; mark the symbol graph stale so the next task
        # rebuilds from the new state instead of a map that predates this edit.
        topo = getattr(project, "topography", None)
        if topo is not None and hasattr(topo, "invalidate"):
            topo.invalidate()
        return ExecutionResult(status="completed", message=msg, logs=logs)

    def _fail_task(
        self, project: Project, task: Task, msg: str, logs: str = ""
    ) -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "failed")
        self._record_invented_tools(project, task, resolved=False)
        return ExecutionResult(status="failed", message=msg, logs=logs)
