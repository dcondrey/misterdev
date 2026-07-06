"""Terminal task-status transitions (complete/fail)."""

from misterdev.core.models import Task, ExecutionResult
from misterdev.core.execution.project import Project


class ResultsMixin:
    def _complete_task(
        self, project: Project, task: Task, msg: str, logs: str
    ) -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "completed")
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
        return ExecutionResult(status="failed", message=msg, logs=logs)
