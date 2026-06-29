"""Terminal task-status transitions (complete/fail)."""

from my_project_orchestrator.core.models import Task, ExecutionResult
from my_project_orchestrator.core.execution.project import Project


class ResultsMixin:
    def _complete_task(
        self, project: Project, task: Task, msg: str, logs: str
    ) -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "completed")
        return ExecutionResult(status="completed", message=msg, logs=logs)

    def _fail_task(
        self, project: Project, task: Task, msg: str, logs: str = ""
    ) -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "failed")
        return ExecutionResult(status="failed", message=msg, logs=logs)
