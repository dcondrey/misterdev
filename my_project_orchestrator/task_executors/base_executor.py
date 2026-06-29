from abc import ABC, abstractmethod
from my_project_orchestrator.core.models import Task, ExecutionResult
from my_project_orchestrator.core.execution.project import Project


class BaseTaskExecutor(ABC):
    @abstractmethod
    def execute(self, task: Task, project_context: Project) -> ExecutionResult:
        """Executes a given task within the project context."""
        pass
