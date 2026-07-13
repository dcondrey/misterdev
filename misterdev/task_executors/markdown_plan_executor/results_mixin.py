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

    def _defer_task(
        self, project: Project, task: Task, msg: str, questions: list, logs: str = ""
    ) -> ExecutionResult:
        """Park a task that needs human input (a missing credential, a judgment
        call, an ambiguity) instead of failing it. The task's work has already
        been reverted by the caller, so a follow-up run redoes it once answered;
        the run itself keeps going. Neither a success nor a failure — it does not
        count toward the consecutive-failure abort."""
        project.task_manager.update_task_status(task.id, "deferred")
        self._record_invented_tools(project, task, resolved=False)
        return ExecutionResult(
            status="deferred",
            message=msg,
            logs=logs or msg,
            questions=[q for q in questions if q],
        )

    def _deferral_reason(self, task: Task, error_logs: str, has_gate: bool):
        """Decide WHY a stuck task should be parked and WHAT to ask, as
        ``(reason, question)``. Three shapes, most-specific first: blocked on an
        external resource (a credential the user must supply), a judgment/review
        task with no automated check, or a genuine inability the user should steer.
        Always returns a pair — in walk-away mode every stuck task becomes a
        question, never a silent failure."""
        from misterdev.core.execution.blocker import blocked_reason

        title = task.title or task.id
        blk = blocked_reason(error_logs or "")
        if blk:
            return (
                f"blocked: {blk}",
                f"'{title}' needs an external resource: {blk}. Provide it (or say "
                "how to proceed), then re-run.",
            )
        if not has_gate:
            return (
                "no automated verification (judgment/review task)",
                f"'{title}' has no automated check to confirm it. I made a best "
                "effort — please review and confirm it is right, or tell me what "
                "to change.",
            )
        tail = ""
        if error_logs and error_logs.strip():
            tail = error_logs.strip().splitlines()[-1][:160]
        return (
            "could not complete after all attempts",
            f"I couldn't finish '{title}'"
            + (f" (last error: {tail})" if tail else "")
            + ". How should I proceed — clarify the requirement, or skip it?",
        )
