import time
from typing import Optional

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class _WorktreeProjectView:
    """A Project facade that overrides only `path` (for worktree execution).

    Everything else (config, llm_client, topography, tools, env) delegates to
    the base project, so the executor reads shared context but writes, builds,
    and commits inside the worktree.
    """

    def __init__(self, base, path):
        self._base = base
        self.path = path

    def __getattr__(self, name):
        return getattr(self._base, name)


class ProgressReporter:
    """Lightweight wave/task progress logger for long runs."""

    def __init__(self, total_tasks: int):
        self.total = total_tasks
        self.completed = 0
        self.failed = 0
        self.current_wave = 0
        self.start_time = time.time()
        self._task_start: Optional[float] = None

    def start_wave(self, wave_num: int, task_ids: list[str]):
        self.current_wave = wave_num
        logger.info(f"=== Wave {wave_num} === [{', '.join(task_ids)}]")

    def start_task(self, task_id: str, title: str):
        self._task_start = time.time()
        logger.info(
            f"[{self.completed + self.failed}/{self.total}] Starting {task_id}: {title}"
        )

    def end_task(self, task_id: str, success: bool):
        elapsed = time.time() - self._task_start if self._task_start else 0
        if success:
            self.completed += 1
            logger.info(
                f"[{self.completed + self.failed}/{self.total}] {task_id} DONE ({elapsed:.0f}s)"
            )
        else:
            self.failed += 1
            logger.warning(
                f"[{self.completed + self.failed}/{self.total}] {task_id} FAILED ({elapsed:.0f}s)"
            )

    def summary(self):
        total_time = time.time() - self.start_time
        logger.info(
            f"=== Complete: {self.completed} done, {self.failed} failed, {total_time:.0f}s total ==="
        )
