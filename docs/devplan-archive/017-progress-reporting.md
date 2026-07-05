---
category: feat
complexity: medium
depends_on:
- '001'
files_to_modify:
- misterdev/agent.py
- misterdev/task_executors/markdown_plan_executor.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add live progress reporting during task execution
---

During long runs, the only feedback is log messages. Add structured progress reporting that shows:
- Which wave is executing
- Which task is active and its attempt number
- Running totals (completed/failed/remaining)
- Elapsed time per task and total

1. Create a simple progress reporter in `agent.py`:
   ```python
   class ProgressReporter:
       def __init__(self, total_tasks: int):
           self.total = total_tasks
           self.completed = 0
           self.failed = 0
           self.current_wave = 0
           self.start_time = time.time()
           self._task_start = None
       
       def start_wave(self, wave_num: int, task_ids: List[str]):
           self.current_wave = wave_num
           logger.info(f"=== Wave {wave_num} === [{', '.join(task_ids)}]")
       
       def start_task(self, task_id: str, title: str):
           self._task_start = time.time()
           remaining = self.total - self.completed - self.failed
           logger.info(f"[{self.completed}/{self.total}] Starting {task_id}: {title}")
       
       def end_task(self, task_id: str, success: bool):
           elapsed = time.time() - self._task_start if self._task_start else 0
           if success:
               self.completed += 1
               logger.info(f"[{self.completed}/{self.total}] {task_id} DONE ({elapsed:.0f}s)")
           else:
               self.failed += 1
               logger.warning(f"[{self.completed}/{self.total}] {task_id} FAILED ({elapsed:.0f}s)")
       
       def summary(self):
           total_time = time.time() - self.start_time
           logger.info(f"=== Complete: {self.completed} done, {self.failed} failed, {total_time:.0f}s total ===")
   ```

2. Use `ProgressReporter` in both `run_project()` and `_execute_tasks()`.

3. Add `import time` if not present.