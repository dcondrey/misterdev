---
category: fix
complexity: medium
context_files:
- misterdev/core/decomposer.py
- misterdev/task_executors/markdown_plan_executor.py
depends_on: []
files_to_modify:
- misterdev/agent.py
- misterdev/core/task.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Fix task ID mismatch between build() decomposed tasks and TaskManager
---

When `build()` decomposes a spec into tasks via LLM, those tasks get IDs like `T-001`, `T-002`. But `TaskManager.update_task_status()` only knows about devplan tasks (IDs like `001-posting-shard`). This causes ERROR-level "Task T-XXX not found" logs on every task start and completion, and means task status is never persisted to disk during `build()` runs.

Fix by registering decomposed tasks with TaskManager before execution:

1. In `agent.py` `_execute_tasks()`, after `topological_sort(tasks)`, register each task with `TaskManager`:
   ```python
   for task in sorted_tasks:
       project.task_manager.tasks[task.id] = task
   ```

2. In `TaskManager.update_task_status()`, change the ERROR log to WARNING level when the task has no `source_ref` (decomposed tasks don't have backing markdown files, so persisting status is expected to be skipped):
   ```python
   if task_id not in self.tasks:
       logger.warning(f"Task {task_id} not in task registry, status update skipped.")
       return
   ```

3. For decomposed tasks that DO get registered, skip the file-persist step if `task.source_ref` is None or empty.

This fix ensures:
- No more ERROR-level noise in logs
- Decomposed tasks can be tracked by TaskManager for progress/contracts
- Status updates for devplan tasks still persist to markdown files