---
category: feat
complexity: small
context_files:
- misterdev/core/decomposer.py
depends_on:
- '001'
files_to_modify:
- misterdev/core/task.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Auto-detect implicit dependencies from file overlap between tasks
---

If two independent tasks both modify the same file, they will conflict when run in the same wave. The decomposer already does implicit dependency detection for `build()` (see `decomposer.py` "Implicit dependency" log lines), but `TaskManager.discover_tasks()` for devplan-based runs doesn't.

Add automatic dependency detection to `TaskManager`:

1. After `discover_tasks()` and `_resolve_dependency_ids()`, add `_detect_file_overlaps()`:
   ```python
   def _detect_file_overlaps(self):
       """Add implicit dependencies when tasks modify the same files."""
       file_to_tasks = {}
       task_list = sorted(self.tasks.values(), key=lambda t: t.id)
       
       for task in task_list:
           for f in task.files_to_modify:
               file_to_tasks.setdefault(f, []).append(task.id)
       
       for file_path, task_ids in file_to_tasks.items():
           if len(task_ids) < 2:
               continue
           # Chain: each task depends on the previous one touching this file
           for i in range(1, len(task_ids)):
               later = self.tasks[task_ids[i]]
               earlier_id = task_ids[i - 1]
               if earlier_id not in later.dependencies:
                   later.dependencies.append(earlier_id)
                   logger.info(
                       f"Implicit dependency: {later.id} depends on {earlier_id} "
                       f"(both modify {file_path})"
                   )
   ```

2. Call `_detect_file_overlaps()` at the end of `discover_tasks()`.

3. Also detect `files_to_create` vs `files_to_modify` overlap: if task B modifies a file that task A creates, B depends on A.

4. Log a summary: `"Detected 3 implicit dependencies from file overlaps"`.