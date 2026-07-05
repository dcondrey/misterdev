---
category: feat
complexity: large
context_files:
- misterdev/tools/git_tool.py
depends_on:
- '001'
files_to_modify:
- misterdev/agent.py
- misterdev/task_executors/markdown_plan_executor.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add automatic regression detection and per-task rollback
---

If the final build/test gate fails after all tasks complete, there's no way to identify which task introduced the regression. The user must manually bisect. Git branch-per-task creates individual commits, making automated bisection possible.

Add regression detection and rollback:

1. After the final SOTA gate check (Phase 5 in `build()`), if tests fail:
   ```python
   if not success and report.completed_tasks:
       logger.warning("Post-build regression detected. Starting bisect...")
       culprit = self._bisect_regression(project, report.completed_tasks, test_command)
       if culprit:
           logger.warning(f"Regression introduced by task {culprit.id}: {culprit.title}")
           report.key_decisions.append(f"Regression from {culprit.id} auto-reverted")
           self._rollback_task(project, culprit)
   ```

2. Implement `_bisect_regression()` using git:
   ```python
   def _bisect_regression(self, project, tasks, test_cmd):
       """Binary search through task commits to find the regression."""
       # Get commit SHAs for each completed task
       commits = self._get_task_commits(project, tasks)
       
       low, high = 0, len(commits) - 1
       while low < high:
           mid = (low + high) // 2
           # Checkout the commit at mid
           self._git(project, f"git checkout {commits[mid]}")
           success, _ = self._run_command(project, test_cmd, timeout=180)
           if success:
               low = mid + 1
           else:
               high = mid
       
       # Restore to HEAD
       self._git(project, "git checkout -")
       return tasks[low] if low < len(tasks) else None
   ```

3. Implement `_rollback_task()`:
   ```python
   def _rollback_task(self, project, task):
       """Revert a specific task's commit."""
       commit = self._find_task_commit(project, task.id)
       if commit:
           self._git(project, f"git revert --no-edit {commit}")
           project.task_manager.update_task_status(task.id, "reverted")
   ```

4. Add a `--no-rollback` flag to disable automatic regression rollback.

5. In the report, include the bisect result:
   ```
   ### Regression Detection
   Post-build tests failed. Bisected to task T-009 (add concurrency module).
   Commit abc123 was automatically reverted.
   ```