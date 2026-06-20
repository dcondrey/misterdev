---
category: feat
complexity: medium
depends_on:
- '001'
files_to_modify:
- my_project_orchestrator/core/progress.py
- my_project_orchestrator/agent.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add content-hash-based incremental task execution
---

Currently, re-running `project-orchestrator run` skips completed tasks (via `ProgressTracker`), but if a task's devplan file or target files change, it still shows as "completed" and is skipped. There's no way to detect that a task needs re-execution.

Add content-hash-based change detection:

1. In `ProgressTracker`, store a hash alongside each completed task:
   ```python
   def mark_completed(self, task_id: str, task_hash: str):
       self._data["completed"][task_id] = {
           "timestamp": datetime.now(timezone.utc).isoformat(),
           "hash": task_hash,
       }
       self._save()
   
   def needs_rerun(self, task_id: str, current_hash: str) -> bool:
       entry = self._data["completed"].get(task_id)
       if not entry:
           return True
       return entry.get("hash") != current_hash
   ```

2. Compute task hash from: devplan file content + target file mtimes + dependency hashes:
   ```python
   import hashlib
   
   def compute_task_hash(task: Task, project_path: Path) -> str:
       h = hashlib.sha256()
       # Hash the devplan file content
       if task.source_ref:
           content = Path(task.source_ref).read_bytes()
           h.update(content)
       # Hash target file mtimes (cheap proxy for content changes)
       for f in sorted(task.files_to_modify):
           fp = project_path / f
           if fp.exists():
               h.update(str(fp.stat().st_mtime_ns).encode())
       return h.hexdigest()[:16]
   ```

3. In `run_project()`, check `progress.needs_rerun(task.id, current_hash)` instead of just `progress.is_done(task.id)`. If a completed task's hash changed, log it and re-run:
   ```
   Task 005-formatter-per-task was previously completed but inputs changed. Re-running.
   ```

4. Add a `--force` flag to bypass the cache and re-run all tasks.

5. Add a `--status` flag that shows which tasks would run vs skip:
   ```
   001-run-uses-topo-sort     SKIP (unchanged, completed 2026-06-19)
   002-silent-errors          RUN  (devplan modified)
   003-command-timeout         RUN  (pending)
   ```