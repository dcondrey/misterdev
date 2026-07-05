---
category: feat
complexity: large
context_files:
- misterdev/core/decomposer.py
- misterdev/core/contracts.py
- misterdev/core/progress.py
- misterdev/core/change_tracker.py
- misterdev/core/scratchpad.py
depends_on: []
files_to_modify:
- misterdev/agent.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Upgrade run_project to use topological sort, contracts, and progress tracking
---

`run_project()` (line 62) is the main entry point for `misterdev run`, but it's severely underpowered compared to `build()`. It runs tasks in file-system order without dependency resolution, contract injection, progress tracking, or scratchpad learning. This means devplan tasks with `depends_on` fields are ignored.

Upgrade `run_project()` to:

1. **Topological sort**: after `discover_tasks()`, call `topological_sort(pending)` to respect `depends_on` ordering.

2. **Progress tracking**: initialize a `ProgressTracker` and skip already-completed tasks. Mark tasks completed/failed as they finish. This enables crash recovery (re-run picks up where it left off).

3. **Contract extraction**: after each successful task, call `contracts.extract_contracts()` on modified files. Before each task, inject `contracts.get_contracts_for_task(task.dependencies)` into `task.processor_data["interface_contracts"]`.

4. **Scratchpad**: create a shared `Scratchpad` and pass it to the `MarkdownPlanExecutor`. Record successes and failures.

5. **Dependency-aware execution loop**: use the same wave-based loop as `_execute_tasks()` in `build()`:
   - Find tasks whose dependencies are all in `completed_ids`
   - Skip tasks whose dependencies are in `failed_ids`
   - Stop on `MAX_CONSECUTIVE_FAILURES`

6. **Change tracking**: initialize `ChangeTracker` and call `changes.record_task_changes()` / `changes.get_recent_changes_for_files()`.

Do NOT add the full 6-phase build workflow (analysis, spec, decomposition, SOTA gates). Keep `run_project` focused on executing pre-written devplan tasks with proper orchestration.

Also upgrade `run_task()` (line 88) to inject contracts for the specific task's dependencies before execution.

Detect the project language from config: `lang = (project.config.get("language") or "python").lower()`