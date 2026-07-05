---
category: feat
complexity: small
depends_on:
- '001'
files_to_modify:
- misterdev/agent.py
- misterdev/cli.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add dry-run mode to run_project for dependency visualization
---

Add a `--dry-run` flag to `misterdev run` that shows the execution plan without running anything. This helps users verify dependency ordering and wave structure before committing to an expensive LLM-powered run.

1. In `cli.py`, add `--dry-run` flag to the `run` subcommand:
   ```python
   run_parser.add_argument("--dry-run", action="store_true", help="Show execution plan without running tasks")
   ```

2. In `agent.py` `run_project()`, add `dry_run: bool = False` parameter. When true:
   - Discover and sort tasks (topological sort from task 001)
   - Print the execution plan:
     ```
     Execution Plan (dry-run):
     
     Wave 1 (parallel):
       [001-run-uses-topo-sort] Upgrade run_project to use topological sort... (large, feat)
       [002-silent-errors] Fix all silent exception swallowing... (small, fix)
       [003-command-timeout] Add timeout to CommandTool (small, fix)
     
     Wave 2 (depends on wave 1):
       [013-contract-extraction] Improve Rust contract extraction... (medium, feat) → depends on 010
     
     Total: 16 tasks, 4 waves, est. 12 LLM calls
     ```
   - Show any dependency issues (cycles, missing deps)
   - Exit without executing

3. Pass through from CLI: `orchestrator.run_project(args.path, dry_run=args.dry_run)`