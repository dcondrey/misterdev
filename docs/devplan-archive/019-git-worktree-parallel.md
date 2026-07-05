---
category: feat
complexity: large
depends_on:
- '011'
files_to_modify:
- misterdev/tools/git_tool.py
- misterdev/agent.py
- misterdev/task_executors/markdown_plan_executor.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add git worktree support for parallel task execution
---

Currently, parallel execution disables git branching to avoid races. This means parallel tasks all modify the same working tree, which can cause conflicts. Git worktrees allow each parallel task to work in an isolated copy.

1. In `git_tool.py`, add worktree management methods:
   ```python
   def create_worktree(self, project, branch_name: str, worktree_path: Path) -> Tuple[bool, str]:
       """Create a git worktree for isolated task execution."""
       cmd = f"git worktree add {worktree_path} -b {branch_name}"
       return self.execute(project, command=cmd)
   
   def remove_worktree(self, project, worktree_path: Path) -> Tuple[bool, str]:
       """Remove a git worktree after task completion."""
       cmd = f"git worktree remove {worktree_path} --force"
       return self.execute(project, command=cmd)
   
   def merge_worktree(self, project, branch_name: str) -> Tuple[bool, str]:
       """Merge a worktree branch back to main."""
       success, out = self.execute(project, command=f"git merge {branch_name} --no-edit")
       if success:
           self.execute(project, command=f"git branch -d {branch_name}")
       return success, out
   ```

2. In `_execute_parallel()`, when git is available:
   - Create a worktree per task in a temporary directory
   - Each task's `MarkdownPlanExecutor` operates on the worktree path
   - After task completion, merge the worktree branch back to main
   - Clean up worktrees

3. In `markdown_plan_executor.py`, accept an optional `working_dir` override that points to the worktree instead of the main project path.

4. Add a config option to enable/disable worktree parallelism:
   ```yaml
   orchestrator:
     parallel_mode: "worktree"  # or "shared" (current behavior)
   ```

5. Handle merge conflicts: if merge fails, mark the task as needing manual resolution and continue.