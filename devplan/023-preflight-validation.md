---
category: feat
complexity: medium
depends_on: []
files_to_create:
- my_project_orchestrator/core/preflight.py
files_to_modify:
- my_project_orchestrator/core/task.py
- my_project_orchestrator/agent.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add pre-flight devplan validation before LLM execution
---

Before spending money on LLM calls, validate that the devplan is well-formed and executable. Currently, errors like missing files, broken dependencies, or invalid test commands are only discovered during execution.

Create `my_project_orchestrator/core/preflight.py`:

```python
class PreflightValidator:
    """Validates devplan tasks before execution to catch issues early."""
    
    def validate(self, tasks: List[Task], project_path: Path, config: dict) -> List[PreflightIssue]:
        issues = []
        task_ids = {t.id for t in tasks}
        
        for task in tasks:
            # 1. Check that context_files exist
            for f in task.context_files:
                if not (project_path / f).exists():
                    issues.append(PreflightIssue(
                        task.id, "warning",
                        f"Context file '{f}' does not exist"
                    ))
            
            # 2. Check that dependencies reference valid task IDs
            for dep in task.dependencies:
                if dep not in task_ids:
                    issues.append(PreflightIssue(
                        task.id, "error",
                        f"Dependency '{dep}' does not match any task"
                    ))
            
            # 3. Check for dependency cycles
            # (topological_sort will catch this, but better to report early)
            
            # 4. Check test_command is a known binary
            test_cmd = task.processor_data.get("test_command")
            if test_cmd:
                binary = test_cmd.split()[0]
                if not shutil.which(binary):
                    issues.append(PreflightIssue(
                        task.id, "warning",
                        f"Test command binary '{binary}' not found in PATH"
                    ))
            
            # 5. Check for duplicate file modifications across tasks
            # (warn if two independent tasks modify the same file)
            
            # 6. Validate frontmatter fields
            if not task.title:
                issues.append(PreflightIssue(
                    task.id, "warning", "Task has no title"
                ))
        
        return issues
```

Integration:

1. In `run_project()`, call `PreflightValidator.validate()` after discovering tasks but before execution.
2. Print all issues. If any are "error" severity, abort with a clear message.
3. "warning" issues are printed but don't block execution.
4. Add `--skip-preflight` flag to bypass validation.