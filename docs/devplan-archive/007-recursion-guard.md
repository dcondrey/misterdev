---
category: fix
complexity: small
depends_on: []
files_to_modify:
- misterdev/task_executors/markdown_plan_executor.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add recursion depth guard to strategy escalation
---

`MarkdownPlanExecutor.execute()` at line ~206 escalates strategy by recursively calling itself:
```python
return self.execute(project, task, strategy="surgical")
```

If the surgical strategy also fails and somehow triggers another escalation, this creates unbounded recursion. In practice it currently only escalates once (iterative/architectural → surgical), but there's no guard preventing deeper recursion if the logic changes.

Fix by adding a depth parameter:

1. Add `_depth: int = 0` parameter to `execute()`:
   ```python
   def execute(self, project, task, strategy=None, _depth=0):
   ```

2. At the escalation point (~line 206), check depth before recursing:
   ```python
   if _depth < 1:
       logger.info(f"Escalating strategy from {strategy} to surgical for final attempt")
       return self.execute(project, task, strategy="surgical", _depth=_depth + 1)
   else:
       return self._fail_task(project, task, f"All strategies exhausted after {_depth + 1} escalation levels", logs)
   ```

3. Log when recursion is blocked:
   ```python
   logger.warning(f"Strategy escalation blocked at depth {_depth}: already exhausted all strategies")
   ```