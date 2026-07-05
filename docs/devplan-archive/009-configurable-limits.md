---
category: feat
complexity: medium
depends_on: []
files_to_modify:
- misterdev/agent.py
- misterdev/core/context_budget.py
- misterdev/core/config.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Make hardcoded operational limits configurable via project.yaml
---

Several operational limits are hardcoded with no way to override them per-project:

- `MAX_CONSECUTIVE_FAILURES = 3` (agent.py)
- `max_workers = 4` (agent.py `_execute_parallel`)
- `max_tokens = 100000` (context_budget.py)

Make these configurable in `project.yaml` under an `orchestrator:` key:

```yaml
orchestrator:
  max_consecutive_failures: 5
  max_workers: 8
  context_budget_tokens: 150000
  max_task_attempts: 3
```

Changes:

1. In `config.py`, add defaults to `DEFAULT_CONFIG`:
   ```python
   "orchestrator": {
       "max_consecutive_failures": 3,
       "max_workers": 4,
       "context_budget_tokens": 100000,
       "max_task_attempts": 3,
   }
   ```

2. In `agent.py`, read from config instead of constants:
   ```python
   max_failures = project.config.get("orchestrator", {}).get("max_consecutive_failures", 3)
   max_workers = project.config.get("orchestrator", {}).get("max_workers", 4)
   ```

3. In `context_budget.py`, accept `max_tokens` as constructor parameter with fallback:
   ```python
   def __init__(self, max_tokens: int = 100000):
   ```
   And in agent.py, pass the configured value when constructing ContextBudget.

4. In `markdown_plan_executor.py`, read `max_task_attempts` from config instead of hardcoded `3`.

Keep existing hardcoded values as defaults so nothing breaks without config.