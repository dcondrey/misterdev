---
category: feat
complexity: medium
depends_on: []
files_to_modify:
- misterdev/llm/client.py
- misterdev/core/report.py
- misterdev/agent.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add per-task LLM cost tracking and budget enforcement
---

The orchestrator makes many LLM calls per run but doesn't track costs or enforce budgets. A single crosstalk run with 30 tasks could cost $50+ with no warning.

1. In `llm/client.py`, add token/cost tracking:
   ```python
   class LLMUsageTracker:
       def __init__(self):
           self.calls = []
           self.total_input_tokens = 0
           self.total_output_tokens = 0
       
       def record(self, model: str, input_tokens: int, output_tokens: int, task_id: str = None):
           cost = self._estimate_cost(model, input_tokens, output_tokens)
           self.calls.append({
               "model": model, "input_tokens": input_tokens,
               "output_tokens": output_tokens, "cost": cost,
               "task_id": task_id,
           })
           self.total_input_tokens += input_tokens
           self.total_output_tokens += output_tokens
       
       def _estimate_cost(self, model, inp, out):
           # Price per 1M tokens (approximate)
           prices = {
               "anthropic/claude-opus-4-8": (15.0, 75.0),
               "anthropic/claude-sonnet-4-6": (3.0, 15.0),
               "anthropic/claude-haiku-4-5": (0.25, 1.25),
           }
           inp_price, out_price = prices.get(model, (10.0, 30.0))
           return (inp * inp_price + out * out_price) / 1_000_000
       
       @property
       def total_cost(self):
           return sum(c["cost"] for c in self.calls)
       
       def per_task_summary(self):
           by_task = {}
           for c in self.calls:
               tid = c.get("task_id", "overhead")
               by_task.setdefault(tid, 0.0)
               by_task[tid] += c["cost"]
           return by_task
   ```

2. After each LLM call, extract token counts from the API response and call `tracker.record()`.

3. Add optional budget enforcement in `project.yaml`:
   ```yaml
   llm:
     budget_limit: 10.00  # USD, abort if exceeded
   ```
   Check `tracker.total_cost` before each LLM call and abort with a clear message if over budget.

4. In `report.py`, include cost breakdown in the build report:
   ```
   LLM Usage:
     Total calls: 47
     Input tokens: 1,234,567
     Output tokens: 89,012
     Estimated cost: $8.42
     Most expensive task: T-009 ($1.23, 3 retries)
   ```