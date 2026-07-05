---
category: feat
complexity: large
context_files:
- misterdev/core/config.py
depends_on: []
files_to_modify:
- misterdev/llm/client.py
- misterdev/core/sovereign.py
- misterdev/task_executors/markdown_plan_executor.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add complexity-based model routing for cost optimization
---

The orchestrator uses a single LLM model for all calls, but task complexity varies wildly. A simple "add logging" fix doesn't need Opus ($15/$75 per 1M tokens) when Haiku ($0.80/$4) would suffice. The `StrategyOptimizer` already classifies tasks by strategy but doesn't adjust the model.

Add model routing based on task complexity and strategy:

1. In `project.yaml`, allow model tiers:
   ```yaml
   llm:
     provider: openrouter
     models:
       default: anthropic/claude-sonnet-4
       simple: anthropic/claude-haiku-4
       complex: anthropic/claude-opus-4-8
     routing:
       small: simple      # small complexity tasks use cheap model
       medium: default    # medium uses default
       large: complex     # large uses expensive model
       surgical: simple   # surgical strategy = minimal change = cheap model
       architectural: complex  # architectural = complex reasoning = expensive model
   ```

2. In `BaseLLMClient`, add a `with_model(model_name)` context method that temporarily overrides the model:
   ```python
   @contextmanager
   def with_model(self, model: str):
       original = self.model
       self.model = model
       try:
           yield
       finally:
           self.model = original
   ```

3. In `MarkdownPlanExecutor.execute()`, resolve the model based on task complexity and strategy before calling `generate_code()`:
   ```python
   routing = project.config.get("llm", {}).get("routing", {})
   models = project.config.get("llm", {}).get("models", {})
   tier = routing.get(task.complexity, routing.get(strategy, "default"))
   model = models.get(tier, models.get("default", self.model))
   with project.llm_client.with_model(model):
       llm_response = project.llm_client.generate_code(prompt, system_prompt)
   ```

4. In `StrategyOptimizer.select_best_strategy()`, always use the cheapest model (it's a classification task, not code generation).

5. Update `COST_PER_1M` in both `OpenRouterLLMClient` and `AnthropicLLMClient` to include the latest models:
   ```python
   "anthropic/claude-opus-4-8": {"input": 15.0, "output": 75.0},
   "anthropic/claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
   "anthropic/claude-haiku-4-5": {"input": 0.80, "output": 4.0},
   ```

6. Log which model is used for each task:
   ```
   [T-003] Using anthropic/claude-haiku-4 (small/surgical)
   [T-001] Using anthropic/claude-opus-4-8 (large/architectural)
   ```

Expected cost reduction: 50-70% on typical runs where most tasks are small/medium.