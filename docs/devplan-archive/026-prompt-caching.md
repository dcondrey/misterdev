---
category: feat
complexity: medium
context_files:
- misterdev/task_executors/markdown_plan_executor.py
depends_on: []
files_to_modify:
- misterdev/llm/client.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add Anthropic prompt caching for repeated context blocks
---

Tasks in the same wave share identical system prompts and large overlapping code context blocks. Anthropic's prompt caching can cache these shared prefixes, reducing latency by ~80% and cost by ~90% on cache hits.

1. In `AnthropicLLMClient._call()`, add `cache_control` markers on the system prompt and large context blocks:
   ```python
   kwargs = {
       "model": self.model,
       "max_tokens": self.max_tokens,
       "messages": [{"role": "user", "content": prompt}],
       "temperature": self.temperature,
   }
   if system_prompt:
       kwargs["system"] = [
           {
               "type": "text",
               "text": system_prompt,
               "cache_control": {"type": "ephemeral"},
           }
       ]
   ```

2. For `OpenRouterLLMClient`, check if the model supports prompt caching via OpenRouter's API and add the equivalent headers/parameters if available.

3. Track cache hit statistics in `LLMUsage`:
   ```python
   @dataclass
   class LLMUsage:
       prompt_tokens: int = 0
       completion_tokens: int = 0
       total_tokens: int = 0
       estimated_cost: float = 0.0
       call_count: int = 0
       cache_creation_tokens: int = 0
       cache_read_tokens: int = 0
   ```

4. Extract cache stats from the API response:
   ```python
   if hasattr(response.usage, 'cache_creation_input_tokens'):
       usage.cache_creation_tokens = response.usage.cache_creation_input_tokens
   if hasattr(response.usage, 'cache_read_input_tokens'):
       usage.cache_read_tokens = response.usage.cache_read_input_tokens
   ```

5. Adjust cost estimation: cache reads cost 10% of normal input tokens, cache writes cost 25% more:
   ```python
   input_cost = (
       (normal_tokens * base_input_price) +
       (cache_read_tokens * base_input_price * 0.1) +
       (cache_creation_tokens * base_input_price * 1.25)
   ) / 1_000_000
   ```

6. Log cache hit rate:
   ```
   LLM call: 12,340 input tokens (8,200 cached, 66% hit rate)
   ```