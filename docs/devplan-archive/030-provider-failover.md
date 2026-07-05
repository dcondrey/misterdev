---
category: feat
complexity: medium
depends_on: []
files_to_modify:
- misterdev/llm/client.py
- misterdev/core/config.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add multi-provider LLM failover chain
---

If the configured LLM provider (OpenRouter) goes down mid-run, all remaining tasks fail. The orchestrator should automatically failover to an alternative provider.

1. Add failover configuration to `project.yaml`:
   ```yaml
   llm:
     provider: openrouter
     model: anthropic/claude-sonnet-4
     failover:
       - provider: anthropic
         model: claude-sonnet-4-20250514
       - provider: openrouter
         model: google/gemini-2.5-pro
   ```

2. Create a `FailoverLLMClient` that wraps multiple providers:
   ```python
   class FailoverLLMClient(BaseLLMClient):
       def __init__(self, config: dict):
           super().__init__(config)
           self.primary = create_llm_client(config)
           self.failover_clients = []
           for fc in config.get("llm", {}).get("failover", []):
               try:
                   fc_config = {**config, "llm": {**config["llm"], **fc}}
                   self.failover_clients.append(create_llm_client(fc_config))
               except (ValueError, ImportError) as e:
                   logger.warning(f"Failover provider unavailable: {e}")
           self._active = self.primary
           self._consecutive_failures = 0
       
       def _call(self, prompt, system_prompt):
           clients = [self.primary] + self.failover_clients
           for client in clients:
               try:
                   return client._call(prompt, system_prompt)
               except LLMCallError as e:
                   if not e.retryable:
                       raise
                   logger.warning(f"Provider {client.__class__.__name__} failed, trying next...")
                   continue
           raise LLMCallError("All LLM providers failed", retryable=False)
   ```

3. Update `create_llm_client()` to return `FailoverLLMClient` when failover config is present.

4. Track which provider actually served each request for cost attribution:
   ```python
   return LLMResponse(content=..., model=client.model, ...)
   ```

5. Log failover events:
   ```
   WARNING: OpenRouter failed (503 overloaded). Failing over to Anthropic direct.
   INFO: Failover successful. Continuing with claude-sonnet-4-20250514.
   ```