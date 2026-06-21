---
category: feat
complexity: large
depends_on: []
files_to_modify:
- my_project_orchestrator/llm/client.py
- my_project_orchestrator/task_executors/markdown_plan_executor.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add streaming LLM responses with early abort on bad patterns
---

The LLM client waits for the full response before returning. A typical code generation response is 2000-8000 tokens, taking 15-45 seconds. During this time, the orchestrator can't detect problems like the LLM generating an essay instead of code, or producing output in the wrong format.

Add streaming support with early abort:

1. In `BaseLLMClient`, add a `generate_stream()` method:
   ```python
   def generate_stream(self, prompt: str, system_prompt: str = "", 
                       abort_check: Callable[[str], bool] = None) -> LLMResponse:
       """Generate with streaming, optionally aborting early on bad patterns."""
       chunks = []
       for chunk in self._call_stream(prompt, system_prompt):
           chunks.append(chunk)
           accumulated = "".join(chunks)
           if abort_check and abort_check(accumulated):
               logger.warning("Aborting LLM stream: bad pattern detected")
               # Return partial content for error reporting
               return LLMResponse(content=accumulated, model=self.model, 
                                  finish_reason="aborted")
       return self._finalize_stream(chunks)
   ```

2. In `OpenRouterLLMClient`, implement `_call_stream()` using `stream=True`:
   ```python
   def _call_stream(self, prompt, system_prompt):
       response = self.client.chat.completions.create(
           model=self.model, messages=messages,
           temperature=self.temperature, stream=True,
       )
       for chunk in response:
           if chunk.choices[0].delta.content:
               yield chunk.choices[0].delta.content
   ```

3. In `AnthropicLLMClient`, implement `_call_stream()` using the Anthropic streaming API.

4. Define abort patterns for code generation:
   ```python
   def code_gen_abort_check(accumulated: str) -> bool:
       # Abort if >500 tokens with no code fence or file path
       if len(accumulated) > 2000 and "```" not in accumulated and "# File:" not in accumulated:
           return True
       # Abort if generating a conversation instead of code
       if "I'll help you" in accumulated[:200] or "Sure, here" in accumulated[:200]:
           return True
       return False
   ```

5. In `MarkdownPlanExecutor`, use `generate_stream()` with abort check. On abort, immediately retry with a more explicit prompt.

6. Make streaming opt-in via config:
   ```yaml
   llm:
     streaming: true
   ```
   Default to non-streaming for backward compatibility.