---
category: feat
complexity: medium
context_files:
- misterdev/core/error_classifier.py
depends_on:
- 008
files_to_modify:
- misterdev/task_executors/markdown_plan_executor.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Inject previous attempt errors into retry prompts
---

When a task fails validation and retries, the LLM gets the same prompt again with no information about what went wrong. This wastes attempts. The crosstalk run showed T-009 failing 3 times on the same Cargo.toml error because the LLM didn't know what broke.

Improve the retry loop in `markdown_plan_executor.py` `execute()`:

1. Accumulate error context across attempts:
   ```python
   prior_errors = []
   for attempt in range(1, max_attempts + 1):
       ...
       if not success:
           classified = error_classifier.classify(error_output)
           prior_errors.append({
               "attempt": attempt,
               "error": error_output[:2000],  # cap size
               "classification": classified.category,
               "suggestion": classified.suggestion,
           })
   ```

2. When building the retry prompt, prepend the error history:
   ```python
   if prior_errors:
       error_section = "\n### Previous Attempt Failures\n"
       for err in prior_errors:
           error_section += f"**Attempt {err['attempt']}** ({err['classification']}): {err['suggestion']}\n"
           error_section += f"```\n{err['error'][:500]}\n```\n"
       prompt = error_section + prompt
   ```

3. Include the classified suggestion in the prompt so the LLM knows specifically what to fix:
   - For `manifest_error`: "Check that Cargo.toml has [package] with name and version"
   - For `type_error`: "Fix the type mismatch shown in the error"
   - For `unknown`: include raw error output so the LLM can self-diagnose

4. Cap the error context to stay within the context budget (use `context_budget.estimate_tokens()` if available).