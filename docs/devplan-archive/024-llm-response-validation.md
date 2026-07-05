---
category: feat
complexity: medium
depends_on: []
files_to_modify:
- misterdev/task_executors/markdown_plan_executor.py
- misterdev/llm/responses.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add LLM response validation before applying edits
---

The orchestrator blindly applies whatever edits the LLM returns. If the LLM hallucinates a file path, writes to a file outside the project, or returns a massive diff that replaces entire files when only a small edit was needed, those edits get applied without checks.

Add pre-apply validation in `markdown_plan_executor.py`:

1. **Path validation**: reject edits to files outside the project root:
   ```python
   def _validate_edit_paths(self, project: Project, edits: Dict[str, str]) -> Dict[str, str]:
       valid = {}
       for path, content in edits.items():
           full = (project.path / path).resolve()
           if not full.is_relative_to(project.path.resolve()):
               logger.error(f"Rejected edit to path outside project: {path}")
               continue
           valid[path] = content
       return valid
   ```

2. **Scope validation**: warn if the LLM modified files not listed in `files_to_modify` or `files_to_create`:
   ```python
   expected = set(task.files_to_modify + task.files_to_create)
   unexpected = set(edits.keys()) - expected
   if unexpected:
       logger.warning(f"LLM modified unexpected files: {unexpected}")
   ```

3. **Size guard**: if an edit replaces more than 80% of a file's content and the file is >100 lines, log a warning (likely the LLM rewrote the file instead of making targeted edits):
   ```python
   for path, new_content in edits.items():
       old_content = read_file(project.path / path)
       if old_content and len(old_content) > 100:
           overlap = _similarity(old_content, new_content)
           if overlap < 0.2:
               logger.warning(f"LLM appears to have rewritten {path} (only {overlap:.0%} overlap)")
   ```

4. **Empty content guard**: reject edits that would create empty files.

5. In `LLMResponseParser`, add validation that parsed file paths don't contain `..` segments.