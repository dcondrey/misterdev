---
category: fix
complexity: small
depends_on: []
files_to_modify:
- misterdev/core/sovereign.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Fix resource leak in sovereign EphemeralCodeManager
---

`EphemeralCodeManager` in `sovereign.py` creates temporary files for code evaluation but doesn't clean them up on all paths.

Read `sovereign.py` and fix:

1. Find the `EphemeralCodeManager` class and its temp file creation.

2. Add cleanup using `tempfile.TemporaryDirectory` as context manager, or ensure cleanup in a `finally` block:
   ```python
   def evaluate(self, code: str, ...):
       tmp_dir = tempfile.mkdtemp(prefix="orchestrator_")
       try:
           # ... write code to tmp_dir, execute, collect results
           return results
       finally:
           shutil.rmtree(tmp_dir, ignore_errors=True)
   ```

3. If `EphemeralCodeManager` is used as a long-lived object, add `__enter__`/`__exit__` for context manager support, or add an explicit `cleanup()` method called from the agent.

4. Add `import shutil` if needed.