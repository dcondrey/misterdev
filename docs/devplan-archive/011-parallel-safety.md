---
category: fix
complexity: medium
context_files:
- misterdev/core/contracts.py
- misterdev/core/progress.py
depends_on:
- '006'
files_to_modify:
- misterdev/agent.py
- misterdev/core/scratchpad.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Fix race conditions in parallel task execution
---

`_execute_parallel()` uses `ThreadPoolExecutor` to run tasks concurrently, but several shared resources have no synchronization:

1. **Scratchpad** (`core/scratchpad.py`): Multiple tasks read/write the scratchpad simultaneously. Read `scratchpad.py` and add a `threading.Lock` around `add_entry()` and `get_relevant_entries()`:
   ```python
   import threading
   
   class Scratchpad:
       def __init__(self):
           self._entries = []
           self._lock = threading.Lock()
       
       def add_entry(self, ...):
           with self._lock:
               self._entries.append(...)
       
       def get_relevant_entries(self, ...):
           with self._lock:
               return [e for e in self._entries if ...]
   ```

2. **Progress tracker**: Already fixed by 006 (atomic writes), but add a lock around `mark_completed()` and `mark_failed()` for in-memory state consistency.

3. **Contract registry**: Already fixed by 006 (atomic writes), but add a lock around `extract_contracts()` and `get_contracts_for_task()`.

4. **Consecutive failure counter** in `_execute_parallel()`: The counter `consecutive_failures` is modified from multiple threads without synchronization. Use `threading.Lock` or `atomic` counter:
   ```python
   import threading
   failure_lock = threading.Lock()
   ```

Do not add locks to `MarkdownPlanExecutor` or `GitTool` since parallel mode already disables git branching.