---
category: fix
complexity: small
depends_on: []
files_to_create:
- tests/test_command_tool.py
files_to_modify:
- misterdev/tools/command.py
status: completed
test_command: uv run pytest tests/test_command_tool.py -v
title: Add timeout and subprocess hardening to CommandTool
---

`CommandTool.execute()` has no timeout, allowing hanging commands to block the entire orchestrator. Every other subprocess call in the codebase has a timeout. Also, it doesn't handle `FileNotFoundError` when the command binary doesn't exist.

Fix `misterdev/tools/command.py`:

1. Add `timeout: int = 120` parameter to `execute()`
2. Pass `timeout=timeout` to `subprocess.run()`
3. Catch `subprocess.TimeoutExpired` and return `(False, f"Command timed out after {timeout}s: {command}")`
4. Catch `FileNotFoundError` and return `(False, f"Command not found: {command}")`

Create `tests/test_command_tool.py`:
1. Successful command (`echo hello`) returns `(True, "hello\n")`
2. Failed command (`false`) returns `(False, ...)`
3. stderr is captured in output
4. Default cwd uses `project.path`
5. Custom cwd overrides default
6. Timeout fires on `sleep 10` with `timeout=1`

Use a `FakeProject` class: `class FakeProject: path = Path(td)`.