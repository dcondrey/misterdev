---
category: fix
complexity: small
depends_on: []
files_to_modify:
- my_project_orchestrator/agent.py
- my_project_orchestrator/core/change_tracker.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Fix all silent exception swallowing across the codebase
---

Fix bare `except Exception:` blocks that swallow errors without logging:

1. **`agent.py` line ~54** in `get_project_status()`:
   Change `except Exception:` to `except Exception as e:` and add `logger.error(f"Failed to load project at {project_path}: {e}")`. Include `str(e)` in the returned error dict: `return {"error": f"Project load failed: {e}"}`.

2. **`agent.py` line ~439** in `_get_or_register()`:
   Change `except Exception:` to `except Exception as e:` and add `logger.error(f"Failed to register project at {project_path}: {e}")`.

3. **`change_tracker.py` line ~122** in `_get_last_diff()`:
   Change `except Exception:` to `except Exception as e:` and add `logger.debug(f"Could not get git diff: {e}")`. Use `debug` level since this is expected to fail in non-git directories.