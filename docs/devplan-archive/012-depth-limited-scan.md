---
category: feat
complexity: small
depends_on: []
files_to_modify:
- my_project_orchestrator/core/project_analyzer.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Add depth-limited file scanning to project analyzer
---

`ProjectAnalyzer` uses `Path.rglob("*")` to scan the entire project tree with no depth limit. On large repos (monorepos, node_modules, target/), this can be extremely slow and consume excessive memory.

Read `project_analyzer.py` and add:

1. A `max_depth` parameter (default 5) to the scan methods.

2. Replace `rglob` with a depth-limited walk:
   ```python
   def _walk_limited(self, root: Path, max_depth: int = 5) -> Iterator[Path]:
       """Walk directory tree with depth limit."""
       def _walk(path: Path, depth: int):
           if depth > max_depth:
               return
           try:
               entries = sorted(path.iterdir())
           except PermissionError:
               return
           for entry in entries:
               yield entry
               if entry.is_dir() and not entry.name.startswith('.'):
                   yield from _walk(entry, depth + 1)
       yield from _walk(root, 0)
   ```

3. Add a default ignore list for known large directories:
   ```python
   _IGNORE_DIRS = {"node_modules", "target", ".git", "__pycache__", ".venv", 
                    "venv", "dist", "build", ".tox", ".mypy_cache", ".ruff_cache"}
   ```
   Skip these in the walker.

4. Make `max_depth` and additional ignore patterns configurable via `project.yaml`:
   ```yaml
   analyzer:
     max_depth: 8
     ignore_dirs:
       - vendor
       - third_party
   ```