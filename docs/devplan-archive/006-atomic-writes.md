---
category: fix
complexity: small
depends_on: []
files_to_modify:
- misterdev/core/contracts.py
- misterdev/core/progress.py
status: completed
test_command: uv run pytest tests/ -x -q
title: Make file writes atomic in contracts and progress tracking
---

Both `ContractRegistry._save()` and `ProgressTracker._save()` use direct `write_text()` to persist state. If the orchestrator crashes (or parallel tasks race) mid-write, the JSON file is left in a corrupt state, breaking crash recovery.

Fix both `_save()` methods to use atomic write-then-rename:

```python
import tempfile

def _save(self):
    data = json.dumps(self._data, indent=2)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(self._path.parent),
        suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, str(self._path))
    except BaseException:
        os.unlink(tmp_path)
        raise
```

`os.replace()` is atomic on POSIX (rename within same filesystem). The temp file is created in the same directory to guarantee same-filesystem rename.

Apply the same pattern to both:
1. `contracts.py` `ContractRegistry._save()`
2. `progress.py` `ProgressTracker._save()`

Add `import tempfile, os` to both files if not already present.