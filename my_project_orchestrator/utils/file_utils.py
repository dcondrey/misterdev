import fnmatch
import os
import re
import tempfile
from pathlib import Path
from typing import List


def is_golden_path(file_path: str, patterns) -> bool:
    """True if a project-relative path is part of the golden (protected) suite.

    Supports an exact path, a directory prefix (``tests/golden/`` matches
    everything beneath it), and a glob (``tests/test_contract_*.py``). Shared
    by the executor (conceal + reject edits) and the symbol graph (exclude from
    indexing) so the model can neither see nor modify these files.
    """
    if not patterns:
        return False
    norm = file_path.replace("\\", "/").lstrip("./")
    for pat in patterns:
        p = str(pat).replace("\\", "/").lstrip("./")
        if not p:
            continue
        if norm == p or norm.startswith(p.rstrip("/") + "/"):
            return True
        if fnmatch.fnmatch(norm, p):
            return True
    return False


def safe_ref_slug(value: str, fallback: str = "x", maxlen: int = 64) -> str:
    """Filesystem- and git-ref-safe slug from an arbitrary (LLM-supplied) string.

    Any LLM-generated identifier that becomes a branch name (``task/<id>``), a
    script/tool filename, or a dict key passes through here first, so a stray
    ``/``, space, or ``:`` can't crash a build (missing-subdir write) or produce
    an invalid git ref. Replaces unsafe chars with ``_``, collapses runs, trims
    leading/trailing separators, and falls back when nothing usable remains.
    """
    s = re.sub(r"[^A-Za-z0-9._-]", "_", str(value))
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:maxlen] or fallback


def read_file(file_path: str | Path) -> str:
    """Reads the content of a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def atomic_write(file_path: str | Path, content: str) -> None:
    """Write content via a temp file + atomic rename.

    Guarantees the destination is never left half-written if the process
    crashes or two writers race: readers see either the old file or the new
    one, never a truncated mix. The temp file is created in the destination
    directory so os.replace stays within one filesystem (atomic on POSIX).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_file(file_path: str | Path, content: str) -> None:
    """Writes content to a file, creating parent directories if needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def glob_files(directory: str | Path, pattern: str) -> List[Path]:
    """Finds files matching a glob pattern within a directory."""
    path = Path(directory)
    return list(path.glob(pattern))


def ensure_directory(directory: str | Path) -> None:
    """Ensures a directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
