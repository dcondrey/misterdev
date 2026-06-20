import os
import tempfile
from pathlib import Path
from typing import List

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
