import fnmatch
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List


def ensure_artifact_dir(directory: Path) -> Path:
    """Create an orchestrator artifact dir that git ignores.

    Runtime artifacts (ledger, response cache, free-model list, reports) live
    under ``.orchestrator/`` and must never dirty the user's working tree. The
    dir self-ignores via a ``.gitignore`` containing ``*`` — the same trick
    ``.pytest_cache`` uses — so users get clean behavior with zero setup.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Place the guard at the .orchestrator root (parent of per-feature subdirs
    # like llm_cache/, or alongside files written directly into .orchestrator).
    root = directory.parent if directory.parent.name == ".orchestrator" else directory
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        try:
            gitignore.write_text("*\n", encoding="utf-8")
        except OSError:
            pass
    return directory


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


def orchestrator_state_file(project_path: str | Path, name: str) -> Path:
    """Path to a JSON state file under the project's ``.orchestrator/`` dir.

    Creates the parent directory and centralizes the state-dir convention shared
    by the progress/contracts/change trackers so it lives in exactly one place.
    """
    f = Path(project_path) / ".orchestrator" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


def read_file(file_path: str | Path) -> str:
    """Reads the content of a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# Cap a model-requested read: a multi-GB in-project file (build artifact, data
# dump) would otherwise load whole into memory and the LLM context, a token/cost
# blowout. 2 MB matches the gatekeeper scan bound.
_MAX_READ_CHARS = 2_000_000


def read_file_capped(file_path: str | Path, max_chars: int = _MAX_READ_CHARS) -> str:
    """Read up to ``max_chars`` characters, appending a truncation marker when the
    file is larger, so the caller (and the model) knows the content is partial."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read(max_chars + 1)
    if len(data) > max_chars:
        return data[:max_chars] + f"\n...[truncated: file exceeds {max_chars} chars]"
    return data


def atomic_write(file_path: str | Path, content: str) -> None:
    """Write content via a temp file + atomic rename.

    Guarantees the destination is never left half-written if the process
    crashes or two writers race: readers see either the old file or the new
    one, never a truncated mix. The temp file is created in the destination
    directory so os.replace stays within one filesystem (atomic on POSIX).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace(path, content)


def _atomic_replace(path: Path, content: str) -> None:
    """Temp-file-then-rename core, assuming ``path.parent`` already exists.

    Split out so callers that have already created the directory (e.g. via
    ``ensure_artifact_dir``) don't mkdir it a second time.
    """
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


def atomic_write_json(
    file_path: str | Path,
    obj: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> None:
    """Serialize ``obj`` to JSON and write it atomically into a gitignored
    artifact dir.

    Centralizes the ``ensure_artifact_dir`` + temp-file-then-rename that the
    cache/ledger writers each reimplemented, reusing the crash-safe write core so
    a partial JSON file is never observed. ``ensure_artifact_dir`` already
    creates the directory, so the write core skips a redundant mkdir.
    """
    path = Path(file_path)
    ensure_artifact_dir(path.parent)
    _atomic_replace(path, json.dumps(obj, indent=indent, sort_keys=sort_keys))


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
