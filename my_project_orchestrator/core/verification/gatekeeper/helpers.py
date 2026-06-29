from pathlib import Path
from typing import Optional

# Cap how much of a single file the scanners read into memory. A generated
# project can contain a multi-GB file (data dumps, bundles); reading it whole to
# grep for secrets would exhaust the gate's memory. Secrets/markers of interest
# live near the top or in small config files, so the head is what matters.
_MAX_SCAN_CHARS = 2_000_000


def _path_in_scope(
    path_str: str,
    extensions: frozenset,
    filenames: frozenset = frozenset(),
) -> bool:
    """True when ``path_str`` is in scope by file extension or exact name."""
    p = Path(path_str)
    return p.suffix in extensions or p.name in filenames


def _read_capped(path: Path, max_chars: int = _MAX_SCAN_CHARS) -> Optional[str]:
    """Read at most ``max_chars`` characters of ``path`` for scanning, or ``None``
    on an OS error. Bounds memory so a huge file can't OOM the gate."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_chars)
    except OSError:
        return None
