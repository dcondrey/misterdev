from pathlib import Path


def _path_in_scope(
    path_str: str,
    extensions: frozenset,
    filenames: frozenset = frozenset(),
) -> bool:
    """True when ``path_str`` is in scope by file extension or exact name."""
    p = Path(path_str)
    return p.suffix in extensions or p.name in filenames
