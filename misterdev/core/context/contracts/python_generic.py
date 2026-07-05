"""Public-symbol extraction for Python and generic C-like languages."""

from typing import Dict, List

from ._text import _extract_name


def _extract_python_symbols(lines: List[str]) -> List[Dict[str, str]]:
    """Extract top-level def and class from Python (non-underscore)."""
    symbols = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") and not stripped.startswith("def _"):
            name = _extract_name(stripped[4:])
            sig = stripped.rstrip(":")
            symbols.append({"kind": "def", "name": name, "signature": sig})
        elif stripped.startswith("class ") and not stripped.startswith("class _"):
            name = _extract_name(stripped[6:])
            symbols.append(
                {"kind": "class", "name": name, "signature": stripped.rstrip(":")}
            )
    return symbols


def _extract_generic_symbols(lines: List[str]) -> List[Dict[str, str]]:
    """Fallback: extract function/type declarations from any C-like language."""
    symbols = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("export ") or stripped.startswith("public "):
            symbols.append(
                {"kind": "export", "name": stripped[:60], "signature": stripped[:80]}
            )
        elif stripped.startswith("func "):
            symbols.append(
                {
                    "kind": "func",
                    "name": _extract_name(stripped[5:]),
                    "signature": stripped[:80],
                }
            )
    return symbols
