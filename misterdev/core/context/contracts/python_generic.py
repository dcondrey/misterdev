"""Public-symbol extraction for Python and generic C-like languages."""

import ast
from typing import Dict, List

from ._text import _extract_name


def _extract_python_symbols(lines: List[str]) -> List[Dict[str, str]]:
    """Extract public def and class from Python using AST, including class methods."""
    source = "\n".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_python_symbols_fallback(lines)

    symbols: List[Dict[str, str]] = []

    def _visit(node, prefix: str = "") -> None:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                return
            full = f"{prefix}.{node.name}" if prefix else node.name
            symbols.append(
                {"kind": "class", "name": full, "signature": f"class {full}"}
            )
            for child in ast.iter_child_nodes(node):
                _visit(child, full)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                return
            full = f"{prefix}.{node.name}" if prefix else node.name
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            symbols.append({"kind": kind, "name": full, "signature": f"{kind} {full}"})
        else:
            for child in ast.iter_child_nodes(node):
                _visit(child, prefix)

    _visit(tree)
    return symbols


def _extract_python_symbols_fallback(lines: List[str]) -> List[Dict[str, str]]:
    """Line-based fallback when AST parsing fails (e.g., Python 2 syntax)."""
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
