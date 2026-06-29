"""Symbol data type and its cache (de)serialization."""

from typing import Dict, List, Set, Any


class SymbolNode:
    def __init__(
        self,
        name: str,
        file_path: str,
        kind: str,
        start_line: int,
        end_line: int,
        content: str,
    ):
        self.name = name
        self.file_path = file_path
        self.kind = kind  # 'function', 'class', 'method'
        self.start_line = start_line
        self.end_line = end_line
        self.content = content
        self.outgoing_calls: Set[str] = set()
        self.incoming_calls: Set[str] = set()
        self.imports: List[Dict[str, str]] = []  # {name: ..., module: ...}

    def __repr__(self):
        return f"Symbol({self.kind}:{self.name} in {self.file_path})"


def _symbol_to_dict(s: "SymbolNode") -> Dict[str, Any]:
    """Serialize the parse-derived fields of a SymbolNode for the disk cache.

    Only the fields produced by parsing are stored; call neighbors (rebuilt by
    ``_resolve_references`` from content) and imports are intentionally omitted
    so the cache never has to track derived/global state.
    """
    return {
        "name": s.name,
        "file_path": s.file_path,
        "kind": s.kind,
        "start_line": s.start_line,
        "end_line": s.end_line,
        "content": s.content,
    }


def _symbol_from_dict(d: Dict[str, Any]) -> "SymbolNode":
    return SymbolNode(
        d["name"],
        d["file_path"],
        d["kind"],
        d["start_line"],
        d["end_line"],
        d["content"],
    )
