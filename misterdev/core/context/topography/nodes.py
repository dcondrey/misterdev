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
        # Callee identifier names invoked within this symbol's span, extracted
        # from the tree-sitter AST at parse time (call/invocation nodes only, so
        # identifiers in strings/comments and `if (`/`while (` keywords are never
        # counted). Resolved into concrete edges by SymbolGraph._resolve_references.
        self.call_names: Set[str] = set()
        self.outgoing_calls: Set[str] = set()
        self.incoming_calls: Set[str] = set()
        self.imports: List[Dict[str, str]] = []  # {name: ..., module: ...}

    def __repr__(self):
        return f"Symbol({self.kind}:{self.name} in {self.file_path})"


def _symbol_to_dict(s: "SymbolNode") -> Dict[str, Any]:
    """Serialize the parse-derived fields of a SymbolNode for the disk cache.

    Stores the parse-derived fields, including ``call_names`` (the AST-extracted
    callees, cached so ``_resolve_references`` need not re-parse). The graph-wide
    call neighbors and imports are omitted so the cache never tracks global state.
    """
    return {
        "name": s.name,
        "file_path": s.file_path,
        "kind": s.kind,
        "start_line": s.start_line,
        "end_line": s.end_line,
        "content": s.content,
        "call_names": sorted(s.call_names),
    }


def _symbol_from_dict(d: Dict[str, Any]) -> "SymbolNode":
    node = SymbolNode(
        d["name"],
        d["file_path"],
        d["kind"],
        d["start_line"],
        d["end_line"],
        d["content"],
    )
    node.call_names = set(d.get("call_names", ()))
    return node
