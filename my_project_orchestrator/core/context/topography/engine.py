"""The Topography Engine: a lazy-loaded facade over the symbol graph."""

from pathlib import Path
from typing import Dict, List, Set, Any

from ._log import logger
from .nodes import SymbolNode
from .graph import SymbolGraph


class TopographyEngine:
    """Topography Engine with Vector Persistence and Lazy Loading."""

    def __init__(self, project_path: Path, llm_client: Any, golden_paths=None):
        self.project_path = project_path
        self.llm = llm_client
        self.graph = SymbolGraph(project_path, golden_paths=golden_paths)
        self._initialized = False

    def initialize(self, force: bool = False):
        if self._initialized and not force:
            return

        logger.info("Initializing Topography Engine...")
        self.graph.build()
        logger.info(f"Symbol graph: {len(self.graph.symbols)} symbols indexed")
        self._initialized = True

    def get_file_outline(self, file_path: str) -> str:
        self.initialize()
        return self.graph.file_outline(file_path)

    def get_file_symbols(self, file_path: str):
        self.initialize()
        return self.graph.file_symbols(file_path)

    def get_project_outline(self) -> str:
        self.initialize()
        return self.graph.project_outline()

    def get_context_for_task(
        self, query: str, related_files: List[str], max_symbols: int = 30, ranker=None
    ) -> str:
        """Retrieves functional neighborhood and semantic context. Triggers lazy init.

        When more candidate symbols are found than fit (``max_symbols``) and a
        semantic ``ranker`` is supplied, the kept symbols are the ones most
        relevant to ``query`` rather than an arbitrary slice.
        """
        self.initialize()

        context_symbols: Set[str] = set()
        for file in related_files:
            for key, sym in self.graph.symbols.items():
                if sym.file_path == file:
                    context_symbols.add(key)
                    context_symbols.update(sym.outgoing_calls)
                    context_symbols.update(sym.incoming_calls)

        if not context_symbols:
            return ""

        # Cap to avoid blowing LLM context, keeping the most task-relevant
        # symbols when a semantic ranker is available.
        if ranker is not None and len(context_symbols) > max_symbols:
            candidates = {
                key: self.graph.symbols[key].content
                for key in context_symbols
                if key in self.graph.symbols
            }
            symbol_list = ranker.top_k(query, candidates, max_symbols)
        else:
            symbol_list = list(context_symbols)[:max_symbols]

        output = "## Topological Context\n"
        by_file: Dict[str, List[SymbolNode]] = {}
        for key in symbol_list:
            if key in self.graph.symbols:
                sym = self.graph.symbols[key]
                by_file.setdefault(sym.file_path, []).append(sym)

        for file_path, syms in by_file.items():
            output += f"\n### Symbols in {file_path}\n"
            for sym in syms:
                output += f"\n# {sym.kind.upper()}: {sym.name}\n{sym.content}\n"

        if len(context_symbols) > max_symbols:
            output += (
                f"\n(... {len(context_symbols) - max_symbols} more symbols omitted)\n"
            )

        return output
