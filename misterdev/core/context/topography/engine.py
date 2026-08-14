"""The Topography Engine: a lazy-loaded facade over the symbol graph."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Any

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

    def invalidate(self) -> None:
        """Mark the symbol graph stale so the next access rebuilds it.

        The graph is built once and then never reflected the edits tasks made, so
        later tasks saw a stale map — missing new symbols, listing deleted ones —
        which weakens reference discovery and the dangling-reference gate. Calling
        this after a task changes the tree makes the next task rebuild from the
        current on-disk state. Rebuild is cheap: the per-file symbol cache is
        content-hashed, so only files that actually changed are re-parsed.
        """
        self._initialized = False

    def get_file_outline(self, file_path: str) -> str:
        self.initialize()
        return self.graph.file_outline(file_path)

    def get_file_symbols(self, file_path: str):
        self.initialize()
        return self.graph.file_symbols(file_path)

    def get_project_outline(self) -> str:
        self.initialize()
        return self.graph.project_outline()

    def localize(self, query: str, top_k: int = 10, ranker=None):
        """Rank symbols by relevance to ``query`` to FIND edit targets when they
        are not given (see :mod:`misterdev.core.context.localizer`). Returns a
        list of ``LocalizationHit`` best-first; lazy-inits the graph."""
        from misterdev.core.context.localizer import localize as _localize

        self.initialize()
        return _localize(query, self.graph.symbols, top_k=top_k, ranker=ranker)

    def localize_files(self, query: str, top_k: int = 5, ranker=None):
        """File-level edit targets for ``query`` (relevance summed per file),
        best-first — the shape decomposition scopes a task to."""
        from misterdev.core.context.localizer import localize_files as _lf

        self.initialize()
        return _lf(query, self.graph.symbols, top_k=top_k, ranker=ranker)

    def get_context_for_task(
        self,
        query: str,
        related_files: List[str],
        max_symbols: int = 30,
        ranker=None,
        exclude_files: Optional[Set[str]] = None,
    ) -> str:
        """Retrieves functional neighborhood and semantic context. Triggers lazy init.

        When more candidate symbols are found than fit (``max_symbols``) and a
        semantic ``ranker`` is supplied, the kept symbols are the ones most
        relevant to ``query`` rather than an arbitrary slice.

        ``exclude_files`` names files already sent verbatim IN FULL by another
        context section (small target files in code_context). Their own symbols
        are dropped here so the same code isn't duplicated across two sections;
        their cross-file call-neighbors are still surfaced. Files shown only as a
        windowed excerpt must NOT be excluded — their out-of-window symbols are
        complementary, not duplicate.
        """
        self.initialize()
        excluded = exclude_files or set()

        # Reuse SymbolGraph's memoized per-file index instead of re-scanning
        # every symbol on each call; it only rebuilds when self.symbols is
        # actually replaced (see SymbolGraph._file_index).
        _by_file = self.graph._file_index()

        context_symbols: Set[str] = set()
        for file in related_files:
            for key, sym in _by_file.get(file, []):
                if file not in excluded:
                    context_symbols.add(key)
                context_symbols.update(sym.outgoing_calls)
                context_symbols.update(sym.incoming_calls)

        if excluded:
            context_symbols = {
                key
                for key in context_symbols
                if key not in self.graph.symbols
                or self.graph.symbols[key].file_path not in excluded
            }

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

    def reference_sites(self, target_files: List[str], max_refs: int = 80) -> str:
        """Every EXTERNAL call site of the symbols defined in ``target_files``.

        A delete/rename/refactor task must update every reference to the symbol
        it changes, but the referencing files often sit outside the task's
        declared scope — so the model discovers them one build-error at a time
        and runs out of attempts (whack-a-mole). Listing all of them up front,
        with exact file:line, lets one attempt update them completely. Only
        references OUTSIDE the target files are listed (in-file uses are already
        visible in code_context). Returns '' when there are none.
        """
        self.initialize()
        targets = set(target_files)
        blocks: List[str] = []
        shown = 0
        _by_file = self.graph._file_index()  # reuse the memoized per-file index
        done = False
        for target_file in target_files:
            if done:
                break
            for _key, sym in _by_file.get(target_file, []):
                if not sym.incoming_calls:
                    continue
                sites = set()
                for caller_key in sym.incoming_calls:
                    caller = self.graph.symbols.get(caller_key)
                    if caller is not None and caller.file_path not in targets:
                        sites.add((caller.file_path, caller.start_line, caller.name))
                if not sites:
                    continue
                lines = [f"- `{sym.name}` ({sym.kind}) is referenced by:"]
                for fp, ln, nm in sorted(sites):
                    if shown >= max_refs:
                        lines.append("    - (... more references omitted)")
                        break
                    lines.append(f"    - {fp}:{ln} (in {nm})")
                    shown += 1
                blocks.append("\n".join(lines))
                if shown >= max_refs:
                    done = True
                    break
        if not blocks:
            return ""
        return (
            "## Complete External Reference Sites\n"
            "Every reference to a symbol defined in the files you are editing "
            "that lives OUTSIDE those files. If this task removes, renames, or "
            "changes the signature of such a symbol, you MUST update ALL of these "
            "sites in this attempt — the build fails on any you miss:\n"
            + "\n".join(blocks)
        )
