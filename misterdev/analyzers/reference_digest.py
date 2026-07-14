"""Read-only structural digest of a reference implementation, for porting.

The ``build`` workflow can be pointed at a *reference* implementation — an
existing project (often in a different language) whose design should be
reproduced idiomatically in the target project. This module renders a compact,
language-agnostic map of that reference's modules, public symbols, and data
models by reusing misterdev's tree-sitter symbol graph (:class:`SymbolGraph`),
so the planner is grounded in the reference's real architecture rather than a
vague prose description.

Design constraints (a donor tool that inspired this had bugs here — we do not
repeat them):

- **Strictly read-only.** ``SymbolGraph.build`` writes a cache to
  ``<tree>/.orchestrator/topography_cache.json``; pointing it at the reference
  tree would MUTATE the donor. We redirect ``cache_path`` off the reference tree
  (into the target project's ``.orchestrator`` or a temp dir) so the reference
  directory is never written to.
- **Bounded output.** A large reference must not blow the planning context: the
  rendered map is truncated to ``max_chars`` with an explicit elision note.
- **Validated input.** The path is resolved and required to be an existing
  directory; a bad path raises ``ValueError`` (the caller fails fast rather than
  planning against nothing).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from misterdev.core.context.topography import SymbolGraph
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# Keep the rendered map well under a typical spec/context budget. A reference is
# a design aid, not the source of truth, so a dense-but-partial map is fine.
_DEFAULT_MAX_CHARS = 12_000

_HEADER = """## Reference implementation to port from: {name}

This is a READ-ONLY structural map of an existing implementation (possibly in a
different language). Use it as a DESIGN reference: reproduce its modules, public
interfaces, and data models idiomatically in THIS project's language and
conventions. Do NOT copy code verbatim, and do NOT create files under the
reference path — it is not part of this project.

### Module / symbol map (path: top-level symbols)
"""


def build_reference_digest(
    reference_dir: str | Path,
    cache_dir: str | Path | None = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Render a compact, read-only digest of ``reference_dir`` for porting.

    Parameters
    ----------
    reference_dir:
        Path to the reference implementation to analyze. Must be an existing
        directory. Resolved to an absolute path (user ``~`` expanded).
    cache_dir:
        Where to write the symbol-graph cache. Redirected here so the reference
        tree is never mutated. Defaults to a throwaway temp directory when
        omitted. Its parent is created if missing.
    max_chars:
        Hard ceiling on the rendered symbol map (excluding the header). The map
        is truncated at a line boundary with an elision note when exceeded.

    Returns
    -------
    A markdown digest string. Never empty on success; raises ``ValueError`` if
    the path is missing/not a directory. Extraction that yields no symbols (an
    empty or unparseable tree) returns the header plus an explicit note so the
    caller can still see the reference was consulted.
    """
    ref = Path(reference_dir).expanduser().resolve()
    if not ref.is_dir():
        raise ValueError(f"reference dir not found or not a directory: {reference_dir}")
    if max_chars <= 0:
        raise ValueError(f"max_chars must be > 0, got {max_chars}")

    graph = SymbolGraph(ref)
    # Redirect the on-disk cache OFF the reference tree: this is the single line
    # that guarantees analysis is read-only. A temp dir when no target cache is
    # given; either way, never ``ref``.
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
    else:
        cache_root = Path(tempfile.mkdtemp(prefix="misterdev-refcache-"))
    graph.cache_path = cache_root / "reference_topography_cache.json"

    graph.build()
    outline = graph.project_outline()

    header = _HEADER.format(name=ref.name)
    if not outline.strip():
        logger.info("Reference digest: no symbols extracted from %s", ref)
        return header + "(no source symbols could be extracted from the reference)"

    if len(outline) > max_chars:
        truncated = outline[:max_chars].rsplit("\n", 1)[0]
        omitted = outline.count("\n") - truncated.count("\n")
        outline = truncated + f"\n(... reference map truncated, ~{omitted} more lines)"

    return header + outline
