"""Raw project gathering: file listing, config/doc reads, source overview, git log."""

import re
from pathlib import Path
from typing import Optional

from my_project_orchestrator.core.gitcmd import run_git
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


_IGNORE_DIRS = {
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
    ".git",
    "target",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".eggs",
    "vendor",
}


def _walk_limited(root: Path, max_depth: int = 6):
    """Yield files under root, pruning ignored dirs and bounding depth.

    Unlike Path.rglob, this prunes large directories (node_modules, target)
    during traversal instead of descending into them and filtering afterward,
    so scanning a monorepo does not stall on vendored trees.
    """
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in _IGNORE_DIRS or entry.name.startswith("."):
                    continue
                if depth < max_depth:
                    stack.append((entry, depth + 1))
            elif entry.is_file():
                yield entry


def _get_file_listing(
    project_path: Path, max_files: int = 200, max_depth: int = 6
) -> str:
    """Get a truncated file listing for the project."""
    files = []
    for item in _walk_limited(project_path, max_depth):
        rel = item.relative_to(project_path)
        files.append(str(rel))
        if len(files) >= max_files:
            files.append(f"... ({max_files}+ files, truncated)")
            break
    return "\n".join(sorted(files))


def _read_config_files(project_path: Path) -> str:
    """Read common config files for structure analysis."""
    config_names = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "CMakeLists.txt",
        "project.yaml",
        "tsconfig.json",
        "webpack.config.js",
    ]
    contents = []
    for name in config_names:
        text = _read_file_safe(project_path / name, max_lines=100)
        if text:
            contents.append(f"### {name}\n{text}")
    return "\n\n".join(contents) if contents else "(no config files found)"


def _read_docs(project_path: Path) -> str:
    """Read documentation files."""
    doc_names = ["README.md", "CLAUDE.md", "SPEC.md", "DESIGN.md", "REQUIREMENTS.md"]
    contents = []
    for name in doc_names:
        text = _read_file_safe(project_path / name, max_lines=200)
        if text:
            contents.append(f"### {name}\n{text}")
    return "\n\n".join(contents) if contents else "(no docs found)"


_OVERVIEW_CODE_EXTS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".swift",
    ".cs",
    ".kt",
}


# Phrases that mark an empty/default/no-op result as deliberate rather than
# unfinished. When a file's leading doc contains one, the sentence carrying it is
# preserved into the overview so the completeness analyzer does not flag the code
# as a stub.
_INTENT_KEYWORDS = (
    "degrade",
    "no-op",
    "noop",
    "fallback",
    "parity",
    "by design",
    "intentional",
    "graceful",
    "historical",
    "never panic",
)


def _leading_doc(path: Path, max_lines: int = 40, max_chars: int = 220) -> str:
    """Extract a source file's leading comment/doc block as one compact line.

    Carries each file's stated *intent* (e.g. "degrades to empty on wasm — the
    historical missing-model contract") into completeness analysis, so a deep
    file is judged by its documented purpose rather than guessed from its symbol
    names. Returns "" when the file opens with code and no leading comment.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = [next(fh, "") for _ in range(max_lines)]
    except OSError:
        return ""

    # `#` is a line comment in Python but a preprocessor directive in C-family
    # files, so only treat it (and `"""` docstrings) as a comment for Python.
    is_py = path.suffix == ".py"
    line_prefixes = ("///", "//!", "//", "#") if is_py else ("///", "//!", "//")
    doc: list[str] = []
    in_block = False
    block_end = ""
    for raw in head:
        line = raw.strip()
        if not in_block:
            if not line:
                # Skip blank lines anywhere in the leading comment region (not just
                # before the first comment), so an intent stated after a blank
                # separator line is still collected; the first real CODE line below
                # still ends the block.
                continue
            if not doc and line.startswith("#!"):
                continue  # shebang
        if in_block:
            end = line.find(block_end)
            seg = (line[:end] if end != -1 else line).lstrip("*").strip()
            if seg:
                doc.append(seg)
            if end != -1:
                break
            continue
        if is_py and (line.startswith('"""') or line.startswith("'''")):
            quote = line[:3]
            rest = line[3:].rstrip()
            if rest.endswith(quote) and len(rest) >= 3:
                doc.append(rest[:-3].strip())
                break
            in_block, block_end = True, quote
            if rest.strip():
                doc.append(rest.strip())
            continue
        if line.startswith("/*"):
            rest = line[2:]
            if "*/" in rest:
                doc.append(rest.split("*/", 1)[0].strip())
                break
            in_block, block_end = True, "*/"
            seg = rest.lstrip("*").strip()
            if seg:
                doc.append(seg)
            continue
        matched = next((p for p in line_prefixes if line.startswith(p)), None)
        if matched is not None:
            doc.append(line[len(matched) :].strip())
            continue
        break  # first line of real code ends the leading doc block

    full = " ".join(d for d in doc if d).strip()
    if not full:
        return ""
    summary = full[:max_chars]
    # The strongest "this empty/no-op is intentional" signal often sits a few
    # sentences into a module doc, past the summary cap. If the block states such
    # intent, graft the sentence that says so onto the summary so it is never
    # truncated away — this is exactly what stops a documented graceful-degrade
    # backend from being misread as an unfinished stub.
    lowered = full.lower()
    if any(k in lowered for k in _INTENT_KEYWORDS):
        for sentence in re.split(r"(?<=[.;])\s+", full):
            sl = sentence.lower()
            if any(k in sl for k in _INTENT_KEYWORDS) and sentence not in summary:
                summary = f"{summary.rstrip()} … {sentence.strip()}"[: max_chars + 180]
                break
    return summary


def _get_source_overview(
    project_path: Path, max_chars: int = 8000, outline: Optional[str] = None
) -> str:
    """Whole-project structural map plus the heads of source files.

    The symbol map (every file and its symbols, from tree-sitter) conveys the
    architecture densely so planning is grounded in the entire project rather
    than the first few files that fit ``max_chars`` of raw heads. A per-file
    intent map (leading doc comments) then carries each file's documented purpose
    — including deliberate graceful-degradation — so deep files are not judged by
    their symbol names alone.

    ``outline`` lets the caller pass an already-built symbol outline (the
    project's TopographyEngine graph). When given, this reuses it instead of
    parsing a second throwaway ``SymbolGraph`` here, so the whole-project graph is
    built once per run rather than once for the overview AND once for the engine.
    """
    parts = []
    if outline is None:
        try:
            from my_project_orchestrator.core.context.topography import SymbolGraph

            graph = SymbolGraph(project_path)
            graph.build()
            outline = graph.project_outline()
        except (ImportError, OSError, ValueError) as e:
            logger.debug(f"symbol-based overview unavailable: {e}")
            outline = ""
    if outline:
        parts.append("## Project structure (files and symbols)\n" + outline)

    intents = []
    intent_total = 0
    for item in _walk_limited(project_path):
        if item.suffix in _OVERVIEW_CODE_EXTS:
            doc = _leading_doc(item)
            if doc:
                line = f"{item.relative_to(project_path)}: {doc}"
                intents.append(line)
                intent_total += len(line) + 1
                if intent_total >= 16000:
                    break
    if intents:
        parts.append(
            "## File intents (leading doc comments)\n"
            "Documented graceful-degradation, platform-gated no-ops, and parity "
            "shims are intentional and COMPLETE — not stubs.\n" + "\n".join(intents)
        )

    head = []
    total = 0
    for item in _walk_limited(project_path):
        if item.suffix in _OVERVIEW_CODE_EXTS:
            text = _read_file_safe(item, max_lines=30)
            if text:
                rel = item.relative_to(project_path)
                chunk = f"### {rel}\n{text}\n"
                head.append(chunk)
                total += len(chunk)
                if total >= max_chars:
                    break
    if head:
        parts.append("## Source heads\n" + "\n".join(head))
    return "\n\n".join(parts) if parts else "(no source files found)"


def _get_git_log(project_path: Path, count: int = 20) -> str:
    """Get recent git log."""
    proc = run_git(f"git log --oneline -n {count}", project_path, timeout=10)
    if proc is None:
        return "(git log unavailable)"
    return proc.stdout.strip() if proc.returncode == 0 else "(not a git repo)"


def _read_file_safe(path: Path, max_lines: int = 500) -> str:
    """Read a file, returning empty string on failure."""
    try:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines)} lines total, truncated)"]
        return "\n".join(lines)
    except Exception:
        return ""
