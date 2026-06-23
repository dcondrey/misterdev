"""Unified Error Attribution Resolver.

Maps raw build/test error output back to specific project symbols and files.
"""

from pathlib import Path
from typing import List, Optional

from my_project_orchestrator.core.topography import SymbolGraph
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class ErrorLocation:
    def __init__(
        self, file_path: str, line: int, message: str, symbol: Optional[str] = None
    ):
        self.file_path = file_path
        self.line = line
        self.message = message
        self.symbol = symbol

    def __repr__(self):
        return f"Error({self.file_path}:{self.line} in {self.symbol or 'unknown'}): {self.message}"


class ErrorResolver:
    """Parses error logs and attributes them to the project topography."""

    def __init__(self, project_path: Path, symbol_graph: SymbolGraph):
        self.project_path = project_path
        self.graph = symbol_graph

    def resolve_errors(self, error_output: str) -> List[ErrorLocation]:
        """Parses the error output and returns attributed locations."""
        locations = []

        for line in error_output.splitlines():
            loc = self._parse_file_line_error(line)
            if loc:
                locations.append(loc)

        # Deduplicate by file:line
        seen = set()
        unique = []
        for loc in locations:
            key = (loc.file_path, loc.line)
            if key not in seen:
                seen.add(key)
                unique.append(loc)

        return unique

    def _parse_file_line_error(self, line: str) -> Optional[ErrorLocation]:
        """Parse a single line for file:line:message patterns.

        Handles:
          - Python: File "path", line N, in func
          - Rust/Go/generic: path.ext:N: message
          - pytest: path.py:N: ErrorType
        """
        stripped = line.strip()

        # Python traceback: File "path/to/file.py", line 42, in func_name
        if stripped.startswith('File "'):
            return self._parse_python_traceback(stripped)

        # Generic file:line:message (Rust, Go, pytest, gcc, etc.)
        parts = stripped.split(":", 2)
        if len(parts) >= 2:
            file_candidate = parts[0].strip()
            line_candidate = parts[1].strip()
            if line_candidate.isdigit() and self._looks_like_source(file_candidate):
                rel_path = self._to_rel_path(file_candidate)
                if rel_path:
                    msg = parts[2].strip() if len(parts) > 2 else ""
                    line_num = int(line_candidate)
                    symbol = self._find_symbol_at_line(rel_path, line_num)
                    return ErrorLocation(rel_path, line_num, msg, symbol)

        return None

    def _parse_python_traceback(self, line: str) -> Optional[ErrorLocation]:
        """Parse: File "path", line N, in func"""
        # Extract path between quotes
        quote_start = line.find('"')
        quote_end = line.find('"', quote_start + 1)
        if quote_start < 0 or quote_end < 0:
            return None
        file_path = line[quote_start + 1 : quote_end]

        # Extract line number
        line_marker = ", line "
        line_idx = line.find(line_marker, quote_end)
        if line_idx < 0:
            return None
        after_marker = line[line_idx + len(line_marker) :]
        # Line number ends at comma or end of string
        num_chars = []
        for c in after_marker:
            if c.isdigit():
                num_chars.append(c)
            else:
                break
        if not num_chars:
            return None
        line_num = int("".join(num_chars))

        rel_path = self._to_rel_path(file_path)
        if not rel_path:
            return None
        symbol = self._find_symbol_at_line(rel_path, line_num)
        return ErrorLocation(rel_path, line_num, line.strip(), symbol)

    def _looks_like_source(self, path_str: str) -> bool:
        """Check if string looks like a source file path."""
        source_exts = {".py", ".rs", ".go", ".js", ".ts", ".c", ".cpp", ".java", ".rb"}
        for ext in source_exts:
            if path_str.endswith(ext):
                return True
        return False

    def _to_rel_path(self, path_str: str) -> Optional[str]:
        try:
            path = Path(path_str)
            if path.is_absolute():
                if path.is_relative_to(self.project_path):
                    return str(path.relative_to(self.project_path))
                return None
            return str(path)
        except (ValueError, OSError):
            return None

    def _find_symbol_at_line(self, file_path: str, line: int) -> Optional[str]:
        """Finds the symbol in the graph that contains this line number."""
        for key, sym in self.graph.symbols.items():
            if sym.file_path == file_path:
                if sym.start_line <= line <= sym.end_line:
                    return sym.name
        return None

    def format_for_llm(self, locations: List[ErrorLocation]) -> str:
        """Formats resolved errors for inclusion in an LLM prompt."""
        if not locations:
            return "No specific error locations could be attributed."

        lines = ["### Structured Error Attribution"]
        for loc in locations:
            lines.append(f"- **Symbol**: `{loc.symbol or 'Module Level'}`")
            lines.append(f"  - **File**: {loc.file_path}:{loc.line}")
            lines.append(f"  - **Error**: {loc.message}")

            if loc.symbol:
                callers = self._get_callers(loc.symbol)
                if callers:
                    lines.append(f"  - **Called by**: {', '.join(callers)}")

        return "\n".join(lines)

    def _get_callers(self, symbol_name: str) -> List[str]:
        """Find symbols that call the given symbol."""
        callers = []
        for key, sym in self.graph.symbols.items():
            if sym.name == symbol_name:
                for caller_key in sym.incoming_calls:
                    caller = self.graph.symbols.get(caller_key)
                    if caller:
                        callers.append(caller.name)
        return callers[:5]
