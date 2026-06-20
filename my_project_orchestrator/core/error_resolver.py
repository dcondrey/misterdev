"""Error attribution and resolution for build/test failures.

Parses compiler and test-runner output to locate the source files and
line numbers responsible for each error, then formats that information
for injection into LLM prompts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Patterns that match common error location formats:
#   file.py:42: error message
#   file.py:42:10: error message
#   File "file.py", line 42
#   --> file.py:42:10
_LOCATION_PATTERNS: list[re.Pattern] = [
    re.compile(r'^(?P<file>[^\s:]+\.(?:py|js|ts|rs|go|java|c|cpp|h|rb|cs))'
               r':(?P<line>\d+)(?::\d+)?[:\s]', re.MULTILINE),
    re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)', re.MULTILINE),
    re.compile(r'-->\s*(?P<file>[^\s:]+):(?P<line>\d+):\d+', re.MULTILINE),
    re.compile(r'at (?P<file>[^\s(]+)\(.*:(?P<line>\d+)\)', re.MULTILINE),
]


class ErrorLocation:
    """A single attributed error location."""

    def __init__(self, file: str, line: int, snippet: str = ""):
        self.file = file
        self.line = line
        self.snippet = snippet

    def __repr__(self) -> str:  # pragma: no cover
        return f"ErrorLocation({self.file}:{self.line})"


class ErrorResolver:
    """Resolves build/test error output to source locations.

    Parameters
    ----------
    project_path:
        Root directory of the project (used to read source snippets).
    dependency_graph:
        Optional topology graph (currently reserved for future use –
        pass ``None`` or an empty dict if unavailable).
    """

    def __init__(self, project_path: Path, dependency_graph: Optional[Any] = None):
        self.project_path = project_path
        self.graph = dependency_graph  # reserved for future topology-aware resolution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_errors(self, error_output: str) -> List[ErrorLocation]:
        """Parse *error_output* and return attributed :class:`ErrorLocation` objects."""
        locations: List[ErrorLocation] = []
        seen: set[tuple[str, int]] = set()

        for pattern in _LOCATION_PATTERNS:
            for match in pattern.finditer(error_output):
                file_str = match.group("file")
                try:
                    line_no = int(match.group("line"))
                except (ValueError, IndexError):
                    continue

                key = (file_str, line_no)
                if key in seen:
                    continue
                seen.add(key)

                snippet = self._read_snippet(file_str, line_no)
                locations.append(ErrorLocation(file=file_str, line=line_no, snippet=snippet))

        if not locations:
            logger.debug("ErrorResolver: no structured locations found in error output.")

        return locations

    def format_for_llm(self, locations: List[ErrorLocation]) -> str:
        """Format attributed locations as a prompt-ready string."""
        if not locations:
            return ""

        lines = ["## Error Attribution"]
        for loc in locations[:10]:  # cap to avoid bloating context
            lines.append(f"\n### {loc.file}:{loc.line}")
            if loc.snippet:
                lines.append(f"```\n{loc.snippet}\n```")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_snippet(self, file_str: str, line_no: int, context: int = 5) -> str:
        """Return a few lines of source around *line_no* for context."""
        # Try relative to project root first, then as absolute path.
        candidates = [
            self.project_path / file_str,
            Path(file_str),
        ]
        for path in candidates:
            if path.exists():
                try:
                    source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    start = max(0, line_no - context - 1)
                    end = min(len(source_lines), line_no + context)
                    numbered = [
                        f"{'>' if i + 1 == line_no else ' '} {i + 1:4d} | {source_lines[i]}"
                        for i in range(start, end)
                    ]
                    return "\n".join(numbered)
                except OSError:
                    return ""
        return ""