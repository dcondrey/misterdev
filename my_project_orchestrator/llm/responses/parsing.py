"""Parse LLM output into structured file edits.

A line-by-line state machine extracts whole-file fenced blocks, anchored
SEARCH/REPLACE hunks, and unified diffs, plus the pure helpers that detect
fences and resolve file paths from fence lines, leading comments, or
surrounding prose.
"""

from typing import Dict, List, Optional

from .models import SearchReplaceEdit, _CodeBlock


class LLMResponseParser:
    """Parses LLM output into structured file edits using a line-by-line
    state machine. No regex; handles nested backticks, variable fence
    widths, and all common LLM output formats.

    Supported formats:
    1. Tagged:    ```python:path/to/file.py
    2. Comment:   ```python\\n# path/to/file.py\\n...
    3. Path-only: ```path/to/file.py
    4. Preceding: Update `path/to/file.py`:\\n```
    5. Unified diff: --- a/file  +++ b/file
    """

    _CODE_EXTENSIONS = frozenset(
        {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".rs",
            ".go",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".zig",
            ".cs",
            ".csproj",
            ".csx",
            ".xaml",
            ".sln",
            ".cc",
            ".cxx",
            ".hh",
            ".hxx",
            ".toml",
            ".yaml",
            ".yml",
            ".json",
            ".md",
            ".html",
            ".css",
            ".scss",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".cfg",
            ".ini",
            ".env",
            ".txt",
            ".xml",
        }
    )

    @staticmethod
    def parse_file_edits(llm_output: str) -> Dict[str, str]:
        edits = LLMResponseParser._parse_unified_diffs(llm_output)
        if edits:
            return edits

        blocks = LLMResponseParser._extract_code_blocks(llm_output)
        result: Dict[str, str] = {}
        for block in blocks:
            path = LLMResponseParser._resolve_path(block)
            if path:
                result[path] = "\n".join(block.content_lines)
        return result

    @staticmethod
    def parse_search_replace_blocks(llm_output: str) -> List[SearchReplaceEdit]:
        """Extract surgical SEARCH/REPLACE hunks from the model's output.

        Recognized format (the path may sit on a fence-open line, on a bare
        line, or in backticks immediately above the marker)::

            ```rust:path/to/file.rs
            <<<<<<< SEARCH
            exact existing snippet
            =======
            replacement snippet
            >>>>>>> REPLACE
            ```

        Multiple hunks may follow one path. Returns an empty list when the
        output contains no markers, in which case the caller falls back to the
        whole-file parser.
        """
        lines = llm_output.split("\n")
        edits: List[SearchReplaceEdit] = []
        current_path: Optional[str] = None
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            fence, remainder = _parse_fence_open(line)
            if fence is not None:
                _, path_hint = _parse_opening(remainder)
                if path_hint:
                    current_path = path_hint.strip().strip("`'\"")
                i += 1
                continue
            if _is_search_marker(line):
                search_lines: List[str] = []
                i += 1
                while i < n and not _is_divider_marker(lines[i]):
                    search_lines.append(lines[i])
                    i += 1
                i += 1  # skip divider
                replace_lines: List[str] = []
                while i < n and not _is_replace_marker(lines[i]):
                    replace_lines.append(lines[i])
                    i += 1
                i += 1  # skip replace marker
                if current_path:
                    edits.append(
                        SearchReplaceEdit(
                            path=current_path,
                            search="\n".join(search_lines),
                            replace="\n".join(replace_lines),
                        )
                    )
                continue
            stripped = line.strip().strip("`'\"")
            if stripped.startswith("./"):
                stripped = stripped[2:]
            if _looks_like_path(stripped):
                current_path = stripped
            i += 1
        return edits

    # ------------------------------------------------------------------
    # State machine: extract code blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_blocks(text: str) -> List[_CodeBlock]:
        blocks = []
        lines = text.split("\n")
        i = 0
        preceding_start = 0

        while i < len(lines):
            fence, remainder = _parse_fence_open(lines[i])
            if fence is None:
                i += 1
                continue

            # Capture the text before this block for path detection
            preceding = "\n".join(lines[preceding_start:i])

            # Parse lang:path or lang or path from the remainder
            lang, path_hint = _parse_opening(remainder)

            # Collect content lines until closing fence
            content_lines = []
            i += 1
            while i < len(lines):
                if _is_fence_close(lines[i], fence):
                    break
                content_lines.append(lines[i])
                i += 1

            blocks.append(
                _CodeBlock(
                    lang=lang,
                    path_hint=path_hint,
                    content_lines=content_lines,
                    preceding_text=preceding,
                )
            )

            preceding_start = i + 1
            i += 1

        return blocks

    # ------------------------------------------------------------------
    # Path resolution: try each strategy in priority order
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(block: _CodeBlock) -> Optional[str]:
        path = None

        # 1. Explicit path from opening line (```python:path or ```path)
        if block.path_hint:
            path = block.path_hint

        # 2. First line of content is a path comment
        if not path and block.content_lines:
            extracted = _extract_path_from_first_line(block.content_lines[0])
            if extracted:
                path = extracted
                block.content_lines = block.content_lines[1:]

        # 3. Preceding text contains a backtick-quoted or labeled path
        if not path:
            path = _extract_path_from_preceding(block.preceding_text)

        if path:
            path = path.strip().strip("`'\"")
            if path.startswith("./"):
                path = path[2:]
            if _looks_like_path(path):
                return path

        return None

    # ------------------------------------------------------------------
    # Unified diff parser (line-by-line, no regex)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_unified_diffs(text: str) -> Dict[str, str]:
        if "--- a/" not in text and "--- " not in text:
            return {}

        edits: Dict[str, str] = {}
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            # Look for --- a/path
            if not lines[i].startswith("--- "):
                i += 1
                continue

            # Next line must be +++ b/path
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                i += 1
                continue

            plus_line = lines[i + 1]
            file_path = plus_line[4:].strip()
            if file_path.startswith("b/"):
                file_path = file_path[2:]

            i += 2
            new_lines = []
            hunk_count = 0
            new_start: Optional[int] = None

            # Collect hunks
            while i < len(lines):
                line = lines[i]
                if line.startswith("--- "):
                    break  # next diff
                if line.startswith("@@"):
                    hunk_count += 1
                    if hunk_count == 1:
                        new_start = _unified_new_start(line)
                    i += 1
                    continue
                if line.startswith("+") and not line.startswith("+++"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    new_lines.append(line[1:])
                elif not line.startswith("-"):
                    break  # end of diff section
                i += 1

            # This parser rebuilds the whole file from the diff's +/context
            # lines, which equals the real file ONLY for a single hunk that
            # starts at line 1 (or a brand-new file). A partial or multi-hunk
            # diff omits the unchanged regions between hunks, so reconstructing
            # from it would TRUNCATE the file. Skip that unsafe case so the
            # caller falls back (or retries) instead of writing a corrupt file.
            if new_lines and hunk_count == 1 and new_start in (0, 1):
                edits[file_path] = "\n".join(new_lines)

        return edits


# ------------------------------------------------------------------
# Pure-function helpers (no regex, no state)
# ------------------------------------------------------------------


def _unified_new_start(header: str) -> Optional[int]:
    """New-file start line from a ``@@ -a,b +c,d @@`` hunk header, or None.

    Returns ``c`` (the first line number of the new side), used to confirm a
    single-hunk diff begins at line 1 before its reconstruction is trusted as
    whole-file content.
    """
    plus = header.find("+")
    if plus == -1:
        return None
    digits = ""
    for ch in header[plus + 1 :]:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def _is_search_marker(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("<<<<<<<") and "SEARCH" in stripped


def _is_replace_marker(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(">>>>>>>") and "REPLACE" in stripped


def _is_divider_marker(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 5 and all(c == "=" for c in stripped)


def _parse_fence_open(line: str) -> tuple:
    """Check if a line is a code fence opener (``` or ~~~~).

    Returns (fence_char, remainder) or (None, None).
    fence_char is the repeated character (` or ~) at the exact width used,
    so the closer can be matched precisely.
    """
    stripped = line.lstrip()
    for char in ("`", "~"):
        if stripped.startswith(char * 3):
            # Count fence width
            width = 0
            for c in stripped:
                if c == char:
                    width += 1
                else:
                    break
            remainder = stripped[width:].strip()
            fence = char * width
            return fence, remainder
    return None, None


def _is_fence_close(line: str, fence: str) -> bool:
    """Check if a line closes the given fence."""
    stripped = line.strip()
    # Must start with at least as many fence chars and have nothing else
    if not stripped.startswith(fence):
        return False
    # Everything after the fence chars must also be the fence char or empty
    rest = stripped[len(fence) :]
    return all(c == fence[0] for c in rest)


def _parse_opening(remainder: str) -> tuple:
    """Parse the text after ``` into (lang, path_hint).

    Examples:
      "python:src/main.py" -> ("python", "src/main.py")
      "python"             -> ("python", "")
      "src/main.py"        -> ("", "src/main.py")
      ""                   -> ("", "")
    """
    if not remainder:
        return "", ""

    # Check for lang:path format
    if ":" in remainder:
        parts = remainder.split(":", 1)
        lang_candidate = parts[0].strip()
        path_candidate = parts[1].strip()
        if _looks_like_path(path_candidate):
            return lang_candidate, path_candidate

    # Is the whole thing a path?
    if _looks_like_path(remainder):
        return "", remainder

    # Otherwise it's a language identifier
    return remainder, ""


def _extract_path_from_first_line(line: str) -> Optional[str]:
    """Extract a file path from a comment on the first line of a code block."""
    stripped = line.strip()

    # Comment prefixes: # path, // path, -- path, /* path */, <!-- path -->
    comment_prefixes = ("#", "//", "--", "/*", "<!--")
    for prefix in comment_prefixes:
        if stripped.startswith(prefix):
            remainder = stripped[len(prefix) :].strip()
            # Strip trailing comment closers
            for closer in ("*/", "-->"):
                if remainder.endswith(closer):
                    remainder = remainder[: -len(closer)].strip()
            if _looks_like_path(remainder):
                return remainder

    # "File: path" or "Filename: path" or "Path: path"
    for label in ("file:", "filename:", "path:"):
        lower = stripped.lower()
        if lower.startswith(label):
            candidate = stripped[len(label) :].strip()
            if _looks_like_path(candidate):
                return candidate

    return None


def _extract_path_from_preceding(text: str) -> Optional[str]:
    """Find a file path in text that precedes a code block.

    Scans backward from the end looking for backtick-quoted paths,
    bold paths, or labeled paths like "Update `file.py`:" or
    "**file.py**:".
    """
    # Only look at the last ~200 chars
    segment = text[-200:] if len(text) > 200 else text

    # Strategy 1: find last backtick-quoted path
    path = _find_last_quoted_path(segment, "`")
    if path:
        return path

    # Strategy 2: find last bold-quoted path (**path**)
    path = _find_last_delimited_path(segment, "**", "**")
    if path:
        return path

    # Strategy 3: labeled path (File: path, Update: path, etc.)
    for label in (
        "file:",
        "update:",
        "create:",
        "modify:",
        "in:",
        "filename:",
        "path:",
    ):
        idx = segment.lower().rfind(label)
        if idx >= 0:
            after = segment[idx + len(label) :].strip()
            # Take first whitespace-delimited token
            candidate = after.split()[0].strip("`'\",:") if after else ""
            if _looks_like_path(candidate):
                return candidate

    return None


def _find_last_quoted_path(text: str, quote: str) -> Optional[str]:
    """Find the last `path.ext` in text."""
    end = len(text)
    while True:
        close = text.rfind(quote, 0, end)
        if close < 0:
            return None
        open_pos = text.rfind(quote, 0, close)
        if open_pos < 0:
            return None
        candidate = text[open_pos + len(quote) : close]
        if _looks_like_path(candidate):
            return candidate
        end = open_pos


def _find_last_delimited_path(text: str, opener: str, closer: str) -> Optional[str]:
    """Find the last **path.ext** in text."""
    end = len(text)
    while True:
        close = text.rfind(closer, 0, end)
        if close < 0:
            return None
        open_pos = text.rfind(opener, 0, close)
        if open_pos < 0:
            return None
        candidate = text[open_pos + len(opener) : close]
        if _looks_like_path(candidate):
            return candidate
        end = open_pos


def _looks_like_path(s: str) -> bool:
    """Check if a string looks like a file path."""
    s = s.strip().strip("`'\"")
    if not s or " " in s or len(s) > 200:
        return False
    dot = s.rfind(".")
    if dot < 0:
        return False
    ext = s[dot:]
    return ext in LLMResponseParser._CODE_EXTENSIONS
