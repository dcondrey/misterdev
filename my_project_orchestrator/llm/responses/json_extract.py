"""Extract JSON values embedded in prose or code fences from LLM output."""

import json
from typing import Any, Optional


def extract_json_array(response: str, default: Optional[Any] = None) -> Any:
    """Extract the outermost JSON array from an LLM response.

    LLMs often wrap a JSON array in prose or code fences; this slices from the
    first ``[`` to the last ``]`` and parses. Returns ``default`` (or ``[]``)
    on any failure. Consolidates the find/slice/loads pattern that was copy-
    pasted across the decomposer, advisor, sovereign, and metacognition.
    """
    if default is None:
        default = []
    text = (response or "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return default
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return default


def extract_balanced_span(text: str, start: int) -> Optional[str]:
    """Return the balanced ``{...}`` substring beginning at ``start`` (a ``{``).

    Brace-counts while honoring braces inside double-quoted strings (with
    escapes), so an object containing ``{`` or ``}`` in a string value is
    extracted whole; tolerates the object spanning multiple lines. Returns None
    if the span never balances. Consolidates the string-aware brace scan that was
    duplicated in the goal check and the MCP gather loop.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(text: str) -> Optional[dict]:
    """Return the first parseable top-level JSON object in ``text``, or None.

    Scans each ``{`` for a balanced, string-aware span and json-loads it; a span
    that fails to parse is skipped and the next ``{`` is tried. Survives leading
    prose or a ```json fence around the object.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        span = extract_balanced_span(text, start)
        if span is not None:
            try:
                parsed = json.loads(span)
                return parsed if isinstance(parsed, dict) else None
            except ValueError:
                pass  # malformed span; try the next '{'
        start = text.find("{", start + 1)
    return None
