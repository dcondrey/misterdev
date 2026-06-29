"""Parse LLM output into file edits.

Two edit shapes are supported: anchored SEARCH/REPLACE hunks (the surgical
default — the model emits only changed regions, applied against the on-disk file
so large files are never truncated) and whole-file fenced blocks (fallback for
new/short files). apply_search_replace() matches exactly, then tolerates
trailing-whitespace/CRLF drift, then wrong indentation (re-indenting the
replacement), always requiring a single unique match so a partial file is never
written.
"""

from .json_extract import (
    extract_json_array,
    extract_balanced_span,
    extract_json_object,
)
from .models import (
    FileEdit,
    SearchReplaceEdit,
    EditConflictError,
)
from .parsing import LLMResponseParser
from .apply import apply_search_replace

__all__ = [
    "extract_json_array",
    "extract_balanced_span",
    "extract_json_object",
    "FileEdit",
    "SearchReplaceEdit",
    "EditConflictError",
    "LLMResponseParser",
    "apply_search_replace",
]
