"""Edit shapes and errors shared across the response parsers."""

from dataclasses import dataclass


@dataclass
class FileEdit:
    path: str
    content: str
    is_new: bool = False


@dataclass
class SearchReplaceEdit:
    """A single anchored, surgical edit to one file.

    ``search`` is the exact existing snippet to locate; ``replace`` is what
    takes its place. An empty ``search`` means "create this file" and is only
    valid when the file does not yet exist. Multiple edits may target the same
    path and are applied in order against the evolving content.
    """

    path: str
    search: str
    replace: str


class EditConflictError(Exception):
    """A search/replace hunk could not be applied unambiguously.

    Raised when the SEARCH snippet is absent, matches more than once, or would
    blank/overwrite an existing file. The caller surfaces the message back to
    the model as an error so it retries with a corrected anchor — a partial or
    truncated file is never written.
    """


@dataclass
class _CodeBlock:
    lang: str
    path_hint: str
    content_lines: list
    preceding_text: str
