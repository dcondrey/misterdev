"""Apply anchored SEARCH/REPLACE hunks against on-disk file content."""

from typing import List

from .models import SearchReplaceEdit, EditConflictError


def apply_search_replace(original: str, edits: List[SearchReplaceEdit]) -> str:
    """Apply anchored hunks to ``original`` and return the full new content.

    Each hunk's SEARCH snippet must resolve to exactly one location; a missing
    or ambiguous anchor raises :class:`EditConflictError` so the caller can
    retry rather than write a corrupt file. An empty SEARCH creates a new file
    and is rejected if the file already has content. Matching is line-anchored
    (never a bare substring, which would splice mid-line), with a per-line
    whitespace/line-ending-tolerant comparison that absorbs the trailing-space
    and CRLF drift models commonly introduce without ever matching more than one
    place.
    """
    content = original
    for edit in edits:
        if not edit.search.strip():
            if content.strip():
                raise EditConflictError(
                    f"{edit.path}: empty SEARCH block is only valid for a new "
                    "file, but this file already has content. Anchor the edit "
                    "to the exact lines you are replacing."
                )
            content = edit.replace
            continue
        content = _apply_one_hunk(content, edit)
    return content


def _apply_one_hunk(content: str, edit: SearchReplaceEdit) -> str:
    spliced, hits = _line_splice(content, edit.search, edit.replace)
    if hits == 1:
        return spliced
    if hits == 0:
        raise EditConflictError(
            f"{edit.path}: SEARCH block not found. It must match the current "
            "file (whitespace aside). Re-read the file and copy the lines "
            "verbatim."
        )
    raise EditConflictError(
        f"{edit.path}: SEARCH block matches {hits} locations. Include more "
        "surrounding context so it identifies exactly one place."
    )


def _line_splice(content: str, search: str, replace: str) -> tuple:
    """Locate ``search`` as a window of whole LINES in ``content`` and replace it.

    Line-anchored, never a bare substring: a single-line hunk like ``x = 1`` is
    matched against whole lines, so it can never splice into the middle of a
    longer line such as ``x = 10``. Two passes, each requiring exactly one
    matching window (any other count is a conflict and the content is returned
    unmutated):

    1. right-stripped / CR-free lines — absorbs trailing-space and CRLF drift,
       keeping indentation significant;
    2. fully-stripped lines — absorbs wrong indentation too, then re-indents the
       replacement to the file's actual indentation so formatting is preserved.

    Replacement lines inherit the file's line ending, so a CRLF file stays CRLF.
    """
    crlf = "\r\n" in content
    o_lines = content.split("\n")
    s_lines = search.split("\n")
    r_lines = replace.split("\n")

    rstripped = [ln.replace("\r", "").rstrip() for ln in o_lines]
    ns_r = [ln.replace("\r", "").rstrip() for ln in s_lines]
    k = len(ns_r)
    if k == 0 or k > len(o_lines):
        return content, 0
    hits = [i for i in range(len(rstripped) - k + 1) if rstripped[i : i + k] == ns_r]
    if len(hits) == 1:
        return _join_splice(o_lines, hits[0], k, r_lines, crlf), 1
    if len(hits) > 1:
        return content, len(hits)

    # Pass 2: ignore indentation, then reapply the file's indentation.
    stripped = [ln.strip() for ln in o_lines]
    ns_s = [ln.strip() for ln in s_lines]
    hits = [i for i in range(len(stripped) - k + 1) if stripped[i : i + k] == ns_s]
    if len(hits) != 1:
        return content, len(hits)
    i = hits[0]
    reindented = _reindent(o_lines[i : i + k], s_lines, r_lines)
    return _join_splice(o_lines, i, k, reindented, crlf), 1


def _join_splice(
    o_lines: List[str], i: int, k: int, r_lines: List[str], crlf: bool
) -> str:
    """Splice ``r_lines`` over ``o_lines[i:i+k]``, giving the replacement lines the
    file's line ending so a CRLF file does not end up with mixed CRLF/LF lines."""
    if crlf:
        r_lines = [ln if ln.endswith("\r") else ln + "\r" for ln in r_lines]
    return "\n".join(o_lines[:i] + r_lines + o_lines[i + k :])


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(matched: List[str], search: List[str], replace: List[str]) -> List[str]:
    """Shift ``replace`` by the indent delta between matched and search anchors.

    The delta is measured on the first non-blank line of each; applying it to
    every non-blank replacement line lands the new code at the file's real
    indentation even when the model used a different indent in its SEARCH.
    """

    def first_indent(lines: List[str]) -> str:
        for ln in lines:
            if ln.strip():
                return _leading_ws(ln.replace("\r", ""))
        return ""

    target = first_indent(matched)
    source = first_indent(search)
    out = []
    for ln in replace:
        if not ln.strip():
            out.append(ln)
            continue
        body = ln[len(source) :] if ln.startswith(source) else ln.lstrip()
        out.append(target + body)
    return out
