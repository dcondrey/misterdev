# Structured tool for edit extraction. Forcing this (when a model supports
# `tools`) replaces brittle markdown-fence parsing: the model returns
# well-formed JSON we render back into the canonical fence format the executor
# already consumes, so nothing downstream changes.
APPLY_EDITS_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_edits",
        "description": (
            "Write the complete final content of each file to create or modify "
            "to satisfy the task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Project-relative file path.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Full final content of the file.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                }
            },
            "required": ["edits"],
        },
    },
}


def _edits_to_markdown(edits: list) -> str:
    """Render structured edits into the canonical ```lang:path fence format.

    The executor's parser keys on the ``path`` after the colon; the language
    token is re-derived from the path downstream, so a placeholder is fine.
    """
    blocks = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        path = edit.get("path")
        content = edit.get("content", "")
        if path:
            blocks.append(f"```text:{path}\n{content}\n```")
    return "\n\n".join(blocks)


def code_gen_abort_check(accumulated: str) -> bool:
    """Heuristic: True when a code-gen stream is clearly going wrong.

    Trips when a lot of text arrives with no code fence or file marker, or when
    the model opens with conversational filler instead of code.
    """
    if (
        len(accumulated) > 2000
        and "```" not in accumulated
        and "# File:" not in accumulated
    ):
        return True
    head = accumulated[:200]
    return ("I'll help you" in head) or ("Sure, here" in head)
