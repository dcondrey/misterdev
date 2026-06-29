"""Shared text helpers used by the language-specific extractors."""


def _extract_name(text: str) -> str:
    """Extract identifier name from text (stops at non-alphanumeric)."""
    name = []
    for ch in text.strip():
        if ch.isalnum() or ch == "_":
            name.append(ch)
        else:
            break
    return "".join(name)
