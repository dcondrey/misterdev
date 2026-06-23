"""Context budget management for LLM prompts.

Estimates token counts and intelligently truncates context sections
to stay within model limits. By task 15+ in a multi-task build,
unmanaged context can exceed 100K tokens and degrade LLM quality.
"""

from typing import Dict
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Approximate tokens per character for English/code (conservative). Used only
# when tiktoken is unavailable.
CHARS_PER_TOKEN = 3.5

_encoder = None
_encoder_loaded = False


def _get_encoder():
    """Lazily load a tiktoken encoder; cache the result (incl. failure)."""
    global _encoder, _encoder_loaded
    if _encoder_loaded:
        return _encoder
    _encoder_loaded = True
    try:
        import tiktoken

        # cl100k_base is a good general proxy for code/English token counts and
        # avoids a per-model lookup that may miss newer model ids.
        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # pragma: no cover - import or offline BPE fetch
        logger.debug(f"tiktoken unavailable; using char heuristic: {e}")
        _encoder = None
    return _encoder


def estimate_tokens(text: str) -> int:
    """Count tokens with tiktoken when available, else a char heuristic.

    Accurate counts keep ContextBudget from over- or under-filling the window
    (the heuristic mis-sizes code, especially the windowed large-file context).
    """
    if not text:
        return 1
    enc = _get_encoder()
    if enc is not None:
        try:
            return max(1, len(enc.encode(text, disallowed_special=())))
        except Exception:  # pragma: no cover - encoder edge cases
            pass
    return max(1, int(len(text) / CHARS_PER_TOKEN))


class ContextBudget:
    """Allocates a token budget across context sections and truncates as needed.

    Usage:
        budget = ContextBudget(max_tokens=100000)
        budget.set("code_context", big_code_string, priority=1)
        budget.set("scratchpad", scratchpad_text, priority=3)
        budget.set("interface_contracts", contracts, priority=2)
        budget.set("error_logs", errors, priority=1)

        # Get truncated versions that fit within budget
        sections = budget.allocate()
        code = sections["code_context"]
    """

    def __init__(self, max_tokens: int = 100000, reserved_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens  # for system prompt + LLM response
        self.available = max_tokens - reserved_tokens
        self._sections: Dict[str, _Section] = {}

    def set(self, name: str, content: str, priority: int = 2, min_lines: int = 10):
        """Register a context section.

        priority: 1 = essential (truncate last), 2 = important, 3 = nice-to-have (truncate first)
        min_lines: minimum lines to keep even under pressure
        """
        self._sections[name] = _Section(
            name=name,
            content=content,
            priority=priority,
            min_lines=min_lines,
            tokens=estimate_tokens(content),
        )

    def allocate(self) -> Dict[str, str]:
        """Allocate budget and return truncated sections."""
        total = sum(s.tokens for s in self._sections.values())

        if total <= self.available:
            return {name: s.content for name, s in self._sections.items()}

        logger.warning(
            f"Context exceeds budget: {total} tokens > {self.available} available. Truncating."
        )

        # Sort by priority (highest number = truncate first)
        by_priority = sorted(self._sections.values(), key=lambda s: -s.priority)

        overflow = total - self.available
        result = {}

        for section in by_priority:
            if overflow <= 0:
                result[section.name] = section.content
                continue

            # How much can we trim from this section?
            lines = section.content.splitlines()
            if len(lines) <= section.min_lines:
                result[section.name] = section.content
                continue

            # Binary search for the right number of lines
            target_tokens = max(
                estimate_tokens("\n".join(lines[: section.min_lines])),
                section.tokens - overflow,
            )

            kept = _truncate_to_tokens(lines, target_tokens, section.name)
            saved = section.tokens - estimate_tokens(kept)
            overflow -= saved
            result[section.name] = kept

        return result

    def summary(self) -> str:
        total = sum(s.tokens for s in self._sections.values())
        parts = [f"budget={self.available}"]
        for s in sorted(self._sections.values(), key=lambda s: s.priority):
            parts.append(f"{s.name}={s.tokens}t(p{s.priority})")
        parts.append(f"total={total}")
        over = total > self.available
        parts.append("OVER" if over else "OK")
        return " | ".join(parts)


class _Section:
    def __init__(
        self, name: str, content: str, priority: int, min_lines: int, tokens: int
    ):
        self.name = name
        self.content = content
        self.priority = priority
        self.min_lines = min_lines
        self.tokens = tokens


def _truncate_to_tokens(lines: list, target_tokens: int, name: str) -> str:
    """Keep lines from the start until we hit the target token count."""
    kept = []
    running = 0
    for line in lines:
        line_tokens = estimate_tokens(line)
        if running + line_tokens > target_tokens and len(kept) > 0:
            break
        kept.append(line)
        running += line_tokens

    total_lines = len(lines)
    omitted = total_lines - len(kept)
    if omitted > 0:
        kept.append(
            f"\n... ({omitted} lines omitted from {name} to fit context budget)"
        )
        logger.info(f"Truncated {name}: kept {len(kept)}/{total_lines} lines")

    return "\n".join(kept)
