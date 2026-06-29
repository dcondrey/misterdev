from dataclasses import dataclass, field


@dataclass
class LLMUsage:
    """Token usage tracking for budget enforcement."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    call_count: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""
    finish_reason: str = ""
