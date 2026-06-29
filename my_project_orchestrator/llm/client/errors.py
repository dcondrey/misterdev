from typing import Optional


class BudgetExceededError(Exception):
    pass


class LLMCallError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# HTTP status codes that denote a transient failure worth retrying / failing
# over. The 4xx members are the standard retryable ones (request timeout,
# conflict, too-early, rate limit); any other 4xx is a hard client error.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Exception class names (across the openai / anthropic / httpx SDKs) that always
# denote a transient failure, regardless of how the message happens to be
# phrased. Matched by bare class name so neither SDK must be importable here.
RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
        "TimeoutError",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
    }
)

# Last-resort substring signals for plain exceptions that carry neither a status
# code nor a recognizable type (e.g. a provider that raises RuntimeError with a
# descriptive message). Checked only after the structured signals above.
RETRYABLE_ERROR_MARKERS = (
    "rate limit",
    "too many requests",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "overloaded",
    "connection",
    "try again",
    "429",
    "500",
    "502",
    "503",
    "504",
    "529",
)


def _error_status_code(error: Exception) -> Optional[int]:
    """Best-effort HTTP status code from a provider exception, or None.

    Covers the openai/anthropic ``status_code`` attribute and the httpx-style
    nested ``response.status_code``. ``bool`` is excluded so a stray ``True``
    isn't read as code 1.
    """
    for attr in ("status_code", "http_status"):
        val = getattr(error, attr, None)
        if isinstance(val, bool):
            continue
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
    resp = getattr(error, "response", None)
    sc = getattr(resp, "status_code", None)
    if isinstance(sc, int) and not isinstance(sc, bool):
        return sc
    return None


def _is_retryable_error(error: Exception) -> bool:
    """Classify an exception as transient (retry / fail over) or terminal.

    Structured signals win: an explicit HTTP status code or a known transient
    exception type is authoritative, so a hard 4xx (e.g. 400/401/404) is never
    retried even if its message text coincidentally contains a retryable marker.
    Only when no structured signal is present do we fall back to substrings.
    """
    code = _error_status_code(error)
    if code is not None:
        if code in RETRYABLE_STATUS_CODES:
            return True
        if 400 <= code < 500:
            return False
        if code >= 500:
            return True
    if type(error).__name__ in RETRYABLE_EXCEPTION_NAMES:
        return True
    text = str(error).lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def _api_error(provider: str, error: Exception) -> "LLMCallError":
    """Wrap a provider exception as an LLMCallError with retryability classified."""
    return LLMCallError(
        f"{provider} API error: {error}", retryable=_is_retryable_error(error)
    )
