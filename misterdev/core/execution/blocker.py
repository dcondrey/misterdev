"""Classify a gate failure as blocked-on-an-external-resource, not a code bug.

Some task failures are not the model's to fix: a command needs a credential
(``wrangler`` is not logged in), an account id, an API token, a paid service, or
the network — retrying the same edit forever cannot resolve it. This detects
those signals so the executor can PARK the task with a specific question for the
user instead of burning attempts and then failing (see
``core/execution/deferral.py`` and the executor's terminal seam).

Pure and signal-only: it reads command/gate output and returns a short human
reason (or None). It never decides policy — the caller decides whether to defer.
A conservative allowlist of phrases keeps a genuine code error (a failing
assertion, a type error) from being misread as "blocked".
"""

import re
from typing import Optional, Tuple

# (compiled signal, human reason). Ordered most-specific first. Every phrase is
# one an EXTERNAL resource emits — never something a normal compile/test failure
# prints — so a real code error is not misclassified as a block.
_SIGNALS: Tuple[Tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"\bnot logged in\b|\bwrangler login\b|please run `?wrangler login", re.I
        ),
        "not authenticated with Cloudflare (run `wrangler login`)",
    ),
    (
        re.compile(
            r"\baccount[_ ]id\b.*(required|missing|not found)|missing account_id", re.I
        ),
        "a Cloudflare account_id is required",
    ),
    # Credential-ish terms (env-var style ``_TOKEN`` / ``_KEY`` included) paired
    # with an ABSENCE word only — never "invalid"/"expired", which are validation
    # RESULTS a normal test emits ("expected api key to be valid, got invalid" is a
    # feature test, not a missing-credential block; a genuinely bad credential
    # surfaces as 401/unauthorized below). This keeps a task that IMPLEMENTS
    # api-key issuance from being misread as needing an external key.
    (
        re.compile(
            r"(?:api[_ ]?key|api[_ ]?token|access[_ ]?token|auth[_ ]?token|"
            r"bearer\s+token|_token\b|_key\b|secret|credentials?)"
            r".{0,30}(?:is required|are required|required\b|missing(?!\s+from)|"
            r"not set|not configured|not provided|unset|undefined)|"
            r"(?:required|missing|no|unset|undefined|set the|provide (?:a|the|your))"
            r"\s.{0,20}(?:api[_ ]?key|api[_ ]?token|access[_ ]?token|auth[_ ]?token|"
            r"_token\b|_key\b|credentials?)",
            re.I,
        ),
        "a required API key / token / secret is missing",
    ),
    (
        re.compile(
            r"\b401\b|\bunauthorized\b|authentication failed|invalid credentials", re.I
        ),
        "authentication failed (401 / invalid credentials)",
    ),
    (
        re.compile(r"\b403\b|\bforbidden\b|access denied", re.I),
        "access forbidden (403) — the account lacks permission",
    ),
    (
        re.compile(
            r"\benv(ironment)?[_ ]?var(iable)?\b.*(required|missing|not set|undefined)|"
            r"(missing|unset).{0,20}environment variable",
            re.I,
        ),
        "a required environment variable / secret is not set",
    ),
    (
        re.compile(
            r"\bENOTFOUND\b|\bEAI_AGAIN\b|getaddrinfo|network is unreachable|"
            r"could not resolve host|connection refused|connection timed out",
            re.I,
        ),
        "a network / external service is unreachable",
    ),
    (
        re.compile(
            r"\bpayment required\b|\b402\b|quota exceeded|insufficient credits|"
            r"billing",
            re.I,
        ),
        "a paid service / quota / billing action is required",
    ),
    (
        re.compile(r"\bpermission denied\b|\bEACCES\b|operation not permitted", re.I),
        "a filesystem / OS permission is required",
    ),
)


def blocked_reason(output: str) -> Optional[str]:
    """A short human reason when ``output`` shows an external-resource block, else
    None. None means "treat as an ordinary (retryable / real) failure"."""
    text = output or ""
    for pattern, reason in _SIGNALS:
        if pattern.search(text):
            return reason
    return None
