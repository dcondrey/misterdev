"""Independent verification of completeness claims before they become tasks.

The completeness analyzer flags features as "incomplete" or files as "stubs"
from a lossy project overview, so it can mislabel deliberate design — a
graceful-degradation path, a platform-gated no-op, a parity shim, a documented
fallback — as unfinished work. Acting on that wastes budget and risks rewriting
code that was correct by design (observed: a documented wasm degrade-to-empty
backend planned as a "fix the stub" task).

This gate gives each such CLAIM a second, INDEPENDENT look at the REAL file (and
whatever evidence the caller assembled — the file body, tests that exercise it,
the verified build/test state) and drops only the claims it can REFUTE with
evidence. Anything it confirms, or is unsure about, is KEPT — the burden of proof
is on dropping, so genuine work is never silently lost on a guess. It mirrors the
edit-time adversarial critic and the goal-completion judge: advisory and
best-effort, routed through an independent model when one is configured, and a
SKIP (keep the claim) on no client, an unparseable verdict, any error, or the
hard timeout. The decomposer and gates remain the ground truth.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional

from my_project_orchestrator.core.bounded import run_bounded
from my_project_orchestrator.core.independent import build_independent_call
from my_project_orchestrator.llm.responses import (
    extract_json_object as _extract_json_object,
)
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# A claim is KEPT (acted on) unless the verifier REFUTES it with evidence.
CONFIRMED = "confirmed"
REFUTED = "refuted"
KEPT = "kept"

# A verifier call takes the assembled prompt text and returns the model's text.
# Injected in tests; defaulted to the project client's generate_code path.
VerifyCall = Callable[[str], str]

_VERIFY_PROMPT = """A project analyzer claimed the following is INCOMPLETE work \
that needs implementing. Decide whether that claim is REAL or a FALSE POSITIVE.

It is a FALSE POSITIVE (refute it) when the code is intentional and complete:
- a documented graceful-degradation or fallback path (returns empty/default BY
  DESIGN, e.g. a missing-model or no-network contract),
- a platform-gated no-op (e.g. a wasm/no-filesystem backend), a parity shim, or
  a deliberate placeholder the design says is correct,
- a capability that already builds and is covered by passing tests.

It is REAL (confirm it) only when there is a concrete, unmet behavior the code is
supposed to provide but does not. Judge the SOURCE and tests, not names or docs
plans. If the evidence does not let you decide, say unsure — do not guess "real".

## Claim ({kind})
{label}: {description}

## Evidence (real source / tests / verified build+test state)
{evidence}

Return ONLY a JSON object, no prose or fences:
{{"real": true|false|null, "reason": "<one sentence citing the evidence>"}}
where false = false positive (intentional/complete), true = genuinely incomplete,
null = unsure."""


@dataclass(frozen=True)
class Claim:
    """One completeness claim to verify.

    ``kind`` is "incomplete" or "stub"; ``label`` is its identity (feature name
    or file path) used to drop it from the assessment; ``description`` and
    ``evidence`` are the text shown to the verifier (real file body, tests, and
    the verified build/test state — assembled by the caller).
    """

    kind: str
    label: str
    description: str = ""
    evidence: str = ""


@dataclass
class ClaimVerdict:
    """Outcome for one claim. ``status`` is CONFIRMED / REFUTED / KEPT.

    REFUTED means drop the claim (a false positive). CONFIRMED and KEPT both keep
    it; KEPT covers "unsure" and every skip/error path, so uncertainty never
    drops real work. ``reason`` is the model's one-line justification (or the
    skip reason); ``raw`` is its text.
    """

    claim: Claim
    status: str
    reason: str = ""
    raw: str = ""

    @property
    def refuted(self) -> bool:
        return self.status == REFUTED


# Evidence cap per claim so a huge file can't blow up the prompt or cost; the
# head carries the most signal (module docs, signatures, the flagged symbols).
_MAX_EVIDENCE_CHARS = 12000


def verify_claims(
    claims: List[Claim],
    verify_call: Optional[VerifyCall] = None,
    llm_client=None,
    model: Optional[str] = None,
    timeout: float = 45,
) -> List[ClaimVerdict]:
    """Verify each claim independently; REFUTE only false positives, else KEEP.

    The model call is performed by ``verify_call`` when supplied (the test seam);
    otherwise one is built from ``llm_client``. ``model``, when given, routes the
    judgment through an INDEPENDENT model so it does not share the generator's
    blind spots (its absence is logged — the same-model verifier is weaker).

    With no callable and no client, every claim is KEPT (no verifier available),
    so the behavior is byte-identical to not running the gate. ``timeout`` is the
    hard ceiling PER claim; an unparseable verdict, any error, or a timeout KEEPS
    that claim. Independent claims fan out across a small thread pool (the
    analyzers do the same) EXCEPT when an independent ``model`` is configured:
    routing through it uses ``with_model``, which mutates shared client state and
    is not thread-safe, so that path stays sequential. Returns one verdict per
    input claim, in order.
    """
    if not claims:
        return []

    call = verify_call or build_independent_call(
        llm_client,
        "You are a precise code reviewer. Return only valid JSON.",
        model,
        "Completeness-claim verifier",
    )
    if call is None:
        return [
            ClaimVerdict(c, KEPT, reason="no LLM verifier available") for c in claims
        ]

    def judge(claim: Claim) -> ClaimVerdict:
        prompt = _VERIFY_PROMPT.format(
            kind=claim.kind,
            label=claim.label or "(unnamed)",
            description=(claim.description or "(no description)").strip(),
            evidence=(claim.evidence or "(no evidence assembled)").strip()[
                :_MAX_EVIDENCE_CHARS
            ],
        )

        def _work() -> ClaimVerdict:
            try:
                return _parse_verdict(claim, call(prompt) or "")
            except Exception as e:  # any model/IO failure is non-fatal -> keep
                logger.debug(f"Claim verifier unavailable for {claim.label!r}: {e}")
                return ClaimVerdict(claim, KEPT, reason=f"error: {e}")

        return run_bounded(
            _work,
            timeout,
            ClaimVerdict(claim, KEPT, reason="timed out"),
            "Completeness-claim verifier",
        )

    # `model` set -> per-call with_model() is not thread-safe -> sequential.
    if model or len(claims) == 1:
        return [judge(c) for c in claims]
    with ThreadPoolExecutor(max_workers=min(4, len(claims))) as pool:
        return list(pool.map(judge, claims))


def _parse_verdict(claim: Claim, text: str) -> ClaimVerdict:
    """Parse the verifier's JSON verdict; default to KEEP on any ambiguity.

    Only an explicit ``{"real": false}`` REFUTES (drops) the claim. ``real`` true
    CONFIRMS it; ``real`` null, a missing/invalid object, or a missing/non-bool
    ``real`` all KEEP it — the claim is acted on unless the verifier positively
    refutes it, so uncertainty never silently removes real work.
    """
    if not text or not text.strip():
        return ClaimVerdict(claim, KEPT, reason="empty verdict", raw=text)

    obj = _extract_json_object(text)
    if obj is None or "real" not in obj:
        return ClaimVerdict(claim, KEPT, reason="unparseable verdict", raw=text)

    real = obj.get("real")
    reason = obj.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else ""
    if real is False:
        # Surface a content-free refute honestly (it dropped a claim with no stated
        # justification) instead of fabricating a confident-sounding rationale.
        return ClaimVerdict(
            claim,
            REFUTED,
            reason=reason or "refuted without a stated reason",
            raw=text,
        )
    if real is True:
        return ClaimVerdict(
            claim, CONFIRMED, reason=reason or "genuinely incomplete", raw=text
        )
    # real is null / non-boolean / anything else -> keep (unsure).
    return ClaimVerdict(claim, KEPT, reason=reason or "unsure", raw=text)
