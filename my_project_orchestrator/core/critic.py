"""Optional adversarial edit critic (independent second component).

A single generator shares its own blind spots: a model that misreads a
requirement, misses an edge case, or "fixes" a symptom instead of the cause will
not catch that by re-reading its own output — the error and the reviewer are the
same component. This gate adds a SECOND, deliberately independent component that
reviews a CANDIDATE edit *before it is applied* and either APPROVES it or returns
concrete, actionable objections. Objections are fed back to the generator as the
next attempt's error context, forming a generate -> critique -> regenerate loop.

Independence is the whole point. A critic that shares the generator's model,
prompt, and training mostly rationalizes the first answer (self-critique is
weakest exactly when the model is confidently wrong). Configure ``critic.model``
to a DIFFERENT model for the real benefit; without it the gate still runs on the
generator's own model with adversarial framing, but the signal is weaker — and
that weakness is logged so it is never silently assumed away.

The critic is ADVISORY, never authoritative. The deterministic build/test/lint
gates remain the ground truth; this only catches semantic blind spots earlier
and more cheaply than a full gate run, and it defers to the gates after
``max_rejections`` regenerations so an over-zealous critic can't starve the loop.

Mirrors :mod:`my_project_orchestrator.core.goal_check`: best-effort and run in a
daemon worker thread with a hard timeout so a slow or unreachable model can NEVER
block a build. No client, no candidate edit, an unparseable verdict, or a timeout
is a SKIP (no opinion) that lets the edit proceed untouched.
"""

import threading
from typing import Callable, Dict, List, Optional

from my_project_orchestrator.core.goal_check import _extract_json_object
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Outcome constants. SKIP means "no opinion" (no client/edit, an unparseable
# verdict, or a timeout) and must let the edit proceed — never a rejection.
SKIP = "skip"
APPROVED = "approved"
REJECTED = "rejected"

_PROMPT = (
    "You are an adversarial code reviewer. Your job is to find what is WRONG with "
    "a candidate change BEFORE it is applied — assume it is flawed and try to "
    "prove it. Judge only the change shown against the task and acceptance "
    "criteria; do not rewrite it.\n\n"
    "## Task\n{task}\n\n"
    "## Acceptance Criteria\n{criteria}\n\n"
    "## Candidate Change (full content of each edited/created file)\n{candidate}\n\n"
    "Look specifically for: misread or unmet requirements, logic errors, "
    "unhandled edge cases (empty/null/zero/max, concurrency), resource leaks, "
    "swallowed errors, and security holes. Ignore pure style.\n\n"
    "Reply with a single JSON object on the first line and nothing else:\n"
    '{{"approved": true|false, "objections": ["one concrete, actionable problem '
    'per item"]}}\n'
    "If approved is true, objections must be an empty list. Only reject for a "
    "concrete defect you can name — not a vague preference. Keep each objection to "
    "one sentence."
)

# Cap the candidate evidence so a large edit can't blow up the prompt or cost;
# the head of each file carries the signatures and new logic that matter most.
_MAX_CANDIDATE_CHARS = 16000

# A critic call takes the assembled prompt text and returns the model's text.
# Injected in tests; defaulted to the project client's generate_code path.
CriticCall = Callable[[str], str]


class CritiqueVerdict:
    """Outcome of an adversarial edit critique.

    ``status`` is SKIP / APPROVED / REJECTED. ``objections`` lists the concrete
    problems (empty unless REJECTED). ``raw`` is the model's text (evidence);
    ``reason`` explains a SKIP.
    """

    def __init__(
        self,
        status: str,
        objections: Optional[List[str]] = None,
        raw: str = "",
        reason: str = "",
    ):
        self.status = status
        self.objections = list(objections or [])
        self.raw = raw
        self.reason = reason

    @property
    def approved(self) -> bool:
        return self.status == APPROVED

    @property
    def rejected(self) -> bool:
        return self.status == REJECTED

    @property
    def skipped(self) -> bool:
        return self.status == SKIP

    def __repr__(self) -> str:
        return (
            f"CritiqueVerdict(status={self.status!r}, "
            f"objections={len(self.objections)})"
        )


def run_edit_critic(
    task_description: Optional[str],
    acceptance_criteria: Optional[str],
    candidate_edits: Optional[Dict[str, str]],
    critic_call: Optional[CriticCall] = None,
    llm_client=None,
    critic_model: Optional[str] = None,
    timeout: float = 60,
) -> CritiqueVerdict:
    """Critique ``candidate_edits`` against the task and acceptance criteria.

    The model call is performed by ``critic_call`` when supplied (the test seam);
    otherwise one is built from ``llm_client``. ``critic_model``, when given,
    selects an INDEPENDENT model so the critic does not share the generator's
    blind spots (its absence is logged once — the same-model critic is weaker).
    With no callable and no client, or with no candidate edit, the critic SKIPs.
    SKIP also on an unparseable verdict, any critic error, or the hard timeout
    (never blocks). Only a parsed rejection returns objections.
    """
    if not candidate_edits:
        return CritiqueVerdict(SKIP, reason="no candidate edit to review")

    call = critic_call or _default_critic_call(llm_client, critic_model)
    if call is None:
        return CritiqueVerdict(SKIP, reason="no LLM critic available")

    prompt = _PROMPT.format(
        task=(task_description or "(none given)").strip(),
        criteria=(acceptance_criteria or "(none given)").strip(),
        candidate=_render_candidate(candidate_edits),
    )

    box = {"result": CritiqueVerdict(SKIP, reason="not started")}

    def _run() -> None:
        try:
            raw = call(prompt) or ""
            box["result"] = _parse_verdict(raw)
        except Exception as e:  # any model/IO failure is non-fatal -> skip
            logger.debug(f"Adversarial critic unavailable: {e}")
            box["result"] = CritiqueVerdict(SKIP, reason=f"error: {e}")

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning(
            f"Adversarial critic exceeded {timeout}s; skipping (never blocks)."
        )
        return CritiqueVerdict(SKIP, reason="timed out")
    return box["result"]


def _render_candidate(edits: Dict[str, str]) -> str:
    """Render the candidate edits as labeled, size-bounded file blocks.

    Files are shown in a stable (sorted) order and the total is capped so a huge
    edit can't blow up the prompt; truncation is marked so the critic knows it
    saw a head, not the whole file.
    """
    parts: List[str] = []
    budget = _MAX_CANDIDATE_CHARS
    for path in sorted(edits):
        if budget <= 0:
            parts.append("... (further files omitted; candidate too large)")
            break
        content = edits[path] or ""
        shown = content[:budget]
        if len(content) > len(shown):
            shown += "\n... (file truncated)"
        budget -= len(shown)
        parts.append(f"### {path}\n{shown}")
    return "\n\n".join(parts)


def _parse_verdict(text: str) -> CritiqueVerdict:
    """Deterministically parse the critic's JSON verdict.

    Tolerates surrounding prose / markdown fences via the balanced-object
    extractor shared with the goal check. A missing/invalid object or a missing
    ``approved`` boolean is a SKIP (no opinion), not a rejection — the critic must
    never block on its own malformed output. A rejection with no listed objection
    still records one generic objection so the loop never regenerates without
    telling the generator why.
    """
    if not text or not text.strip():
        return CritiqueVerdict(SKIP, raw=text, reason="empty verdict")

    obj = _extract_json_object(text)
    if obj is None or "approved" not in obj:
        return CritiqueVerdict(SKIP, raw=text, reason="unparseable verdict")

    approved = obj.get("approved")
    if not isinstance(approved, bool):
        return CritiqueVerdict(SKIP, raw=text, reason="non-boolean 'approved'")

    if approved:
        return CritiqueVerdict(APPROVED, objections=[], raw=text)

    raw_objections = obj.get("objections")
    objections: List[str] = []
    if isinstance(raw_objections, list):
        for o in raw_objections:
            if isinstance(o, str) and o.strip():
                objections.append(o.strip())
    elif isinstance(raw_objections, str) and raw_objections.strip():
        objections.append(raw_objections.strip())
    if not objections:
        objections = ["change rejected (no specific objection reported)"]
    return CritiqueVerdict(REJECTED, objections=objections, raw=text)


def _default_critic_call(
    llm_client, critic_model: Optional[str]
) -> Optional[CriticCall]:
    """Build a critic call from the project's LLM client, or None if unusable.

    Uses the client's ``generate_code(prompt, system)`` text interface. When
    ``critic_model`` is given and the client supports ``with_model``, the call is
    routed through that INDEPENDENT model; otherwise it runs on the generator's
    own model and the weaker independence is logged once. Tolerant of client
    shape so an absent/limited client degrades to SKIP rather than raising. No
    network is touched until the returned callable is invoked in the worker.
    """
    if llm_client is None or not hasattr(llm_client, "generate_code"):
        return None

    system = (
        "You are a precise adversarial code reviewer. Output only the requested "
        "JSON object. Reject only for a concrete, nameable defect."
    )

    use_independent = bool(critic_model) and hasattr(llm_client, "with_model")
    if critic_model and not use_independent:
        logger.warning(
            "Adversarial critic: critic.model set but the client cannot switch "
            "models; running on the generator's own model (weaker independence)."
        )
    elif not critic_model:
        logger.info(
            "Adversarial critic: no critic.model configured; running on the "
            "generator's own model (weaker independence — set critic.model to a "
            "different model for a true second component)."
        )

    def _call(prompt: str) -> str:
        if use_independent:
            with llm_client.with_model(critic_model):
                return llm_client.generate_code(prompt, system) or ""
        return llm_client.generate_code(prompt, system) or ""

    return _call
