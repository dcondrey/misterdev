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

Mirrors :mod:`my_project_orchestrator.core.verification.goal_check`: best-effort and run in a
daemon worker thread with a hard timeout so a slow or unreachable model can NEVER
block a build. No client, no candidate edit, an unparseable verdict, or a timeout
is a SKIP (no opinion) that lets the edit proceed untouched.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from my_project_orchestrator.core.execution.bounded import run_bounded
from my_project_orchestrator.core.verification.goal_check import _extract_json_object
from my_project_orchestrator.core.verification.independent import build_independent_call
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Perspective lenses for a multi-member panel. Each member reviews the same
# change through a different lens so the panel catches failure modes a single
# reviewer (or N identical reviewers) would share — diversity, not redundancy.
_LENSES = (
    "Focus above all on CORRECTNESS: does the logic actually do what the task asks?",
    "Focus above all on EDGE CASES: empty/null/zero/maximum inputs and concurrency.",
    "Focus above all on SAFETY: resource leaks, swallowed errors, and security holes.",
    "Focus above all on REQUIREMENTS: is any acceptance criterion unmet or misread?",
)

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
    "## Candidate Change\n{candidate}\n\n"
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
    candidate_diffs: Optional[Dict[str, str]] = None,
    panel: int = 1,
    timeout: float = 60,
) -> CritiqueVerdict:
    """Critique ``candidate_edits`` against the task and acceptance criteria.

    The model call is performed by ``critic_call`` when supplied (the test seam);
    otherwise one is built from ``llm_client``. ``critic_model``, when given,
    selects an INDEPENDENT model so the critic does not share the generator's
    blind spots (its absence is logged — the same-model critic is weaker).

    When ``candidate_diffs`` is given, the critic reviews the unified diff of each
    change (what actually changed, with a little context) instead of whole files —
    sharper signal and far less prompt for a small edit to a large file.

    ``panel`` > 1 runs that many reviewers concurrently, each through a different
    perspective lens, and the change is rejected only on a MAJORITY of decisive
    votes (ties approve), which suppresses a lone false rejection.

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
        candidate=_render_candidate(candidate_edits, candidate_diffs),
    )
    members = max(1, int(panel))

    def _work() -> CritiqueVerdict:
        try:
            if members == 1:
                return _parse_verdict(call(prompt) or "")
            return _aggregate_panel(_run_panel(call, prompt, members))
        except Exception as e:  # any model/IO failure is non-fatal -> skip
            logger.debug(f"Adversarial critic unavailable: {e}")
            return CritiqueVerdict(SKIP, reason=f"error: {e}")

    return run_bounded(
        _work, timeout, CritiqueVerdict(SKIP, reason="timed out"), "Adversarial critic"
    )


def _run_panel(
    call: CriticCall, base_prompt: str, members: int
) -> List["CritiqueVerdict"]:
    """Run ``members`` lens-diversified critiques concurrently; return verdicts.

    A member whose call fails contributes a SKIP (abstention) rather than sinking
    the panel, so one flaky reviewer can't force a decision either way.
    """
    prompts = [f"{base_prompt}\n{_LENSES[i % len(_LENSES)]}\n" for i in range(members)]

    def _one(p: str) -> "CritiqueVerdict":
        try:
            return _parse_verdict(call(p) or "")
        except Exception as e:
            logger.debug(f"Critic panel member failed: {e}")
            return CritiqueVerdict(SKIP, reason=f"member error: {e}")

    with ThreadPoolExecutor(max_workers=members) as pool:
        return list(pool.map(_one, prompts))


def _aggregate_panel(verdicts: List["CritiqueVerdict"]) -> "CritiqueVerdict":
    """Combine panel verdicts: reject only on a strict majority of decisive votes.

    Abstentions (SKIP) don't count. A tie approves — the critic is advisory and
    defers to the real gates, so consensus is required to block. A rejection
    unions the objections of the rejecting members (deduplicated, order-stable).
    """
    decisive = [v for v in verdicts if v.status != SKIP]
    if not decisive:
        return CritiqueVerdict(SKIP, reason="panel reached no verdict")
    rejects = [v for v in decisive if v.status == REJECTED]
    approves = [v for v in decisive if v.status == APPROVED]
    if len(rejects) > len(approves):
        objections: List[str] = []
        seen = set()
        for v in rejects:
            for o in v.objections:
                if o not in seen:
                    seen.add(o)
                    objections.append(o)
        return CritiqueVerdict(
            REJECTED,
            objections=objections,
            reason=f"panel rejected {len(rejects)}/{len(decisive)}",
        )
    return CritiqueVerdict(
        APPROVED, reason=f"panel approved {len(approves)}/{len(decisive)}"
    )


def _render_candidate(
    edits: Dict[str, str], diffs: Optional[Dict[str, str]] = None
) -> str:
    """Render the candidate as labeled, size-bounded blocks.

    Shows unified diffs when ``diffs`` is provided (sharper, smaller signal —
    just what changed), otherwise full file contents. Files are shown in a stable
    (sorted) order and the total is capped so a huge edit can't blow up the
    prompt; truncation is marked so the critic knows it saw a head, not all.
    """
    source = diffs if diffs else edits
    note = (
        "(shown as unified diffs — unchanged lines omitted)"
        if diffs
        else "(full content of each edited/created file)"
    )
    parts: List[str] = [note]
    budget = _MAX_CANDIDATE_CHARS
    for path in sorted(source):
        if budget <= 0:
            parts.append("... (further files omitted; candidate too large)")
            break
        body = source[path] or ""
        shown = body[:budget]
        if len(body) > len(shown):
            shown += "\n... (truncated)"
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
    system = (
        "You are a precise adversarial code reviewer. Output only the requested "
        "JSON object. Reject only for a concrete, nameable defect."
    )
    return build_independent_call(
        llm_client, system, critic_model, "Adversarial critic"
    )
