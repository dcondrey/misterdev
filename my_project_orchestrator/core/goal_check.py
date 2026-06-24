"""Optional goal-completion check (LLM judge).

Green gates prove the code compiles, tests pass, and the diff is hygienic — but
not that the work actually accomplished what was asked. "Gates green != goal
met": a build can satisfy every mechanical gate while quietly omitting a feature
the goal called for. This check closes that gap. Given the goal text, the
acceptance criteria, and the cumulative diff/summary of the build, an LLM judge
returns a verdict {satisfied: bool, gaps: [str]}.

It mirrors :mod:`my_project_orchestrator.core.vision_verify`: best-effort and run
in a daemon worker thread with a hard timeout so a slow or unreachable model can
NEVER block the build. It is ADVISORY by default — the verdict's gaps are
recorded into the report and logged, and the build is NOT failed. It blocks only
when ``orchestrator.block_on_goal_gap`` is true (the caller's choice). Absent a
goal / criteria, absent an LLM client, an unparseable verdict, or a timeout is a
SKIP (no opinion), never a failure.

Like the vision gate, the signal IS a model judgment, so the judge is shown the
concrete evidence the build produced (the diff/summary) rather than asked to
self-report on intent.
"""

import json
from typing import Callable, List, Optional

from my_project_orchestrator.core.bounded import run_bounded
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# Outcome constants. SKIP means "no opinion" (no goal/criteria/client, an
# unparseable verdict, or a timeout) and must never be treated as a pass/fail
# signal by callers.
SKIP = "skip"
SATISFIED = "satisfied"
GAP = "gap"

_PROMPT = (
    "You are a strict acceptance reviewer. Decide whether the WORK that was done "
    "actually satisfies the stated GOAL and ACCEPTANCE CRITERIA. Judge only from "
    "the evidence given; do not assume work that is not shown.\n\n"
    "## Goal\n{goal}\n\n"
    "## Acceptance Criteria\n{criteria}\n\n"
    "## Work Done (diff / summary)\n{evidence}\n\n"
    "Reply with a single JSON object on the first line and nothing else:\n"
    '{{"satisfied": true|false, "gaps": ["short description of each unmet '
    'requirement"]}}\n'
    "If satisfied is true, gaps must be an empty list. Keep each gap to one "
    "sentence."
)

# Cap the evidence we feed the judge so a huge diff can't blow up the prompt or
# cost; the head of a diff carries the most signal (new files, signatures).
_MAX_EVIDENCE_CHARS = 16000

# A judge call takes the assembled prompt text and returns the model's text.
# Injected in tests; defaulted to the project client's generate_code path.
JudgeCall = Callable[[str], str]


class GoalVerdict:
    """Outcome of a goal-completion check.

    ``status`` is SKIP / SATISFIED / GAP. ``gaps`` lists the unmet requirements
    (empty when satisfied or skipped). ``raw`` is the model's text (evidence);
    ``reason`` explains a SKIP.
    """

    def __init__(
        self,
        status: str,
        gaps: Optional[List[str]] = None,
        raw: str = "",
        reason: str = "",
    ):
        self.status = status
        self.gaps = list(gaps or [])
        self.raw = raw
        self.reason = reason

    @property
    def satisfied(self) -> bool:
        return self.status == SATISFIED

    @property
    def skipped(self) -> bool:
        return self.status == SKIP

    @property
    def has_gap(self) -> bool:
        return self.status == GAP

    def __repr__(self) -> str:
        return f"GoalVerdict(status={self.status!r}, gaps={len(self.gaps)})"


def run_goal_check(
    goal: Optional[str],
    criteria: Optional[str],
    evidence: Optional[str],
    judge_call: Optional[JudgeCall] = None,
    llm_client=None,
    judge_model: Optional[str] = None,
    timeout: float = 60,
) -> GoalVerdict:
    """Judge whether ``evidence`` satisfies ``goal`` + ``criteria``.

    The model call is performed by ``judge_call`` when supplied (the test seam);
    otherwise one is built from ``llm_client`` if provided. ``judge_model``, when
    given, routes the judgment through an INDEPENDENT model so it does not share
    the generator's blind spots (its absence is logged — the same-model judge is
    weaker). With neither a callable nor a client, the check SKIPs (no model).
    SKIP also when there is no goal AND no criteria (no target to judge against),
    on an unparseable verdict, on any judge error, or on the hard timeout (never
    blocks). ``timeout`` is the hard ceiling for the whole run.
    """
    if not (goal or "").strip() and not (criteria or "").strip():
        return GoalVerdict(SKIP, reason="no goal or acceptance criteria")

    call = judge_call or _default_judge_call(llm_client, judge_model)
    if call is None:
        return GoalVerdict(SKIP, reason="no LLM judge available")

    prompt = _PROMPT.format(
        goal=(goal or "(none given)").strip(),
        criteria=(criteria or "(none given)").strip(),
        evidence=(evidence or "(no diff/summary captured)").strip()[
            :_MAX_EVIDENCE_CHARS
        ],
    )

    def _work() -> GoalVerdict:
        try:
            return _parse_verdict(call(prompt) or "")
        except Exception as e:  # any model/IO failure is non-fatal -> skip
            logger.debug(f"Goal-completion check unavailable: {e}")
            return GoalVerdict(SKIP, reason=f"error: {e}")

    return run_bounded(
        _work, timeout, GoalVerdict(SKIP, reason="timed out"), "Goal-completion check"
    )


def _parse_verdict(text: str) -> GoalVerdict:
    """Deterministically parse the judge's JSON verdict.

    Tolerates surrounding prose and markdown fences by extracting the first
    balanced ``{...}`` object. A missing/invalid object, or a missing
    ``satisfied`` boolean, is a SKIP (no opinion), not a failure. When
    ``satisfied`` is false, ``gaps`` is normalized to a list of non-empty
    strings; an unsatisfied verdict with no listed gaps still records one
    generic gap so the report never claims a gap exists without saying which.
    """
    if not text or not text.strip():
        return GoalVerdict(SKIP, raw=text, reason="empty verdict")

    obj = _extract_json_object(text)
    if obj is None or "satisfied" not in obj:
        return GoalVerdict(SKIP, raw=text, reason="unparseable verdict")

    satisfied = obj.get("satisfied")
    if not isinstance(satisfied, bool):
        return GoalVerdict(SKIP, raw=text, reason="non-boolean 'satisfied'")

    if satisfied:
        return GoalVerdict(SATISFIED, gaps=[], raw=text)

    gaps_raw = obj.get("gaps")
    gaps: List[str] = []
    if isinstance(gaps_raw, list):
        for g in gaps_raw:
            if isinstance(g, str) and g.strip():
                gaps.append(g.strip())
    elif isinstance(gaps_raw, str) and gaps_raw.strip():
        gaps.append(gaps_raw.strip())
    if not gaps:
        gaps = ["goal not satisfied (no specific gap reported)"]
    return GoalVerdict(GAP, gaps=gaps, raw=text)


def _extract_json_object(text: str) -> Optional[dict]:
    """Return the first parseable top-level JSON object in ``text``, or None.

    Scans for a balanced ``{...}`` span (respecting string literals so a brace
    inside a string does not break balancing) and json-loads it. This survives
    leading prose or a ```json fence around the object.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                    except ValueError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def _default_judge_call(
    llm_client, judge_model: Optional[str] = None
) -> Optional[JudgeCall]:
    """Build a judge call from the project's LLM client, or None if unusable.

    Uses the client's ``generate_code(prompt, system)`` text interface. When
    ``judge_model`` is given and the client supports ``with_model``, the call is
    routed through that INDEPENDENT model; otherwise it runs on the generator's
    own model and the weaker independence is logged. Kept tolerant of client
    shape so an absent/limited client degrades to SKIP rather than raising. No
    network is touched until the returned callable is invoked inside the worker.
    """
    if llm_client is None or not hasattr(llm_client, "generate_code"):
        return None

    system = (
        "You are a precise acceptance reviewer. Output only the requested JSON "
        "object. Do not invent work that is not shown in the evidence."
    )

    use_independent = bool(judge_model) and hasattr(llm_client, "with_model")
    if judge_model and not use_independent:
        logger.warning(
            "Goal-completion check: judge.model set but the client cannot switch "
            "models; judging on the generator's own model (weaker independence)."
        )

    def _call(prompt: str) -> str:
        if use_independent:
            with llm_client.with_model(judge_model):
                return llm_client.generate_code(prompt, system) or ""
        return llm_client.generate_code(prompt, system) or ""

    return _call


def build_evidence(diff: str = "", summary: str = "") -> str:
    """Compose the judge's evidence from a diff and/or a summary.

    Either may be empty; the summary (what the build reports it did) is shown
    first because it is the most compact statement of intent-realized, with the
    raw diff following as the ground truth. Returned empty when both are empty,
    which the caller treats as "no evidence" (the judge still runs but is told
    so, and a satisfied verdict on no evidence is unlikely).
    """
    parts = []
    if summary and summary.strip():
        parts.append(f"### Summary\n{summary.strip()}")
    if diff and diff.strip():
        parts.append(f"### Diff\n{diff.strip()}")
    return "\n\n".join(parts)
