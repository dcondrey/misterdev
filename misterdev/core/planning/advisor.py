"""Work recommendation for interactive planning.

Given a ProjectAssessment, ask the LLM to propose a short, ranked list of
concrete work items it would recommend next. Used by the interactive planner
so the orchestrator composes the plan from the project's live state instead of
a predefined devplan.
"""

from dataclasses import dataclass
from typing import List

from misterdev.core.planning.assessment import ProjectAssessment
from misterdev.llm.client import BaseLLMClient
from misterdev.llm.responses import extract_json_array
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

VALID_WORK_TYPES = {"debug", "complete", "feature", "refactor", "test", "docs"}

RECOMMEND_PROMPT = """You are advising on what to work on next in this project.

Project assessment:
{summary}

Incomplete features: {incomplete}
Broken/stub code: {broken}
Open TODO/FIXME: {todos}

Propose 3-6 concrete, high-value work items, ranked best-first. Favor fixing
what is broken and completing what is started over new features. Each item must
be specific enough to act on (name the area or behavior), not a vague theme.

Return ONLY a JSON array, each element:
  {{"title": "...", "rationale": "one sentence why it matters now",
    "work_type": "debug|complete|feature|refactor|test|docs"}}
"""


@dataclass
class Recommendation:
    title: str
    rationale: str
    work_type: str


def recommend_work(
    assessment: ProjectAssessment, llm_client: BaseLLMClient
) -> List[Recommendation]:
    """Return a ranked list of recommended work items, or [] on failure."""
    features = assessment.features
    prompt = RECOMMEND_PROMPT.format(
        summary=assessment.summary(),
        incomplete=_names(getattr(features, "incomplete", [])),
        broken=_names(getattr(features, "broken", [])),
        todos=len(getattr(features, "todos", []) or []),
    )
    try:
        response = llm_client.generate_code(
            prompt, "You are a pragmatic engineering lead. Return only JSON."
        )
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        return []
    raw = extract_json_array(response)

    recs: List[Recommendation] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        wt = str(item.get("work_type", "complete")).lower()
        if wt not in VALID_WORK_TYPES:
            wt = "complete"
        recs.append(
            Recommendation(
                title=str(item["title"]).strip(),
                rationale=str(item.get("rationale", "")).strip(),
                work_type=wt,
            )
        )
    return recs


def _names(items) -> str:
    out = []
    for it in items:
        name = getattr(it, "name", None) or getattr(it, "title", None) or str(it)
        out.append(str(name))
    return ", ".join(out[:12]) if out else "none"
