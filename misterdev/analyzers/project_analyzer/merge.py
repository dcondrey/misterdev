"""Coercion helpers and mergers that fold analyzer JSON into the assessment."""

from misterdev.core.planning.assessment import (
    FeatureInfo,
    ProjectAssessment,
)


def _health_ground_truth(health) -> str:
    """One-line verified-state preamble to anchor the completeness analyzer."""
    build = "passes" if health.builds else "FAILS"
    if health.test_count:
        passing = health.test_count - health.test_failures
        tests = f"{passing}/{health.test_count} tests passing"
    elif health.tests_pass:
        tests = "test suite passes"
    else:
        tests = "no test results"
    return f"VERIFIED ground truth: build {build}; {tests}."


def _as_str_list(value, default: list) -> list:
    """Coerce an LLM-provided value to a ``list[str]``, else return ``default``.

    The analyzers occasionally emit ``null`` (or a bare string) for a field the
    schema types as a list. ``dict.get(k, default)`` only falls back when the key
    is ABSENT, so a present ``null`` would store ``None`` on the typed field
    (Pydantic does not validate on assignment) and later crash a ``for``/``join``
    over it. Normalizing here keeps the trust boundary sound.
    """
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return default


def _as_int(value, default: int) -> int:
    """Coerce an LLM-provided value to ``int``, else return ``default``.

    Tolerates a numeric string ("75") or float (75.0); rejects bools, ``null``,
    and unparseable text so a typed ``int`` field never holds a string.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _merge_structure(assessment: ProjectAssessment, data: dict) -> None:
    if not data:
        return
    s = assessment.structure
    s.project_type = data.get("project_type") or s.project_type
    s.languages = _as_str_list(data.get("languages"), s.languages)
    s.frameworks = _as_str_list(data.get("frameworks"), s.frameworks)
    s.build_command = data.get("build_command", s.build_command)
    s.test_command = data.get("test_command", s.test_command)
    s.lint_command = data.get("lint_command", s.lint_command)
    s.package_manager = data.get("package_manager", s.package_manager)
    s.entry_points = _as_str_list(data.get("entry_points"), s.entry_points)
    s.directory_structure = data.get("directory_structure") or s.directory_structure


def _merge_completeness(assessment: ProjectAssessment, data: dict) -> None:
    if not data:
        return
    f = assessment.features
    for item in data.get("existing") or []:
        if isinstance(item, dict):
            f.existing.append(
                FeatureInfo(
                    name=item.get("name", ""), description=item.get("description", "")
                )
            )
    for item in data.get("incomplete") or []:
        if isinstance(item, dict):
            f.incomplete.append(
                FeatureInfo(
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    complexity=item.get("complexity", "medium"),
                )
            )
    for item in data.get("missing") or []:
        if isinstance(item, dict):
            f.missing.append(
                FeatureInfo(
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    complexity=item.get("complexity", "medium"),
                )
            )
    f.dead_code = _as_str_list(data.get("dead_code"), f.dead_code)
    f.stubs = _as_str_list(data.get("stubs"), f.stubs)
    f.broken = _as_str_list(data.get("broken"), f.broken)
    todos = data.get("todos")
    f.todos = todos if isinstance(todos, list) else f.todos


def _merge_context(assessment: ProjectAssessment, data: dict) -> None:
    if not data:
        return
    c = assessment.context
    c.purpose = data.get("purpose", c.purpose)
    c.goals = data.get("goals", c.goals)
    c.conventions = data.get("conventions", c.conventions)
    c.constraints = data.get("constraints", c.constraints)
    c.recent_activity = data.get("recent_activity", c.recent_activity)
    c.stated_requirements = data.get("stated_requirements", c.stated_requirements)


def _merge_debt_risk(assessment: ProjectAssessment, data: dict) -> None:
    if not data:
        return

    debt_data = data.get("tech_debt") or {}
    if debt_data:
        assessment.tech_debt.score = _as_int(debt_data.get("score"), 0)
        assessment.tech_debt.description = debt_data.get("description") or ""
        assessment.tech_debt.critical_issues = _as_str_list(
            debt_data.get("critical_issues"), assessment.tech_debt.critical_issues
        )

    risk_data = data.get("risk") or {}
    if risk_data:
        assessment.risk.level = risk_data.get("level") or "low"
        assessment.risk.factors = _as_str_list(
            risk_data.get("factors"), assessment.risk.factors
        )
        assessment.risk.mitigations = _as_str_list(
            risk_data.get("mitigations"), assessment.risk.mitigations
        )
