"""Project assessment models ported from /build skill Phase 1.

Structured representations of project analysis results used to drive
all subsequent build phases.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class _AssessmentModel(BaseModel):
    """Base for the assessment models.

    ``validate_assignment`` makes attribute writes (not just construction)
    type-checked. The pipeline mutates these models field-by-field straight from
    LLM JSON (see the analyzer's merge step), so without this a ``null`` or
    wrong-typed write would be stored silently and only crash much later; with it
    a bad write fails fast, at the assignment site.
    """

    model_config = ConfigDict(validate_assignment=True)


class HealthCheck(_AssessmentModel):
    """Result of running build, test, and lint commands."""

    builds: bool = False
    build_output: str = ""
    tests_pass: bool = False
    test_count: int = 0
    test_failures: int = 0
    test_output: str = ""
    lint_clean: bool = False
    lint_warnings: int = 0
    lint_output: str = ""
    missing_dependencies: list[str] = Field(default_factory=list)


class FeatureInfo(_AssessmentModel):
    """A single feature with evidence of its state."""

    name: str
    description: str = ""
    evidence_files: list[str] = Field(default_factory=list)
    complexity: str = "medium"  # trivial, small, medium, large, architectural


class FeatureInventory(_AssessmentModel):
    """Completeness analysis from /build Phase 1b."""

    existing: list[FeatureInfo] = Field(default_factory=list)
    incomplete: list[FeatureInfo] = Field(default_factory=list)
    missing: list[FeatureInfo] = Field(default_factory=list)
    dead_code: list[str] = Field(default_factory=list)
    stubs: list[str] = Field(default_factory=list)
    broken: list[str] = Field(default_factory=list)
    todos: list[dict] = Field(default_factory=list)


class ProjectStructure(_AssessmentModel):
    """Structural profile from /build Phase 1a."""

    project_type: str = "unknown"  # web-api, web-app, cli, library, etc.
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    build_system: Optional[str] = None
    build_command: Optional[str] = None
    test_system: Optional[str] = None
    test_command: Optional[str] = None
    lint_command: Optional[str] = None
    package_manager: Optional[str] = None
    entry_points: list[str] = Field(default_factory=list)
    directory_structure: str = ""


class ProjectContext(_AssessmentModel):
    """Contextual information from /build Phase 1c."""

    purpose: str = ""
    goals: str = ""
    conventions: str = ""
    constraints: str = ""
    recent_activity: str = ""
    stated_requirements: str = ""
    reference_impl: Optional[str] = None


class TechnicalDebt(_AssessmentModel):
    """Technical debt estimation from /build Phase 1."""

    score: int = 0  # 0-100
    description: str = ""
    critical_issues: list[str] = Field(default_factory=list)


class RiskAssessment(_AssessmentModel):
    """Risk analysis for the proposed build."""

    level: str = "low"  # low, medium, high, critical
    factors: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)


class ProjectAssessment(_AssessmentModel):
    """Merged assessment from all Phase 1 agents.

    This is the central data structure that drives Phases 2-6.
    """

    structure: ProjectStructure = Field(default_factory=ProjectStructure)
    health: HealthCheck = Field(default_factory=HealthCheck)
    features: FeatureInventory = Field(default_factory=FeatureInventory)
    context: ProjectContext = Field(default_factory=ProjectContext)
    tech_debt: TechnicalDebt = Field(default_factory=TechnicalDebt)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)

    def summary(self) -> str:
        """One-line summary for logging."""
        s = self.structure
        h = self.health
        lang = ", ".join(s.languages) if s.languages else "unknown"
        build_status = "OK" if h.builds else "FAIL"
        # Clamp the passing count: some runner parsers fill test_count and
        # test_failures from independent regexes, so a malformed log can yield
        # failures > count, which would otherwise render a negative "passed".
        test_status = (
            f"{max(0, h.test_count - h.test_failures)}/{h.test_count}"
            if h.test_count
            else "none"
        )
        return (
            f"[{s.project_type}] {lang} | "
            f"build={build_status} tests={test_status} "
            f"lint_warnings={h.lint_warnings}"
        )
