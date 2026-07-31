"""SpecGenMixin — mode-driven spec synthesis for ProjectOrchestrator.

Extracted from agent.py. Both methods are pure functions of their arguments
(assessment data + project metadata + an optional LLM client); they reference
no other self methods and carry no mutable state.
"""

from misterdev.core.modes import BuildMode
from misterdev.core.planning.assessment import ProjectAssessment
from misterdev.core.execution.project import Project
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


class SpecGenMixin:
    def _ground_completion_spec(
        self, assessment: ProjectAssessment, project: Project
    ) -> str:
        """Build a COMPLETE-mode spec grounded in objective signals.

        A vague "complete everything" goal on a real codebase otherwise churns:
        the completeness analyzer flags "incomplete"/"stub" items from a lossy
        overview and mislabels deliberate design (graceful degradation, platform
        no-ops) as work, so the spec becomes a pile of speculative tasks. Instead
        lead with HARD signals — a failing build, failing tests, located
        TODO/FIXME markers, broken references — which are objective and
        verifiable, add features the docs promise but are absent, and demote the
        analyzer's guesses to an explicit "do NOT task these unless corroborated"
        advisory. When nothing hard or documented exists, the goal is ill-posed:
        emit zero-task guidance rather than fabricate work.
        """
        h = assessment.health
        f = assessment.features
        hard: list[str] = []
        if not h.builds and h.build_output:
            hard.append(
                f"- The build is FAILING; fix it first:\n{h.build_output[:400]}"
            )
        if not h.tests_pass and h.test_output:
            hard.append(f"- Tests are FAILING:\n{h.test_output[:400]}")
        hard.extend(f"- Broken: {item}" for item in f.broken)
        hard.extend(
            f"- {t.get('file', '?')}:{t.get('line', '?')} {t.get('text', '')}"
            for t in f.todos[:20]
        )
        documented = [f"- {m.name}: {m.description}" for m in f.missing]
        speculative = [f"- {i.name}: {i.description}" for i in f.incomplete]
        speculative += [f"- Stub: {s}" for s in f.stubs]

        parts = [f"# Completion Spec\n## Project: {project.name}\n"]
        if hard:
            parts.append(
                "## Must Fix — objective, verifiable failures\n" + "\n".join(hard)
            )
        if documented:
            parts.append(
                "\n## Should Add — promised by the docs but absent\n"
                + "\n".join(documented)
            )
        if not hard and not documented:
            parts.append(
                "## No concrete objective found\n"
                "The build and tests pass and there are no TODO/FIXME markers or "
                "documented-but-missing features. A vague 'complete everything' goal "
                "has no well-posed work here. Do NOT fabricate tasks from speculation: "
                "produce ZERO tasks and report that a specific objective (a feature, a "
                "bug to fix, or --focus <area>) is required."
            )
        if speculative:
            parts.append(
                "\n## Advisory — analyzer guesses, NOT tasks\n"
                "Inferred as incomplete/stub from a lossy overview; these often "
                "mislabel deliberate design. Do NOT create a task for any of these "
                "unless a failing test or build error above corroborates it.\n"
                + "\n".join(speculative[:15])
            )
        return "\n".join(parts)

    def _generate_spec(
        self,
        mode: BuildMode,
        prompt: str,
        assessment: ProjectAssessment,
        project: Project,
        facts: str = "",
    ) -> str:
        """Phase 2: Generate a spec based on mode."""
        if mode == BuildMode.DEBUG:
            parts = ["# Debug Spec\n## Broken Items"]
            for item in assessment.features.broken:
                parts.append(f"- {item}")
            if assessment.features.stubs:
                parts.append("\n## Stubs")
                for item in assessment.features.stubs:
                    parts.append(f"- {item}")
            if not assessment.health.builds:
                parts.append(
                    f"\n## Build Failure\n{assessment.health.build_output[:500]}"
                )
            if not assessment.health.tests_pass:
                parts.append(
                    f"\n## Test Failures\n{assessment.health.test_output[:500]}"
                )
            return "\n".join(parts)

        if mode == BuildMode.COMPLETE:
            return self._ground_completion_spec(assessment, project)

        if mode == BuildMode.SPEC:
            spec_path = project.path / prompt.strip()
            try:
                spec_path = spec_path.resolve()
                spec_path.relative_to(project.path.resolve())
            except ValueError:
                return f"Spec file not found: {prompt}"
            if spec_path.exists():
                return spec_path.read_text(encoding="utf-8")
            return f"Spec file not found: {prompt}"

        if mode == BuildMode.CREATE:
            expand_prompt = (
                f"Expand the following into a comprehensive project spec.\n"
                f"Include: features with acceptance criteria, error handling, "
                f"input validation, testing strategy, architecture decisions.\n\n"
                f"Project context: {assessment.context.purpose}\n"
                f"Conventions: {assessment.context.conventions}\n"
                f"Languages: {assessment.structure.languages}\n"
                f"Frameworks: {assessment.structure.frameworks}\n"
                f"Existing features: {[f.name for f in assessment.features.existing]}\n"
                f"Verified facts: {facts}\n\n"
                f"Description: {prompt}\n\nReturn the spec as markdown."
            )
            return project.llm_client.generate_code(
                expand_prompt,
                "You are a software architect writing a project specification.",
            )

        if mode == BuildMode.SMART:
            # SMART is a SPECIFIC instruction on an EXISTING project — not a
            # from-scratch build. The goal is the scope boundary: implement
            # exactly what it asks plus only what is strictly necessary to make
            # THAT correct and tested. Never expand into a whole-project spec
            # (that is CREATE's job) — doing so makes the decomposer invent
            # unrelated tasks and rewrite pre-existing files it was only meant
            # to read (observed: a "create region.py" goal ballooned into
            # rewriting the harness and inventing conftest/config tasks).
            scoped_prompt = (
                f"Write a tightly-scoped implementation spec for EXACTLY this "
                f"goal — nothing more.\n"
                f"Rules:\n"
                f"- Implement only what the goal asks. Add only what is strictly "
                f"necessary to make the goal correct, tested, and safe (its own "
                f"error handling, input validation, and tests).\n"
                f"- Do NOT expand scope: no unrelated features, no 'completing' "
                f"or 'improving' the project, no refactors the goal did not "
                f"request.\n"
                f"- Existing files are CONTEXT, not work items. Do not modify or "
                f"rewrite them unless the goal explicitly requires it.\n"
                f"- Prefer the smallest change set that fully satisfies the "
                f"goal.\n\n"
                f"Project context: {assessment.context.purpose}\n"
                f"Conventions: {assessment.context.conventions}\n"
                f"Languages: {assessment.structure.languages}\n"
                f"Existing files (context only — do not modify unless the goal "
                f"requires it): {[f.name for f in assessment.features.existing]}\n"
                f"Verified facts: {facts}\n\n"
                f"Goal: {prompt}\n\nReturn the spec as markdown."
            )
            return project.llm_client.generate_code(
                scoped_prompt,
                "You are a software engineer writing a tightly-scoped "
                "implementation spec. You implement exactly what is asked and "
                "resist scope creep.",
            )

        if mode == BuildMode.REVIEW:
            return (
                f"# Review Spec\nReview and fix all issues found in the project.\n"
                f"Broken: {assessment.features.broken}\n"
                f"Stubs: {assessment.features.stubs}\n"
                f"TODOs: {len(assessment.features.todos)} items"
            )

        return f"# Auto Spec\n{prompt}"
