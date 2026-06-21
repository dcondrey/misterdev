"""Task decomposition with dependency resolution, ported from /build Phase 3.

Breaks a spec into ordered, atomic tasks with dependency tracking.
Uses topological sort to determine execution order.
"""

import json
from collections import deque
from typing import Optional

from my_project_orchestrator.core.models import Task
from my_project_orchestrator.utils.file_utils import safe_ref_slug
from my_project_orchestrator.core.assessment import ProjectAssessment
from my_project_orchestrator.core.modes import BuildMode
from my_project_orchestrator.llm.client import BaseLLMClient
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

# /build ordering: infrastructure > core types > core logic > features >
#                   integration > tests > fixes > cleanup
CATEGORY_ORDER = {
    "infrastructure": 0,
    "core": 1,
    "feature": 2,
    "integration": 3,
    "test": 4,
    "fix": 5,
    "docs": 6,
    "cleanup": 7,
}

MAX_TASKS = 30

DECOMPOSE_PROMPT = """You are a task decomposer for a software project.

Given the project assessment and spec below, break the work into an ordered
list of atomic, verifiable tasks. Each task must be completable in one editing
session (1-20 files) and have a concrete "done" condition.

## Project Assessment

### Structure
- Type: {project_type}
- Languages: {languages}
- Frameworks: {frameworks}
- Build: {build_command}
- Test: {test_command}
- Package manager: {package_manager}

### Health
- Builds: {builds}
- Tests pass: {tests_pass} ({test_count} tests, {test_failures} failures)
- Lint clean: {lint_clean}

### Features
- Existing: {existing_features}
- Incomplete: {incomplete_features}
- Missing: {missing_features}
- Broken: {broken_items}
- Stubs: {stub_items}

### Context
- Purpose: {purpose}
- Conventions: {conventions}

### Risk
- Tech debt score: {debt_score}/100
- Risk level: {risk_level}

## Spec
{spec}

## Mode
{mode}

## Rules
- Max {max_tasks} tasks. Prioritize: must-fix > must-complete > should-add.
- Order: infrastructure > core types > core logic > features > integration > tests > fixes > cleanup.
- Each task's files_to_modify must not overlap with another task's files_to_create unless a dependency is declared.
- For DEBUG mode: order by build-blocking > test-blocking > runtime errors > warnings.
- For COMPLETE mode: order by infrastructure > core > features > tests > docs > cleanup.
- acceptance_criteria must be specific and testable (e.g., "pytest tests/test_auth.py passes" not "auth works").
- context_files should include imports, interfaces, or types that the task needs to read but not modify.

Return a JSON array of task objects. Each object must have exactly these fields:
  id (string, like "T-001"),
  title (string),
  description (string),
  acceptance_criteria (string),
  files_to_create (array of strings),
  files_to_modify (array of strings),
  context_files (array of strings, files needed for context but not modified),
  dependencies (array of task id strings),
  complexity ("trivial", "small", "medium", "large", or "architectural"),
  category ("infrastructure", "core", "feature", "integration", "test", "fix", "docs", or "cleanup")

Return ONLY the JSON array, no markdown fences or other text."""


def decompose_spec(
    spec: str,
    assessment: ProjectAssessment,
    mode: BuildMode,
    llm_client: BaseLLMClient,
    project_ref: str,
) -> list[Task]:
    """Use LLM to decompose a spec into ordered tasks with dependencies."""
    s = assessment.structure
    h = assessment.health
    f = assessment.features
    c = assessment.context

    prompt = DECOMPOSE_PROMPT.format(
        project_type=s.project_type,
        languages=", ".join(s.languages) if s.languages else "unknown",
        frameworks=", ".join(s.frameworks) if s.frameworks else "none",
        build_command=s.build_command or "none",
        test_command=s.test_command or "none",
        package_manager=s.package_manager or "unknown",
        builds="yes" if h.builds else "no",
        tests_pass="yes" if h.tests_pass else "no",
        test_count=h.test_count,
        test_failures=h.test_failures,
        lint_clean="yes" if h.lint_clean else "no",
        existing_features=", ".join(fi.name for fi in f.existing)
        if f.existing
        else "none",
        incomplete_features=", ".join(
            f"{fi.name} ({fi.complexity})" for fi in f.incomplete
        )
        if f.incomplete
        else "none",
        missing_features=", ".join(f"{fi.name} ({fi.complexity})" for fi in f.missing)
        if f.missing
        else "none",
        broken_items=", ".join(f.broken) if f.broken else "none",
        stub_items=", ".join(f.stubs) if f.stubs else "none",
        purpose=c.purpose or "unknown",
        conventions=c.conventions or "none specified",
        debt_score=assessment.tech_debt.score,
        risk_level=assessment.risk.level,
        spec=spec,
        mode=mode.value,
        max_tasks=MAX_TASKS,
    )

    logger.info("Decomposing spec into tasks via LLM...")
    response = llm_client.generate_code(
        prompt, "You are a precise task decomposer. Return only valid JSON."
    )

    tasks = _parse_tasks(response, project_ref)

    # Detect implicit dependencies from file overlaps
    _add_implicit_dependencies(tasks)

    # Validate and cap
    if len(tasks) > MAX_TASKS:
        logger.warning(f"LLM returned {len(tasks)} tasks, capping at {MAX_TASKS}")
        tasks = tasks[:MAX_TASKS]

    return tasks


def _parse_tasks(response: str, project_ref: str) -> list[Task]:
    """Parse LLM JSON response into Task objects."""
    # Strip markdown fences if present
    text = response.strip()
    # Strip markdown fences
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        raw_tasks = json.loads(text)
    except json.JSONDecodeError:
        # LLM may have wrapped JSON in prose; find the array
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                raw_tasks = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return []
        else:
            logger.error("No JSON array found in LLM response")
            return []

    tasks = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(raw_tasks):
        try:
            # Sanitize the LLM-supplied id (it becomes a git branch name, commit
            # grep key, progress/contract dict key) and dependency refs with the
            # SAME function so cross-references stay consistent. Drop duplicates
            # so they don't collapse in dependency maps.
            tid = safe_ref_slug(raw.get("id") or "", fallback=f"T-{i + 1:03d}")
            if tid in seen_ids:
                logger.warning(f"Skipping duplicate task id {tid!r}")
                continue
            seen_ids.add(tid)
            deps = [
                slug
                for d in raw.get("dependencies", [])
                if (slug := safe_ref_slug(d, fallback=""))
            ]
            task = Task(
                id=tid,
                title=raw.get("title", ""),
                description=raw.get("description", ""),
                acceptance_criteria=raw.get("acceptance_criteria", ""),
                files_to_create=raw.get("files_to_create", []),
                files_to_modify=raw.get("files_to_modify", []),
                context_files=raw.get("context_files", []),
                dependencies=deps,
                complexity=raw.get("complexity", "medium"),
                category=raw.get("category", "feature"),
                type="markdown_planner",
                status="pending",
                project_ref=project_ref,
                processor_data=raw,
            )
            tasks.append(task)
        except Exception as e:
            logger.warning(f"Skipping malformed task: {e}")

    return tasks


def _add_implicit_dependencies(tasks: list[Task]) -> None:
    """If task B modifies a file that task A creates, B depends on A."""
    creates_map: dict[str, str] = {}
    for t in tasks:
        for f in t.files_to_create:
            creates_map[f] = t.id

    for t in tasks:
        for f in t.files_to_modify:
            creator_id = creates_map.get(f)
            if creator_id and creator_id != t.id and creator_id not in t.dependencies:
                t.dependencies.append(creator_id)
                logger.info(
                    f"Implicit dependency: {t.id} depends on {creator_id} (file {f})"
                )


def topological_sort(tasks: list[Task]) -> list[Task]:
    """Sort tasks respecting dependencies. Falls back to category order on ties.

    Returns tasks in execution order. Raises ValueError on cycles.
    """
    task_map = {t.id: t for t in tasks}
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    dependents: dict[str, list[str]] = {t.id: [] for t in tasks}

    for t in tasks:
        for dep_id in t.dependencies:
            if dep_id in task_map:
                in_degree[t.id] += 1
                dependents[dep_id].append(t.id)

    # Seed queue with zero-dependency tasks, sorted by category order
    queue: deque[str] = deque()
    ready = [tid for tid, deg in in_degree.items() if deg == 0]
    ready.sort(key=lambda tid: CATEGORY_ORDER.get(task_map[tid].category, 99))
    queue.extend(ready)

    result = []
    while queue:
        tid = queue.popleft()
        result.append(task_map[tid])
        for dep_tid in dependents[tid]:
            in_degree[dep_tid] -= 1
            if in_degree[dep_tid] == 0:
                queue.append(dep_tid)

    if len(result) != len(tasks):
        # Cycle detected; append remaining tasks anyway with a warning
        remaining = [t for t in tasks if t.id not in {r.id for r in result}]
        logger.warning(
            f"Dependency cycle detected involving tasks: "
            f"{[t.id for t in remaining]}. Appending in category order."
        )
        remaining.sort(key=lambda t: CATEGORY_ORDER.get(t.category, 99))
        result.extend(remaining)

    return result


def format_plan(tasks: list[Task], mode: BuildMode) -> str:
    """Format task list as a markdown plan table (for --dry-run)."""
    lines = [
        "## Build Plan\n",
        f"**Mode**: {mode.value} | **Tasks**: {len(tasks)}\n",
        "| # | ID | Title | Category | Complexity | Depends On | Files |",
        "|---|------|-------|----------|------------|------------|-------|",
    ]
    for i, t in enumerate(tasks, 1):
        deps = ", ".join(t.dependencies) if t.dependencies else "-"
        n_create = len(t.files_to_create)
        n_modify = len(t.files_to_modify)
        files = f"{n_create} create, {n_modify} modify"
        lines.append(
            f"| {i} | {t.id} | {t.title} | {t.category} | "
            f"{t.complexity} | {deps} | {files} |"
        )
    return "\n".join(lines)
