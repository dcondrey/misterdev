"""Project analyzer ported from /build Phase 1.

Uses LLM to analyze project structure, completeness, and context,
then merges results into a ProjectAssessment. In /build these run
as 3 parallel Claude sub-agents; here they are 3 sequential LLM
calls (or concurrent via threading if desired).
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from my_project_orchestrator.core.assessment import (
    FeatureInfo,
    FeatureInventory,
    HealthCheck,
    ProjectAssessment,
    ProjectContext,
    ProjectStructure,
)
from my_project_orchestrator.core.validator import run_health_check
from my_project_orchestrator.llm.client import BaseLLMClient
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)

STRUCTURE_PROMPT = """Analyze the project at the path below. Return a JSON object with these exact fields:
  project_type (string: web-api, web-app, cli, library, worker, static-site, monorepo, or unknown),
  languages (array of strings),
  frameworks (array of strings),
  build_command (string or null),
  test_command (string or null),
  lint_command (string or null),
  package_manager (string or null),
  entry_points (array of file path strings),
  directory_structure (string, tree of src dirs, max 30 lines)

Project files:
{file_listing}

Key config files:
{config_contents}

Return ONLY valid JSON, no markdown fences."""

COMPLETENESS_PROMPT = """Analyze this project for completeness. Return a JSON object with:
  existing (array of objects with "name" and "description"),
  incomplete (array of objects with "name", "description", and "complexity"),
  missing (array of objects with "name", "description", and "complexity"),
  dead_code (array of file path strings),
  stubs (array of file path strings),
  broken (array of file path strings),
  todos (array of objects with "file", "line", "text")

Project docs:
{docs}

Project source overview:
{source_overview}

Return ONLY valid JSON, no markdown fences."""

CONTEXT_PROMPT = """Gather context for this project. Return a JSON object with:
  purpose (string, 1-2 sentences),
  goals (string),
  conventions (string, coding conventions found),
  constraints (string, requirements or compatibility needs),
  recent_activity (string, summary of recent work),
  stated_requirements (string, from spec/design docs)

README:
{readme}

CLAUDE.md / config:
{config}

Recent git log:
{git_log}

Return ONLY valid JSON, no markdown fences."""


DEBT_RISK_PROMPT = """Analyze this project for technical debt and implementation risk. Return a JSON object with:
  tech_debt (object with "score" [0-100], "description", "critical_issues"),
  risk (object with "level" [low, medium, high, critical], "factors", "mitigations")

Project assessment so far:
{assessment_summary}

Source code overview:
{source_overview}

Return ONLY valid JSON, no markdown fences."""


def analyze_project(
    project_path: Path,
    llm_client: BaseLLMClient,
    build_command: Optional[str] = None,
    test_command: Optional[str] = None,
    lint_command: Optional[str] = None,
    env_activate: Optional[str] = None,
    parallel: bool = True,
) -> ProjectAssessment:
    """Run all Phase 1 analyses and merge into a ProjectAssessment."""
    assessment = ProjectAssessment()

    # Gather raw project info for prompts
    file_listing = _get_file_listing(project_path)
    config_contents = _read_config_files(project_path)
    docs = _read_docs(project_path)
    source_overview = _get_source_overview(project_path)
    readme = _read_file_safe(project_path / "README.md")
    claude_md = _read_file_safe(project_path / "CLAUDE.md")
    git_log = _get_git_log(project_path)

    def analyze_structure():
        prompt = STRUCTURE_PROMPT.format(
            file_listing=file_listing,
            config_contents=config_contents,
        )
        return _call_llm_json(llm_client, prompt, "project structure analyzer")

    def analyze_completeness():
        prompt = COMPLETENESS_PROMPT.format(
            docs=docs,
            source_overview=source_overview,
        )
        return _call_llm_json(llm_client, prompt, "completeness analyzer")

    def analyze_context():
        prompt = CONTEXT_PROMPT.format(
            readme=readme,
            config=claude_md or config_contents,
            git_log=git_log,
        )
        return _call_llm_json(llm_client, prompt, "context analyzer")

    def analyze_debt_risk(current_summary: str):
        prompt = DEBT_RISK_PROMPT.format(
            assessment_summary=current_summary,
            source_overview=source_overview,
        )
        return _call_llm_json(llm_client, prompt, "debt and risk analyzer")

    # Phase 1a-1c: run analyses (parallel or sequential)
    results = {}
    if parallel:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(analyze_structure): "structure",
                pool.submit(analyze_completeness): "completeness",
                pool.submit(analyze_context): "context",
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"Analysis failed for {key}: {e}")
                    results[key] = {}
            
            # Merge preliminary results to feed into debt/risk analyzer
            _merge_structure(assessment, results.get("structure", {}))
            _merge_completeness(assessment, results.get("completeness", {}))
            _merge_context(assessment, results.get("context", {}))
            
            future_debt = pool.submit(analyze_debt_risk, assessment.summary())
            results["debt_risk"] = future_debt.result()
    else:
        results["structure"] = analyze_structure()
        results["completeness"] = analyze_completeness()
        results["context"] = analyze_context()
        _merge_structure(assessment, results.get("structure", {}))
        _merge_completeness(assessment, results.get("completeness", {}))
        _merge_context(assessment, results.get("context", {}))
        results["debt_risk"] = analyze_debt_risk(assessment.summary())

    # Phase 1d: merge remaining into assessment
    _merge_debt_risk(assessment, results.get("debt_risk", {}))

    # Run actual health check using detected or provided commands
    bc = build_command or assessment.structure.build_command
    tc = test_command or assessment.structure.test_command
    lc = lint_command or assessment.structure.lint_command
    assessment.health = run_health_check(
        project_path, bc, tc, lc, env_activate=env_activate,
    )

    logger.info(f"Assessment complete: {assessment.summary()}")
    return assessment


def _call_llm_json(llm_client: BaseLLMClient, prompt: str, role: str) -> dict:
    """Call LLM and parse JSON response."""
    logger.info(f"Running {role}...")
    try:
        response = llm_client.generate_code(prompt, f"You are a {role}. Return only valid JSON.")
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"{role} returned non-JSON response")
        return {}
    except Exception as e:
        logger.error(f"{role} failed: {e}")
        return {}


def _merge_structure(assessment: ProjectAssessment, data: dict) -> None:
    if not data:
        return
    s = assessment.structure
    s.project_type = data.get("project_type", s.project_type)
    s.languages = data.get("languages", s.languages)
    s.frameworks = data.get("frameworks", s.frameworks)
    s.build_command = data.get("build_command", s.build_command)
    s.test_command = data.get("test_command", s.test_command)
    s.lint_command = data.get("lint_command", s.lint_command)
    s.package_manager = data.get("package_manager", s.package_manager)
    s.entry_points = data.get("entry_points", s.entry_points)
    s.directory_structure = data.get("directory_structure", s.directory_structure)


def _merge_completeness(assessment: ProjectAssessment, data: dict) -> None:
    if not data:
        return
    f = assessment.features
    for item in data.get("existing", []):
        if isinstance(item, dict):
            f.existing.append(FeatureInfo(name=item.get("name", ""), description=item.get("description", "")))
    for item in data.get("incomplete", []):
        if isinstance(item, dict):
            f.incomplete.append(FeatureInfo(
                name=item.get("name", ""), description=item.get("description", ""),
                complexity=item.get("complexity", "medium"),
            ))
    for item in data.get("missing", []):
        if isinstance(item, dict):
            f.missing.append(FeatureInfo(
                name=item.get("name", ""), description=item.get("description", ""),
                complexity=item.get("complexity", "medium"),
            ))
    f.dead_code = data.get("dead_code", f.dead_code)
    f.stubs = data.get("stubs", f.stubs)
    f.broken = data.get("broken", f.broken)
    f.todos = data.get("todos", f.todos)


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
    
    debt_data = data.get("tech_debt", {})
    if debt_data:
        assessment.tech_debt.score = debt_data.get("score", 0)
        assessment.tech_debt.description = debt_data.get("description", "")
        assessment.tech_debt.critical_issues = debt_data.get("critical_issues", [])

    risk_data = data.get("risk", {})
    if risk_data:
        assessment.risk.level = risk_data.get("level", "low")
        assessment.risk.factors = risk_data.get("factors", [])
        assessment.risk.mitigations = risk_data.get("mitigations", [])


_IGNORE_DIRS = {
    "venv", ".venv", "node_modules", "__pycache__", ".git",
    "target", "build", "dist", ".tox", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", ".eggs", "vendor",
}


def _walk_limited(root: Path, max_depth: int = 6):
    """Yield files under root, pruning ignored dirs and bounding depth.

    Unlike Path.rglob, this prunes large directories (node_modules, target)
    during traversal instead of descending into them and filtering afterward,
    so scanning a monorepo does not stall on vendored trees.
    """
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in _IGNORE_DIRS or entry.name.startswith("."):
                    continue
                if depth < max_depth:
                    stack.append((entry, depth + 1))
            elif entry.is_file():
                yield entry


def _get_file_listing(project_path: Path, max_files: int = 200, max_depth: int = 6) -> str:
    """Get a truncated file listing for the project."""
    files = []
    for item in _walk_limited(project_path, max_depth):
        rel = item.relative_to(project_path)
        files.append(str(rel))
        if len(files) >= max_files:
            files.append(f"... ({max_files}+ files, truncated)")
            break
    return "\n".join(sorted(files))


def _read_config_files(project_path: Path) -> str:
    """Read common config files for structure analysis."""
    config_names = [
        "pyproject.toml", "setup.py", "setup.cfg", "package.json",
        "Cargo.toml", "go.mod", "Makefile", "CMakeLists.txt",
        "project.yaml", "tsconfig.json", "webpack.config.js",
    ]
    contents = []
    for name in config_names:
        text = _read_file_safe(project_path / name, max_lines=100)
        if text:
            contents.append(f"### {name}\n{text}")
    return "\n\n".join(contents) if contents else "(no config files found)"


def _read_docs(project_path: Path) -> str:
    """Read documentation files."""
    doc_names = ["README.md", "CLAUDE.md", "SPEC.md", "DESIGN.md", "REQUIREMENTS.md"]
    contents = []
    for name in doc_names:
        text = _read_file_safe(project_path / name, max_lines=200)
        if text:
            contents.append(f"### {name}\n{text}")
    return "\n\n".join(contents) if contents else "(no docs found)"


def _get_source_overview(project_path: Path, max_chars: int = 8000) -> str:
    """Read first N chars of source files for completeness analysis."""
    code_exts = {".py", ".js", ".ts", ".rs", ".go", ".java"}
    parts = []
    total = 0
    for item in _walk_limited(project_path):
        if item.suffix in code_exts:
            text = _read_file_safe(item, max_lines=30)
            if text:
                rel = item.relative_to(project_path)
                chunk = f"### {rel}\n{text}\n"
                parts.append(chunk)
                total += len(chunk)
                if total >= max_chars:
                    break
    return "\n".join(parts) if parts else "(no source files found)"


def _get_git_log(project_path: Path, count: int = 20) -> str:
    """Get recent git log."""
    try:
        proc = subprocess.run(
            f"git log --oneline -n {count}",
            shell=True, cwd=project_path,
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "(not a git repo)"
    except Exception:
        return "(git log unavailable)"


def _read_file_safe(path: Path, max_lines: int = 500) -> str:
    """Read a file, returning empty string on failure."""
    try:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines)} lines total, truncated)"]
        return "\n".join(lines)
    except Exception:
        return ""
