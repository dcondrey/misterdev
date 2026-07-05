"""Project analyzer ported from /build Phase 1.

Uses LLM to analyze project structure, completeness, and context,
then merges results into a ProjectAssessment. In /build these run
as 3 parallel Claude sub-agents; here they are 3 sequential LLM
calls (or concurrent via threading if desired).
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from misterdev.core.planning.assessment import ProjectAssessment
from misterdev.core.verification.validator import run_health_check
from misterdev.llm.client import BaseLLMClient
from misterdev.logging_setup import setup_logger

from .detection import (
    detect_build_command,
    detect_test_command,
    has_test_files,
    _file_mentions,
    _has_node_tests,
    _has_python_tests,
    _json_has_test_script,
)
from .merge import (
    _as_int,
    _as_str_list,
    _health_ground_truth,
    _merge_completeness,
    _merge_context,
    _merge_debt_risk,
    _merge_structure,
)
from .overview import (
    _IGNORE_DIRS,
    _INTENT_KEYWORDS,
    _OVERVIEW_CODE_EXTS,
    _get_file_listing,
    _get_git_log,
    _get_source_overview,
    _leading_doc,
    _read_config_files,
    _read_docs,
    _read_file_safe,
    _walk_limited,
)
from .prompts import (
    COMPLETENESS_PROMPT,
    CONTEXT_PROMPT,
    DEBT_RISK_PROMPT,
    STRUCTURE_PROMPT,
)

logger = setup_logger(__name__)


def analyze_project(
    project_path: Path,
    llm_client: BaseLLMClient,
    build_command: Optional[str] = None,
    test_command: Optional[str] = None,
    lint_command: Optional[str] = None,
    env_activate: Optional[str] = None,
    parallel: bool = True,
    build_timeout: Optional[int] = None,
    test_timeout: Optional[int] = None,
    lint_timeout: Optional[int] = None,
    project_outline: Optional[str] = None,
) -> ProjectAssessment:
    """Run all Phase 1 analyses and merge into a ProjectAssessment.

    ``project_outline``, when supplied, is the project's already-built symbol
    outline (its TopographyEngine graph); passing it avoids parsing a second
    throwaway symbol graph just for the source overview.
    """
    assessment = ProjectAssessment()

    # Gather raw project info for prompts
    file_listing = _get_file_listing(project_path)
    config_contents = _read_config_files(project_path)
    docs = _read_docs(project_path)
    source_overview = _get_source_overview(project_path, outline=project_outline)
    readme = _read_file_safe(project_path / "README.md")
    claude_md = _read_file_safe(project_path / "CLAUDE.md")
    git_log = _get_git_log(project_path)

    # Run the health check FIRST, using reliable config + deterministic
    # detection (not LLM-guessed commands), so the completeness analyzer is
    # grounded in what actually builds and passes — otherwise it reads the
    # from-scratch docs and hallucinates that implemented features are missing.
    bc = build_command or detect_build_command(project_path)
    tc = test_command or detect_test_command(project_path)
    logger.info(
        "Running health check (build + tests) to ground the analysis; "
        "this can take a few minutes on a large project..."
    )
    assessment.health = run_health_check(
        project_path,
        bc,
        tc,
        lint_command,
        env_activate=env_activate,
        build_timeout=build_timeout,
        test_timeout=test_timeout,
        lint_timeout=lint_timeout,
    )
    assessment.structure.build_command = bc
    assessment.structure.test_command = tc
    health_ground = _health_ground_truth(assessment.health)

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
            health_ground=health_ground,
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

    # Health already ran (before the analyzers, to ground them). Re-assert the
    # deterministic commands in case the structure analyzer overwrote them with
    # nulls during merge, since downstream safety gates read them.
    if not assessment.structure.test_command:
        assessment.structure.test_command = tc
    if not assessment.structure.build_command:
        assessment.structure.build_command = bc

    logger.info(f"Assessment complete: {assessment.summary()}")
    return assessment


def _call_llm_json(llm_client: BaseLLMClient, prompt: str, role: str) -> dict:
    """Call LLM and parse JSON response."""
    logger.info(f"Running {role}...")
    try:
        response = llm_client.generate_code(
            prompt, f"You are a {role}. Return only valid JSON."
        )
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"{role} returned non-JSON response")
        return {}
    except Exception as e:
        logger.error(f"{role} failed: {e}")
        return {}


__all__ = [
    "STRUCTURE_PROMPT",
    "COMPLETENESS_PROMPT",
    "CONTEXT_PROMPT",
    "DEBT_RISK_PROMPT",
    "analyze_project",
    "_call_llm_json",
    "detect_test_command",
    "detect_build_command",
    "has_test_files",
    "_has_python_tests",
    "_has_node_tests",
    "_file_mentions",
    "_json_has_test_script",
    "_health_ground_truth",
    "_as_str_list",
    "_as_int",
    "_merge_structure",
    "_merge_completeness",
    "_merge_context",
    "_merge_debt_risk",
    "_IGNORE_DIRS",
    "_OVERVIEW_CODE_EXTS",
    "_INTENT_KEYWORDS",
    "_walk_limited",
    "_get_file_listing",
    "_read_config_files",
    "_read_docs",
    "_leading_doc",
    "_get_source_overview",
    "_get_git_log",
    "_read_file_safe",
    "run_health_check",
]
