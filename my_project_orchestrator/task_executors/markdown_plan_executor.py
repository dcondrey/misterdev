"""Markdown plan executor - executes tasks via LLM with Try-Test-Fix loop."""

import ast
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from my_project_orchestrator.core.context_budget import ContextBudget
from my_project_orchestrator.core.models import Task, ExecutionResult
from my_project_orchestrator.core.project import Project
from my_project_orchestrator.core.scratchpad import Scratchpad
from my_project_orchestrator.llm.prompt_manager import PromptManager
from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.task_executors.base_executor import BaseTaskExecutor
from my_project_orchestrator.llm.responses import LLMResponseParser
from my_project_orchestrator.llm.client import code_gen_abort_check
from my_project_orchestrator.core.validator import (
    CodeValidator,
    CertaintyScorer,
    StallDetector,
    _run_cmd,
)
from my_project_orchestrator.core.error_resolver import ErrorResolver
from my_project_orchestrator.core.error_classifier import (
    format_classified_error,
    classify_error,
)
from my_project_orchestrator.config import get_setting
from my_project_orchestrator.utils.file_utils import write_file, is_golden_path

_is_golden_path = is_golden_path

logger = setup_logger(__name__)

# Maps file extensions to language identifiers for syntax validation and
# contract extraction. Unknown extensions fall back to "text".
_LANG_MAP = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell",
    ".bash": "shell",
}


def _bisect_first_failing(n: int, passes_at) -> int:
    """Binary-search [0, n) for the first index where passes_at(i) is False.

    Assumes a monotonic pass->fail boundary (all-pass prefix, then failures).
    Returns n-1 if nothing fails; callers should re-check that index.
    """
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if passes_at(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _detect_language(file_path: str) -> str:
    """Detect a source language from a file path's extension.

    Returns "text" for extensions with no known language mapping so callers
    fall back to language-agnostic validation rather than guessing.
    """
    ext = Path(file_path).suffix.lower()
    return _LANG_MAP.get(ext, "text")


# Path patterns that mark a file as holding tests, covering the languages in
# _LANG_MAP. Used to decide which edits get the tamper-resistance check.
_TEST_FILE_PATTERNS = (
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r"_test\.py$"),
    re.compile(r"(^|/)conftest\.py$"),
    re.compile(r"\.test\.(js|jsx|ts|tsx)$"),
    re.compile(r"\.spec\.(js|jsx|ts|tsx)$"),
    re.compile(r"_test\.go$"),
    re.compile(r"_test\.(rs|rb)$"),
    re.compile(r"(^|/)test_[^/]+\.(rb|c|cpp|cc)$"),
    re.compile(r"Test[^/]*\.java$"),
    re.compile(r"(^|/)tests?/"),
)

# Skip/ignore markers per language. Their COUNT must not grow across an edit:
# more skips is the cheapest way to make a suite "pass" by not running it.
_SKIP_PATTERNS = (
    re.compile(
        r"@(?:pytest\.mark\.skip|pytest\.mark\.skipif|unittest\.skip"
        r"|unittest\.SkipTest|skip|skipif)\b"
    ),
    re.compile(r"\b(?:it|describe|test|context)\.skip\b"),
    re.compile(r"\bxit\b|\bxdescribe\b"),
    re.compile(r"#\[ignore\b"),
    re.compile(r"\bt\.Skip(?:Now)?\b|\bt\.SkipNow\b"),
    re.compile(r"@(?:Ignore|Disabled)\b"),
    re.compile(r"\.skip\s*\(|\.todo\s*\("),
)

# Things that count as a "test" definition per language. We compare the total
# across all patterns rather than per-pattern so an edit can't dodge the check
# by converting one form of test to another.
_TEST_DEF_PATTERNS = (
    re.compile(r"(?m)^\s*def\s+test\w*\s*\("),
    re.compile(r"\b(?:it|test)\s*\(\s*['\"`]"),
    re.compile(r"#\[test\]"),
    re.compile(r"(?m)^\s*func\s+Test\w*\s*\("),
    re.compile(r"@Test\b"),
)

# Assertion-ish forms. Weakening a test often keeps the function but guts its
# checks, so a drop in assertion count is just as suspicious as a dropped test.
_ASSERT_PATTERNS = (
    re.compile(r"(?m)^\s*assert\b"),
    re.compile(r"\bself\.assert\w+\s*\("),
    re.compile(r"\bexpect\s*\("),
    re.compile(r"\bassert(?:_eq|_ne|_matches)?!\s*\("),
    re.compile(r"\bpytest\.raises\b|\bassertRaises\b"),
)

# Trivially-true assertions: a cheap way to keep the assertion COUNT steady
# while removing the actual check ("assert True", "expect(true).toBe(true)").
# An increase in these across an edit is treated as weakening, cross-language.
_TAUTOLOGY_PATTERNS = (
    re.compile(r"(?m)^\s*assert\s+(?:True|1)\s*(?:,|$)"),
    re.compile(r"\bself\.assertTrue\s*\(\s*True\s*\)"),
    re.compile(r"\bself\.assertFalse\s*\(\s*False\s*\)"),
    re.compile(r"\bassert!\s*\(\s*true\s*\)"),
    re.compile(r"\bexpect\s*\(\s*true\s*\)\s*\.\s*to(?:Be|Equal)\s*\(\s*true\s*\)"),
    re.compile(r"\bassert\s+1\s*===?\s*1\b"),
)

# In-body skip calls (vs the decorator forms in _SKIP_PATTERNS): inserting one
# of these short-circuits a test at runtime while leaving it visibly defined.
_INBODY_SKIP_PATTERNS = (
    re.compile(r"\bpytest\.skip\s*\("),
    re.compile(r"\bself\.skipTest\s*\("),
    re.compile(r"\bunittest\.SkipTest\b"),
    re.compile(r"\bt\.Skip(?:Now)?\s*\("),
)

# Fraction of the build budget that must remain before the LLM acceptance judge
# (default-on) is allowed to spend; below it, free-text criteria pass for free.
JUDGE_MIN_BUDGET_FRACTION = 0.1


def _is_test_file(file_path: str) -> bool:
    """True if the path names a test file in one of the supported languages."""
    norm = file_path.replace("\\", "/")
    return any(p.search(norm) for p in _TEST_FILE_PATTERNS)


# Known test/build runners that mark the START of a runnable acceptance command.
# We only treat acceptance_criteria as a command when one of these verbs appears,
# so free-text criteria ("the login form rejects empty passwords") are never
# mis-parsed into a command. Ordered/anchored so multi-word runners match before
# their first word (e.g. "python -m pytest" before bare "python").
_ACCEPTANCE_RUNNERS = (
    r"python3?\s+-m\s+pytest",
    r"pytest",
    r"cargo\s+test",
    r"cargo\s+build",
    r"cargo\s+check",
    r"cargo\s+clippy",
    r"go\s+test",
    r"go\s+build",
    r"npm\s+test",
    r"npm\s+run\s+\S+",
    r"npx\s+\S+",
    r"yarn\s+\S+",
    r"pnpm\s+\S+",
    r"make\s+\S+",
    r"make",
    r"ruff(?:\s+\S+)?",
    r"mypy",
    r"pyright",
    r"tsc",
    r"jest",
    r"vitest",
    r"tox",
    r"phpunit",
    r"rspec",
    r"gradle\s+\S+",
    r"\./gradlew\s+\S+",
    r"mvn\s+\S+",
)

# A runnable command starts at a known runner and runs to the end of the line (or
# a sentence-terminating boundary). Anything before the runner (e.g. "Verify that
# ") is dropped. The trailing tail is trimmed by _extract_acceptance_command.
_ACCEPTANCE_COMMAND_RE = re.compile(
    r"(?P<cmd>(?:" + "|".join(_ACCEPTANCE_RUNNERS) + r")[^\n]*)",
    re.IGNORECASE,
)

# Trailing prose that commonly follows a quoted command and is not part of it,
# e.g. "pytest tests/test_auth.py passes". Stripped from the extracted command.
_ACCEPTANCE_TAIL_RE = re.compile(
    r"\s+(?:passes?|succeeds?|should\s+pass|must\s+pass|exits?\s+0|"
    r"returns?\s+0|is\s+green|all\s+green|cleanly|without\s+errors?)\b.*$",
    re.IGNORECASE,
)


def _extract_acceptance_command(criteria: str) -> Optional[str]:
    """Extract a single runnable command from an acceptance-criteria string.

    Conservative: returns a command only when the text begins (after optional
    lead-in prose) with a known test/build runner. Trailing prose like
    "... passes" is trimmed so the runner sees just the command. Returns None
    for free-text criteria so un-parseable sentences never fail a task.
    """
    if not criteria:
        return None
    # Prefer a command fenced in backticks if present, but still require it to
    # start with a known runner so prose in backticks isn't run blindly.
    for candidate in re.findall(r"`([^`]+)`", criteria):
        m = _ACCEPTANCE_COMMAND_RE.match(candidate.strip())
        if m:
            return _ACCEPTANCE_TAIL_RE.sub("", m.group("cmd")).strip()
    m = _ACCEPTANCE_COMMAND_RE.search(criteria)
    if not m:
        return None
    cmd = m.group("cmd").strip()
    # Cut at the first sentence boundary so a trailing English sentence on the
    # same line doesn't get fed to the shell.
    cmd = re.split(r"(?<=\S)[.;]\s+[A-Z]", cmd, maxsplit=1)[0].strip()
    cmd = _ACCEPTANCE_TAIL_RE.sub("", cmd).strip()
    return cmd or None


def _test_metrics(content: str) -> Tuple[int, int, int]:
    """Cheap structural metrics for a test file: (tests, asserts, skips).

    Deterministic regex counts only, no parsing. Used to compare a test file
    before vs after an edit; growth is fine, shrinkage/more-skips is tamper.
    """
    tests = sum(len(p.findall(content)) for p in _TEST_DEF_PATTERNS)
    asserts = sum(len(p.findall(content)) for p in _ASSERT_PATTERNS)
    skips = sum(
        len(p.findall(content)) for p in (*_SKIP_PATTERNS, *_INBODY_SKIP_PATTERNS)
    )
    return tests, asserts, skips


def _count_tautologies(content: str) -> int:
    """Count trivially-true assertions (regex, cross-language)."""
    return sum(len(p.findall(content)) for p in _TAUTOLOGY_PATTERNS)


def _diagnose_tampering(before: str, after: str) -> Optional[str]:
    """Return a reason string if `after` weakens the test file, else None.

    Tampering = fewer tests, fewer assertions, more skip markers (decorator or
    in-body), or more trivially-true assertions. Pure additions (new
    tests/assertions) are allowed and return None.
    """
    bt, ba, bs = _test_metrics(before)
    at, aa, as_ = _test_metrics(after)
    reasons = []
    if at < bt:
        reasons.append(f"test count dropped {bt}->{at}")
    if aa < ba:
        reasons.append(f"assertion count dropped {ba}->{aa}")
    if as_ > bs:
        reasons.append(f"skip/ignore markers increased {bs}->{as_}")
    btaut, ataut = _count_tautologies(before), _count_tautologies(after)
    if ataut > btaut:
        reasons.append(f"trivially-true assertions increased {btaut}->{ataut}")
    return "; ".join(reasons) if reasons else None


def _ast_call_name(func: ast.AST) -> Optional[str]:
    """Best-effort dotted-call leaf name: ``self.assertEqual`` -> assertEqual."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _assertion_key(node: ast.AST) -> Optional[str]:
    """A structure-based identity for one assertion, or None if it isn't one.

    Keys on what the assertion checks (the AST of the condition / call args),
    not on its position or enclosing function, so the same check has the same
    key wherever it lives.
    """
    if isinstance(node, ast.Assert):
        return "assert:" + ast.dump(node.test)
    if isinstance(node, ast.Call):
        name = _ast_call_name(node.func)
        if name and (name.startswith("assert") or name in ("expect", "raises")):
            args = ",".join(ast.dump(a) for a in node.args)
            return f"{name}:{args}"
    return None


def _py_assertion_multiset(content: str) -> Optional["Counter"]:
    """Structural multiset of every assertion inside Python ``test*`` functions.

    Because assertions are keyed by structure (see ``_assertion_key``) rather
    than by their enclosing test's name or order, renaming, moving, reordering,
    splitting, or merging tests leaves the multiset unchanged. Only an assertion
    that disappears without an identical one reappearing is a real loss of
    coverage. Returns None when the content does not parse.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    counts: Counter = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        for sub in ast.walk(node):
            key = _assertion_key(sub)
            if key:
                counts[key] += 1
    return counts


def _parametrize_count(content: str) -> int:
    """Count ``parametrize``-style decorators (a legit way to reshape asserts)."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 0
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if _ast_call_name(target) == "parametrize":
                total += 1
    return total


def _py_skip_count(content: str) -> int:
    return sum(
        len(p.findall(content)) for p in (*_SKIP_PATTERNS, *_INBODY_SKIP_PATTERNS)
    )


def _diagnose_py_tampering(before: str, after: str) -> Optional[str]:
    """Python tamper diagnosis keyed on assertion survival, not test identity.

    Flags only genuine weakening: assertions that vanish without an identical
    check reappearing anywhere, an increase in trivially-true assertions, or
    more skip markers. Renames, moves, reorders, splits, and merges all
    preserve the assertion multiset and pass. Parametrization legitimately
    reshapes assertions, so a net loss is not held against an edit that adds a
    parametrize decorator. Returns None when either revision fails to parse.
    """
    before_asserts = _py_assertion_multiset(before)
    after_asserts = _py_assertion_multiset(after)
    if before_asserts is None or after_asserts is None:
        return None
    reasons = []
    if _parametrize_count(after) <= _parametrize_count(before):
        lost = sum((before_asserts - after_asserts).values())
        if lost:
            reasons.append(f"{lost} assertion(s) removed or weakened")
    if _count_tautologies(after) > _count_tautologies(before):
        reasons.append("trivially-true assertions increased")
    if _py_skip_count(after) > _py_skip_count(before):
        reasons.append("skip markers increased")
    return "; ".join(reasons) if reasons else None


class MarkdownPlanExecutor(BaseTaskExecutor):
    """Executes tasks with a Try-Test-Fix loop.

    Uses git branch-per-task for atomic execution: each task runs on a
    temporary branch. Success merges to the current branch. Failure
    deletes the branch, leaving the repo clean.
    """

    def __init__(self, scratchpad: Optional[Scratchpad] = None):
        self.scratchpad = scratchpad or Scratchpad()
        self.stall_detector = StallDetector()

    def execute(
        self, task: Task, project: Project, use_git_branch: bool = True, _depth: int = 0
    ) -> ExecutionResult:
        logger.info(f"Starting execution of task: {task.id}")
        project.task_manager.update_task_status(task.id, "in_progress")

        prompt_manager = PromptManager(project.config)
        processor_config = self._get_processor_config(project)

        # Read max_task_attempts from orchestrator config, falling back to processor config, then hardcoded default
        default_attempts = get_setting(
            project.config, "orchestrator", "max_task_attempts"
        )
        max_retries = processor_config.get("max_retries_per_task", default_attempts)

        # Read context budget tokens from orchestrator config or task processor_data
        context_budget_tokens = task.processor_data.get(
            "context_budget_tokens",
            get_setting(project.config, "orchestrator", "context_budget_tokens"),
        )

        # Minimum LLM certainty required to accept a task as completed when no
        # test gate ran. Deterministic: compared against the heuristic certainty
        # score only, never another LLM call.
        certainty_threshold = get_setting(
            project.config, "orchestrator", "certainty_threshold"
        )

        # Per-task acceptance gate. Cheap deterministic command path is on by
        # default; the LLM-judge fallback is off by default so the default path
        # adds zero extra LLM calls. When no command is found and the judge is
        # off, acceptance is a no-op (behaviour identical to before this gate).
        verify_acceptance = get_setting(
            project.config, "orchestrator", "verify_acceptance"
        )
        llm_acceptance_judge = get_setting(
            project.config, "orchestrator", "llm_acceptance_judge"
        )

        files_key = processor_config.get("target_files_key", "files_to_modify")
        test_cmd_key = processor_config.get("test_command_key", "test_command")
        typecheck_cmd_key = processor_config.get(
            "typecheck_command_key", "typecheck_command"
        )

        target_files = task.processor_data.get(files_key, [])
        if isinstance(target_files, str):
            target_files = [target_files]
        target_files = list(
            set(target_files + task.files_to_modify + task.files_to_create)
        )

        # Golden suite: never task the model with these files and never read
        # them into its context. Combined with the edit-time rejection in
        # _validate_edit_paths, the model cannot see or alter them.
        golden_paths = get_setting(project.config, "orchestrator", "golden_paths")
        target_files = [f for f in target_files if not _is_golden_path(f, golden_paths)]
        context_files = [
            f for f in task.context_files if not _is_golden_path(f, golden_paths)
        ]

        test_command = task.processor_data.get(test_cmd_key)
        typecheck_command = task.processor_data.get(
            typecheck_cmd_key
        ) or project.config.get(typecheck_cmd_key)
        build_timeout = get_setting(project.config, "build", "build_timeout")
        test_timeout = get_setting(project.config, "build", "test_timeout")

        # Atomic execution: git branch per task (disabled in parallel mode to avoid races)
        can_branch = use_git_branch and self._is_git_repo(project)
        branch_name = f"task/{task.id}" if can_branch else None
        base_branch = None

        if branch_name:
            base_branch = self._get_current_branch(project)
            if not self._create_task_branch(project, branch_name):
                logger.warning(
                    "Failed to create task branch; falling back to file snapshots"
                )
                branch_name = None

        # Fallback: file snapshots if git branching isn't available
        snapshot = None
        if not branch_name:
            snapshot = self._snapshot_files(project, target_files)

        error_logs = None
        prior_errors: List[str] = []
        # Build the symbol graph now (idempotent, lazy): it isn't constructed at
        # project registration anymore, and task execution is the first consumer.
        try:
            project.topography.initialize()
        except Exception as e:
            logger.warning(f"Topography init failed (continuing without graph): {e}")
        resolver = ErrorResolver(project.path, project.topography.graph)
        # Every in-root path the LLM actually wrote, across all attempts. Commit
        # exactly these (plus declared targets) so an out-of-scope-but-valid
        # edit isn't applied-then-orphaned by staging only the declared files.
        edited_files: set = set()

        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1}/{max_retries} for task {task.id}")

            code_context = self._get_code_context(project, target_files, context_files)
            topo_context = project.topography.get_context_for_task(
                task.description, target_files
            )
            scratchpad_context = self.scratchpad.format_context(
                files=target_files + context_files,
                tags=[task.category],
            )

            strategy = task.processor_data.get("strategy", "iterative")
            consensus = task.processor_data.get("consensus_context", "None")
            interface_contracts = task.processor_data.get("interface_contracts", "")
            recent_changes = task.processor_data.get("recent_changes", "")

            # Budget-aware context allocation using configured token limit
            budget = ContextBudget(max_tokens=context_budget_tokens)
            budget.set("code_context", code_context, priority=1)
            budget.set("topo_context", topo_context, priority=3)
            budget.set("error_logs", error_logs or "", priority=1, min_lines=20)
            budget.set("scratchpad", scratchpad_context, priority=3)
            budget.set("interface_contracts", interface_contracts, priority=2)
            budget.set("recent_changes", recent_changes, priority=2)
            budget.set("consensus_context", consensus, priority=3)
            allocated = budget.allocate()

            full_code_context = allocated["code_context"]
            if allocated["topo_context"]:
                full_code_context += "\n\n" + allocated["topo_context"]
            if allocated["recent_changes"]:
                full_code_context += "\n\n" + allocated["recent_changes"]

            context_dict = {
                "project": project,
                "task": task,
                "code_context": full_code_context,
                "error_logs": allocated["error_logs"] or None,
                "task.description": task.description,
                "task.target_files": ", ".join(target_files)
                if target_files
                else "None explicitly specified",
                "scratchpad": allocated["scratchpad"],
                "acceptance_criteria": task.acceptance_criteria,
                "consensus_context": allocated["consensus_context"],
                "interface_contracts": allocated["interface_contracts"],
                "strategy": strategy.upper(),
                "invariants": (
                    f"Strategy: {strategy.upper()}. Output MUST be syntactically valid. "
                    "Provide certainty indicators."
                ),
            }

            system_prompt = prompt_manager.format_prompt("system", context_dict)
            if error_logs and attempt > 0:
                prompt = prompt_manager.format_prompt(
                    "error_correction_instruction", context_dict
                )
            else:
                prompt = prompt_manager.format_prompt(
                    "task_completion_instruction", context_dict
                )

            routed_model = self._resolve_model(project, task, strategy)
            aborted = False
            try:
                with project.llm_client.track_task(task.id):
                    if routed_model:
                        logger.info(
                            f"[{task.id}] routing to {routed_model} ({task.complexity}/{strategy})"
                        )
                        with project.llm_client.with_model(routed_model):
                            llm_response, aborted = self._invoke_llm(
                                project, prompt, system_prompt
                            )
                    else:
                        llm_response, aborted = self._invoke_llm(
                            project, prompt, system_prompt
                        )
            except Exception as e:
                msg = f"LLM generation failed: {e}"
                logger.error(msg)
                self._abort_task(project, branch_name, base_branch, snapshot)
                return self._fail_task(project, task, msg)

            if aborted:
                logger.warning(
                    f"LLM stream aborted for {task.id}; retrying with stricter instruction."
                )
                error_logs = "ERROR: response was not code. Output ONLY file edits as code blocks with file paths."
                continue

            certainty = CertaintyScorer.compute_score(llm_response)
            logger.info(f"LLM Certainty Score: {certainty:.2f}")

            edits = LLMResponseParser.parse_file_edits(llm_response)
            edits = self._validate_edit_paths(project, task, edits)
            if not edits:
                logger.warning("No file edits detected in LLM response.")
            else:
                stall_risk = self.stall_detector.push_edit(edits)
                if stall_risk > 0.7:
                    logger.warning(f"High stall risk detected ({stall_risk:.2f}).")
                    if attempt > 1:
                        error_logs = "ERROR: Stalling detected. Try a fundamentally different approach."
                        continue

                validation_failed = False
                for file_path, content in edits.items():
                    lang = _detect_language(file_path)
                    valid, error = CodeValidator.validate_code(content, language=lang)
                    if not valid:
                        logger.error(f"Validation failed for {file_path}: {error}")
                        error_logs = f"SYNTAX ERROR in {file_path}:\n{error}"
                        validation_failed = True
                        break

                if validation_failed:
                    continue

                tamper = self._detect_test_tampering(project, edits)
                if tamper:
                    logger.error(f"Test tampering rejected: {tamper}")
                    error_logs = (
                        "ERROR: test files were weakened. Do not delete "
                        "tests, weaken assertions, or add skip/ignore markers "
                        "to make the suite pass. Fix the real code instead. "
                        f"Detected: {tamper}"
                    )
                    continue

                self._apply_edits(project, edits)
                self._run_formatters(project, edits.keys())
                edited_files.update(edits.keys())

            build_cmd = project.config.get("build_command")
            if build_cmd:
                success, output = self._run_command(
                    project, build_cmd, timeout=build_timeout
                )
                if not success:
                    logger.warning(f"Build failed on attempt {attempt + 1}")
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors, attempt, output, classified, attributed_error
                    )
                    continue

            if typecheck_command:
                success, output = self._run_command(
                    project, typecheck_command, timeout=build_timeout
                )
                if not success:
                    logger.warning(f"Type check failed on attempt {attempt + 1}")
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors, attempt, output, classified, attributed_error
                    )
                    continue

            if test_command:
                success, output = self._run_command(
                    project, test_command, timeout=test_timeout
                )
                if success:
                    logger.info("Tests passed successfully.")
                    acc_ok, acc_output = self._verify_acceptance(
                        project,
                        task,
                        verify_acceptance,
                        llm_acceptance_judge,
                        test_timeout,
                    )
                    if not acc_ok:
                        logger.warning(
                            f"Acceptance criteria not met on attempt {attempt + 1}."
                        )
                        classified = format_classified_error(acc_output)
                        error_logs = self._build_acceptance_error_context(
                            prior_errors, attempt, task, classified
                        )
                        continue
                    self._record_success(task, target_files)
                    self._commit_task(
                        project,
                        branch_name,
                        base_branch,
                        task,
                        sorted(set(target_files) | edited_files),
                    )
                    return self._complete_task(
                        project, task, "Task completed and tests passed.", output
                    )
                else:
                    logger.warning(f"Tests failed on attempt {attempt + 1}.")
                    locations = resolver.resolve_errors(output)
                    attributed_error = resolver.format_for_llm(locations)
                    classified = format_classified_error(output)
                    error_logs = self._build_error_context(
                        prior_errors, attempt, output, classified, attributed_error
                    )
            elif certainty < certainty_threshold:
                # No test gate ran, so completion rests entirely on the LLM's
                # word. With low certainty that's not enough to trust: force a
                # retry (which escalates to surgical once attempts run out)
                # instead of silently accepting unverified code.
                logger.warning(
                    f"No tests ran and certainty {certainty:.2f} < "
                    f"{certainty_threshold:.2f}; refusing silent completion."
                )
                error_logs = (
                    "ERROR: no tests verify this change and the response "
                    "expressed low certainty. Provide a higher-confidence, "
                    "syntactically valid edit with explicit verification."
                )
                continue
            else:
                acc_ok, acc_output = self._verify_acceptance(
                    project,
                    task,
                    verify_acceptance,
                    llm_acceptance_judge,
                    test_timeout,
                )
                if not acc_ok:
                    logger.warning(
                        f"Acceptance criteria not met on attempt {attempt + 1}."
                    )
                    classified = format_classified_error(acc_output)
                    error_logs = self._build_acceptance_error_context(
                        prior_errors, attempt, task, classified
                    )
                    continue
                self._record_success(task, target_files)
                self._commit_task(
                    project,
                    branch_name,
                    base_branch,
                    task,
                    sorted(set(target_files) | edited_files),
                )
                return self._complete_task(
                    project, task, "Task completed (no tests run).", llm_response
                )

        # Strategy escalation: if current strategy failed, try one more attempt
        # with "surgical". Guarded by _depth so escalation can never recurse more
        # than once, even if the strategy-selection logic changes later.
        current_strategy = task.processor_data.get("strategy", "iterative")
        if current_strategy != "surgical" and _depth < 1:
            logger.info(
                f"Escalating strategy from {current_strategy} to surgical for final attempt"
            )
            self._abort_task(project, branch_name, base_branch, snapshot)

            task.processor_data["strategy"] = "surgical"
            task.processor_data["invariants"] = (
                "ESCALATED: Previous strategy failed. Use SURGICAL approach: "
                "make the smallest possible change to fix the immediate error. "
                "Do not refactor or restructure. Minimal, targeted fix only."
            )
            # Recursive single attempt with surgical strategy
            return self.execute(
                task, project, use_git_branch=use_git_branch, _depth=_depth + 1
            )
        elif _depth >= 1:
            logger.warning(
                f"Strategy escalation blocked at depth {_depth}: already exhausted all strategies"
            )

        logger.warning(
            f"Task {task.id} failed after all attempts including escalation. Reverting."
        )
        self._abort_task(project, branch_name, base_branch, snapshot)
        self.scratchpad.record(
            category="pitfall",
            discovery=(
                f"Failed after {max_retries} attempts + escalation: "
                f"{error_logs[:200] if error_logs else 'unknown'}"
            ),
            task_id=task.id,
            files=target_files,
        )
        return self._fail_task(
            project,
            task,
            f"Task failed after {max_retries} attempts + escalation.",
            error_logs,
        )

    # ----------------------------------------------------------------
    # Git branch-per-task operations
    # ----------------------------------------------------------------

    def _is_git_repo(self, project: Project) -> bool:
        return (project.path / ".git").exists()

    # ----------------------------------------------------------------
    # Regression bisection (post-build gate failure)
    # ----------------------------------------------------------------

    def find_task_commit(self, project: Project, task_id: str) -> Optional[str]:
        """SHA of the commit recording a task (message 'task(<id>):'), or None."""
        ok, out = self._git(
            project,
            f"git log --all -n 1 --format=%H --fixed-strings --grep={shlex.quote(f'task({task_id}):')}",
        )
        sha = out.strip().splitlines()[0] if ok and out.strip() else ""
        return sha or None

    def bisect_regression(
        self,
        project: Project,
        task_commits: List,
        test_command: str,
        timeout: int = 180,
    ) -> Optional[str]:
        """Find the earliest task commit whose checkout fails the test command.

        task_commits is [(task_id, sha)] ordered oldest->newest. Returns the
        culprit task_id, or None if no checked-out commit actually fails (so a
        flaky/non-task regression isn't misattributed). Restores HEAD after.
        """
        if not task_commits:
            return None
        # Restore to the BRANCH, not a bare SHA: checking out a SHA detaches
        # HEAD, and a mid-build gate that left HEAD detached would make every
        # subsequent task branch from / merge into a detached head instead of
        # the working branch, silently diverging the build from the branch ref.
        okb, branch = self._git(project, "git rev-parse --abbrev-ref HEAD")
        branch = branch.strip() if okb else ""
        if branch and branch != "HEAD":
            restore = branch
        else:
            ok, head = self._git(project, "git rev-parse HEAD")
            restore = head.strip() if ok else None

        def passes_at(i: int) -> bool:
            self._git(project, f"git checkout {shlex.quote(task_commits[i][1])}")
            success, _ = self._run_command(project, test_command, timeout=timeout)
            return success

        try:
            idx = _bisect_first_failing(len(task_commits), passes_at)
            culprit = None if passes_at(idx) else task_commits[idx][0]
        finally:
            if restore:
                self._git(project, f"git checkout {shlex.quote(restore)}")
        return culprit

    def revert_task_commit(self, project: Project, sha: str) -> bool:
        """Revert a task's commit, leaving an explicit revert commit.

        Aborts cleanly on conflict so a failed revert never leaves the working
        tree in a half-reverted, conflict-marked state that would break later
        suite runs and checkouts.
        """
        ok, _ = self._git(project, f"git revert --no-edit {shlex.quote(sha)}")
        if not ok:
            self._git(project, "git revert --abort")
        return ok

    def _get_current_branch(self, project: Project) -> Optional[str]:
        ok, output = self._git(project, "git rev-parse --abbrev-ref HEAD")
        return output.strip() if ok else None

    def _create_task_branch(self, project: Project, branch_name: str) -> bool:
        ok, _ = self._git(project, f"git checkout -b {shlex.quote(branch_name)}")
        if ok:
            logger.info(f"Created task branch: {branch_name}")
        return ok

    def _commit_task(
        self,
        project: Project,
        branch_name: Optional[str],
        base_branch: Optional[str],
        task: Task,
        files: Optional[List[str]] = None,
    ):
        """Commit the task's own changes and merge the task branch back to base.

        Stages only the named files, never `git add -A`: a blanket add sweeps
        unrelated untracked files (other uncommitted user work, or files carried
        onto the task branch) into the commit, which then get destroyed if the
        task is later reverted. With no files, commit empty rather than sweep.
        """
        msg = f"task({task.id}): {task.title or task.description[:50]}"
        if files:
            quoted = " ".join(shlex.quote(f) for f in files)
            self._git(project, f"git add -- {quoted}")
        self._git(project, f"git commit -m {shlex.quote(msg)} --allow-empty")

        if branch_name and base_branch:
            # Drop tracked spillover before switching branches: a project-wide
            # formatter (e.g. `ruff format .`) reformats files outside the task,
            # which aren't committed and would otherwise be carried across the
            # checkout and accumulate as a permanently dirty tree.
            self._git(project, "git checkout -- .")
            self._git(project, f"git checkout {shlex.quote(base_branch)}")
            ok, output = self._git(
                project,
                f"git merge --no-ff {shlex.quote(branch_name)} -m {shlex.quote(f'Merge {branch_name}')}",
            )
            if ok:
                self._git(project, f"git branch -d {shlex.quote(branch_name)}")
                logger.info(f"Merged and cleaned up branch: {branch_name}")
            else:
                logger.error(f"Merge failed for {branch_name}: {output}")

    def _abort_task(
        self,
        project: Project,
        branch_name: Optional[str],
        base_branch: Optional[str],
        snapshot: Optional[Dict],
    ):
        """Revert a failed task: delete branch or restore file snapshots."""
        if branch_name and base_branch:
            self._git(project, "git reset --hard")
            self._git(project, f"git checkout {shlex.quote(base_branch)}")
            self._git(project, f"git branch -D {shlex.quote(branch_name)}")
            logger.info(f"Aborted and deleted branch: {branch_name}")
        elif snapshot is not None:
            self._revert_files(project, snapshot)

    def _git(self, project: Project, command: str) -> tuple:
        ok, output = _run_cmd(command, project.path, None, timeout=30)
        return ok, output.strip()

    # ----------------------------------------------------------------
    # Command execution and file operations
    # ----------------------------------------------------------------

    def _run_command(self, project: Project, command: str, timeout: int = 120) -> tuple:
        logger.info(f"Running: {command} (timeout={timeout}s)")
        activation = (
            project.env_manager.activate_command() if project.env_manager else None
        )
        return _run_cmd(command, project.path, activation, timeout)

    def _snapshot_files(
        self, project: Project, files: List[str]
    ) -> Dict[str, Optional[str]]:
        snapshot = {}
        for file_path in files:
            full_path = project.path / file_path
            if full_path.exists():
                try:
                    snapshot[file_path] = full_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    snapshot[file_path] = None
            else:
                snapshot[file_path] = None
        return snapshot

    def _revert_files(
        self, project: Project, snapshot: Dict[str, Optional[str]]
    ) -> None:
        for file_path, content in snapshot.items():
            full_path = project.path / file_path
            if content is None:
                if full_path.exists():
                    full_path.unlink()
            else:
                write_file(full_path, content)

    def _record_success(self, task: Task, files: List[str]) -> None:
        self.scratchpad.record(
            category="pattern",
            discovery=f"Task {task.id} completed successfully",
            task_id=task.id,
            files=files,
            tags=[task.category],
        )

    def _get_processor_config(self, project: Project) -> dict:
        processors = project.config.get("task_processors", [])
        for p in processors:
            if p.get("type") == "markdown_planner":
                return p.get("settings", {})
        return {}

    def _get_code_context(
        self,
        project: Project,
        target_files: List[str],
        context_files: List[str],
        max_lines: int = 500,
    ) -> str:
        context = ""
        if target_files:
            context += "### Files to Modify/Create\n"
            for file_path in target_files:
                context += self._read_file_for_context(
                    project.path / file_path, file_path, max_lines
                )
        if context_files:
            context += "\n### Reference/Context Files (Read-Only)\n"
            for file_path in context_files:
                context += self._read_file_for_context(
                    project.path / file_path, file_path, max_lines
                )
        return context

    def _read_file_for_context(
        self, full_path: Path, rel_path: str, max_lines: int
    ) -> str:
        if not full_path.exists():
            return f"\n# File: {rel_path} (Does not exist yet)\n"
        try:
            content = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return f"\n# File: {rel_path} (binary or unreadable, skipped)\n"
        lines = content.splitlines()
        if len(lines) > max_lines:
            content = (
                "\n".join(lines[:max_lines])
                + f"\n... ({len(lines)} lines total, truncated)"
            )
        return f"\n# File: {rel_path}\n{content}\n"

    def _invoke_llm(self, project: Project, prompt: str, system_prompt: str):
        """Call the LLM, optionally streaming with early abort (config opt-in).

        Returns (text, aborted). Streaming is enabled via llm.streaming=true.
        """
        if get_setting(project.config, "llm", "streaming"):
            resp = project.llm_client.generate_stream(
                prompt, system_prompt, abort_check=code_gen_abort_check
            )
            return resp.content, resp.finish_reason == "aborted"
        return project.llm_client.generate_code(prompt, system_prompt), False

    def _resolve_model(
        self, project: Project, task: Task, strategy: str
    ) -> Optional[str]:
        """Pick a model for this task from llm.routing/llm.models config.

        Routes by task complexity first, then strategy. Returns None when no
        routing is configured so the client keeps its default model.
        """
        routing = get_setting(project.config, "llm", "routing")
        models = get_setting(project.config, "llm", "models")
        if not routing or not models:
            return None
        tier = routing.get(task.complexity) or routing.get(strategy)
        if not tier:
            return None
        return models.get(tier) or models.get("default")

    def _build_error_context(
        self,
        prior_errors: List[str],
        attempt: int,
        output: str,
        classified: str,
        attributed_error: str,
    ) -> str:
        """Combine the current error with a summary of prior failed attempts.

        Surfacing what already failed (and how it was classified) stops the LLM
        from re-submitting the same broken fix across retries.
        """
        prior_errors.append(f"Attempt {attempt + 1}: {classify_error(output)}")
        history = ""
        if len(prior_errors) > 1:
            past = "\n".join(f"- {e}" for e in prior_errors[:-1])
            history = (
                "### Previous Attempt Failures (a different approach is required)\n"
                f"{past}\n\n"
            )
        return f"{history}{classified}\n\n{attributed_error}"

    def _verify_acceptance(
        self,
        project: Project,
        task: Task,
        verify_acceptance: bool,
        llm_acceptance_judge: bool,
        timeout: int,
    ) -> Tuple[bool, str]:
        """Verify the task's acceptance_criteria after build/test gates pass.

        Returns (passed, output). Deterministic primary path: extract an
        explicit runnable command from acceptance_criteria and run it; a
        non-zero exit fails acceptance. When acceptance_criteria is empty, the
        gate is disabled, or no command can be confidently extracted, this is a
        no-op that passes (behaviour identical to before this gate) unless the
        default-off ``orchestrator.llm_acceptance_judge`` flag is set, in which
        case an LLM judge is consulted. Never blocks on un-parseable free text.
        """
        if not verify_acceptance:
            return True, ""
        criteria = (task.acceptance_criteria or "").strip()
        if not criteria:
            return True, ""
        command = _extract_acceptance_command(criteria)
        if command:
            logger.info(f"Verifying acceptance criteria via command: {command}")
            success, output = self._run_command(project, command, timeout=timeout)
            if success:
                logger.info("Acceptance criteria command passed.")
                return True, ""
            return False, (
                f"Acceptance criterion not met: `{criteria}`\n"
                f"Ran: {command}\n"
                f"Command exited non-zero:\n{output}"
            )
        if llm_acceptance_judge and self._judge_affordable(project):
            return self._llm_acceptance_judge(project, task, criteria)
        return True, ""

    def _judge_affordable(self, project: Project) -> bool:
        """True while enough budget remains to spend on the LLM acceptance judge.

        Cost control for the (now default-on) judge: once the run has burned
        through all but ``JUDGE_MIN_BUDGET_FRACTION`` of the budget, stop paying
        for free-text judging and let those criteria pass, reserving the last
        funds for actually fixing code. Fail-open when budget can't be read.
        """
        client = project.llm_client
        remaining = getattr(client, "budget_remaining", None)
        total = getattr(client, "_budget", None)
        if not isinstance(remaining, (int, float)) or not isinstance(
            total, (int, float)
        ):
            return True
        if total <= 0:
            return True
        return remaining > total * JUDGE_MIN_BUDGET_FRACTION

    def _llm_acceptance_judge(
        self, project: Project, task: Task, criteria: str
    ) -> Tuple[bool, str]:
        """Default-off LLM fallback judging free-text acceptance criteria.

        Only reached when ``orchestrator.llm_acceptance_judge`` is true and no
        runnable command could be extracted. A failure to reach a confident
        verdict passes (fail-open) so an unreliable judge never blocks a task.
        """
        try:
            prompt = (
                "A code task has just passed its build and test gates. Judge "
                "ONLY whether the stated acceptance criterion is satisfied by "
                "the task's implementation. Reply with PASS or FAIL on the "
                "first line, then a brief reason.\n\n"
                f"Task: {task.description}\n"
                f"Acceptance criterion: {criteria}\n"
            )
            verdict = project.llm_client.generate_code(prompt, "")
        except Exception as e:
            logger.warning(f"LLM acceptance judge failed, passing open: {e}")
            return True, ""
        first = (verdict or "").strip().splitlines()
        if first and first[0].strip().upper().startswith("FAIL"):
            return False, (
                f"Acceptance criterion not met (LLM judge): `{criteria}`\n{verdict}"
            )
        return True, ""

    def _build_acceptance_error_context(
        self,
        prior_errors: List[str],
        attempt: int,
        task: Task,
        classified: str,
    ) -> str:
        """Format an acceptance failure into the same retry context as other gates.

        Makes the unmet criterion explicit so the next attempt targets it rather
        than re-submitting a change that only satisfies the build/test gates.
        """
        prior_errors.append(f"Attempt {attempt + 1}: acceptance criteria not met")
        history = ""
        if len(prior_errors) > 1:
            past = "\n".join(f"- {e}" for e in prior_errors[:-1])
            history = (
                "### Previous Attempt Failures (a different approach is required)\n"
                f"{past}\n\n"
            )
        return (
            f"{history}### Acceptance criterion not met\n"
            f"The build and tests passed, but the task's acceptance criterion "
            f"was not satisfied:\n{task.acceptance_criteria}\n\n{classified}"
        )

    def _validate_edit_paths(
        self, project: Project, task: Task, edits: Dict[str, str]
    ) -> Dict[str, str]:
        """Reject hallucinated or out-of-scope edits before they touch disk.

        Drops edits that escape the project root (absolute paths, ``..``
        traversal) or that would create empty files, and warns when the LLM
        touches files outside the task's declared scope. This is what prevents
        a misrouted edit from clobbering files outside the project.
        """
        project_root = project.path.resolve()
        expected = set(task.files_to_modify + task.files_to_create)
        golden_paths = get_setting(project.config, "orchestrator", "golden_paths")
        valid: Dict[str, str] = {}
        for path, content in edits.items():
            if ".." in Path(path).parts or Path(path).is_absolute():
                logger.error(f"Rejected edit with unsafe path: {path}")
                continue
            if _is_golden_path(path, golden_paths):
                logger.error(f"Rejected edit to protected golden file: {path}")
                continue
            full = (project.path / path).resolve()
            try:
                inside = full.is_relative_to(project_root)
            except ValueError:
                inside = False
            if not inside:
                logger.error(f"Rejected edit to path outside project root: {path}")
                continue
            if not content.strip():
                logger.warning(f"Rejected empty-content edit: {path}")
                continue
            if expected and path not in expected:
                logger.warning(f"LLM modified file outside task scope: {path}")
            valid[path] = content
        return valid

    def _detect_test_tampering(
        self, project: Project, edits: Dict[str, str]
    ) -> Optional[str]:
        """Reject edits that weaken existing test files (deterministic gate).

        For each edited TEST file that already exists on disk, compares the
        current (pre-edit) content against the proposed content. An edit that
        reduces tests/assertions or adds skip markers is tampering: the test
        gate must not be satisfied by gutting the gate. New test files and
        purely additive edits pass. Set ``orchestrator.allow_test_edits`` to
        skip the check (escape hatch); it defaults to off (check enforced).
        Must be called BEFORE ``_apply_edits`` so the on-disk content still
        reflects the pre-edit state.
        """
        if get_setting(project.config, "orchestrator", "allow_test_edits"):
            return None
        reasons = []
        for file_path, content in edits.items():
            if not _is_test_file(file_path):
                continue
            full_path = project.path / file_path
            if not full_path.exists():
                continue
            try:
                before = full_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            # Python uses precise assertion-survival diagnosis (resistant to
            # rename/move/split/merge); other languages fall back to the
            # cross-language regex totals, which can't parse structure.
            if file_path.endswith((".py", ".pyi")):
                reason = _diagnose_py_tampering(before, content)
            else:
                reason = _diagnose_tampering(before, content)
            if reason:
                reasons.append(f"{file_path} ({reason})")
        return "; ".join(reasons) if reasons else None

    def _apply_edits(self, project: Project, edits: Dict[str, str]):
        for file_path, content in edits.items():
            full_path = project.path / file_path
            write_file(full_path, content)

    def _run_formatters(self, project: Project, files):
        # Per-file formatters substitute {path}; project-wide formatters run
        # once. Pass each file only to formatters whose command templates use
        # {path}; otherwise invoke the formatter a single time.
        file_list = list(files)
        for tool_name, tool in project.tool_manager.tools.items():
            if getattr(tool, "type", None) != "formatter":
                continue
            template = (
                getattr(tool, "config", {}).get("command", "")
                if hasattr(tool, "config")
                else ""
            )
            if "{path}" in template:
                for file_path in file_list:
                    tool.execute(project, file_path=file_path)
            else:
                tool.execute(project)

    def _complete_task(
        self, project: Project, task: Task, msg: str, logs: str
    ) -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "completed")
        return ExecutionResult(status="completed", message=msg, logs=logs)

    def _fail_task(
        self, project: Project, task: Task, msg: str, logs: str = ""
    ) -> ExecutionResult:
        project.task_manager.update_task_status(task.id, "failed")
        return ExecutionResult(status="failed", message=msg, logs=logs)
