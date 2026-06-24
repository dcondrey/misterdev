import ast
import re
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path

from my_project_orchestrator.core.assessment import HealthCheck
from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


# ----------------------------------------------------------------
# Build/Test/Lint Validation (Phase 5 from /build skill)
# ----------------------------------------------------------------


class ValidationResult:
    def __init__(self):
        self.build_ok: bool = False
        self.build_output: str = ""
        self.tests_ok: bool = False
        self.test_output: str = ""
        self.lint_ok: bool = False
        self.lint_output: str = ""
        self.diff_stats: str = ""
        self.issues: list[str] = []
        # Whether each gate actually executed a command. A gate that did not
        # run must not be reported as "OK" (that hides untested code).
        self.build_ran: bool = True
        self.tests_ran: bool = True
        self.lint_ran: bool = True

    @property
    def passed(self) -> bool:
        return self.build_ok and self.tests_ok and not self.issues

    def _status(self, ran: bool, ok: bool, warn: bool = False) -> str:
        if not ran:
            return "SKIP"
        if ok:
            return "OK"
        return "WARN" if warn else "FAIL"

    def summary(self) -> str:
        parts = [
            f"build={self._status(self.build_ran, self.build_ok)}",
            f"tests={self._status(self.tests_ran, self.tests_ok)}",
            f"lint={self._status(self.lint_ran, self.lint_ok, warn=True)}",
        ]
        if self.issues:
            parts.append(f"issues={len(self.issues)}")
        return " | ".join(parts)


def _run_cmd(
    cmd: str,
    cwd: Path,
    env_activate: Optional[str] = None,
    timeout: int = 180,
    runner: Optional[Callable[[str, int], Tuple[bool, str]]] = None,
    policy: Optional[object] = None,
    audit: Optional[object] = None,
) -> Tuple[bool, str]:
    # Governance gate (opt-in): when a policy is supplied AND it refuses the
    # command, return a refusal without executing. Both args default to None, so
    # with no policy/audit this is byte-identical to the prior behavior. The
    # classifier sees the raw command, before the host activation prefix.
    if policy is not None:
        try:
            decision = policy.authorize(cmd)
        except Exception:  # a policy bug must never break the command seam
            decision = None
        if decision is not None and not decision.allowed:
            msg = f"Command refused by governance: {decision.reason}"
            logger.warning(f"{msg}: {cmd}")
            return False, msg
    # When a runner is supplied (e.g. a container engine), the command executes
    # through it instead of the local subprocess. Activation prefixes are
    # host-venv concepts, so they are only applied to local execution.
    if runner is not None:
        ok, output = runner(cmd, timeout)
        _audit_command(audit, cmd, ok, cwd)
        return ok, output
    if env_activate:
        full_cmd = f"{env_activate} && {cmd}"
    else:
        full_cmd = cmd
    try:
        proc = subprocess.run(
            full_cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout
        if proc.stderr:
            output += "\n" + proc.stderr
        ok = proc.returncode == 0
        _audit_command(audit, cmd, ok, cwd)
        return ok, output
    except subprocess.TimeoutExpired:
        _audit_command(audit, cmd, False, cwd)
        return False, f"Command timed out after {timeout}s: {full_cmd}"
    except Exception as e:
        _audit_command(audit, cmd, False, cwd)
        return False, f"Command failed: {e}"


def _audit_command(audit: Optional[object], cmd: str, ok: bool, cwd: Path) -> None:
    """Record a command event if an audit trail is wired. Never raises."""
    if audit is None:
        return
    try:
        audit.record_command(cmd, ok=ok, cwd=str(cwd))
    except Exception as e:  # audit is observability; it must never break execution
        # Swallow so a broken audit sink can't fail a command, but leave a trace
        # so silently-vanishing audit records are diagnosable.
        logger.debug(f"Audit record_command failed (ignored): {e}")


def run_validation(
    project_path: Path,
    build_command: Optional[str],
    test_command: Optional[str],
    lint_command: Optional[str],
    env_activate: Optional[str] = None,
    timeout: int = 180,
) -> ValidationResult:
    """Run full validation suite (Phase 5a-5d from /build)."""
    result = ValidationResult()

    if build_command:
        logger.info(f"Validation: running build command: {build_command}")
        result.build_ok, result.build_output = _run_cmd(
            build_command,
            project_path,
            env_activate,
            timeout,
        )
        if not result.build_ok:
            result.issues.append("Build failed during validation")
    else:
        result.build_ok = True

    if test_command:
        logger.info(f"Validation: running test command: {test_command}")
        result.tests_ok, result.test_output = _run_cmd(
            test_command,
            project_path,
            env_activate,
            timeout,
        )
        if not result.tests_ok:
            result.issues.append("Tests failed during validation")
    else:
        result.tests_ok = True

    if lint_command:
        logger.info(f"Validation: running lint command: {lint_command}")
        result.lint_ok, result.lint_output = _run_cmd(
            lint_command,
            project_path,
            env_activate,
            timeout,
        )
        if not result.lint_ok:
            result.issues.append("Lint warnings found during validation")
    else:
        result.lint_ok = True

    try:
        proc = subprocess.run(
            "git diff --stat",
            shell=True,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        result.diff_stats = proc.stdout.strip()
    except Exception:
        result.diff_stats = "(unable to get diff stats)"

    return result


def run_health_check(
    project_path: Path,
    build_command: Optional[str],
    test_command: Optional[str],
    lint_command: Optional[str],
    env_activate: Optional[str] = None,
    timeout: int = 120,
    build_timeout: Optional[int] = None,
    test_timeout: Optional[int] = None,
    lint_timeout: Optional[int] = None,
) -> HealthCheck:
    """Run health check and return structured result for assessment.

    Per-command timeouts override the shared ``timeout`` so a slow compiler or
    linter (e.g. cargo with ort/tokenizers, clippy over a workspace) isn't
    falsely reported as failing when it merely exceeds the default.
    """
    health = HealthCheck()
    bt = build_timeout if build_timeout is not None else timeout
    tt = test_timeout if test_timeout is not None else timeout
    lt = lint_timeout if lint_timeout is not None else tt

    if build_command:
        health.builds, health.build_output = _run_cmd(
            build_command,
            project_path,
            env_activate,
            bt,
        )

    if test_command:
        health.tests_pass, health.test_output = _run_cmd(
            test_command,
            project_path,
            env_activate,
            tt,
        )
        health.test_count, health.test_failures = _parse_test_counts(health.test_output)

    if lint_command:
        health.lint_clean, health.lint_output = _run_cmd(
            lint_command,
            project_path,
            env_activate,
            lt,
        )

    return health


def _parse_test_counts(output: str) -> Tuple[int, int]:
    """Extract (total, failures) from common test-runner output.

    Without this, test_count stays 0 and assessment.summary() renders
    'tests=none' even on a fully-passing suite, which misleads the analyzers
    and recommender into believing no tests exist.
    """
    passed = failed = 0
    # pytest: "N passed", "N failed", "N error(s)", "N skipped"
    for kind, target in (
        ("passed", "p"),
        ("failed", "f"),
        ("error", "f"),
        ("errors", "f"),
    ):
        m = re.search(rf"(\d+) {kind}\b", output)
        if m:
            if target == "p":
                passed += int(m.group(1))
            else:
                failed += int(m.group(1))
    if passed or failed:
        return passed + failed, failed
    # cargo: "test result: ok. N passed; M failed"
    m = re.search(r"test result:.*?(\d+) passed; (\d+) failed", output)
    if m:
        p, f = int(m.group(1)), int(m.group(2))
        return p + f, f
    # swift test / XCTest: "Executed N tests, with M failures"
    m = re.search(r"Executed (\d+) tests?, with (\d+) failure", output)
    if m:
        total, f = int(m.group(1)), int(m.group(2))
        return total, f
    # ctest: "X% tests passed, M tests failed out of N"
    m = re.search(r"tests passed,\s*(\d+) tests? failed out of (\d+)", output)
    if m:
        f, total = int(m.group(1)), int(m.group(2))
        return total, f
    # dotnet test (VSTest): "Failed: N, Passed: M, Skipped: K, Total: T"
    fm = re.search(r"Failed:\s*(\d+)", output)
    tm = re.search(r"Total:\s*(\d+)", output)
    if fm and tm:
        return int(tm.group(1)), int(fm.group(1))
    # dotnet test (alt): "Total tests: T. Passed: M. Failed: N."
    m = re.search(r"Total tests:\s*(\d+)\..*?Failed:\s*(\d+)", output, re.DOTALL)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


# ----------------------------------------------------------------
# Code Quality Validators
# ----------------------------------------------------------------


class CodeValidator:
    """Validates code syntax using AST parsing and structural checks."""

    @staticmethod
    def validate_code(
        content: str, language: str = "python"
    ) -> Tuple[bool, Optional[str]]:
        lang = language.lower()
        if lang in ["python", "py"]:
            try:
                ast.parse(content)
                return True, None
            except SyntaxError as e:
                error_msg = f"Syntax error at line {e.lineno}, col {e.offset}: {e.msg}"
                return False, error_msg
        # Shell syntax (command substitution $(), arithmetic (()), here-docs)
        # uses parentheses in ways the delimiter matcher misreports. Skip the
        # bracket check for shell rather than emit false "unclosed delimiter".
        if lang in ["shell", "sh", "bash", "zsh"]:
            return True, None
        # Real parse-based check for languages with a trustworthy grammar. It
        # understands strings/comments (so braces in a literal don't false-trip)
        # and catches actual syntax errors that brace-balancing misses.
        try:
            from my_project_orchestrator.core.topography import check_syntax

            result = check_syntax(content, lang)
        except ImportError:
            result = None
        if result is not None:
            return result
        return CodeValidator._basic_integrity_check(content)

    @staticmethod
    def _basic_integrity_check(content: str) -> Tuple[bool, Optional[str]]:
        delimiters = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for i, char in enumerate(content):
            if char in delimiters:
                stack.append((char, i))
            elif char in delimiters.values():
                if not stack:
                    return False, f"Unmatched closing delimiter '{char}' at index {i}"
                top, _ = stack.pop()
                if delimiters[top] != char:
                    return False, f"Mismatched delimiter '{char}' at index {i}"
        if stack:
            top, idx = stack[0]
            return False, f"Unclosed delimiter '{top}' at index {idx}"
        return True, None


class CertaintyScorer:
    """Heuristic-based LLM certainty analyzer."""

    HEDGES = [
        "maybe",
        "perhaps",
        "might",
        "could be",
        "not sure",
        "i think",
        "possibly",
        "unclear",
        "hard to say",
        "it depends",
        "arguably",
        "i'm not certain",
        "one possibility",
    ]

    ASSERTIONS = [
        "verified",
        "confirmed",
        "optimal",
        "correct",
        "proven",
        "tests pass",
        "all tests",
        "successfully",
        "no issues",
        "the solution is",
        "this fixes",
        "this resolves",
        "implemented",
        "works",
        "complete",
        "done",
    ]

    @staticmethod
    def compute_score(content: str) -> float:
        lower_content = content.lower()
        score = 0.5
        hedge_count = sum(
            1 for hedge in CertaintyScorer.HEDGES if hedge in lower_content
        )
        score -= min(hedge_count * 0.1, 0.4)
        assertion_count = sum(
            1 for a in CertaintyScorer.ASSERTIONS if a in lower_content
        )
        score += min(assertion_count * 0.1, 0.4)
        if "```" in content:
            score += 0.1
        if len(content.split()) < 20:
            score -= 0.1
        return max(0.0, min(1.0, score))


class StallDetector:
    """Detects oscillation or lack of progress in task execution history.

    Uses token-based Jaccard Similarity to detect 'semantic' stalling where
    edits are nearly identical but not byte-perfect matches.
    """

    def __init__(self, history_limit: int = 5, similarity_threshold: float = 0.95):
        self.history_limit = history_limit
        self.similarity_threshold = similarity_threshold
        self._history: List[str] = []

    def push_edit(self, edits: Dict[str, str]) -> float:
        """Records an edit and returns a stall risk score [0.0 - 1.0]."""
        sorted_keys = sorted(edits.keys())
        current_content = " ".join(f"{k} {edits[k]}" for k in sorted_keys)
        current_tokens = _tokenize(current_content)

        if not current_tokens:
            return 0.0

        max_similarity = 0.0
        for prev_content in self._history:
            prev_tokens = _tokenize(prev_content)
            intersection = len(current_tokens & prev_tokens)
            union = len(current_tokens | prev_tokens)
            similarity = intersection / union if union > 0 else 0.0
            max_similarity = max(max_similarity, similarity)

        self._history.append(current_content)
        if len(self._history) > self.history_limit:
            self._history.pop(0)

        if max_similarity > self.similarity_threshold:
            return 0.8 + (
                0.2
                * (max_similarity - self.similarity_threshold)
                / (1.0 - self.similarity_threshold)
            )

        return max_similarity * 0.5


def _tokenize(text: str) -> set:
    """Extract word tokens from text without regex."""
    tokens = set()
    current = []
    for char in text.lower():
        if char.isalnum() or char == "_":
            current.append(char)
        elif current:
            tokens.add("".join(current))
            current = []
    if current:
        tokens.add("".join(current))
    return tokens
