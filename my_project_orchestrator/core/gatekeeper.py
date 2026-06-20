import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from my_project_orchestrator.logging_setup import setup_logger
from my_project_orchestrator.core.assessment import HealthCheck
from my_project_orchestrator.core.validator import _run_cmd

logger = setup_logger(__name__)

# Patterns that indicate incomplete or debug code
BANNED_MARKERS = ("todo!", "FIXME", "HACK", "XXX", "placeholder", "dummy")

# High-signal patterns: a bare substring match is enough to flag a file.
SECRET_PATTERNS = (
    "PRIVATE KEY", "BEGIN RSA", "BEGIN EC", "BEGIN DSA",
    "sk-", "ghp_", "gho_", "AKIA",
)

# Low-signal credential keys. These appear constantly in ordinary source
# (struct fields, function params, config keys), so they are only flagged when
# assigned a concrete quoted literal, not a variable/env reference.
ASSIGNMENT_SECRET_KEYS = (
    "password", "passwd", "secret", "api_key", "apikey", "access_key", "token",
)

# Extensions to skip during file scanning
SKIP_DIRS = frozenset({
    ".venv", "venv", ".git", "node_modules", "__pycache__",
    "target", "build", "dist", ".tox", ".mypy_cache", ".eggs",
})

CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".rb", ".php", ".swift", ".kt", ".sh",
})


class SOTAGateKeeper:
    """Implements the SOTA gate sequence for project validation.

    Gates:
      G1: Build compiles
      G2: Lint passes
      G3: Tests pass
      G4: Type check (if available)
      G5: Completeness scan (no banned markers in source)
      G6: Secrets scan (no leaked credentials)
      G9: Diff hygiene (no debug artifacts in staged changes)
    """

    def __init__(self, project_path: Path, env_activate: Optional[str] = None):
        self.project_path = project_path
        self.env_activate = env_activate

    def run_gates(self, commands: Dict[str, Optional[str]]) -> Tuple[bool, List[str], HealthCheck]:
        """Run the gate sequence. Returns (success, issues, final_health)."""
        issues: List[str] = []
        health = HealthCheck()

        # G1: Build
        build_cmd = commands.get("build_command")
        if build_cmd:
            success, output = _run_cmd(build_cmd, self.project_path, self.env_activate, timeout=180)
            health.builds = success
            health.build_output = output
            if not success:
                issues.append("G1: Build failed")
                return False, issues, health
        else:
            health.builds = True

        # G2: Lint
        lint_cmd = commands.get("lint_command")
        if lint_cmd:
            success, output = _run_cmd(lint_cmd, self.project_path, self.env_activate, timeout=120)
            health.lint_clean = success
            health.lint_output = output
            if not success:
                issues.append("G2: Lint warnings or errors")
        else:
            health.lint_clean = True

        # G3: Tests
        test_cmd = commands.get("test_command")
        if test_cmd:
            success, output = _run_cmd(test_cmd, self.project_path, self.env_activate, timeout=180)
            health.tests_pass = success
            health.test_output = output
            if not success:
                issues.append("G3: Tests failed")
                return False, issues, health
        else:
            health.tests_pass = True

        # G4: Type check (optional)
        typecheck_cmd = commands.get("typecheck_command")
        if typecheck_cmd:
            success, output = _run_cmd(typecheck_cmd, self.project_path, self.env_activate, timeout=120)
            if not success:
                issues.append("G4: Type check failed")

        # G5: Completeness scan
        banned_found = self._scan_banned_markers()
        if banned_found:
            issues.append(f"G5: Banned markers found: {', '.join(banned_found)}")

        # G6: Secrets scan
        secrets_found = self._scan_secrets()
        if secrets_found:
            issues.append(f"G6: Possible secrets in {len(secrets_found)} file(s)")
            for path in secrets_found[:5]:
                logger.warning(f"G6: Possible secret in {path}")

        # G9: Diff hygiene
        diff_issues = self._check_diff_hygiene()
        if diff_issues:
            issues.extend(diff_issues)

        return len(issues) == 0, issues, health

    def _scan_banned_markers(self) -> List[str]:
        """G5: Scan source files for banned markers (TODO!, FIXME, HACK, etc.)."""
        found = set()
        for path in self._iter_source_files():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                for marker in BANNED_MARKERS:
                    if marker in content:
                        found.add(marker)
            except OSError:
                continue
        return sorted(found)

    def _scan_secrets(self) -> List[str]:
        """G6: Scan for patterns that look like leaked secrets."""
        flagged = []
        for path in self._iter_source_files():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if self._content_has_secret(content):
                flagged.append(str(path.relative_to(self.project_path)))
        return flagged

    @staticmethod
    def _content_has_secret(content: str) -> bool:
        for pattern in SECRET_PATTERNS:
            if pattern in content:
                return True
        for line in content.splitlines():
            if SOTAGateKeeper._is_secret_assignment(line):
                return True
        return False

    @staticmethod
    def _is_secret_assignment(line: str) -> bool:
        """True only for a credential key assigned a concrete quoted literal.

        Skips variable/env references (``token = get_token()``,
        ``api_key = os.environ[...]``) and type annotations (``token: String``)
        that would otherwise produce false positives on ordinary source.
        """
        if "=" not in line:
            return False
        key_part, _, value = line.partition("=")
        if not any(k in key_part.lower() for k in ASSIGNMENT_SECRET_KEYS):
            return False
        value = value.strip()
        for quote in ('"', "'"):
            if value.startswith(quote):
                literal = value[1:].split(quote, 1)[0]
                low = literal.lower()
                if len(literal) >= 6 and not literal.startswith("${") and "env" not in low:
                    return True
        return False

    def _check_diff_hygiene(self) -> List[str]:
        """G9: Check staged/unstaged changes for debug artifacts."""
        issues = []
        try:
            proc = subprocess.run(
                "git diff --cached --diff-filter=ACMR -U0",
                shell=True, cwd=self.project_path,
                capture_output=True, text=True, timeout=30,
            )
            diff_text = proc.stdout
        except Exception:
            return issues

        debug_markers = (
            "console.log(", "print(", "println!", "dbg!(",
            "debugger", "binding.pry", "import pdb",
        )
        for line in diff_text.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            added = line[1:]
            for marker in debug_markers:
                if marker in added:
                    issues.append(f"G9: Debug artifact in staged change: {marker}")
                    break

        return issues

    def _iter_source_files(self):
        """Yield source code file paths, skipping excluded directories."""
        stack = [self.project_path]
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(directory.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.name in SKIP_DIRS:
                    continue
                if entry.is_dir():
                    stack.append(entry)
                elif entry.suffix in CODE_EXTENSIONS:
                    yield entry
