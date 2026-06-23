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
    "PRIVATE KEY",
    "BEGIN RSA",
    "BEGIN EC",
    "BEGIN DSA",
    "sk-",
    "ghp_",
    "gho_",
    "AKIA",
)

# Low-signal credential keys. These appear constantly in ordinary source
# (struct fields, function params, config keys), so they are only flagged when
# assigned a concrete quoted literal, not a variable/env reference.
ASSIGNMENT_SECRET_KEYS = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_key",
    "token",
)

# Extensions to skip during file scanning
SKIP_DIRS = frozenset(
    {
        ".venv",
        "venv",
        ".git",
        "node_modules",
        "__pycache__",
        "target",
        "build",
        "dist",
        ".tox",
        ".mypy_cache",
        ".eggs",
    }
)

CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".sh",
    }
)


class GateKeeper:
    """Implements the gate sequence for project validation.

    Gates:
      G1: Build compiles
      G2: Lint passes
      G3: Tests pass
      G3.5: Golden suite (model-blind, immutable; if configured)
      G4: Type check (if available)
      G5: Completeness scan (no banned markers in source)
      G6: Secrets scan (no leaked credentials)
      G9: Diff hygiene (no debug artifacts in staged changes)
    """

    def __init__(
        self,
        project_path: Path,
        env_activate: Optional[str] = None,
        build_timeout: int = 180,
        test_timeout: int = 180,
        lint_timeout: Optional[int] = None,
        lsp_diagnostics: bool = False,
        lsp_language: Optional[str] = None,
        lsp_timeout: int = 30,
    ):
        self.project_path = Path(project_path)
        self.env_activate = env_activate
        # Honor the project's configured timeouts so a slow compiler or linter
        # isn't falsely failed by the gate the way the analyzer once was.
        self.build_timeout = build_timeout
        self.test_timeout = test_timeout
        self.lint_timeout = lint_timeout if lint_timeout is not None else test_timeout
        self.lsp_diagnostics = lsp_diagnostics
        self.lsp_language = lsp_language
        self.lsp_timeout = lsp_timeout

    def run_gates(
        self, commands: Dict[str, Optional[str]]
    ) -> Tuple[bool, List[str], HealthCheck]:
        """Run the gate sequence. Returns (success, issues, final_health)."""
        issues: List[str] = []
        health = HealthCheck()

        # G1: Build
        build_cmd = commands.get("build_command")
        if build_cmd:
            success, output = _run_cmd(
                build_cmd,
                self.project_path,
                self.env_activate,
                timeout=self.build_timeout,
            )
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
            success, output = _run_cmd(
                lint_cmd,
                self.project_path,
                self.env_activate,
                timeout=self.lint_timeout,
            )
            health.lint_clean = success
            health.lint_output = output
            if not success:
                issues.append("G2: Lint warnings or errors")
        else:
            health.lint_clean = True

        # G3: Tests
        test_cmd = commands.get("test_command")
        if test_cmd:
            success, output = _run_cmd(
                test_cmd,
                self.project_path,
                self.env_activate,
                timeout=self.test_timeout,
            )
            health.tests_pass = success
            health.test_output = output
            if not success:
                issues.append("G3: Tests failed")
                return False, issues, health
        else:
            health.tests_pass = True

        # G3.5: Golden suite. Tests the model never sees and cannot edit, so a
        # gamed visible suite can't hide a regression. Blocking like G1/G3.
        golden_cmd = commands.get("golden_command")
        if golden_cmd:
            success, output = _run_cmd(
                golden_cmd,
                self.project_path,
                self.env_activate,
                timeout=self.test_timeout,
            )
            if not success:
                issues.append("G3.5: Golden suite failed")
                health.tests_pass = False
                health.test_output = output
                return False, issues, health

        # G4: Type check (optional). Blocking like G1/G3: a configured
        # typecheck command that fails short-circuits the gate so broken types
        # can't slip through. When none is configured, skip with no penalty.
        typecheck_cmd = commands.get("typecheck_command")
        if typecheck_cmd:
            success, _output = _run_cmd(
                typecheck_cmd,
                self.project_path,
                self.env_activate,
                timeout=self.test_timeout,
            )
            if not success:
                issues.append("G4: Type check failed")
                return False, issues, health

        # G4.5: LSP semantic diagnostics (optional, off by default). Catches
        # errors a syntax check misses; best-effort and timeout-bounded, so a
        # None result (unsupported/slow/unavailable) is a skip, not a failure.
        if self.lsp_diagnostics and self.lsp_language:
            from my_project_orchestrator.core.lsp import (
                collect_diagnostics,
                find_source_files,
            )

            files = find_source_files(self.project_path, self.lsp_language)
            diags = collect_diagnostics(
                self.project_path, self.lsp_language, files, self.lsp_timeout
            )
            if diags:
                for d in diags[:5]:
                    issues.append(
                        f"G4.5: LSP error {d['file']}:{d['line']}: {d['message'][:80]}"
                    )
                return False, issues, health

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
        """G5: Scan ADDED lines for banned markers (TODO!, FIXME, HACK, etc.).

        Only flags markers this build introduced. In a git repo we look at the
        added lines of the diff (staged + unstaged vs HEAD); a pre-existing
        marker in an untouched region of a modified file is left alone. Outside
        a git repo there is no diff to scope to, so we fall back to scanning the
        whole tree (the previous behavior) rather than skipping the gate.
        """
        added = self._iter_diff_added_lines()
        if added is None:
            return self._scan_banned_markers_whole_tree()
        found = set()
        for _path, line in added:
            for marker in BANNED_MARKERS:
                if marker in line:
                    found.add(marker)
        return sorted(found)

    def _scan_banned_markers_whole_tree(self) -> List[str]:
        """Non-git fallback for G5: scan every source file."""
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
        """G6: Scan ADDED lines for patterns that look like leaked secrets.

        Like G5, this is scoped to the diff in a git repo so only secrets this
        build introduced are flagged, and falls back to a whole-tree scan when
        not in a git repo. Returns the changed file paths that contain a secret.
        """
        added = self._iter_diff_added_lines()
        if added is None:
            return self._scan_secrets_whole_tree()
        # Group added lines per file, then reuse the existing content check so
        # the multi-line / per-line secret detection stays identical.
        by_file: Dict[str, List[str]] = {}
        for path, line in added:
            by_file.setdefault(path, []).append(line)
        flagged = []
        for path, lines in by_file.items():
            if self._content_has_secret("\n".join(lines)):
                flagged.append(path)
        return flagged

    def _scan_secrets_whole_tree(self) -> List[str]:
        """Non-git fallback for G6: scan every source file."""
        flagged = []
        for path in self._iter_source_files():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if self._content_has_secret(content):
                flagged.append(str(path.relative_to(self.project_path)))
        return flagged

    def _iter_diff_added_lines(self) -> Optional[List[Tuple[str, str]]]:
        """Return ``(file_path, added_line)`` pairs from the git diff.

        Covers both staged and unstaged changes against HEAD (the union of
        ``git diff --cached`` and ``git diff``), so any line this build added is
        seen regardless of whether it was staged yet. Returns ``None`` when the
        project is not a git repo, signalling callers to fall back to a
        whole-tree scan. Only added lines (``+``) of code-extension files are
        included; the diff header lines (``+++``) are skipped.
        """
        try:
            check = subprocess.run(
                "git rev-parse --is-inside-work-tree",
                shell=True,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return None
        if check.returncode != 0 or check.stdout.strip() != "true":
            return None
        added: List[Tuple[str, str]] = []
        for cmd in (
            "git diff --cached --diff-filter=ACMR -U0",
            "git diff --diff-filter=ACMR -U0",
        ):
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception:
                continue
            current_file = ""
            for line in proc.stdout.splitlines():
                if line.startswith("+++ b/"):
                    current_file = line[len("+++ b/") :]
                    continue
                if line.startswith("+++") or not line.startswith("+"):
                    continue
                if current_file and Path(current_file).suffix not in CODE_EXTENSIONS:
                    continue
                added.append((current_file, line[1:]))
        return added

    @staticmethod
    def _content_has_secret(content: str) -> bool:
        for pattern in SECRET_PATTERNS:
            if pattern in content:
                return True
        for line in content.splitlines():
            if GateKeeper._is_secret_assignment(line):
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
                if (
                    len(literal) >= 6
                    and not literal.startswith("${")
                    and "env" not in low
                ):
                    return True
        return False

    def _check_diff_hygiene(self) -> List[str]:
        """G9: Check staged/unstaged changes for debug artifacts."""
        issues = []
        try:
            proc = subprocess.run(
                "git diff --cached --diff-filter=ACMR -U0",
                shell=True,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            diff_text = proc.stdout
        except Exception:
            return issues

        debug_markers = (
            "console.log(",
            "print(",
            "println!",
            "dbg!(",
            "debugger",
            "binding.pry",
            "import pdb",
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
