from pathlib import Path
from typing import Callable, List, Optional, Dict, Tuple, TYPE_CHECKING

from misterdev.logging_setup import setup_logger
from misterdev.core.planning.assessment import HealthCheck
from misterdev.core.gitcmd import run_git
from misterdev.core.verification.validator import _run_cmd

from .constants import (
    BANNED_MARKERS,
    SECRET_PATTERNS,
    SECRET_REGEXES,
    ASSIGNMENT_SECRET_KEYS,
    ENV_REFERENCE_MARKERS,
    SKIP_DIRS,
    CODE_EXTENSIONS,
    SECRET_SCAN_EXTENSIONS,
    SECRET_SCAN_FILENAMES,
    ENV_LITERAL_EXTENSIONS,
)
from .helpers import _path_in_scope, _read_capped

if TYPE_CHECKING:
    from misterdev.core.execution.container import ContainerEngine

logger = setup_logger(__name__)


class GateKeeper:
    """Implements the gate sequence for project validation.

    Gates:
      G1: Build compiles
      G2: Lint passes
      G3: Tests pass
      G3.5: Golden suite (model-blind, immutable; if configured)
      G3.6: Mutation-score gate (optional; suite must kill injected faults)
      G4: Type check (if available)
      G4.5: LSP semantic diagnostics (optional)
      G4.6: Runtime smoke (optional)
      G4.7: Web verification, headless browser (optional)
      G4.8: Vision verification, VLM judgment (optional)
      G5: Completeness scan (no banned markers in source)
      G6: Secrets scan (no leaked credentials, incl. config/env files)
      G9: Diff hygiene (no debug artifacts in staged + unstaged changes)
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
        container: Optional["ContainerEngine"] = None,
        mutation_gate: bool = False,
        mutation_config: Optional[Dict] = None,
        runtime_smoke: bool = False,
        runtime_config: Optional[Dict] = None,
        web_verify: bool = False,
        vision_verify: bool = False,
        vision_client=None,
    ):
        self.project_path = Path(project_path)
        self.env_activate = env_activate
        # Optional container engine: when present and usable, gate commands
        # (build/lint/test/golden/typecheck) run inside it instead of locally.
        # Git stays host-side (never routed here). None -> run locally as before.
        self.container = container if container and container.is_available() else None
        # Optional mutation-score gate (off by default). mutation_config carries
        # the top-level `mutation` mapping (command/min_score/timeout). Like the
        # runtime gates it is timeout-bounded and SKIP-on-unparseable so it can
        # never block a build except on a parsed score below the configured floor.
        self.mutation_gate = mutation_gate
        self.mutation_config = mutation_config or {}
        # Honor the project's configured timeouts so a slow compiler or linter
        # isn't falsely failed by the gate the way the analyzer once was.
        self.build_timeout = build_timeout
        self.test_timeout = test_timeout
        self.lint_timeout = lint_timeout if lint_timeout is not None else test_timeout
        self.lsp_diagnostics = lsp_diagnostics
        self.lsp_language = lsp_language
        self.lsp_timeout = lsp_timeout
        # Optional runtime smoke gate (off by default). runtime_config carries
        # the top-level `runtime` mapping; the smoke spec lives under its
        # `smoke` key. Timeout-bounded so it can never block the build.
        self.runtime_smoke = runtime_smoke
        self.runtime_config = runtime_config or {}
        # Optional web (headless-browser) and vision (VLM) verification gates,
        # both off by default. Their specs live under the top-level `runtime`
        # mapping (`runtime.web`, `runtime.vision`). Like the smoke gate they are
        # daemon-threaded and timeout-bounded so they can never block the build;
        # a SKIP (no/incomplete config, missing dep, timeout) never fails.
        self.web_verify = web_verify
        self.vision_verify = vision_verify
        self.vision_client = vision_client

    @property
    def _runner(self) -> Optional[Callable[[str, int], Tuple[bool, str]]]:
        """The command runner for gate commands: the container's ``run`` when a
        usable engine is attached, else ``None`` (local execution)."""
        return self.container.run if self.container else None

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
                runner=self._runner,
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
                runner=self._runner,
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
                runner=self._runner,
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
                runner=self._runner,
            )
            if not success:
                issues.append("G3.5: Golden suite failed")
                health.tests_pass = False
                # Golden tests are model-blind by design; their failure output
                # must stay blind too. Otherwise the convergence fix-spec
                # (_build_fix_spec) would feed golden assertions/values back to
                # the model, letting it target the very tests it can't see. Keep
                # the detail in the log (human-debuggable) and expose only a
                # generic signal in the model-facing health output.
                logger.warning(
                    "G3.5 golden suite failed; output withheld from the model "
                    "(model-blind). Detail:\n" + (output or "")[:2000]
                )
                health.test_output = (
                    "Golden suite failed (details withheld — the golden suite is "
                    "model-blind; see orchestrator logs)."
                )
                return False, issues, health

        # G3.6: Mutation-score gate (optional, off by default). Runs the project's
        # configured mutation command and asserts the parsed score meets a floor —
        # catching a suite that passes but kills few injected faults. Best-effort
        # and timeout-bounded: a SKIP (no config, unparseable score, timeout)
        # never fails; only a parsed score below the floor is a RED that blocks.
        if self.mutation_gate:
            from misterdev.core.verification.mutation_gate import (
                run_mutation_gate,
            )

            mutation = run_mutation_gate(
                self.project_path, self.mutation_config, runner=self._runner
            )
            if mutation.status == "red":
                issues.append(
                    f"G3.6: Mutation score gate failed ({mutation.reason or 'no detail'})"
                )
                health.tests_pass = False
                if mutation.evidence:
                    health.test_output = mutation.evidence
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
                runner=self._runner,
            )
            if not success:
                issues.append("G4: Type check failed")
                return False, issues, health

        # G4.5: LSP semantic diagnostics (optional, off by default). Catches
        # errors a syntax check misses; best-effort and timeout-bounded, so a
        # None result (unsupported/slow/unavailable) is a skip, not a failure.
        if self.lsp_diagnostics and self.lsp_language:
            from misterdev.core.context.lsp import (
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

        # G4.6: Runtime smoke gate (optional, off by default). Launches the
        # built artifact and asserts it responds. Best-effort and
        # timeout-bounded: a SKIP (no/incomplete config, timeout) never fails;
        # only a RED (non-zero exit or missing expectation) blocks the build.
        if self.runtime_smoke:
            from misterdev.core.execution.runtime import run_smoke_gate

            smoke = run_smoke_gate(self.project_path, self.runtime_config.get("smoke"))
            if smoke.status == "red":
                issues.append(
                    f"G4.6: Runtime smoke failed ({smoke.reason or 'no detail'})"
                )
                health.tests_pass = False
                if smoke.evidence:
                    health.test_output = smoke.evidence
                return False, issues, health

        # G4.7: Web verification gate (optional, off by default). Drives a real
        # headless browser against the running web artifact and runs declarative
        # checks (DOM/text presence, no console errors, axe, screenshot diff).
        # Best-effort and timeout-bounded: a SKIP (no/incomplete config, no
        # Playwright/browser, timeout) never fails; only a RED (a failed check)
        # blocks the build. Real screenshot evidence is captured.
        web_evidence: Optional[str] = None
        if self.web_verify:
            from misterdev.core.verification.web_verify import (
                run_web_gate,
            )

            web = run_web_gate(self.project_path, self.runtime_config.get("web"))
            # The captured screenshot doubles as the vision gate's input below, so
            # the two gates compose: web renders + captures, vision judges it.
            web_evidence = web.evidence or None
            if web.status == "red":
                issues.append(f"G4.7: Web verify failed ({web.reason or 'no detail'})")
                health.tests_pass = False
                if web.evidence:
                    health.test_output = f"web evidence: {web.evidence}"
                return False, issues, health

        # G4.8: Vision verification gate (optional, off by default). Asks a
        # vision model whether a captured screenshot satisfies a stated visual
        # requirement. Best-effort and timeout-bounded: a SKIP (no/incomplete
        # config, no model/network, unparseable verdict, timeout) never fails;
        # only a RED (the model denies the assertion) blocks the build.
        if self.vision_verify:
            from misterdev.core.verification.vision_verify import (
                run_vision_gate,
            )

            # Default the screenshot to the web gate's freshly captured evidence
            # when the vision config doesn't name its own ``capture``, so enabling
            # both gates "just works" without duplicating the path. An explicit
            # capture in config still wins.
            vision_config = dict(self.runtime_config.get("vision") or {})
            if not vision_config.get("capture") and web_evidence:
                vision_config["capture"] = web_evidence
            vision = run_vision_gate(
                self.project_path,
                vision_config or None,
                llm_client=self.vision_client,
            )
            if vision.status == "red":
                issues.append(
                    f"G4.8: Vision verify failed ({vision.reason or 'no detail'})"
                )
                health.tests_pass = False
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

        # Plugin gates: third-party gates registered on misterdev.plugins.GATES
        # run after the built-ins and can only ADD blocking issues (a RED
        # outcome), never remove one.
        issues.extend(self._run_plugin_gates(commands))

        return len(issues) == 0, issues, health

    def _run_plugin_gates(self, commands: Dict[str, Optional[str]]) -> List[str]:
        """Run each registered plugin gate; collect a blocking issue per RED.

        A gate that raises or returns nothing is skipped (best-effort, like the
        optional built-in gates) so a third-party gate can never break the build
        pipeline itself — only fail it deliberately with a RED outcome.
        """
        from misterdev.plugins import GATES
        from misterdev.core.execution.outcomes import GateContext, RED

        ctx = GateContext(self.project_path, commands, self.env_activate)
        found: List[str] = []
        for name in GATES.names():
            gate = GATES.get(name)
            try:
                outcome = gate(ctx)
            except Exception as e:
                logger.warning(f"Plugin gate {name!r} raised, skipping: {e}")
                continue
            if outcome is not None and outcome.status == RED:
                found.append(f"G-plugin[{name}]: {outcome.reason or 'failed'}")
        return found

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
        for path, line in added:
            if not _path_in_scope(path, CODE_EXTENSIONS):
                continue
            for marker in BANNED_MARKERS:
                if marker in line:
                    found.add(marker)
        return sorted(found)

    def _scan_banned_markers_whole_tree(self) -> List[str]:
        """Non-git fallback for G5: scan every source file."""
        found = set()
        for path in self._iter_source_files():
            content = _read_capped(path)
            if content is None:
                continue
            for marker in BANNED_MARKERS:
                if marker in content:
                    found.add(marker)
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
            if not _path_in_scope(path, SECRET_SCAN_EXTENSIONS, SECRET_SCAN_FILENAMES):
                continue
            by_file.setdefault(path, []).append(line)
        flagged = []
        for path, lines in by_file.items():
            if self._content_has_secret(
                "\n".join(lines), is_env_file=self._is_env_literal_file(path)
            ):
                flagged.append(path)
        return flagged

    def _scan_secrets_whole_tree(self) -> List[str]:
        """Non-git fallback for G6: scan every source and config/env file."""
        flagged = []
        for path in self._iter_source_files(
            SECRET_SCAN_EXTENSIONS, SECRET_SCAN_FILENAMES
        ):
            content = _read_capped(path)
            if content is None:
                continue
            if self._content_has_secret(
                content, is_env_file=self._is_env_literal_file(path)
            ):
                flagged.append(str(path.relative_to(self.project_path)))
        return flagged

    def _iter_diff_added_lines(self) -> Optional[List[Tuple[str, str]]]:
        """Return ``(file_path, added_line)`` pairs from the git diff.

        Covers both staged and unstaged changes against HEAD (the union of
        ``git diff --cached`` and ``git diff``), so any line this build added is
        seen regardless of whether it was staged yet. Returns ``None`` when the
        project is not a git repo, signalling callers to fall back to a
        whole-tree scan. All added lines (``+``) are included regardless of file
        type; the diff header lines (``+++``) are skipped. Callers apply their
        own per-gate scope (code-only for G5/G9, code+config for G6).
        """
        check = run_git("git rev-parse --is-inside-work-tree", self.project_path)
        if check is None or check.returncode != 0 or check.stdout.strip() != "true":
            return None
        added: List[Tuple[str, str]] = []
        for cmd in (
            "git diff --cached --diff-filter=ACMR -U0",
            "git diff --diff-filter=ACMR -U0",
        ):
            proc = run_git(cmd, self.project_path)
            if proc is None:
                continue
            current_file = ""
            for line in proc.stdout.splitlines():
                if line.startswith("+++ b/"):
                    current_file = line[len("+++ b/") :]
                    continue
                if line.startswith("+++") or not line.startswith("+"):
                    continue
                added.append((current_file, line[1:]))
        # A brand-new file is untracked and never appears in `git diff`, yet its
        # every line was introduced by this build. Enumerate untracked files
        # (honoring .gitignore via --exclude-standard, so build artifacts are
        # skipped) and treat their whole content as added lines.
        others = run_git("git ls-files --others --exclude-standard", self.project_path)
        if others is not None and others.returncode == 0:
            for rel in others.stdout.splitlines():
                rel = rel.strip()
                if not rel:
                    continue
                content = _read_capped(self.project_path / rel)
                if content is None:
                    continue
                for line in content.splitlines():
                    added.append((rel, line))
        return added

    @staticmethod
    def _is_env_literal_file(path) -> bool:
        """True when ``path`` is an env/ini-style config file where ``KEY=value``
        is a literal assignment (so an unquoted value may be a real secret)."""
        p = Path(path)
        return p.suffix in ENV_LITERAL_EXTENSIONS or p.name in SECRET_SCAN_FILENAMES

    @staticmethod
    def _content_has_secret(content: str, is_env_file: bool = False) -> bool:
        for pattern in SECRET_PATTERNS:
            if pattern in content:
                return True
        for rx in SECRET_REGEXES:
            if rx.search(content):
                return True
        for line in content.splitlines():
            if GateKeeper._is_secret_assignment(line):
                return True
            if is_env_file and GateKeeper._is_unquoted_env_secret(line):
                return True
        return False

    @staticmethod
    def _is_unquoted_env_secret(line: str) -> bool:
        """True for an UNQUOTED credential value in an env/ini-style config file.

        Only used for ENV_LITERAL_EXTENSIONS files, where ``KEY=value`` is a
        literal — never on source code, where an unquoted RHS is a variable
        reference (``token = get_token()``) and flagging it would block real code.
        The value must look like an actual secret: long, no whitespace, no
        variable/template prefix, and mixing letters with digits — so ordinary
        config (ports, hosts, booleans, ``${VAR}`` refs, plain words) is ignored.
        """
        stripped = line.lstrip()
        if not stripped or stripped[0] in ("#", ";"):  # comment
            return False
        if "=" not in stripped:
            return False
        key_part, _, value = stripped.partition("=")
        if not any(k in key_part.lower() for k in ASSIGNMENT_SECRET_KEYS):
            return False
        value = value.strip()
        # Quoted literals are handled by _is_secret_assignment; here we want bare
        # values only, and never a variable/template reference.
        if (
            not value
            or value[0] in ('"', "'")
            or value.startswith(("${", "$", "%", "{{"))
        ):
            return False
        if len(value) < 16 or any(c.isspace() for c in value) or "://" in value:
            return False
        return any(c.isalpha() for c in value) and any(c.isdigit() for c in value)

    @staticmethod
    def _is_secret_assignment(line: str) -> bool:
        """True only for a credential key assigned a concrete quoted literal.

        Handles both ``key = "..."`` (source/ini/env) and ``key: "..."``
        (YAML/JSON) forms. Skips variable/env references (``token = get_token()``,
        ``api_key = os.environ[...]``) and type annotations (``token: String``)
        that would otherwise produce false positives on ordinary source, because
        the value must be a quoted literal of >=6 chars that is not a ``${...}``
        interpolation or an env reference.
        """
        for sep in ("=", ":"):
            if sep not in line:
                continue
            key_part, _, value = line.partition(sep)
            if not any(k in key_part.lower() for k in ASSIGNMENT_SECRET_KEYS):
                continue
            value = value.strip()
            for quote in ('"', "'"):
                if value.startswith(quote):
                    literal = value[1:].split(quote, 1)[0]
                    low = literal.lower()
                    if (
                        len(literal) >= 6
                        and not literal.startswith("${")
                        and not any(m in low for m in ENV_REFERENCE_MARKERS)
                    ):
                        return True
        return False

    def _check_diff_hygiene(self) -> List[str]:
        """G9: Check added lines (staged + unstaged) for debug artifacts.

        Scoped to the same union diff as G5/G6 (``_iter_diff_added_lines``) so an
        unstaged debug line is caught, not just a staged one. Code files only —
        a ``print(`` in a config/doc file is not an artifact. Each distinct
        marker is reported once even if it appears on several lines.
        """
        added = self._iter_diff_added_lines()
        if added is None:
            return []

        debug_markers = (
            "console.log(",
            "print(",
            "println!",
            "dbg!(",
            "debugger",
            "binding.pry",
            "import pdb",
        )
        found: List[str] = []
        for path, line in added:
            if not _path_in_scope(path, CODE_EXTENSIONS):
                continue
            for marker in debug_markers:
                if marker in line and marker not in found:
                    found.append(marker)
        return [f"G9: Debug artifact in diff: {marker}" for marker in found]

    def _iter_source_files(
        self,
        extensions: frozenset = CODE_EXTENSIONS,
        filenames: frozenset = frozenset(),
    ):
        """Yield in-scope file paths, skipping excluded directories.

        Defaults to source-code files; pass a wider ``extensions``/``filenames``
        scope (e.g. for the G6 secret scan) to include config/env files.
        """
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
                # Do not follow directory symlinks: a link back to an ancestor
                # would make this walk loop forever, and a link outside the repo
                # would scan unrelated files. Real subdirectories only.
                if entry.is_dir() and not entry.is_symlink():
                    stack.append(entry)
                elif entry.suffix in extensions or entry.name in filenames:
                    yield entry
