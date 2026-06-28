"""Project analyzer ported from /build Phase 1.

Uses LLM to analyze project structure, completeness, and context,
then merges results into a ProjectAssessment. In /build these run
as 3 parallel Claude sub-agents; here they are 3 sequential LLM
calls (or concurrent via threading if desired).
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from my_project_orchestrator.core.assessment import (
    FeatureInfo,
    ProjectAssessment,
)
from my_project_orchestrator.core.gitcmd import run_git
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

{health_ground}
Treat code that builds and is covered by passing tests as IMPLEMENTED. Docs may
describe a from-scratch plan; do NOT report already-built, tested capabilities
as missing or incomplete. Base "missing"/"broken" on the source, not the docs.

Code documented as intentional is COMPLETE, not incomplete: graceful-degradation
and fallback paths, platform-gated no-ops (e.g. a wasm/no-filesystem backend that
returns empty by design), and shims "retained for parity". A function returning an
empty or default value is NOT a stub when a comment or the design states that empty
result is the contract — check the "File intents" section before flagging. For each
item you place in "incomplete"/"stubs"/"missing", name the specific file and the
concrete unmet behavior; if you cannot, omit it. Never infer incompleteness from a
symbol name or a default return alone.

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


def detect_test_command(project_path: Path) -> Optional[str]:
    """Detect a test command from project markers, independent of the LLM.

    Returns a runnable command string, or None if no recognized test setup is
    found. Used as a fallback when the structure analyzer leaves test_command
    null, which would otherwise leave the suite un-run and the health check
    blind (reporting tests=none on a project with a passing suite).
    """
    p = project_path
    # Python FIRST, but only on a genuine Python signal — a bare ``tests/`` dir
    # is NOT pytest (Rust uses tests/ for integration tests, Node for its own
    # runner), so it must not shadow the language-specific runners below.
    if _has_python_tests(p):
        return "uv run pytest -q" if (p / "uv.lock").exists() else "pytest -q"
    if (p / "Cargo.toml").exists():
        return "cargo test"
    # Node: an explicit `test` script wins; otherwise the built-in node test
    # runner over a discoverable *.test.* suite (the case that, when missed, left
    # a real suite ungated and silently rewritten).
    if (p / "package.json").exists():
        if _json_has_test_script(p / "package.json"):
            return "npm test"
        if _has_node_tests(p):
            return "node --test"
    if (p / "Package.swift").exists():
        return "swift test"
    # .NET: a solution runs every test project; a bare test csproj runs itself.
    if any(p.glob("*.sln")) or any(p.glob("*[Tt]est*.csproj")):
        return "dotnet test"
    # GTK/meson tests run against a configured build dir; cmake/C++ via ctest.
    if (p / "meson.build").exists():
        return "meson test -C build"
    if (p / "CMakeLists.txt").exists():
        return "ctest --test-dir build --output-on-failure"
    if (p / "go.mod").exists():
        return "go test ./..."
    return None


def _has_python_tests(p: Path) -> bool:
    """True when ``p`` looks like a Python project with a pytest-runnable suite.

    Explicit pytest config is definitive. A ``tests/`` directory only counts when
    it holds Python test files OR the project is clearly Python (uv.lock /
    pyproject.toml / setup.py / setup.cfg) — so a Node or Rust project that merely
    uses ``tests/`` for its own framework is never misrouted to pytest.
    """
    if (
        (p / "pytest.ini").exists()
        or (p / "conftest.py").exists()
        or _file_mentions(p / "pyproject.toml", "pytest")
        or _file_mentions(p / "setup.cfg", "pytest")
    ):
        return True
    is_python_project = (
        (p / "uv.lock").exists()
        or (p / "pyproject.toml").exists()
        or (p / "setup.py").exists()
        or (p / "setup.cfg").exists()
    )
    for d in (p / "tests", p / "test"):
        if d.is_dir():
            if any(d.rglob("test_*.py")) or any(d.rglob("*_test.py")):
                return True
            if is_python_project:
                return True
    return False


def _has_node_tests(p: Path) -> bool:
    """True when a ``test/``/``tests/`` dir holds node-runner test files.

    Looks only for the unambiguous ``*.test.{js,mjs,cjs}`` naming the built-in
    ``node --test`` runner discovers, and only inside conventional test dirs so
    ``node_modules`` is never scanned.
    """
    for d in (p / "test", p / "tests"):
        if d.is_dir():
            for pat in ("*.test.js", "*.test.mjs", "*.test.cjs"):
                if any(d.rglob(pat)):
                    return True
    return False


def has_test_files(project_path: Path) -> bool:
    """True when the project appears to contain a test suite of any kind.

    Used to warn when a build is about to run with no test gate even though tests
    exist — the safety hole behind a real run that rewrote an ungated suite.
    """
    p = project_path
    if _has_python_tests(p) or _has_node_tests(p):
        return True
    for d in (p / "tests", p / "test"):
        if d.is_dir() and any(d.rglob("*")):
            return True
    return False


def detect_build_command(project_path: Path) -> Optional[str]:
    """Detect a build/compile-check command from project markers."""
    p = project_path
    if (p / "Cargo.toml").exists():
        return "cargo build"
    if (p / "package.json").exists():
        # Prefer a `typecheck` script: for a TS project it's the fast, deterministic
        # gate (tsc --noEmit), whereas `build` often runs heavy bundling/wasm.
        if _json_has_test_script(p / "package.json", key="typecheck"):
            return "npm run typecheck"
        if _json_has_test_script(p / "package.json", key="build"):
            return "npm run build"
    if (p / "Package.swift").exists():
        return "swift build"
    if any(p.glob("*.sln")) or any(p.glob("*.csproj")):
        return "dotnet build"
    if (p / "meson.build").exists():
        return "meson compile -C build"
    if (p / "CMakeLists.txt").exists():
        return "cmake --build build"
    if (p / "Makefile").exists() or (p / "makefile").exists():
        return "make"
    if (p / "pyproject.toml").exists() or (p / "setup.py").exists():
        return "python -m compileall -q ."
    return None


def _health_ground_truth(health) -> str:
    """One-line verified-state preamble to anchor the completeness analyzer."""
    build = "passes" if health.builds else "FAILS"
    if health.test_count:
        passing = health.test_count - health.test_failures
        tests = f"{passing}/{health.test_count} tests passing"
    elif health.tests_pass:
        tests = "test suite passes"
    else:
        tests = "no test results"
    return f"VERIFIED ground truth: build {build}; {tests}."


def _file_mentions(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _json_has_test_script(path: Path, key: str = "test") -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("scripts", {}).get(key))


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
            f.existing.append(
                FeatureInfo(
                    name=item.get("name", ""), description=item.get("description", "")
                )
            )
    for item in data.get("incomplete", []):
        if isinstance(item, dict):
            f.incomplete.append(
                FeatureInfo(
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    complexity=item.get("complexity", "medium"),
                )
            )
    for item in data.get("missing", []):
        if isinstance(item, dict):
            f.missing.append(
                FeatureInfo(
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    complexity=item.get("complexity", "medium"),
                )
            )
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
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
    ".git",
    "target",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".eggs",
    "vendor",
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


def _get_file_listing(
    project_path: Path, max_files: int = 200, max_depth: int = 6
) -> str:
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
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "CMakeLists.txt",
        "project.yaml",
        "tsconfig.json",
        "webpack.config.js",
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


_OVERVIEW_CODE_EXTS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".swift",
    ".cs",
    ".kt",
}


# Phrases that mark an empty/default/no-op result as deliberate rather than
# unfinished. When a file's leading doc contains one, the sentence carrying it is
# preserved into the overview so the completeness analyzer does not flag the code
# as a stub.
_INTENT_KEYWORDS = (
    "degrade",
    "no-op",
    "noop",
    "fallback",
    "parity",
    "by design",
    "intentional",
    "graceful",
    "historical",
    "never panic",
)


def _leading_doc(path: Path, max_lines: int = 40, max_chars: int = 220) -> str:
    """Extract a source file's leading comment/doc block as one compact line.

    Carries each file's stated *intent* (e.g. "degrades to empty on wasm — the
    historical missing-model contract") into completeness analysis, so a deep
    file is judged by its documented purpose rather than guessed from its symbol
    names. Returns "" when the file opens with code and no leading comment.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = [next(fh, "") for _ in range(max_lines)]
    except OSError:
        return ""

    # `#` is a line comment in Python but a preprocessor directive in C-family
    # files, so only treat it (and `"""` docstrings) as a comment for Python.
    is_py = path.suffix == ".py"
    line_prefixes = ("///", "//!", "//", "#") if is_py else ("///", "//!", "//")
    doc: list[str] = []
    in_block = False
    block_end = ""
    for raw in head:
        line = raw.strip()
        if not in_block:
            if not line:
                # Skip blank lines anywhere in the leading comment region (not just
                # before the first comment), so an intent stated after a blank
                # separator line is still collected; the first real CODE line below
                # still ends the block.
                continue
            if not doc and line.startswith("#!"):
                continue  # shebang
        if in_block:
            end = line.find(block_end)
            seg = (line[:end] if end != -1 else line).lstrip("*").strip()
            if seg:
                doc.append(seg)
            if end != -1:
                break
            continue
        if is_py and (line.startswith('"""') or line.startswith("'''")):
            quote = line[:3]
            rest = line[3:].rstrip()
            if rest.endswith(quote) and len(rest) >= 3:
                doc.append(rest[:-3].strip())
                break
            in_block, block_end = True, quote
            if rest.strip():
                doc.append(rest.strip())
            continue
        if line.startswith("/*"):
            rest = line[2:]
            if "*/" in rest:
                doc.append(rest.split("*/", 1)[0].strip())
                break
            in_block, block_end = True, "*/"
            seg = rest.lstrip("*").strip()
            if seg:
                doc.append(seg)
            continue
        matched = next((p for p in line_prefixes if line.startswith(p)), None)
        if matched is not None:
            doc.append(line[len(matched) :].strip())
            continue
        break  # first line of real code ends the leading doc block

    full = " ".join(d for d in doc if d).strip()
    if not full:
        return ""
    summary = full[:max_chars]
    # The strongest "this empty/no-op is intentional" signal often sits a few
    # sentences into a module doc, past the summary cap. If the block states such
    # intent, graft the sentence that says so onto the summary so it is never
    # truncated away — this is exactly what stops a documented graceful-degrade
    # backend from being misread as an unfinished stub.
    lowered = full.lower()
    if any(k in lowered for k in _INTENT_KEYWORDS):
        for sentence in re.split(r"(?<=[.;])\s+", full):
            sl = sentence.lower()
            if any(k in sl for k in _INTENT_KEYWORDS) and sentence not in summary:
                summary = f"{summary.rstrip()} … {sentence.strip()}"[: max_chars + 180]
                break
    return summary


def _get_source_overview(
    project_path: Path, max_chars: int = 8000, outline: Optional[str] = None
) -> str:
    """Whole-project structural map plus the heads of source files.

    The symbol map (every file and its symbols, from tree-sitter) conveys the
    architecture densely so planning is grounded in the entire project rather
    than the first few files that fit ``max_chars`` of raw heads. A per-file
    intent map (leading doc comments) then carries each file's documented purpose
    — including deliberate graceful-degradation — so deep files are not judged by
    their symbol names alone.

    ``outline`` lets the caller pass an already-built symbol outline (the
    project's TopographyEngine graph). When given, this reuses it instead of
    parsing a second throwaway ``SymbolGraph`` here, so the whole-project graph is
    built once per run rather than once for the overview AND once for the engine.
    """
    parts = []
    if outline is None:
        try:
            from my_project_orchestrator.core.topography import SymbolGraph

            graph = SymbolGraph(project_path)
            graph.build()
            outline = graph.project_outline()
        except (ImportError, OSError, ValueError) as e:
            logger.debug(f"symbol-based overview unavailable: {e}")
            outline = ""
    if outline:
        parts.append("## Project structure (files and symbols)\n" + outline)

    intents = []
    intent_total = 0
    for item in _walk_limited(project_path):
        if item.suffix in _OVERVIEW_CODE_EXTS:
            doc = _leading_doc(item)
            if doc:
                line = f"{item.relative_to(project_path)}: {doc}"
                intents.append(line)
                intent_total += len(line) + 1
                if intent_total >= 16000:
                    break
    if intents:
        parts.append(
            "## File intents (leading doc comments)\n"
            "Documented graceful-degradation, platform-gated no-ops, and parity "
            "shims are intentional and COMPLETE — not stubs.\n" + "\n".join(intents)
        )

    head = []
    total = 0
    for item in _walk_limited(project_path):
        if item.suffix in _OVERVIEW_CODE_EXTS:
            text = _read_file_safe(item, max_lines=30)
            if text:
                rel = item.relative_to(project_path)
                chunk = f"### {rel}\n{text}\n"
                head.append(chunk)
                total += len(chunk)
                if total >= max_chars:
                    break
    if head:
        parts.append("## Source heads\n" + "\n".join(head))
    return "\n\n".join(parts) if parts else "(no source files found)"


def _get_git_log(project_path: Path, count: int = 20) -> str:
    """Get recent git log."""
    proc = run_git(f"git log --oneline -n {count}", project_path, timeout=10)
    if proc is None:
        return "(git log unavailable)"
    return proc.stdout.strip() if proc.returncode == 0 else "(not a git repo)"


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
