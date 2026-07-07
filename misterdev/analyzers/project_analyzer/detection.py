"""Deterministic project-marker detection of build/test commands."""

import json
from pathlib import Path
from typing import Optional


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


def detect_audit_command(project_path: Path) -> Optional[str]:
    """Detect a dependency/security-audit command from project markers.

    Supply-chain scanning (the layer above lint): surfaces known-vulnerable
    dependencies. Returns a runnable command, or None when no recognized
    ecosystem is found. The audit gate is advisory and SKIPs when the tool is
    absent, so returning a command the environment may lack is safe.
    """
    p = project_path
    if (p / "Cargo.toml").exists():
        return "cargo audit"
    if (p / "package.json").exists():
        return "npm audit --omit=dev"
    if (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
        return "uv run pip-audit" if (p / "uv.lock").exists() else "pip-audit"
    if any(p.glob("*.sln")) or any(p.glob("*.csproj")):
        return "dotnet list package --vulnerable --include-transitive"
    return None


def detect_lint_command(project_path: Path) -> Optional[str]:
    """Detect a strict linter command from project markers.

    Fallback used when no ``lint_command`` is configured — so the strict tools
    the guidance recommends actually run in the G2 gate. Conservative: only
    returns a command when the tool is standard for the ecosystem (clippy ships
    with rustup) or a config file proves it is set up, so an absent linter never
    produces a false lint failure.
    """
    p = project_path
    if (p / "Cargo.toml").exists():
        return "cargo clippy --all-targets --all-features -- -D warnings"
    ruff_cfg = (p / "ruff.toml").exists() or (p / ".ruff.toml").exists()
    if ruff_cfg or (
        (p / "pyproject.toml").exists() and _file_mentions(p / "pyproject.toml", "ruff")
    ):
        return "uv run ruff check ." if (p / "uv.lock").exists() else "ruff check ."
    if (p / "package.json").exists() and any(
        any(p.glob(pat)) for pat in ("eslint.config.*", ".eslintrc", ".eslintrc.*")
    ):
        return "npx --no-install eslint . --max-warnings 0"
    if (p / "Package.swift").exists() and (p / ".swiftlint.yml").exists():
        return "swiftlint --strict"
    if (p / "detekt.yml").exists() or (p / "detekt-config.yml").exists():
        return "detekt"
    return None


def detect_typecheck_command(project_path: Path) -> Optional[str]:
    """Detect a standalone type-check command, distinct from the build.

    Only for ecosystems where type-checking is separate from compilation
    (TypeScript's ``tsc --noEmit``, Python's mypy/pyright) — for compiled
    languages the build gate already type-checks, so returning one would just
    duplicate G1. Gated on a config file; the G4 gate SKIPs when the tool is
    absent, so an uninstalled checker never blocks.
    """
    p = project_path
    if (p / "tsconfig.json").exists():
        return "npx --no-install tsc --noEmit"
    mypy_cfg = (
        (p / "mypy.ini").exists()
        or (p / ".mypy.ini").exists()
        or (
            (p / "pyproject.toml").exists()
            and _file_mentions(p / "pyproject.toml", "[tool.mypy]")
        )
    )
    if mypy_cfg:
        return "uv run mypy ." if (p / "uv.lock").exists() else "mypy ."
    if (p / "pyrightconfig.json").exists():
        return "pyright"
    return None


def dependency_add_command(project_path: Path, package: str) -> Optional[str]:
    """Return the command to add a dependency, respecting the lock/manager in use.

    None when the ecosystem edits its manifest by hand (SwiftPM) or is
    unrecognized. Callers use this for deliberate dependency additions; a
    refactor must not touch the lock file, only a genuine add.
    """
    p = project_path
    if (p / "Cargo.toml").exists():
        return f"cargo add {package}"
    if (p / "package.json").exists():
        if (p / "pnpm-lock.yaml").exists():
            return f"pnpm add {package}"
        if (p / "yarn.lock").exists():
            return f"yarn add {package}"
        if (p / "bun.lockb").exists():
            return f"bun add {package}"
        return f"npm install {package}"
    if (p / "pyproject.toml").exists():
        return (
            f"uv add {package}"
            if (p / "uv.lock").exists()
            else f"pip install {package}"
        )
    if any(p.glob("*.csproj")):
        return f"dotnet add package {package}"
    return None


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
