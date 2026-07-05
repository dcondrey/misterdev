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
