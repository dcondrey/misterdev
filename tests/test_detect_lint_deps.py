import tempfile
from pathlib import Path

from misterdev.analyzers.project_analyzer.detection import (
    dependency_add_command,
    detect_lint_command,
    detect_typecheck_command,
)


def _mk(files: dict) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, body in files.items():
        (d / name).write_text(body, encoding="utf-8")
    return d


def test_lint_rust_is_strict_clippy():
    d = _mk({"Cargo.toml": "[package]\nname='x'\n"})
    assert (
        detect_lint_command(d)
        == "cargo clippy --all-targets --all-features -- -D warnings"
    )


def test_lint_python_only_when_ruff_configured():
    plain = _mk({"pyproject.toml": "[project]\nname='x'\n"})
    assert detect_lint_command(plain) is None  # no ruff configured -> no false linter
    ruff = _mk({"pyproject.toml": "[tool.ruff]\nline-length=88\n"})
    assert detect_lint_command(ruff) == "ruff check ."
    ruff_uv = _mk({"pyproject.toml": "[tool.ruff]\n", "uv.lock": ""})
    assert detect_lint_command(ruff_uv) == "uv run ruff check ."


def test_lint_node_only_with_eslint_config():
    no_cfg = _mk({"package.json": "{}"})
    assert detect_lint_command(no_cfg) is None
    cfg = _mk({"package.json": "{}", "eslint.config.js": "export default []"})
    assert detect_lint_command(cfg) == "npx --no-install eslint . --max-warnings 0"


def test_lint_none_when_unrecognized():
    assert detect_lint_command(_mk({})) is None


def test_dependency_add_respects_manager():
    assert dependency_add_command(_mk({"Cargo.toml": ""}), "serde") == "cargo add serde"
    assert (
        dependency_add_command(_mk({"package.json": "{}"}), "left-pad")
        == "npm install left-pad"
    )
    assert (
        dependency_add_command(_mk({"package.json": "{}", "pnpm-lock.yaml": ""}), "x")
        == "pnpm add x"
    )
    assert (
        dependency_add_command(_mk({"pyproject.toml": ""}), "httpx")
        == "pip install httpx"
    )
    assert (
        dependency_add_command(_mk({"pyproject.toml": "", "uv.lock": ""}), "httpx")
        == "uv add httpx"
    )
    assert (
        dependency_add_command(_mk({"App.csproj": "<Project/>"}), "Serilog")
        == "dotnet add package Serilog"
    )


def test_dependency_add_none_for_swift_and_unknown():
    assert dependency_add_command(_mk({"Package.swift": ""}), "X") is None
    assert dependency_add_command(_mk({}), "X") is None


def test_typecheck_typescript_and_python():
    ts = _mk({"package.json": "{}", "tsconfig.json": "{}"})
    assert detect_typecheck_command(ts) == "npx --no-install tsc --noEmit"
    mypy = _mk({"pyproject.toml": "[tool.mypy]\nstrict=true\n"})
    assert detect_typecheck_command(mypy) == "mypy ."
    mypy_uv = _mk({"pyproject.toml": "[tool.mypy]\n", "uv.lock": ""})
    assert detect_typecheck_command(mypy_uv) == "uv run mypy ."


def test_typecheck_none_for_compiled_langs_and_unconfigured():
    # Rust/Swift build already type-checks; a bare Python project has no checker.
    assert detect_typecheck_command(_mk({"Cargo.toml": ""})) is None
    assert (
        detect_typecheck_command(_mk({"pyproject.toml": "[project]\nname='x'\n"}))
        is None
    )
    assert detect_typecheck_command(_mk({})) is None
