import subprocess
import tempfile
from pathlib import Path

from misterdev.analyzers.project_analyzer.detection import detect_audit_command
from misterdev.core.verification.gatekeeper import _audit_tool_missing


def _touch(d: Path, name: str, body: str = "") -> None:
    (d / name).write_text(body, encoding="utf-8")


def test_detect_audit_command_per_ecosystem():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _touch(d, "Cargo.toml", "[package]\nname='x'\n")
        assert detect_audit_command(d) == "cargo audit"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _touch(d, "package.json", "{}")
        assert detect_audit_command(d) == "npm audit --omit=dev"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _touch(d, "pyproject.toml", "[project]\nname='x'\n")
        assert detect_audit_command(d) == "pip-audit"
        _touch(d, "uv.lock", "")
        assert detect_audit_command(d) == "uv run pip-audit"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _touch(d, "App.csproj", "<Project/>")
        assert detect_audit_command(d).startswith("dotnet list package --vulnerable")


def test_detect_audit_command_none_when_unrecognized():
    with tempfile.TemporaryDirectory() as td:
        assert detect_audit_command(Path(td)) is None


def test_audit_tool_missing_signals():
    assert _audit_tool_missing("sh: cargo-audit: command not found")
    assert _audit_tool_missing("pip-audit: not found")
    assert not _audit_tool_missing("found 3 vulnerabilities in 2 packages")
    assert not _audit_tool_missing("")


def _run_audit_gate(project: Path, audit_cmd: str):
    from misterdev.core.verification.gatekeeper import GateKeeper

    gk = GateKeeper(project, build_timeout=30, test_timeout=30, lint_timeout=30)
    # Only the audit command is set — build/lint/test are skipped (health True),
    # isolating the G2.5 behavior.
    _, issues, _ = gk.run_gates({"audit_command": audit_cmd})
    return issues


def test_audit_gate_reports_vulnerabilities_but_does_not_block():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        subprocess.run(["git", "init", "-q"], cwd=td, check=True)
        # A non-zero exit with no "tool missing" signal == vulnerabilities found.
        issues = _run_audit_gate(d, "false")
        assert any("G2.5" in i for i in issues)


def test_audit_gate_skips_when_tool_absent():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        subprocess.run(["git", "init", "-q"], cwd=td, check=True)
        issues = _run_audit_gate(d, "nonexistent_audit_tool_xyz_123")
        assert not any("G2.5" in i for i in issues)


def test_audit_gate_clean_run_no_issue():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        subprocess.run(["git", "init", "-q"], cwd=td, check=True)
        issues = _run_audit_gate(d, "true")
        assert not any("G2.5" in i for i in issues)


def _run_typecheck_gate(project: Path, cmd: str):
    from misterdev.core.verification.gatekeeper import GateKeeper

    gk = GateKeeper(project, build_timeout=30, test_timeout=30, lint_timeout=30)
    return gk.run_gates({"typecheck_command": cmd})


def test_typecheck_gate_blocks_on_real_type_error():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        subprocess.run(["git", "init", "-q"], cwd=td, check=True)
        # A real type error: the checker ran and exited non-zero (no "missing").
        success, issues, _ = _run_typecheck_gate(d, "false")
        assert success is False
        assert any("G4" in i for i in issues)


def test_typecheck_gate_skips_when_checker_absent():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        subprocess.run(["git", "init", "-q"], cwd=td, check=True)
        # An uninstalled type-checker must SKIP, not block the build.
        success, issues, _ = _run_typecheck_gate(d, "nonexistent_tsc_xyz_123 --noEmit")
        assert not any("G4" in i for i in issues)
