import tempfile
import types
from pathlib import Path

from misterdev.tools.dependency import DependencyTool


def _project(files: dict):
    d = Path(tempfile.mkdtemp())
    for name, body in files.items():
        (d / name).write_text(body, encoding="utf-8")
    return types.SimpleNamespace(path=str(d))


def test_resolve_builds_manager_command():
    tool = DependencyTool({})
    cmd, reason = tool._resolve(_project({"Cargo.toml": ""}), "serde")
    assert cmd == "cargo add serde" and reason == ""
    cmd, _ = tool._resolve(
        _project({"package.json": "{}", "pnpm-lock.yaml": ""}), "left-pad"
    )
    assert cmd == "pnpm add left-pad"
    cmd, _ = tool._resolve(_project({"pyproject.toml": "", "uv.lock": ""}), "httpx")
    assert cmd == "uv add httpx"


def test_resolve_allows_scoped_and_pinned_names():
    tool = DependencyTool({})
    cmd, _ = tool._resolve(_project({"package.json": "{}"}), "@scope/pkg")
    assert cmd == "npm install @scope/pkg"
    cmd, _ = tool._resolve(_project({"Cargo.toml": ""}), "serde@1.0")
    assert cmd == "cargo add serde@1.0"


def test_resolve_rejects_shell_injection():
    tool = DependencyTool({})
    for evil in (
        "serde; rm -rf /",
        "a && b",
        "$(whoami)",
        "x`id`",
        "pkg >file",
        "a b",
        "",
    ):
        cmd, reason = tool._resolve(_project({"Cargo.toml": ""}), evil)
        assert cmd == "" and "invalid package name" in reason


def test_resolve_reports_unrecognized_ecosystem():
    tool = DependencyTool({})
    cmd, reason = tool._resolve(_project({"Package.swift": ""}), "Alamofire")
    assert cmd == "" and "no recognized package manager" in reason
    cmd, reason = tool._resolve(_project({}), "anything")
    assert cmd == "" and reason


def test_tool_is_registered():
    from misterdev.plugins import TOOLS

    assert TOOLS.get("dependency") is DependencyTool
