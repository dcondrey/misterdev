"""T2.4 — the documentation tool (context7 etc.) is mounted by default.

A default build (no mcp config) should give the model a version-pinned library-docs
server so it can look things up instead of hallucinating APIs. Only the docs-category
core servers are mounted by default (not the whole core stack); opt out with
mcp.docs_tool: false, override with mcp.curated.
"""

from pathlib import Path

from misterdev.core.execution.project import Project
from misterdev.core.integration.mcp_registry import select_curated


class _Duck:
    def __init__(self, config):
        self.config = config

    def _host_exec_isolated(self):
        return False

    def _mcp_cache_path(self):
        return Path("/tmp/mcp-none.json")


_Duck._build_mcp = Project._build_mcp


def _mounted(config):
    mgr = _Duck(config)._build_mcp()
    return {s["name"] for s in mgr.servers} if mgr else set()


def test_select_curated_category_filter():
    docs = {s["name"] for s in select_curated(("core",), categories={"docs"})}
    assert "context7" in docs
    assert "git" not in docs  # git is vcs, not docs


def test_docs_tool_mounted_by_default():
    names = _mounted({})
    docs = {s["name"] for s in select_curated(("core",), categories={"docs"})}
    assert docs and docs <= names
    # Only docs by default, not the whole curated core stack.
    assert "git" not in names


def test_docs_tool_opt_out():
    assert _mounted(
        {"mcp": {"docs_tool": False}}
    ) == set() or "context7" not in _mounted({"mcp": {"docs_tool": False}})


def test_explicit_curated_still_mounts_full_core():
    names = _mounted({"mcp": {"curated": True}})
    assert "git" in names and "context7" in names
