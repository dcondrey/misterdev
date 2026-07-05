"""Multi-target (polyglot) gate routing.

A monorepo can hold several sub-projects in different languages — e.g. a Rust
core plus a TypeScript web client and a Swift app — each with its OWN build/test/
lint toolchain. The orchestrator otherwise assumes ONE gate for the whole repo,
so a task editing the web client would be (mis)gated with the Rust commands.

A ``targets`` list in project.yaml declares those sub-projects::

    targets:
      - name: core
        path: emathy-core
        build_command: "cargo build -p emathy-core"
        test_command: "cargo test -p emathy-core --lib"
      - name: web
        path: clients/web
        build_command: "npm run typecheck"

For each task, :func:`select_target` picks the target that owns the task's files,
and the executor gates that task with THAT target's commands. With no ``targets``
declared (or no match), everything falls back to the top-level commands, so the
single-target path is byte-identical to before.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# Directories that never contain a sub-project worth gating (vendored deps, build
# output, VCS), so discovery skips them.
_SKIP_DIRS = {
    "node_modules",
    "target",
    "dist",
    "build",
    "pkg",
    "vendor",
    "Pods",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".gradle",
    ".cargo",
}

# A directory holding any of these is the root of a sub-project (one toolchain).
_BUILD_MARKERS = (
    "Cargo.toml",
    "package.json",
    "Package.swift",
    "go.mod",
    "meson.build",
    "build.gradle",
    "build.gradle.kts",
    "pyproject.toml",
    "CMakeLists.txt",
)


def _has_marker(d: Path) -> bool:
    if any((d / m).exists() for m in _BUILD_MARKERS):
        return True
    return any(d.glob("*.csproj")) or any(d.glob("*.sln"))


def discover_targets(project_path: str, max_depth: int = 3) -> List[Dict[str, Any]]:
    """Auto-detect sub-projects (targets) in a polyglot monorepo.

    Walks the tree and records the SHALLOWEST build marker in each subtree (a
    Cargo workspace at ``rust/`` is one target, not one per crate — discovery does
    not descend past a found marker). The repo root itself is never a target (it
    is the top-level fallback). Build/test commands are detected per sub-project.

    Conservative on purpose: returns [] unless at least TWO distinct sub-projects
    are found, so a normal single-project repo is never turned into "targets" and
    behavior is unchanged. Commands are best-effort — explicit ``targets`` in
    project.yaml override and tune them.
    """
    from misterdev.analyzers.project_analyzer import (
        detect_build_command,
        detect_test_command,
    )

    root = Path(project_path)
    targets: List[Dict[str, Any]] = []

    def scan(d: Path, depth: int, rel: str) -> None:
        if depth > max_depth:
            return
        if d.name in _SKIP_DIRS or (rel and d.name.startswith(".")):
            return
        if rel and _has_marker(d):
            build = detect_build_command(d)
            test = detect_test_command(d)
            if build or test:
                targets.append(
                    {
                        "name": rel.replace("/", "-"),
                        "path": rel,
                        "build_command": build,
                        "test_command": test,
                    }
                )
            return  # do not descend into a found sub-project
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for child in children:
            scan(child, depth + 1, f"{rel}/{child.name}" if rel else child.name)

    scan(root, 0, "")
    return targets if len(targets) >= 2 else []


def _norm(path: str) -> str:
    return (path or "").strip().strip("/")


def _owns(target_path: str, file_path: str) -> bool:
    """True when ``file_path`` lives under ``target_path`` (or equals it)."""
    tp = _norm(target_path)
    fp = _norm(file_path)
    if not tp:
        return False
    return fp == tp or fp.startswith(tp + "/")


def select_target(
    targets: List[Dict[str, Any]], file_paths: List[str]
) -> Optional[Dict[str, Any]]:
    """Pick the declared target that owns the most of ``file_paths``, or None.

    Ties break toward the more specific (longer) target path, so a nested target
    (``clients/web/sub``) wins over its parent. Returns None when there are no
    targets, no files, or no target owns any file — the caller then uses the
    top-level commands.
    """
    if not targets or not file_paths:
        return None
    best: Optional[Dict[str, Any]] = None
    best_count = 0
    best_len = -1
    for t in targets:
        tp = _norm(t.get("path", ""))
        if not tp:
            continue
        count = sum(1 for f in file_paths if _owns(tp, f))
        if count == 0:
            continue
        if count > best_count or (count == best_count and len(tp) > best_len):
            best, best_count, best_len = t, count, len(tp)
    return best


_GATE_KEYS = ("build_command", "test_command", "lint_command", "typecheck_command")


def target_commands(
    target: Optional[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    """Resolve the effective build/test/lint/typecheck commands for a task.

    A MATCHED target is self-contained: only the commands it declares apply, and
    any it omits are skipped (None) — NOT inherited from the top-level, which is
    usually a different toolchain (inheriting ``cargo test`` onto a web task would
    be meaningless). With NO matched target, the top-level commands are used
    unchanged (the single-target path).
    """
    source = target if target is not None else config
    return {key: source.get(key) for key in _GATE_KEYS}
