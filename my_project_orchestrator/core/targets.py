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

from typing import Any, Dict, List, Optional


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


def target_commands(
    target: Optional[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    """Resolve the effective build/test/lint/typecheck commands for a task.

    Uses the matched ``target``'s commands when present, falling back per-command
    to the top-level config — so a target may override just its build command and
    inherit the rest. With no target, returns the top-level commands unchanged.
    """

    def pick(key: str) -> Optional[str]:
        if target is not None and target.get(key) is not None:
            return target.get(key)
        return config.get(key)

    return {
        "build_command": pick("build_command"),
        "test_command": pick("test_command"),
        "lint_command": pick("lint_command"),
        "typecheck_command": pick("typecheck_command"),
    }
