"""Prime a git worktree's dependencies by CLONING the base checkout, not reinstalling.

Priming a fresh worktree with a full ``pnpm install`` costs ~15-20s per worktree
per wave. The base checkout already holds a fully resolved ``node_modules``; on a
copy-on-write filesystem that tree can be duplicated into the worktree near-
instantly (APFS ``clonefile(2)`` clones a whole directory hierarchy in ONE
syscall — measured at sub-millisecond — sharing storage until a file is written).

Why the symlinks stay correct: a git worktree is a checkout of the SAME repo at
the SAME paths, and pnpm links workspace packages with RELATIVE symlinks
(``apps/server/node_modules/@scope/pkg -> ../../../../packages/pkg``). Cloned
verbatim, that relative link resolves against the WORKTREE, so it points at the
worktree's own package — not the base — with no reinstall and no cross-tree bleed.

Everything here is fail-safe: ``clone_supported`` returns False on any non-CoW
filesystem (e.g. HFS+, where ``clonefile`` is ENOTSUP), and ``clone_dependencies``
returns False on the first clone that fails. The caller then runs the normal
install, and the P3 sanity probe still gates whatever ends up in the worktree —
so a wrong guess costs an install, never a broken build.
"""

import ctypes
import ctypes.util
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

# node_modules lives at the repo root and under each workspace package; scanning
# deeper than this (or into these dirs) only finds nested deps that a parent clone
# already carries.
_MAX_DEPTH = 6
_SKIP = {".git", ".orchestrator", ".hg", ".svn", "node_modules"}


def _macos_clonefile(src: str, dst: str) -> None:
    """Raw ``clonefile(2)``: recursive CoW clone of ``src`` to ``dst`` (which must
    not exist). Raises OSError on failure — notably ENOTSUP on a non-APFS volume,
    which the caller treats as "cloning unavailable, install instead"."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
    if libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), dst)


def _clone_tree(src: Path, dst: Path) -> None:
    """Clone one directory tree ``src`` -> ``dst`` (``dst`` must not exist).

    macOS uses ``clonefile`` (recursive, one syscall). Linux uses ``cp -al``
    (hardlinked archive copy, same-filesystem). Raises OSError on failure.
    """
    system = platform.system()
    if system == "Darwin":
        _macos_clonefile(str(src), str(dst))
    elif system == "Linux":
        r = subprocess.run(
            ["cp", "-al", "--", str(src), str(dst)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise OSError(r.returncode, (r.stderr or "cp -al failed").strip())
    else:
        raise OSError(f"dependency clone unsupported on {system}")


def clone_supported(base) -> bool:
    """Whether ``base``'s filesystem supports a fast CoW/hardlink clone.

    Probes with a single tiny file — a raw ``clonefile`` on macOS (ENOTSUP on
    HFS+), a hardlink on Linux — so a non-CoW volume is detected up front and the
    caller never pays a slow full-copy masquerading as a clone. Never raises.
    """
    system = platform.system()
    if system not in ("Darwin", "Linux"):
        return False
    probe_dir = Path(base) / ".orchestrator"
    src = probe_dir / ".cloneprobe_src"
    dst = probe_dir / ".cloneprobe_dst"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"x")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if system == "Darwin":
            _macos_clonefile(str(src), str(dst))
        else:
            os.link(src, dst)  # same-fs hardlink; EXDEV/ENOTSUP -> unsupported
        return dst.exists()
    except OSError:
        return False
    finally:
        for p in (src, dst):
            try:
                p.unlink()
            except OSError:
                pass


def find_node_modules_dirs(base, max_depth: int = _MAX_DEPTH) -> List[Path]:
    """Relative paths of every ``node_modules`` dir to clone: the repo root's and
    each workspace package's. Prunes descent into ``node_modules`` (its nested deps
    ride along with the parent clone) and VCS/state dirs. Returns paths relative to
    ``base``, so they can be recreated under the worktree."""
    base = Path(base)
    found: List[Path] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        for e in entries:
            if e.is_symlink() or not e.is_dir():
                continue
            if e.name == "node_modules":
                found.append(Path(e.path).relative_to(base))
                continue  # nested deps come with the parent clone
            if e.name in _SKIP:
                continue
            walk(Path(e.path), depth + 1)

    walk(base, 0)
    return found


def clone_dependencies(base, worktree) -> Tuple[bool, List[str]]:
    """Clone every base ``node_modules`` dir into ``worktree`` at the same path.

    Returns ``(ok, cloned_rel_paths)``. ``ok`` is False (with whatever was cloned
    so far) when there is nothing to clone or ANY clone fails — the caller then
    falls back to a normal install. A fresh worktree has no ``node_modules`` (it is
    gitignored), but any stale destination is removed first so the clone is clean.
    """
    base, worktree = Path(base).resolve(), Path(worktree).resolve()
    rels = find_node_modules_dirs(base)
    if not rels:
        return False, []
    cloned: List[str] = []
    for rel in rels:
        src, dst = base / rel, worktree / rel
        try:
            src = src.resolve()
            dst = dst.resolve()
            src.relative_to(base)
            dst.relative_to(worktree)
        except ValueError:
            logger.warning(f"Skipping out-of-bounds dep path: {rel}")
            continue
        if not src.is_dir():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                dst.unlink()
            elif dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            _clone_tree(src, dst)
        except OSError as e:
            logger.info(
                f"Dependency clone failed at {rel} ({e}); falling back to install."
            )
            return False, cloned
        cloned.append(str(rel))
    return bool(cloned), cloned
