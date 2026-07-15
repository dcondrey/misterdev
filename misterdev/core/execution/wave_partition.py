"""Partition a wave of tasks into parallel-safe batches by declared file overlap.

Independent tasks in a wave run concurrently in isolated worktrees, then merge
back one at a time. But two tasks that both edit a SHARED file (``env.ts``, a
route registry, a shared schema) can't safely run in parallel: their merges race
and the second either conflicts or silently clobbers the first. The plan already
declares each task's files (``files_to_create``/``files_to_modify``), so we can
detect that up front and put conflicting tasks in DIFFERENT sub-waves — run
serially — while keeping genuinely disjoint tasks parallel.

This module is the pure decision: (item, file_set) pairs in -> list of
parallel-safe batches out. Side-effect-free so the conflict policy is directly
unit-testable; the orchestrator supplies the file sets and runs each batch.
"""

from typing import Iterable, List, Set, Tuple, TypeVar

T = TypeVar("T")


def partition_parallel_safe(items: Iterable[Tuple[T, Iterable[str]]]) -> List[List[T]]:
    """Group ``(item, file_set)`` pairs into batches with pairwise-disjoint files.

    Two items whose declared file sets intersect land in DIFFERENT batches, so
    they never run concurrently and can't clobber a shared file on merge; items
    with disjoint sets share a batch and run in parallel. First-fit: each item
    joins the earliest batch it does not conflict with, else starts a new one —
    which keeps the batch count (and thus the serial depth) small. Order is
    preserved within and across batches. An item with an empty/unknown file set
    makes no claim and joins the first batch.
    """
    batches: List[Tuple[List[T], Set[str]]] = []
    for item, files in items:
        fs = {str(f) for f in (files or ())}
        for batch_items, claimed in batches:
            if fs.isdisjoint(claimed):
                batch_items.append(item)
                claimed |= fs
                break
        else:
            batches.append(([item], set(fs)))
    return [batch_items for batch_items, _ in batches]
