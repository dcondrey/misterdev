"""Conflict-graph partitioning of a wave into parallel-safe batches."""

from misterdev.core.execution.wave_partition import partition_parallel_safe


def test_overlapping_files_land_in_different_batches():
    """Two tasks that declare a shared file are never in the same batch."""
    batches = partition_parallel_safe([("A", {"env.ts"}), ("B", {"env.ts"})])
    assert len(batches) == 2
    a = next(b for b in batches if "A" in b)
    assert "B" not in a


def test_disjoint_tasks_share_a_batch():
    """Tasks with disjoint file sets run in parallel (one batch)."""
    batches = partition_parallel_safe(
        [("A", {"a.ts"}), ("B", {"b.ts"}), ("C", {"c.ts"})]
    )
    assert batches == [["A", "B", "C"]]


def test_mixed_overlap_keeps_disjoint_parallel():
    """A conflicts with B on a shared file; C is disjoint from both, so C joins
    A's batch and only B is pushed to a second sub-wave."""
    batches = partition_parallel_safe(
        [("A", {"env.ts"}), ("B", {"env.ts"}), ("C", {"routes.ts"})]
    )
    assert batches == [["A", "C"], ["B"]]


def test_empty_file_set_makes_no_claim():
    """A task with no declared files claims nothing and joins the first batch."""
    batches = partition_parallel_safe(
        [("A", {"shared.ts"}), ("B", set()), ("C", {"shared.ts"})]
    )
    # A and B share batch 0 (B claims nothing); C conflicts with A -> batch 1.
    assert batches == [["A", "B"], ["C"]]


def test_transitive_chain_two_batches():
    """A-B share f1, B-C share f2, A and C are disjoint: A and C can run together,
    only B is isolated."""
    batches = partition_parallel_safe(
        [("A", {"f1"}), ("B", {"f1", "f2"}), ("C", {"f2"})]
    )
    assert batches == [["A", "C"], ["B"]]


def test_three_way_shared_file_serializes_fully():
    """Three tasks all editing the same file must run in three separate batches."""
    batches = partition_parallel_safe(
        [("A", {"g.ts"}), ("B", {"g.ts"}), ("C", {"g.ts"})]
    )
    assert batches == [["A"], ["B"], ["C"]]


def test_empty_input():
    assert partition_parallel_safe([]) == []


def test_order_preserved_and_all_items_present():
    items = [(f"T{i}", {f"file{i % 3}.ts"}) for i in range(9)]
    batches = partition_parallel_safe(items)
    flat = [t for b in batches for t in b]
    assert sorted(flat) == sorted(t for t, _ in items)
    # No batch contains two tasks sharing a file (file0/3/6, file1/4/7, ...).
    fset = dict(items)
    for batch in batches:
        seen: set = set()
        for t in batch:
            assert seen.isdisjoint(fset[t])
            seen |= fset[t]
