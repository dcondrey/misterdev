"""Per-run failure taxonomy (core.execution): classifier routing + aggregation.

Distinct from core.learning.failure_taxonomy (the cognitive-cause classifier);
this is the operational end-of-run breakdown that feeds run_summary.json.
"""

from misterdev.core.execution.failure_taxonomy import (
    CATEGORIES,
    build_run_summary,
    classify_failure,
)


def test_classify_each_category():
    assert classify_failure("failed", "Command timed out after 120s") == "infra"
    assert (
        classify_failure("failed", "please run `wrangler login`, not logged in")
        == "blocked-external"
    )
    assert (
        classify_failure("failed", "merge conflict: CONFLICT (content) in env.ts")
        == "merge-conflict"
    )
    assert (
        classify_failure("failed", "### Acceptance criterion not met\n...")
        == "acceptance-unmet"
    )
    assert (
        classify_failure("failed", "error TS2345: bad type") == "genuine-code-failure"
    )
    assert (
        classify_failure("deferred", "how should I proceed with this?")
        == "deferred-needs-input"
    )


def test_signal_wins_over_status():
    """A blocked/infra signal is labelled as such regardless of status — that IS
    why the task parked or failed."""
    assert (
        classify_failure("deferred", "a required API key is missing")
        == "blocked-external"
    )
    assert classify_failure("failed", "ENOSPC: no space left on device") == "infra"


def test_deferred_without_signal_is_needs_input():
    assert (
        classify_failure("deferred", "please clarify the requirement")
        == "deferred-needs-input"
    )


def test_every_category_reachable():
    produced = {
        classify_failure("failed", "Command timed out after 120s"),
        classify_failure("failed", "wrangler login required"),
        classify_failure("failed", "merge conflict in a.ts"),
        classify_failure("failed", "Acceptance criteria not met"),
        classify_failure("failed", "AssertionError: 1 != 2"),
        classify_failure("deferred", "clarify please"),
    }
    assert produced == set(CATEGORIES)


def test_build_run_summary_counts_and_breakdown():
    summary = build_run_summary(
        completed=5,
        failed_items=[
            ("T1", "Command timed out after 120s"),
            ("T2", "error TS2345: bad type"),
            ("T3", "another AssertionError: x"),
        ],
        deferred_items=[("T4", "a required API key is missing")],
        elapsed_seconds=93.44,
    )
    assert summary["completed"] == 5
    assert summary["failed"] == 3
    assert summary["deferred"] == 1
    assert summary["elapsed_seconds"] == 93.4  # rounded to one decimal
    assert summary["failure_breakdown"] == {
        "infra": 1,
        "blocked-external": 1,
        "genuine-code-failure": 2,
    }
    assert summary["top_obstacle"] == "genuine-code-failure"  # the 2 outweigh the 1s
    assert "genuine-code-failure" in summary["exemplars"]
    assert summary["exemplars"]["infra"] == "Command timed out after 120s"


def test_top_obstacle_tie_breaks_to_more_specific():
    """One of each category → the tie breaks toward the earliest (most specific)
    category, infra."""
    summary = build_run_summary(
        completed=0,
        failed_items=[
            ("T1", "Command timed out after 120s"),
            ("T2", "error TS2345"),
        ],
        deferred_items=[("T3", "clarify please")],
        elapsed_seconds=1.0,
    )
    assert summary["top_obstacle"] == "infra"


def test_empty_run_has_no_breakdown_or_obstacle():
    summary = build_run_summary(3, [], [], 10.0)
    assert summary["failure_breakdown"] == {}
    assert summary["top_obstacle"] is None
    assert summary["exemplars"] == {}


def test_exemplar_skips_fences_and_headers():
    summary = build_run_summary(
        0, [("T1", "```\n### header\nreal error line here\n")], [], 1.0
    )
    assert summary["exemplars"]["genuine-code-failure"] == "real error line here"
