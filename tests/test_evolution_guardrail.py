import pytest

from misterdev.core.evolution import (
    ProtectedPathError,
    assert_mutation_allowed,
    is_protected,
)


@pytest.mark.parametrize(
    "path",
    [
        "evaluation/polyglot/grader.py",  # the judge
        "evaluation/swebench/harness.py",
        "misterdev/core/verification/spec_tests.py",  # a gate
        "misterdev/core/verification/gatekeeper/constants.py",
        "misterdev/core/evolution/fitness.py",  # the loop's own judge
        "tests/test_evolution_loop.py",  # held-out/regression tests
        "./tests/test_topography.py",
        "../sibling/thing.py",  # escapes the repo
        "/etc/passwd",  # absolute
        "misterdev/../evaluation/x.py",  # traversal into a walled dir
    ],
)
def test_protected_paths_are_refused(path):
    assert is_protected(path)


@pytest.mark.parametrize(
    "path",
    [
        "misterdev/core/planning/sovereign.py",
        "misterdev/task_executors/markdown_plan_executor/execute_mixin.py",
        "misterdev/config.py",
        "misterdev/core/context/topography/graph.py",
    ],
)
def test_ordinary_source_is_mutable(path):
    assert not is_protected(path)


def test_assert_raises_listing_the_bad_paths():
    good = "misterdev/config.py"
    bad = "evaluation/polyglot/grader.py"
    assert_mutation_allowed([good])  # no raise
    with pytest.raises(ProtectedPathError) as exc:
        assert_mutation_allowed([good, bad])
    assert "evaluation/polyglot/grader.py" in str(exc.value)


def test_verification_prefix_is_not_overmatched():
    # A path that merely starts with the same letters but is a different dir must
    # not be walled off (prefix must respect the directory boundary).
    assert not is_protected("misterdev/core/verification_helpers.py")
