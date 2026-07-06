"""Aider polyglot benchmark harness.

Runs misterdev on the Exercism-derived polyglot exercises (C++, Go, Java,
JavaScript, Python, Rust) and grades each by running the exercise's own test
file: an exercise is RESOLVED when the test command passes after misterdev edits
the solution stub. Aligned with misterdev's edit->gate->verify loop and far
cheaper than SWE-bench, especially with free-model routing.

The grader is the ground truth (resolved == test command exit 0) and is
unit-tested offline against a synthetic exercise with real pytest.
"""

from .instance import PolyglotInstance
from .grader import GradeResult, grade

__all__ = ["PolyglotInstance", "GradeResult", "grade"]
