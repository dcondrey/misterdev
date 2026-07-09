"""Native-language (Swift, C#) benchmark harness.

Mirrors :mod:`evaluation.polyglot` but for the two toolchains the polyglot
harness never covered — Swift and C# had zero empirical validation, so
misterdev's behavior on them was asserted only by unit-level contract tests,
never by end-to-end "did the model make a real compiler + test suite pass".

An exercise is RESOLVED when its own test command (``swift test`` /
``dotnet test``) passes after misterdev edits the solution stub. The grader is
the ground truth (resolved == test command exit 0, corroborated by the
pass/fail summary the toolchain prints), and the misterdev-driving parts are
injectable so the pure discovery + grading logic is unit-testable offline with
no swift/dotnet toolchain present.
"""

from .harness import (
    NativeExercise,
    SuiteReport,
    discover_exercises,
    grade_output,
    run_suite,
)

__all__ = [
    "NativeExercise",
    "SuiteReport",
    "discover_exercises",
    "grade_output",
    "run_suite",
]
