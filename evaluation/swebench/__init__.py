"""SWE-bench harness: run misterdev on real GitHub-issue tasks and grade the
patch against the task's own hidden tests.

Turns "is this the best we can do?" from a judgement call into a number: the
fraction of tasks resolved (every FAIL_TO_PASS test goes to pass and every
PASS_TO_PASS test stays passing). The grader is the ground truth and is fully
unit-tested against a synthetic instance; the runner wires misterdev's build to
the setup/grade cycle.
"""

from .instance import SWEBenchInstance
from .grader import GradeResult, grade

__all__ = ["SWEBenchInstance", "GradeResult", "grade"]
