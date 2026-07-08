"""Cross-cutting learning substrate — the state misterdev evolves across runs.

This package holds the pieces that make misterdev *learn from its own use*, as
opposed to the per-build planning memory (``core.planning.lesson_store``) and the
manual benchmark-driven code evolver (``core.evolution``):

* :mod:`.failure_log` — a durable, fingerprinted stream of real build failures.
  Every finished build appends what actually broke (error, language, category)
  so the code evolver can be aimed at real weaknesses, not only the synthetic
  benchmark. This is the seam that turns "self-improves in a lab" into
  "self-improves from what actually breaks".
* :mod:`.warm_start` — an index of solved tasks so a new task can start from the
  approach of its nearest solved neighbour instead of cold.

Both degrade to no-ops on any I/O or dependency failure: learning is an
optimization layered over the build, never a precondition for it.
"""

from .failure_log import FailureLog, FailureRecord, language_of
from .reproduction import Case, ReproductionCorpus
from .warm_start import SolvedTask, SolvedTaskIndex

__all__ = [
    "FailureLog",
    "FailureRecord",
    "language_of",
    "SolvedTask",
    "SolvedTaskIndex",
    "Case",
    "ReproductionCorpus",
]
