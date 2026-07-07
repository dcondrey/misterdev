"""Empirical self-improvement loop (AlphaEvolve / Darwin-Gödel style).

The spine, built safe-first: a machine-gradeable fitness function over the
existing benchmark harness, a MAP-Elites archive so stepping-stones are never
forgotten, a reward-hacking guardrail that walls off the evaluator/gates/held-out
tests, and a keep-if-better loop that promotes a self-edit only when its measured
delta beats the harness noise band with zero regressions.

The parts that spend money or mutate source (real benchmark runs, applying an LLM
diff in a worktree) are injected into :class:`EvolutionLoop`, not wired here — so
the decision spine is pure, testable, and guardrailed before tier-2 activation.
"""

from .archive import Candidate, EvolutionArchive
from .attribution import Blame, attribute, top_target
from .fitness import FitnessScore, estimate_noise_band
from .guardrail import (
    ProtectedPathError,
    assert_mutation_allowed,
    is_protected,
)
from .loop import EvolutionLoop, Mutation, StepResult
from .prior import KindWeight, MutationPrior
from .proposer import LLMProposer, build_instruction, parse_paths, parse_tag
from .sandbox import SandboxEvaluator

__all__ = [
    "FitnessScore",
    "estimate_noise_band",
    "EvolutionArchive",
    "Candidate",
    "is_protected",
    "assert_mutation_allowed",
    "ProtectedPathError",
    "EvolutionLoop",
    "Mutation",
    "StepResult",
    "Blame",
    "attribute",
    "top_target",
    "MutationPrior",
    "KindWeight",
    "SandboxEvaluator",
    "LLMProposer",
    "build_instruction",
    "parse_paths",
    "parse_tag",
]
