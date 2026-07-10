"""The tool-promotion pass — the deliberate step that CLOSES the two-timescale
loop (``docs/two-timescale-evolution.md``).

Capture is automatic (every build folds invented tools + outcomes into the tool
corpus). Promotion is deliberate, like the scaffold-evolution run: this pass reads
the tool corpus, derives the WITHOUT-tool baseline PER NICHE from the reproduction
corpus (the resolve-rate on comparable tasks with no self-authored tool), and
admits the tools whose with-tool success-association holds on a held-out task
split into the persistent :class:`~.tool_library.ToolLibrary`. Promoted tools then
seed future runs, so capability compounds.

The two corpora together are the fitness signal that a single run cannot give:
reproduction corpus = without-tool outcomes, tool corpus = with-tool outcomes,
their difference = a tool's (as-causal-as-observational-data-allows) effect. It is
correlational, not a controlled A/B — so this ranks tools by held-out association
and sharpens as data accumulates; it does not certify causation. Pure of LLM/cost:
it only reads the two JSON corpora and writes the library.

CLI: ``python -m misterdev.core.evolution.tool_promotion <project_path>``.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

from misterdev.core.evolution.tool_corpus import ToolCorpus, promote_from_corpus
from misterdev.core.evolution.tool_library import ToolLibrary
from misterdev.core.learning.reproduction import ReproductionCorpus
from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)


def run_tool_promotion(
    project_path,
    *,
    default_baseline: float = 0.5,
    min_observations: int = 5,
    holdout_fraction: float = 0.3,
    noise_band: float = 0.0,
) -> Dict:
    """Run the promotion pass for a project; return a summary dict.

    ``default_baseline`` is the prior used for a niche the reproduction corpus has
    no data on (so an empty baseline never spuriously promotes). Never raises: a
    missing/corrupt corpus degrades to "nothing promoted".
    """
    ev = Path(project_path) / ".orchestrator" / "evolution"
    corpus = ToolCorpus(ev / "tool_corpus.json")
    repro = ReproductionCorpus(ev / "reproduction.json")
    library = ToolLibrary(ev / "tool_library.json")

    def baseline_for(niche: str) -> float:
        rate = repro.resolve_rate(niche)
        return rate if rate is not None else default_baseline

    promoted = promote_from_corpus(
        corpus,
        library,
        baseline_rate=baseline_for,
        min_observations=min_observations,
        holdout_fraction=holdout_fraction,
        noise_band=noise_band,
    )
    stats = corpus.stats()
    logger.info(
        f"Tool promotion: {len(promoted)} promoted from {stats['tools']} tool(s) / "
        f"{stats['observations']} observation(s); library now {len(library.elites())}."
    )
    return {
        "promoted": promoted,
        "corpus": stats,
        "library_size": len(library.elites()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="misterdev.core.evolution.tool_promotion")
    parser.add_argument("project_path", help="project whose .orchestrator/ to promote")
    parser.add_argument("--default-baseline", type=float, default=0.5)
    parser.add_argument("--min-observations", type=int, default=5)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--noise-band", type=float, default=0.0)
    args = parser.parse_args(argv)
    result = run_tool_promotion(
        args.project_path,
        default_baseline=args.default_baseline,
        min_observations=args.min_observations,
        holdout_fraction=args.holdout_fraction,
        noise_band=args.noise_band,
    )
    sys.stdout.write(
        f"promoted {len(result['promoted'])} tool(s); "
        f"corpus={result['corpus']}; library={result['library_size']}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
