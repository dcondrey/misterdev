"""Scored, self-reinforcing lesson memory for cross-run learning.

A lesson learned once is a guess; a lesson re-learned across runs is a pattern.
This store makes misterdev smarter the more it is used, without forgetting what
it already knows:

* **Reinforce** — when an audit re-derives a lesson it already holds (recurrence
  is the usefulness signal, since a genuine pitfall keeps recurring), its score
  rises instead of a near-duplicate being appended.
* **Dedup** — rewordings of the same lesson merge (token overlap coefficient),
  so "run black before commit" reinforces "always run black before committing".
* **Decay** — lessons not reinforced in a run drift down; long-stale one-offs
  fall below a floor and are dropped as noise.
* **Evict by value, not age** — when over the cap, the LOWEST-scoring lessons go,
  so a proven keystone rule survives an influx of newer noise (the old recency
  eviction forgot exactly the lessons worth keeping).
* **Retrieve by relevance** — injection returns the lessons most relevant to the
  current goal, weighted by proven value, not the whole file.

The on-disk format is a dict ``{"run": int, "lessons": [...]}``; a legacy plain
``list[str]`` file is migrated on load so existing projects lose nothing.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from misterdev.logging_setup import setup_logger
from misterdev.utils.file_utils import atomic_write_json

logger = setup_logger(__name__)

_MAX_LESSONS = 40  # hard cap on stored lessons (bounds file + injection)
_MAX_INJECT = 12  # lessons injected per build, relevance-ranked
_DEDUP_THRESHOLD = 0.6  # overlap coefficient above which two rules are "the same"
_REINFORCE = 1.0  # score gain when a lesson recurs
_DECAY = 0.9  # multiplicative decay on lessons not reinforced this run
_MIN_SCORE = 0.15  # below this a decayed lesson is dropped as noise
_NEW_SCORE = 1.0  # starting score for a freshly-learned lesson

# -- efficacy (Tier 3): reinforce on MEASURED help, not mere recurrence --------
# A lesson recurring in audits proves it keeps being noticed; it does NOT prove
# injecting it helped. Efficacy is an EWMA of the outcome delta (run success rate
# minus the project's trailing baseline) of the runs a lesson was injected into.
# Consistently-in-above-baseline runs -> positive; consistently-in-worse runs ->
# negative. This is correlational credit assignment, not an isolated causal A/B —
# but over many runs it separates a lesson that pulls its weight from a generic
# one that just rides along, which pure recurrence cannot.
_OUTCOME_ALPHA = 0.3  # EWMA weight for the project's trailing success baseline
_EFFICACY_BETA = 0.34  # EWMA weight for a lesson's per-run efficacy update
_REGRESS_BAND = (
    0.05  # a run this far below baseline counts as a regression for its lessons
)
_MIN_EFFICACY_EVIDENCE = 3  # injections before efficacy is trusted enough to quarantine
_QUARANTINE_EFFICACY = (
    -0.34
)  # at/below this (with evidence) a lesson stops being injected
_REGRESS_QUARANTINE_FRAC = (
    0.6  # fraction of injections that must be regressions to quarantine on regress_hits
)
_LEXICAL_WEIGHT = 0.3  # lexical share of hybrid relevance when an embedder is present

# Common words carry no signal for similarity; drop them so overlap reflects the
# lesson's actual subject, not shared boilerplate ("always", "must", "the").
_STOP = frozenset(
    "a an the is are was were be been being to of in on for and or not no with "
    "without you your this that it its as at by from must always never should "
    "run use using when then than into out over each any all only".split()
)


def _tokens(text: str) -> frozenset:
    """Content-word token set for similarity (lowercased, destopped, len>2)."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return frozenset(w for w in words if len(w) > 2 and w not in _STOP)


def _similarity(a: frozenset, b: frozenset) -> float:
    """Overlap coefficient — catches reworded/shortened restatements better than
    Jaccard, since a subset restatement scores high on ``inter / min``."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass
class Lesson:
    text: str
    score: float = _NEW_SCORE
    hits: int = 1  # times reinforced (recurrences across runs)
    run: int = 0  # last run in which it was reinforced (recency tiebreak)
    id: int = 0  # stable identity across rewordings, so credit survives a refresh
    injected: int = 0  # times this lesson was injected into a build
    efficacy: float = 0.0  # EWMA of injected-run outcome deltas; 0 = no evidence yet
    regress_hits: int = 0  # times injected into a run that came in below baseline

    def tokens(self) -> frozenset:
        return _tokens(self.text)

    @property
    def quarantined(self) -> bool:
        """True when enough evidence shows this lesson correlates with worse runs,
        so it should stop being injected (kept on disk, but out of the prompt).

        Two independent triggers: a strongly-negative efficacy EWMA, OR presence
        in a majority of below-baseline runs (``regress_hits``) — the latter
        catches a lesson that regresses on most injections yet whose averaged
        efficacy hovers just above the band, which efficacy alone would miss."""
        if self.injected < _MIN_EFFICACY_EVIDENCE:
            return False
        return (
            self.efficacy <= _QUARANTINE_EFFICACY
            or self.regress_hits >= self.injected * _REGRESS_QUARANTINE_FRAC
        )


class LessonStore:
    """Persistent scored lesson memory backed by a single JSON file.

    An optional ``embedder`` upgrades relevance ranking from lexical token
    overlap to a hybrid dense+lexical score, so a lesson relevant to the goal by
    MEANING (no shared literal tokens) still surfaces. It degrades to lexical-only
    when no embedder is supplied or embedding fails, so ranking never breaks.
    """

    def __init__(self, path: Path, embedder=None):
        self.path = Path(path)
        self.embedder = embedder

    # -- persistence -------------------------------------------------------
    def _load(self) -> Tuple[List[Lesson], dict]:
        """Return (lessons, meta). ``meta`` carries the run counter, the trailing
        success baseline, and the next free lesson id. Migrates a legacy list[str]
        or id-less file and degrades to empty on a missing/corrupt file (learning
        is best-effort). Every returned lesson has a stable non-zero ``id``, so
        efficacy credit survives a text refresh."""
        empty_meta = {"run": 0, "baseline": None, "next_id": 1}
        if not self.path.exists():
            return [], dict(empty_meta)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return [], dict(empty_meta)

        lessons: List[Lesson] = []
        meta = dict(empty_meta)
        if isinstance(raw, list):
            lessons = [Lesson(text=_as_text(r)) for r in raw]
        elif isinstance(raw, dict):
            meta["run"] = int(raw.get("run", 0))
            meta["baseline"] = (
                float(raw["baseline"]) if raw.get("baseline") is not None else None
            )
            meta["next_id"] = int(raw.get("next_id", 1))
            for item in raw.get("lessons", []):
                if isinstance(item, dict) and item.get("text"):
                    lessons.append(
                        Lesson(
                            text=str(item["text"]),
                            score=float(item.get("score", _NEW_SCORE)),
                            hits=int(item.get("hits", 1)),
                            run=int(item.get("run", 0)),
                            id=int(item.get("id", 0)),
                            injected=int(item.get("injected", 0)),
                            efficacy=float(item.get("efficacy", 0.0)),
                            regress_hits=int(item.get("regress_hits", 0)),
                        )
                    )
                elif isinstance(item, str):
                    lessons.append(Lesson(text=item))
        else:
            return [], dict(empty_meta)

        # Backfill stable ids for any legacy/id-less lesson so credit can key on id.
        next_id = max(meta["next_id"], max((le.id for le in lessons), default=0) + 1)
        for le in lessons:
            if le.id <= 0:
                le.id = next_id
                next_id += 1
        meta["next_id"] = next_id
        return lessons, meta

    def _save(self, lessons: List[Lesson], meta: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": meta.get("run", 0),
            "baseline": meta.get("baseline"),
            "next_id": meta.get("next_id", 1),
            "lessons": [
                {
                    **asdict(le),
                    "score": round(le.score, 4),
                    "efficacy": round(le.efficacy, 4),
                }
                for le in lessons
            ],
        }
        atomic_write_json(self.path, payload, indent=2)

    # -- learning ----------------------------------------------------------
    def record(self, new_rules: List) -> int:
        """Fold newly-audited rules into the store: reinforce matches, add the
        rest, decay the untouched, drop noise, evict by value. Returns the number
        of genuinely new (non-duplicate) lessons added."""
        lessons, meta = self._load()
        run = meta["run"] + 1
        meta["run"] = run

        # Age everything by one step; reinforced lessons more than recover below.
        for le in lessons:
            le.score *= _DECAY

        added = 0
        for rule in new_rules:
            text = _as_text(rule).strip()
            if not text:
                continue
            toks = _tokens(text)
            match = _best_match(toks, lessons)
            if match is not None:
                match.score += _REINFORCE
                match.hits += 1
                match.run = run
                match.text = text  # refresh to the newest wording
            else:
                lessons.append(
                    Lesson(
                        text=text,
                        score=_NEW_SCORE,
                        hits=1,
                        run=run,
                        id=meta["next_id"],
                    )
                )
                meta["next_id"] += 1
                added += 1

        # Drop faded noise, then evict by RETENTION VALUE over the cap: a
        # quarantined (proven-harmful) lesson is dropped before any usable one, and
        # among usable lessons efficacy-weighted score decides — so the efficacy
        # signal protects a proven-helpful lesson from eviction, not just injection.
        lessons = [le for le in lessons if le.score >= _MIN_SCORE]
        lessons.sort(key=_retention_value, reverse=True)
        lessons = lessons[:_MAX_LESSONS]

        self._save(lessons, meta)
        logger.info(
            f"Lessons: {added} new, {len(new_rules) - added} reinforced; "
            f"{len(lessons)} retained (run {run})."
        )
        return added

    # -- retrieval ---------------------------------------------------------
    def retrieve_lessons(
        self, query: str = "", limit: int = _MAX_INJECT
    ) -> List[Lesson]:
        """The highest-value lessons for ``query`` as :class:`Lesson` objects.

        Ranking blends proven value, relevance, and MEASURED efficacy:
        ``score * (1 + 2*overlap) * (1 + efficacy)`` — a lesson demonstrated to
        help (efficacy > 0) is boosted, one that rode along in worse-than-baseline
        runs (efficacy < 0) is damped, and one with enough evidence of harm is
        quarantined out entirely so it stops crowding the prompt. Newest breaks
        ties so a fresh lesson isn't buried. Returned objects carry ``id`` so the
        caller can credit exactly these lessons with the run's outcome."""
        lessons, _ = self._load()
        candidates = [le for le in lessons if not le.quarantined]
        if not candidates:
            return []
        relevance = self._relevance_map(query, candidates)

        def rank(le: Lesson) -> Tuple[float, int]:
            # efficacy in [-1, 1] -> multiplier in [0, 2]; clamped so a single bad
            # run can't drive a lesson's weight negative and invert the ranking.
            efficacy_mult = 1.0 + max(-1.0, min(1.0, le.efficacy))
            return (le.score * relevance[le.id] * efficacy_mult, le.run)

        candidates.sort(key=rank, reverse=True)
        return candidates[:limit]

    def _relevance_map(self, query: str, lessons: List[Lesson]) -> dict:
        """Map lesson id -> relevance multiplier in [1, 3] for ``query``.

        Lexical overlap always contributes; when an embedder is available the
        dense cosine similarity is blended in (weight ``1 - _LEXICAL_WEIGHT``), so
        a semantically-relevant lesson with no shared tokens still ranks. Empty
        query or any embedding failure falls back to lexical (or a flat 1.0 with
        no query). Never raises — ranking is an optimization, not correctness."""
        if not query:
            return {le.id: 1.0 for le in lessons}
        q = _tokens(query)
        lexical = {le.id: _similarity(q, le.tokens()) for le in lessons}
        dense = self._dense_relevance(query, lessons)
        out = {}
        for le in lessons:
            if dense is not None:
                blend = (1 - _LEXICAL_WEIGHT) * dense[
                    le.id
                ] + _LEXICAL_WEIGHT * lexical[le.id]
            else:
                blend = lexical[le.id]
            out[le.id] = 1.0 + 2.0 * blend
        return out

    def _dense_relevance(self, query: str, lessons: List[Lesson]):
        """Map id -> cosine relevance in [0, 1] via the embedder, or None on any
        failure (caller then uses lexical alone). Vectors are cached on disk."""
        if self.embedder is None:
            return None
        try:
            from misterdev.core.economics.embeddings import (
                EmbeddingCache,
                cosine_similarity,
            )

            cache = EmbeddingCache(
                self.path.with_name("lesson_embeddings.json"),
                getattr(self.embedder, "model", "unknown"),
            )
            texts = [query] + [le.text for le in lessons]
            need = [t for t in dict.fromkeys(texts) if cache.get(t) is None]
            if need:
                cache.put_many(dict(zip(need, self.embedder.embed(need))))
            qv = cache.get(query)
            if qv is None:
                return None
            return {
                le.id: (cosine_similarity(qv, cache.get(le.text) or []) + 1.0) / 2.0
                for le in lessons
            }
        except Exception as e:  # embedding is best-effort; fall back to lexical
            logger.debug(f"Dense lesson relevance unavailable: {e}")
            return None

    def retrieve(self, query: str = "", limit: int = _MAX_INJECT) -> List[str]:
        """The highest-value lesson texts for ``query`` (see :meth:`retrieve_lessons`)."""
        return [le.text for le in self.retrieve_lessons(query, limit)]

    # -- efficacy credit (Tier 3) ------------------------------------------
    def credit(self, injected_ids, outcome: Optional[float]) -> None:
        """Fold a finished run's outcome back onto the lessons it was injected into.

        ``outcome`` is the run's success rate in [0, 1] (completed / attempted);
        ``injected_ids`` are the ids :meth:`retrieve_lessons` returned for that run.
        The delta against the project's trailing baseline updates each injected
        lesson's efficacy EWMA, and a run materially below baseline marks its
        lessons as regression-correlated. First run (no baseline yet) establishes
        the baseline with zero delta, since there is nothing to compare against.

        Best-effort and self-contained: no injected ids, no outcome, or a load
        failure is a no-op — efficacy is a refinement, never a build precondition.
        """
        if outcome is None or not injected_ids:
            return
        ids = {int(i) for i in injected_ids}
        lessons, meta = self._load()
        baseline = meta.get("baseline")
        delta = 0.0 if baseline is None else float(outcome) - float(baseline)
        # Update the trailing baseline toward this run's outcome.
        meta["baseline"] = (
            float(outcome)
            if baseline is None
            else (1 - _OUTCOME_ALPHA) * float(baseline)
            + _OUTCOME_ALPHA * float(outcome)
        )
        regressed = baseline is not None and delta <= -_REGRESS_BAND
        for le in lessons:
            if le.id in ids:
                le.injected += 1
                le.efficacy = (
                    1 - _EFFICACY_BETA
                ) * le.efficacy + _EFFICACY_BETA * delta
                if regressed:
                    le.regress_hits += 1
        self._save(lessons, meta)


def _retention_value(le: "Lesson") -> Tuple[bool, float, int]:
    """Eviction ranking key (higher survives): usable lessons first, then
    efficacy-weighted score, then recency. Quarantined lessons sort last so they
    are the first dropped over the cap, and a proven-helpful lesson (positive
    efficacy) outranks a merely-recurring one of equal raw score."""
    eff = max(-1.0, min(1.0, le.efficacy))
    return (not le.quarantined, le.score * (1.0 + eff), le.run)


def _as_text(rule) -> str:
    """Coerce a rule to text — the LLM sometimes returns JSON objects, not plain
    strings, and those must not crash the store."""
    return rule if isinstance(rule, str) else json.dumps(rule, ensure_ascii=False)


def _best_match(toks: frozenset, lessons: List[Lesson]):
    """The existing lesson most similar to ``toks`` above the dedup threshold, or
    None. Ties resolve to the highest-scoring lesson so reinforcement concentrates
    on the established one."""
    best = None
    best_sim = _DEDUP_THRESHOLD
    for le in lessons:
        sim = _similarity(toks, le.tokens())
        if sim >= best_sim and (
            best is None or le.score > best.score or sim > best_sim
        ):
            best = le
            best_sim = max(best_sim, sim)
    return best
