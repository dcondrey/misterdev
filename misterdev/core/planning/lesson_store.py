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
from typing import List, Tuple

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_MAX_LESSONS = 40  # hard cap on stored lessons (bounds file + injection)
_MAX_INJECT = 12  # lessons injected per build, relevance-ranked
_DEDUP_THRESHOLD = 0.6  # overlap coefficient above which two rules are "the same"
_REINFORCE = 1.0  # score gain when a lesson recurs
_DECAY = 0.9  # multiplicative decay on lessons not reinforced this run
_MIN_SCORE = 0.15  # below this a decayed lesson is dropped as noise
_NEW_SCORE = 1.0  # starting score for a freshly-learned lesson

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

    def tokens(self) -> frozenset:
        return _tokens(self.text)


class LessonStore:
    """Persistent scored lesson memory backed by a single JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    # -- persistence -------------------------------------------------------
    def _load(self) -> Tuple[List[Lesson], int]:
        """Return (lessons, run_counter). Migrates a legacy list[str] file and
        degrades to empty on a missing/corrupt file (learning is best-effort)."""
        if not self.path.exists():
            return [], 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return [], 0
        if isinstance(raw, list):
            # Legacy plain-string format: seed each as a fresh lesson.
            return [Lesson(text=_as_text(r)) for r in raw], 0
        if isinstance(raw, dict):
            run = int(raw.get("run", 0))
            lessons = []
            for item in raw.get("lessons", []):
                if isinstance(item, dict) and item.get("text"):
                    lessons.append(
                        Lesson(
                            text=str(item["text"]),
                            score=float(item.get("score", _NEW_SCORE)),
                            hits=int(item.get("hits", 1)),
                            run=int(item.get("run", 0)),
                        )
                    )
                elif isinstance(item, str):
                    lessons.append(Lesson(text=item))
            return lessons, run
        return [], 0

    def _save(self, lessons: List[Lesson], run: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": run,
            "lessons": [{**asdict(le), "score": round(le.score, 4)} for le in lessons],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- learning ----------------------------------------------------------
    def record(self, new_rules: List) -> int:
        """Fold newly-audited rules into the store: reinforce matches, add the
        rest, decay the untouched, drop noise, evict by value. Returns the number
        of genuinely new (non-duplicate) lessons added."""
        lessons, run = self._load()
        run += 1

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
                lessons.append(Lesson(text=text, score=_NEW_SCORE, hits=1, run=run))
                added += 1

        # Drop faded noise, then evict by value (lowest score first) over the cap.
        lessons = [le for le in lessons if le.score >= _MIN_SCORE]
        lessons.sort(key=lambda le: (le.score, le.run), reverse=True)
        lessons = lessons[:_MAX_LESSONS]

        self._save(lessons, run)
        logger.info(
            f"Lessons: {added} new, {len(new_rules) - added} reinforced; "
            f"{len(lessons)} retained (run {run})."
        )
        return added

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, query: str = "", limit: int = _MAX_INJECT) -> List[str]:
        """Return the highest-value lessons, boosted by relevance to ``query``.

        With a query, ranking is ``score * (1 + 2*overlap(query, lesson))`` so a
        proven lesson relevant to the current goal ranks first; without one, by
        score alone. Newest breaks ties so a fresh lesson isn't buried."""
        lessons, _ = self._load()
        if not lessons:
            return []
        q = _tokens(query) if query else frozenset()

        def rank(le: Lesson) -> Tuple[float, int]:
            weight = (
                le.score * (1.0 + 2.0 * _similarity(q, le.tokens())) if q else le.score
            )
            return (weight, le.run)

        lessons.sort(key=rank, reverse=True)
        return [le.text for le in lessons[:limit]]


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
