"""Tier 4: semantic retrieval + warm-start, with lexical fallback."""

from dataclasses import dataclass, field
from typing import List

from misterdev.core.learning.warm_start import SolvedTaskIndex
from misterdev.core.planning.lesson_store import LessonStore

# A toy embedder: projects text onto two topic axes (db vs formatting) so two
# texts about the same topic are close even with NO shared tokens — exactly the
# case lexical overlap misses.
_DB = {"database", "connection", "pool", "persistence", "layer", "sql", "query"}
_FMT = {"format", "linter", "style", "whitespace", "indent"}


def _vec(text: str) -> List[float]:
    words = set(text.lower().replace(".", " ").split())
    return [float(len(words & _DB)), float(len(words & _FMT)), 0.1]


@dataclass
class _Embedder:
    model: str = "toy"

    def embed(self, texts):
        return [_vec(t) for t in texts]


def test_semantic_surfaces_relevant_lesson_without_shared_tokens(tmp_path):
    store = LessonStore(tmp_path / "lessons.json", embedder=_Embedder())
    store.record(
        [
            "close the connection pool after each sql query",  # db topic, no query tokens
            "keep whitespace and indent consistent with the linter",  # formatting
        ]
    )
    ranked = store.retrieve_lessons("database persistence layer")
    # The db lesson shares no literal tokens with the query, yet ranks first by
    # meaning. Lexical-only ranking could not do this.
    assert "connection pool" in ranked[0].text


def test_lexical_fallback_without_embedder(tmp_path):
    store = LessonStore(tmp_path / "lessons.json")  # no embedder
    store.record(["run the linter before commit", "handle the null case"])
    ranked = store.retrieve_lessons("linter")
    assert "linter" in ranked[0].text


def test_semantic_failure_degrades_to_lexical(tmp_path):
    @dataclass
    class _Broken:
        model: str = "broken"

        def embed(self, texts):
            raise RuntimeError("embedding backend down")

    store = LessonStore(tmp_path / "lessons.json", embedder=_Broken())
    store.record(["run the linter before commit", "handle the null case"])
    # Must not raise; falls back to lexical ranking.
    ranked = store.retrieve_lessons("linter")
    assert "linter" in ranked[0].text


# -- warm-start index --------------------------------------------------------


@dataclass
class _Task:
    id: str
    title: str = ""
    description: str = ""
    files_to_modify: List[str] = field(default_factory=list)
    files_to_create: List[str] = field(default_factory=list)
    category: str = "feature"


def test_index_records_and_retrieves_nearest(tmp_path):
    idx = SolvedTaskIndex(tmp_path / "solved.jsonl")
    idx.record(
        [
            _Task(
                "T-1", title="add JSON export to the report", files_to_modify=["r.py"]
            ),
            _Task("T-2", title="fix the flaky timeout retry", files_to_modify=["t.py"]),
        ]
    )
    near = idx.nearest("add JSON output format", k=1)
    assert near and near[0].task_id == "T-1"


def test_index_dedupes_exact_resolve_not_numbered_variants(tmp_path):
    idx = SolvedTaskIndex(tmp_path / "solved.jsonl")
    assert idx.record([_Task("T-1", title="add JSON export")]) == 1
    # Re-solving the SAME description refreshes in place, no duplicate.
    assert idx.record([_Task("T-1", title="add JSON export")]) == 0
    assert len(idx.load()) == 1
    # Tasks differing only by a NUMBER are distinct work and must NOT collapse
    # (the old error-fingerprint dedup wrongly merged them).
    assert idx.record([_Task("T-2", title="add migration 0001 for users")]) == 1
    assert idx.record([_Task("T-3", title="add migration 0002 for orders")]) == 1
    assert len(idx.load()) == 3


def test_index_retrieval_survives_repeated_task_ids(tmp_path):
    # Task ids like "T-001" repeat across builds; the index must not collapse
    # distinct solved tasks that happen to share an id.
    idx = SolvedTaskIndex(tmp_path / "solved.jsonl")
    idx.record([_Task("T-001", title="add JSON export to report")])
    idx.record([_Task("T-001", title="fix flaky timeout retry logic")])
    assert len(idx.load()) == 2
    near = idx.nearest("timeout retry", k=1)
    assert near and "timeout" in near[0].description


def test_index_infers_language_from_files(tmp_path):
    idx = SolvedTaskIndex(tmp_path / "solved.jsonl")
    idx.record([_Task("T-1", title="do a thing", files_to_modify=["src/main.rs"])])
    assert idx.load()[0].language == "rust"


def test_context_block_is_injectable_or_empty(tmp_path):
    idx = SolvedTaskIndex(tmp_path / "solved.jsonl")
    assert idx.context("anything") == ""  # empty index -> no block
    idx.record([_Task("T-1", title="add JSON export")])
    block = idx.context("json output")
    assert "warm-start" in block.lower()
    assert "JSON export" in block


def test_missing_index_is_empty(tmp_path):
    idx = SolvedTaskIndex(tmp_path / "nope.jsonl")
    assert idx.load() == []
    assert idx.nearest("x") == []
    assert idx.context("x") == ""
