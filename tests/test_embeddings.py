import tempfile
from pathlib import Path

import pytest

from my_project_orchestrator.core.embeddings import (
    EmbeddingCache,
    SemanticRanker,
    cosine_similarity,
    pick_embedding_model,
)


class FakeEmbedder:
    """Embeds by keyword overlap into a tiny fixed vector space."""

    model = "fake/embed"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        # 3-dim vector: counts of marker words a/b/c.
        return [[t.count("a"), t.count("b"), t.count("c")] for t in texts]


def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([], [1]) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_ranker_orders_by_relevance():
    ranker = SemanticRanker(FakeEmbedder())
    # Query leans 'a'; candidates vary. Expect the 'a'-heavy ones first.
    candidates = {"x": "c c c", "y": "a a a", "z": "a b"}
    top = ranker.top_k("a a", candidates, 2)
    assert top[0] == "y"
    assert set(top) == {"y", "z"}


def test_ranker_returns_all_when_k_exceeds_candidates():
    embedder = FakeEmbedder()
    ranker = SemanticRanker(embedder)
    ids = ranker.top_k("a", {"x": "a", "y": "b"}, 5)
    assert set(ids) == {"x", "y"}
    # No embedding work needed when everything fits.
    assert embedder.calls == []


def test_ranker_falls_back_on_embed_failure():
    class Boom:
        model = "boom"

        def embed(self, texts):
            raise RuntimeError("embeddings down")

    ranker = SemanticRanker(Boom())
    ids = ranker.top_k("q", {"x": "1", "y": "2", "z": "3"}, 2)
    # Graceful: arbitrary first-k, never raises.
    assert len(ids) == 2
    assert set(ids).issubset({"x", "y", "z"})


def test_cache_roundtrip_and_reuse():
    with tempfile.TemporaryDirectory() as d:
        embedder = FakeEmbedder()
        cache = EmbeddingCache(
            Path(d) / ".orchestrator" / "embeddings.json", embedder.model
        )
        ranker = SemanticRanker(embedder, cache)
        cands = {"x": "a a", "y": "b"}
        ranker.top_k("a", cands, 1)
        first_calls = len(embedder.calls)
        # A fresh ranker over the same cache re-embeds only the new query.
        ranker2 = SemanticRanker(
            embedder,
            EmbeddingCache(
                Path(d) / ".orchestrator" / "embeddings.json", embedder.model
            ),
        )
        embedder.calls.clear()
        ranker2.top_k("a", cands, 1)
        # The query "a" plus candidate texts were cached; only a cache miss embeds.
        embedded = [t for batch in embedder.calls for t in batch]
        assert "a a" not in embedded and "b" not in embedded  # candidates cached
        assert first_calls >= 1


def test_pick_embedding_model_honors_explicit_config():
    assert pick_embedding_model("vendor/pinned") == "vendor/pinned"


def test_pick_embedding_model_prefers_cheapest_free():
    models = [
        {"id": "paid/big", "pricing": {"prompt": "0.0001"}},
        {"id": "free/q", "pricing": {"prompt": "0"}},
        {"id": "paid/small", "pricing": {"prompt": "0.00002"}},
    ]
    assert pick_embedding_model("", fetcher=lambda: models) == "free/q"


def test_pick_embedding_model_none_on_discovery_failure():
    def boom():
        raise RuntimeError("no network")

    assert pick_embedding_model("", fetcher=boom) is None


def test_pick_prefers_code_model_among_equal_price():
    models = [
        {"id": "vendor/text-embed", "pricing": {"prompt": "0"}},
        {"id": "vendor/code-embed", "pricing": {"prompt": "0"}},
    ]
    assert pick_embedding_model("", fetcher=lambda: models) == "vendor/code-embed"


def test_pick_price_beats_preference():
    models = [
        {"id": "vendor/code-embed", "pricing": {"prompt": "0.001"}},
        {"id": "vendor/text-embed", "pricing": {"prompt": "0"}},
    ]
    # Cheaper non-code wins: price dominates the preference tiebreaker.
    assert pick_embedding_model("", fetcher=lambda: models) == "vendor/text-embed"


def test_tokenize_splits_identifiers():
    from my_project_orchestrator.core.embeddings import _tokenize

    toks = _tokenize("def getUserName(user_id): db_lookup")
    for expected in ("get", "user", "name", "id", "db", "lookup", "def"):
        assert expected in toks
    assert "a" not in toks  # single chars dropped


def test_lexical_overlap_recall():
    from my_project_orchestrator.core.embeddings import _lexical_overlap, _tokenize

    q = _tokenize("parse config file")
    assert _lexical_overlap(q, _tokenize("def parse_config(): ...")) > 0
    assert _lexical_overlap(q, _tokenize("unrelated stuff here")) == 0.0
    assert _lexical_overlap(set(), _tokenize("anything")) == 0.0


def test_lexical_only_ranker_without_embedder():
    ranker = SemanticRanker(embedder=None)
    cands = {
        "a": "def parse_config(path): ...",
        "b": "def render_html(): ...",
        "c": "x = 1",
    }
    assert ranker.top_k("parse the config", cands, 1) == ["a"]


def test_hybrid_lexical_breaks_uniform_dense():
    class FlatEmbedder:
        model = "flat"

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]  # uniform dense -> lexical decides

    ranker = SemanticRanker(FlatEmbedder(), lexical_weight=0.5)
    cands = {"match": "validate_token user", "other": "zzz qqq"}
    assert ranker.top_k("validate token", cands, 1) == ["match"]
