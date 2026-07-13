"""Localizer: find target files/functions for a bare query over the symbol graph."""

from misterdev.core.context.localizer import (
    LocalizationHit,
    localize,
    localize_files,
)
from misterdev.core.context.topography.nodes import SymbolNode


def _sym(name, file, content="", kind="function", out=(), inc=()):
    s = SymbolNode(name, file, kind, 1, 5, content)
    s.outgoing_calls = set(out)
    s.incoming_calls = set(inc)
    return s


def _graph(*syms):
    return {f"{s.file_path}:{s.name}": s for s in syms}


def test_matches_by_name_with_camel_and_snake_split():
    g = _graph(
        _sym("getUserName", "auth/user.py"),
        _sym("compute_tax", "billing/tax.py"),
        _sym("unrelated", "misc.py"),
    )
    hits = localize("the user name is wrong", g)
    assert hits[0].symbol == "getUserName"  # camelCase split matches "user"/"name"
    assert "unrelated" not in [h.symbol for h in hits]


def test_exact_name_hit_outranks_partial():
    g = _graph(
        _sym("parse", "a.py"),
        _sym("parser_helper", "b.py"),
    )
    hits = localize("parse the input", g)
    assert hits[0].symbol == "parse"  # exact whole-name match wins


def test_file_rollup_picks_the_owning_file():
    g = _graph(
        _sym("rate_limit", "lib/rate_limit.py"),
        _sym("apply_rate_limit", "lib/rate_limit.py"),
        _sym("render", "lib/html.py"),
    )
    files = localize_files("add rate limit to requests", g)
    assert files[0][0] == "lib/rate_limit.py"  # two matching symbols -> top file


def test_call_graph_expansion_surfaces_a_neighbor():
    # 'handler' matches the query; 'validate' does not lexically, but 'handler'
    # calls it, so expansion should surface it as a candidate.
    g = _graph(
        _sym("request_handler", "server.py", out=("validate",)),
        _sym("validate", "checks.py"),
    )
    hits = localize("the request handler crashes", g, expand=True)
    symbols = [h.symbol for h in hits]
    assert "request_handler" in symbols and "validate" in symbols
    no_expand = [
        h.symbol for h in localize("the request handler crashes", g, expand=False)
    ]
    assert "validate" not in no_expand  # only surfaces via the call edge


def test_no_terms_or_no_match_returns_empty():
    g = _graph(_sym("foo", "a.py"))
    assert localize("the a of to", g) == []  # all stopwords
    assert localize("zzzznomatch", g) == []


class _FakeRanker:
    """Stub matching the real SemanticRanker.top_k(query, {id: text}, k) contract."""

    def __init__(self, order):
        self.order = order  # candidate keys, best-first
        self.calls = 0

    def top_k(self, query, candidates, k):
        self.calls += 1
        return [key for key in self.order if key in candidates][:k]


def test_ranker_selects_topk_from_lexical_overfetch():
    # Both names match "alpha beta" lexically; the ranker picks which one wins the
    # single slot. top_k reorders only when selecting a strict subset, so top_k=1.
    g = _graph(_sym("alpha", "a.py", content="x"), _sym("beta", "b.py", content="y"))
    base = [h.symbol for h in localize("alpha beta", g, top_k=1, ranker=None)]
    assert base == ["alpha"]  # lexical tie broken by key order
    ranker = _FakeRanker(order=["b.py:beta", "a.py:alpha"])
    ranked = [h.symbol for h in localize("alpha beta", g, top_k=1, ranker=ranker)]
    assert ranked == ["beta"] and ranker.calls == 1


def test_semantic_fallback_rescues_vocabulary_mismatch():
    # The issue's words ("login is slow") share NO token with the target symbol
    # `authenticate`, so lexical scoring is empty. A semantic ranker that knows the
    # mapping rescues it; without a ranker the query localizes to nothing.
    g = _graph(
        _sym("authenticate", "auth.py", content="verify user credentials session"),
        _sym("render_page", "html.py", content="template html output"),
    )
    assert localize("login is slow", g, ranker=None) == []  # pure lexical miss
    ranker = _FakeRanker(order=["auth.py:authenticate"])
    hits = localize("login is slow", g, ranker=ranker)
    assert [h.symbol for h in hits] == ["authenticate"]
    assert hits[0].score > 0  # synthetic descending score, usable for file rollup


def test_semantic_fallback_not_used_when_lexical_is_strong():
    # A confident name-level hit must NOT trigger the fallback (no wasted embed).
    g = _graph(_sym("authenticate", "auth.py"), _sym("render", "html.py"))
    ranker = _FakeRanker(order=["html.py:render"])
    hits = localize("authenticate the user", g, ranker=ranker)
    assert hits[0].symbol == "authenticate" and ranker.calls == 0


def test_semantic_fallback_degrades_when_ranker_raises():
    g = _graph(_sym("authenticate", "auth.py"))

    class _Boom:
        def top_k(self, *a):
            raise RuntimeError("embedding backend down")

    # Fallback swallows the error; a pure lexical miss still returns empty, no raise.
    assert localize("login is slow", g, ranker=_Boom()) == []


def test_hit_is_structured():
    g = _graph(_sym("target", "pkg/mod.py", kind="method"))
    hit = localize("target", g)[0]
    assert isinstance(hit, LocalizationHit)
    assert hit.file_path == "pkg/mod.py" and hit.kind == "method" and hit.score > 0
