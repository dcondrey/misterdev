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


def test_ranker_reorders_shortlist():
    g = _graph(_sym("alpha", "a.py", content="x"), _sym("beta", "b.py", content="y"))

    # A ranker that reverses the lexical order.
    def ranker(query, snippets):
        return list(range(len(snippets)))[::-1]

    base = [h.symbol for h in localize("alpha beta", g, ranker=None)]
    ranked = [h.symbol for h in localize("alpha beta", g, ranker=ranker)]
    assert ranked == base[::-1]


def test_hit_is_structured():
    g = _graph(_sym("target", "pkg/mod.py", kind="method"))
    hit = localize("target", g)[0]
    assert isinstance(hit, LocalizationHit)
    assert hit.file_path == "pkg/mod.py" and hit.kind == "method" and hit.score > 0
