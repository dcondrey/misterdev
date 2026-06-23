"""Incremental, content-hash-keyed caching of the symbol graph.

These gate the cache as a pure optimization: symbols built WITH the cache must be
byte-identical to a from-scratch parse, a second build of an unchanged tree must
re-parse nothing, an edit must re-parse exactly the changed file, a deleted file
must drop out, and any cache corruption must degrade silently to a full parse.
"""

import json
from pathlib import Path

import pytest

from my_project_orchestrator.core.topography import (
    SymbolGraph,
    _get_ts_parsers,
    _TopographyCache,
)


def _fingerprint(graph: SymbolGraph):
    """A stable, comparable snapshot of every symbol's parse-derived fields."""
    return {
        key: (s.name, s.file_path, s.kind, s.start_line, s.end_line, s.content)
        for key, s in graph.symbols.items()
    }


def _write_fixture(root: Path):
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "app.py").write_text(
        "class App:\n    def run(self):\n        helper()\n\n"
        "def helper():\n    return 1\n",
        encoding="utf-8",
    )
    (root / "src" / "lib.rs").write_text(
        "pub fn one() {}\nstruct S { x: i32 }\nfn two() { one(); }\n",
        encoding="utf-8",
    )
    (root / "src" / "ui.ts").write_text(
        "export interface Cmd { n: string }\n"
        "export const run = (x: number) => x + 1;\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _require_grammars():
    if not _get_ts_parsers():
        pytest.skip("no tree-sitter grammars installed")


def _build_with_counter(root: Path):
    """Build a graph, counting how many files actually hit the parser."""
    graph = SymbolGraph(root)
    calls = {"n": 0, "files": []}
    real = graph._parse_file

    def counting(file_path, lang, **kw):
        calls["n"] += 1
        calls["files"].append(str(file_path.relative_to(root)))
        return real(file_path, lang, **kw)

    graph._parse_file = counting
    graph.build()
    return graph, calls


def test_cached_symbols_identical_to_from_scratch(tmp_path):
    _write_fixture(tmp_path)

    # From-scratch parse with the cache disabled by pointing it at a nonexistent
    # path and discarding writes: simulate "no cache" via a private subclass.
    scratch = SymbolGraph(tmp_path)
    scratch.cache_path = tmp_path / ".orchestrator" / "scratch_only.json"
    scratch.build()
    scratch_fp = _fingerprint(scratch)

    # Real cache path. First build populates the cache and parses everything.
    g1, c1 = _build_with_counter(tmp_path)
    assert c1["n"] >= 1
    assert _fingerprint(g1) == scratch_fp

    # Second build serves entirely from cache: identical symbols, zero parses.
    g2, c2 = _build_with_counter(tmp_path)
    assert _fingerprint(g2) == scratch_fp
    assert c2["n"] == 0, f"expected 0 re-parses, got {c2['files']}"


def test_edit_reparses_only_changed_file(tmp_path):
    _write_fixture(tmp_path)
    _build_with_counter(tmp_path)  # warm the cache

    # Edit exactly one file; its symbols must change, others stay intact.
    (tmp_path / "src" / "app.py").write_text(
        "class App:\n    def run(self):\n        pass\n\n"
        "def renamed_helper():\n    return 2\n",
        encoding="utf-8",
    )

    # Reference lib.rs symbols from an independent from-scratch build.
    ref = SymbolGraph(tmp_path)
    ref.cache_path = tmp_path / ".orchestrator" / "ref_only.json"
    ref.build()
    before_rs = {k: v for k, v in _fingerprint(ref).items() if v[1] == "src/lib.rs"}

    g2, c2 = _build_with_counter(tmp_path)
    assert c2["n"] == 1, f"expected exactly 1 re-parse, got {c2['files']}"
    assert c2["files"] == ["src/app.py"]

    fp = _fingerprint(g2)
    # New symbol present, old one gone.
    assert any(v[0] == "renamed_helper" for v in fp.values())
    assert not any(v[0] == "helper" for v in fp.values())
    # Other files' symbols untouched.
    rs_now = {k: v for k, v in fp.items() if v[1] == "src/lib.rs"}
    assert rs_now == before_rs


def test_deleted_file_symbols_disappear(tmp_path):
    _write_fixture(tmp_path)
    _build_with_counter(tmp_path)

    (tmp_path / "src" / "lib.rs").unlink()

    g2, c2 = _build_with_counter(tmp_path)
    fp = _fingerprint(g2)
    assert not any(v[1] == "src/lib.rs" for v in fp.values())
    assert c2["n"] == 0  # the surviving files are still cache hits

    # The pruned entry is gone from the persisted cache too.
    cache = _TopographyCache(tmp_path / ".orchestrator" / "topography_cache.json")
    cache.load()
    assert "src/lib.rs" not in cache.entries


def test_new_file_parses_without_touching_others(tmp_path):
    _write_fixture(tmp_path)
    _build_with_counter(tmp_path)

    (tmp_path / "src" / "extra.py").write_text(
        "def brand_new():\n    return 7\n", encoding="utf-8"
    )

    g2, c2 = _build_with_counter(tmp_path)
    assert c2["files"] == ["src/extra.py"]
    assert any(v[0] == "brand_new" for v in _fingerprint(g2).values())


def test_corrupt_cache_degrades_to_full_parse(tmp_path):
    _write_fixture(tmp_path)
    scratch = SymbolGraph(tmp_path)
    scratch.cache_path = tmp_path / ".orchestrator" / "scratch_only.json"
    scratch.build()
    expected = _fingerprint(scratch)

    cache_file = tmp_path / ".orchestrator" / "topography_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{ this is not valid json ]]", encoding="utf-8")

    g, c = _build_with_counter(tmp_path)
    # Corrupt cache => every file re-parsed, symbols still correct, no raise.
    assert _fingerprint(g) == expected
    assert c["n"] >= 1
    # And the build rewrote a valid cache for next time.
    g2, c2 = _build_with_counter(tmp_path)
    assert c2["n"] == 0
    assert _fingerprint(g2) == expected


def test_unreadable_cache_path_does_not_crash(tmp_path):
    _write_fixture(tmp_path)
    # A directory where the cache file should be: read/write both fail, build
    # must still produce correct symbols without raising.
    cache_dir = tmp_path / ".orchestrator"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "topography_cache.json").mkdir()

    scratch = SymbolGraph(tmp_path)
    scratch.cache_path = tmp_path / ".orchestrator" / "scratch_only.json"
    scratch.build()
    expected = _fingerprint(scratch)

    g, c = _build_with_counter(tmp_path)
    assert _fingerprint(g) == expected
    assert c["n"] >= 1


def test_cache_get_rejects_malformed_entry(tmp_path):
    cache = _TopographyCache(tmp_path / "c.json")
    # Wrong-shape entries must never deserialize into a partial symbol set.
    cache.entries["a.py"] = {"key": "k", "symbols": "not-a-list"}
    assert cache.get("a.py", "k") is None
    cache.entries["b.py"] = {"key": "k", "symbols": [{"name": "x"}]}  # missing fields
    assert cache.get("b.py", "k") is None
    # Key mismatch is a clean miss, not a stale serve.
    cache.entries["c.py"] = {"key": "other", "symbols": []}
    assert cache.get("c.py", "k") is None


def test_read_bytes_returns_none_on_unreadable_file(tmp_path):
    graph = SymbolGraph(tmp_path)
    # A directory (not a file) makes read_file raise; must yield None, not crash.
    (tmp_path / "adir").mkdir()
    assert graph._read_bytes(tmp_path / "adir") is None
    assert graph._read_bytes(tmp_path / "missing.py") is None
    # A None content path means _parse_file yields no symbols and never raises.
    assert graph._parse_file(tmp_path / "missing.py", "python") == []


def test_format_version_bump_invalidates(tmp_path):
    _write_fixture(tmp_path)
    _build_with_counter(tmp_path)

    cache_file = tmp_path / ".orchestrator" / "topography_cache.json"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    data["format"] = data["format"] + 999  # simulate an old/foreign format
    cache_file.write_text(json.dumps(data), encoding="utf-8")

    g2, c2 = _build_with_counter(tmp_path)
    # Stale format => discarded => full re-parse, still correct.
    assert c2["n"] >= 1
    assert any(v[0] == "helper" for v in _fingerprint(g2).values())
