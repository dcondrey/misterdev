"""Reference-implementation digest: read-only structural map for porting."""

import tempfile
from pathlib import Path

import pytest

from misterdev.analyzers.reference_digest import build_reference_digest


def _ref_tree() -> Path:
    """A small multi-file reference tree under a fresh temp dir."""
    root = Path(tempfile.mkdtemp(prefix="ref-src-"))
    (root / "core").mkdir()
    (root / "core" / "engine.py").write_text(
        "class Engine:\n"
        "    def start(self):\n        return 1\n\n"
        "def boot():\n    return Engine()\n"
    )
    (root / "util.py").write_text("def helper(x):\n    return x + 1\n")
    return root


def test_digest_names_reference_and_lists_symbols():
    ref = _ref_tree()
    cache = Path(tempfile.mkdtemp(prefix="cache-"))
    digest = build_reference_digest(ref, cache_dir=cache)

    assert ref.name in digest
    assert "Reference implementation to port from" in digest
    # Symbols from both files/dirs appear in the map.
    assert "Engine" in digest
    assert "boot" in digest
    assert "helper" in digest
    assert "core/engine.py" in digest


def test_digest_never_writes_into_reference_tree():
    ref = _ref_tree()
    before = {p.relative_to(ref) for p in ref.rglob("*")}
    cache = Path(tempfile.mkdtemp(prefix="cache-"))

    build_reference_digest(ref, cache_dir=cache)

    after = {p.relative_to(ref) for p in ref.rglob("*")}
    assert before == after, "reference tree must not be mutated"
    assert not (ref / ".orchestrator").exists()
    # The cache landed in the redirected location instead.
    assert (cache / "reference_topography_cache.json").exists()


def test_digest_default_cache_dir_stays_off_reference_tree():
    ref = _ref_tree()
    # No cache_dir: a throwaway temp dir is used, never the reference.
    build_reference_digest(ref)
    assert not (ref / ".orchestrator").exists()


def test_digest_missing_path_raises():
    with pytest.raises(ValueError):
        build_reference_digest("/no/such/reference/dir/xyz")


def test_digest_file_path_raises():
    ref = _ref_tree()
    with pytest.raises(ValueError):
        build_reference_digest(ref / "util.py")


def test_digest_rejects_nonpositive_max_chars():
    ref = _ref_tree()
    with pytest.raises(ValueError):
        build_reference_digest(ref, max_chars=0)


def test_digest_empty_tree_returns_header_with_note():
    empty = Path(tempfile.mkdtemp(prefix="ref-empty-"))
    digest = build_reference_digest(empty)
    assert "Reference implementation to port from" in digest
    assert "no source symbols" in digest


def test_digest_truncates_large_map_with_note():
    ref = _ref_tree()
    digest = build_reference_digest(ref, max_chars=40)
    assert "truncated" in digest
    # Header is always present; only the map body is bounded.
    assert "Reference implementation to port from" in digest
