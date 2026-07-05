from misterdev.core.context.scratchpad import Scratchpad


def test_record_and_query():
    sp = Scratchpad()
    sp.record("pattern", "use thiserror", "T-001", files=["src/lib.rs"])
    sp.record("env_quirk", "needs --features test", "T-002", tags=["test"])

    assert len(sp) == 2
    assert len(sp.query(files=["src/lib.rs"])) == 1
    assert len(sp.query(tags=["test"])) == 1
    assert len(sp.query()) == 2


def test_query_by_category():
    sp = Scratchpad()
    sp.record("pattern", "d1", "T-1")
    sp.record("pitfall", "d2", "T-2")
    assert len(sp.query(category="pattern")) == 1
    assert len(sp.query(category="pitfall")) == 1
    assert len(sp.query(category="nonexistent")) == 0


def test_format_context_empty():
    sp = Scratchpad()
    assert sp.format_context() == ""


def test_format_context_cap():
    sp = Scratchpad()
    for i in range(50):
        sp.record("pattern", f"discovery {i}", f"T-{i}")
    ctx = sp.format_context(max_entries=5)
    assert ctx.count("- [") == 5


def test_format_context_content():
    sp = Scratchpad()
    sp.record("convention", "use parking_lot", "T-1", files=["src/core.rs"])
    ctx = sp.format_context(files=["src/core.rs"])
    assert "parking_lot" in ctx
    assert "[convention]" in ctx
