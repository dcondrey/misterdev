"""Held-out oracle: reject an acceptance test that passes against a stub."""

from misterdev.core.execution.outcomes import GREEN, RED, SKIP
from misterdev.core.verification.held_out_oracle import (
    check_oracle,
    looks_trivial,
    stub_python,
)

_OLD = "def classify(score):\n    return 'x'\n"
_NEW = "def classify(score):\n    if score >= 60:\n        return 'pass'\n    return 'fail'\n"


def test_stub_python_blanks_changed_function():
    stub = stub_python(_NEW, {1, 2, 3})
    assert "raise NotImplementedError" in stub
    assert "return 'pass'" not in stub


def test_stub_python_skips_untouched_function():
    src = "def a():\n    return 1\ndef b():\n    return 2\n"
    stub = stub_python(src, {1})  # only a() changed
    assert stub.count("raise NotImplementedError") == 1
    assert "return 2" in stub  # b() untouched


def test_stub_python_none_when_no_changed_function():
    assert stub_python(_NEW, set()) is None


def test_real_oracle_passes_when_stub_fails(tmp_path):
    (tmp_path / "m.py").write_text(_NEW)

    def runner(cmd, timeout):
        # A REAL test: fails on the stub (NotImplementedError), passes on the fix.
        return "raise NotImplementedError" not in (tmp_path / "m.py").read_text(), ""

    res = check_oracle(tmp_path, "m.py", _OLD, _NEW, "pytest", runner=runner)
    assert res.status == GREEN
    assert (tmp_path / "m.py").read_text() == _NEW  # restored


def test_trivial_oracle_flagged_when_stub_passes(tmp_path):
    (tmp_path / "m.py").write_text(_NEW)

    def runner(cmd, timeout):
        return True, ""  # passes no matter what -> trivial oracle

    res = check_oracle(tmp_path, "m.py", _OLD, _NEW, "pytest", runner=runner)
    assert res.status == RED
    assert "trivial oracle" in res.reason
    assert (tmp_path / "m.py").read_text() == _NEW  # restored even on RED


def test_skips_non_python_and_no_command(tmp_path):
    (tmp_path / "m.rs").write_text("fn f() {}")
    assert (
        check_oracle(
            tmp_path, "m.rs", "a", "b", "cargo test", lambda c, t: (True, "")
        ).status
        == SKIP
    )
    assert (
        check_oracle(tmp_path, "m.py", _OLD, _NEW, "", lambda c, t: (True, "")).status
        == SKIP
    )


def test_looks_trivial_smell():
    assert looks_trivial("def t():\n    assert isinstance(r, str)\n")
    assert looks_trivial("def t():\n    assert r is not None\n")
    assert looks_trivial("def t():\n    pass\n")  # no assertion at all
    assert not looks_trivial("def t():\n    assert classify(60) == 'pass'\n")
