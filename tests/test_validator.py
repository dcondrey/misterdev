from my_project_orchestrator.core.validator import (
    CodeValidator, CertaintyScorer, StallDetector, _tokenize,
)


def test_code_validator_valid_python():
    valid, err = CodeValidator.validate_code("x = 1 + 2\ndef f(): pass")
    assert valid and err is None


def test_code_validator_invalid_python():
    valid, err = CodeValidator.validate_code("def f(")
    assert not valid and "Syntax error" in err


def test_code_validator_balanced_braces():
    valid, err = CodeValidator.validate_code("fn main() { let x = [1, 2]; }", language="rust")
    assert valid and err is None


def test_code_validator_unbalanced():
    valid, err = CodeValidator.validate_code("fn main() { let x = [1, 2; }", language="rust")
    assert not valid


def test_certainty_high():
    score = CertaintyScorer.compute_score(
        "I have verified that this is correct. Tests pass successfully. The solution is complete."
    )
    assert score > 0.7


def test_certainty_low():
    score = CertaintyScorer.compute_score("Maybe this could work, not sure, possibly wrong.")
    assert score < 0.3


def test_certainty_code_boost():
    score_no_code = CertaintyScorer.compute_score("Here is a solution.")
    score_code = CertaintyScorer.compute_score("Here is a solution.\n```python\nx = 1\n```")
    assert score_code > score_no_code


def test_stall_detector_no_stall():
    sd = StallDetector()
    r1 = sd.push_edit({"a.py": "def foo(): return 1"})
    assert r1 < 0.5


def test_stall_detector_identical():
    sd = StallDetector()
    sd.push_edit({"a.py": "def foo(): return 1"})
    r2 = sd.push_edit({"a.py": "def foo(): return 1"})
    assert r2 > 0.7


def test_stall_detector_similar():
    sd = StallDetector()
    sd.push_edit({"a.py": "def foo(): return 1"})
    r2 = sd.push_edit({"a.py": "def foo(): return 2"})
    assert r2 < 0.8  # similar but not identical


def test_tokenize():
    tokens = _tokenize("hello world_foo bar123 x")
    assert tokens == {"hello", "world_foo", "bar123", "x"}


def test_tokenize_empty():
    assert _tokenize("") == set()
    assert _tokenize("   ") == set()
