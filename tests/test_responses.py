from my_project_orchestrator.llm.responses import LLMResponseParser


def test_tagged_code_block():
    output = "```python:src/main.py\ndef hello():\n    return 42\n```"
    edits = LLMResponseParser.parse_file_edits(output)
    assert "src/main.py" in edits
    assert "return 42" in edits["src/main.py"]


def test_comment_header():
    output = "```python\n# src/utils.py\ndef add(a, b):\n    return a + b\n```"
    edits = LLMResponseParser.parse_file_edits(output)
    assert "src/utils.py" in edits
    assert "return a + b" in edits["src/utils.py"]


def test_preceding_backtick():
    output = "Update `app/config.py`:\n\n```python\nDEBUG = False\n```"
    edits = LLMResponseParser.parse_file_edits(output)
    assert "app/config.py" in edits


def test_unified_diff():
    output = "--- a/src/lib.py\n+++ b/src/lib.py\n@@ -1,2 +1,2 @@\n-old = True\n+new = True\n same = True\n"
    edits = LLMResponseParser.parse_file_edits(output)
    assert "src/lib.py" in edits
    assert "new = True" in edits["src/lib.py"]


def test_tilde_fence():
    output = "~~~~python:tests/test_foo.py\nassert True\n~~~~"
    edits = LLMResponseParser.parse_file_edits(output)
    assert "tests/test_foo.py" in edits


def test_file_label():
    output = "```\nFile: src/models.py\nclass Foo:\n    pass\n```"
    edits = LLMResponseParser.parse_file_edits(output)
    assert "src/models.py" in edits


def test_multiple_blocks():
    output = "```python:a.py\nx = 1\n```\n\n```python:b.py\ny = 2\n```"
    edits = LLMResponseParser.parse_file_edits(output)
    assert len(edits) == 2
    assert "a.py" in edits and "b.py" in edits


def test_no_path():
    output = "```python\nx = 1\n```"
    edits = LLMResponseParser.parse_file_edits(output)
    assert len(edits) == 0


def test_empty_input():
    assert LLMResponseParser.parse_file_edits("") == {}
    assert LLMResponseParser.parse_file_edits("just text, no code") == {}
