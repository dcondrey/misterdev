import pytest

from my_project_orchestrator.llm.responses import (
    EditConflictError,
    LLMResponseParser,
    SearchReplaceEdit,
    apply_search_replace,
)


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


def test_unified_diff_multi_hunk_rejected():
    # A partial multi-hunk diff omits the unchanged regions between hunks, so
    # rebuilding the file from it would truncate it. Must NOT be parsed as a
    # whole-file edit (returns nothing -> caller falls back / retries).
    output = (
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a = 1\n"
        "+a = 2\n"
        "@@ -50,1 +50,1 @@\n"
        "-z = 9\n"
        "+z = 10\n"
    )
    assert LLMResponseParser.parse_file_edits(output) == {}


def test_unified_diff_not_starting_at_line_one_rejected():
    # A single hunk that starts deep in the file is also partial -> rejected.
    output = (
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -40,2 +40,2 @@\n"
        "-old = True\n"
        "+new = True\n"
        " same = True\n"
    )
    assert LLMResponseParser.parse_file_edits(output) == {}


def test_unified_diff_new_file_accepted():
    # A brand-new file is a single hunk from line 1 -> safe to reconstruct.
    output = (
        "--- /dev/null\n"
        "+++ b/src/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+x = 1\n"
        "+y = 2\n"
    )
    edits = LLMResponseParser.parse_file_edits(output)
    assert edits.get("src/new.py") == "x = 1\ny = 2"


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


# --- surgical SEARCH/REPLACE parsing & application --------------------------


def test_parse_search_replace_single_hunk():
    output = (
        "```rust:src/engine.rs\n"
        "<<<<<<< SEARCH\n"
        "let x = 1;\n"
        "=======\n"
        "let x = 2;\n"
        ">>>>>>> REPLACE\n"
        "```"
    )
    edits = LLMResponseParser.parse_search_replace_blocks(output)
    assert len(edits) == 1
    assert edits[0].path == "src/engine.rs"
    assert edits[0].search == "let x = 1;"
    assert edits[0].replace == "let x = 2;"


def test_parse_search_replace_multiple_hunks_same_file():
    output = (
        "```rust:src/engine.rs\n"
        "<<<<<<< SEARCH\nlet a = 1;\n=======\nlet a = 9;\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nlet b = 2;\n=======\nlet b = 8;\n>>>>>>> REPLACE\n"
        "```"
    )
    edits = LLMResponseParser.parse_search_replace_blocks(output)
    assert len(edits) == 2
    assert {e.path for e in edits} == {"src/engine.rs"}


def test_parse_search_replace_bare_path_line():
    output = "src/lib.rs\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"
    edits = LLMResponseParser.parse_search_replace_blocks(output)
    assert len(edits) == 1
    assert edits[0].path == "src/lib.rs"


def test_parse_search_replace_none_when_no_markers():
    output = "```python:a.py\nx = 1\n```"
    assert LLMResponseParser.parse_search_replace_blocks(output) == []


def test_parse_search_replace_csharp_path():
    # .cs must be recognized as a path or the edit silently fails to parse.
    output = (
        "```csharp:clients/windows/Engine.cs\n"
        "<<<<<<< SEARCH\npublic void Start() {}\n"
        "=======\npublic void Start() { Init(); }\n>>>>>>> REPLACE\n```"
    )
    edits = LLMResponseParser.parse_search_replace_blocks(output)
    assert len(edits) == 1
    assert edits[0].path == "clients/windows/Engine.cs"


def test_parse_whole_file_csharp_and_xaml_paths():
    cs = "```csharp:App.cs\nclass A {}\n```"
    assert "App.cs" in LLMResponseParser.parse_file_edits(cs)
    xaml = "```xml:MainWindow.xaml\n<Window/>\n```"
    assert "MainWindow.xaml" in LLMResponseParser.parse_file_edits(xaml)


def test_apply_search_replace_applies_hunk_in_large_file():
    original = "\n".join(f"line {i}" for i in range(5000))
    edit = SearchReplaceEdit(path="big.rs", search="line 4999", replace="LAST")
    result = apply_search_replace(original, [edit])
    assert result.endswith("LAST")
    assert "line 4998" in result
    assert result.count("\n") == original.count("\n")


def test_apply_search_replace_not_found_raises():
    with pytest.raises(EditConflictError, match="not found"):
        apply_search_replace("real content", [SearchReplaceEdit("f.rs", "absent", "x")])


def test_apply_search_replace_ambiguous_raises():
    with pytest.raises(EditConflictError, match="matches 2"):
        apply_search_replace("dup\ndup\n", [SearchReplaceEdit("f.rs", "dup", "x")])


def test_apply_search_replace_empty_search_creates_new_file():
    result = apply_search_replace("", [SearchReplaceEdit("new.rs", "", "fn main() {}")])
    assert result == "fn main() {}"


def test_apply_search_replace_empty_search_on_existing_raises():
    with pytest.raises(EditConflictError, match="empty SEARCH"):
        apply_search_replace("existing", [SearchReplaceEdit("f.rs", "", "wipe")])


def test_apply_search_replace_tolerates_trailing_whitespace():
    original = "fn main() {\n    let x = 1;   \n}\n"  # trailing spaces on disk
    edit = SearchReplaceEdit("f.rs", "    let x = 1;", "    let x = 2;")
    result = apply_search_replace(original, [edit])
    assert "let x = 2;" in result
    assert "let x = 1;" not in result


def test_apply_search_replace_tolerates_crlf():
    original = "a\r\nb\r\nc\r\n"  # CRLF on disk, LF in the SEARCH block
    edit = SearchReplaceEdit("f.rs", "b", "B")
    result = apply_search_replace(original, [edit])
    assert "B" in result


def test_apply_search_replace_tolerant_still_rejects_ambiguous():
    original = "x = 1 \nx = 1\n"  # two near-identical lines (trailing-space drift)
    with pytest.raises(EditConflictError, match="2 locations"):
        apply_search_replace(original, [SearchReplaceEdit("f.rs", "x = 1", "y")])


def test_apply_search_replace_tolerates_wrong_indent_and_reindents():
    # File uses 4-space indent; model wrote SEARCH and REPLACE at 2 spaces.
    original = "fn f() {\n    let total = compute();\n}\n"
    edit = SearchReplaceEdit(
        "f.rs", "  let total = compute();", "  let total = compute() + 1;"
    )
    result = apply_search_replace(original, [edit])
    # matched despite indent drift, and re-indented to the file's 4 spaces
    assert "    let total = compute() + 1;" in result
    assert "compute();" not in result


def test_apply_search_replace_sequential_hunks():
    original = "a = 1\nb = 2\nc = 3\n"
    edits = [
        SearchReplaceEdit("f.rs", "a = 1", "a = 9"),
        SearchReplaceEdit("f.rs", "c = 3", "c = 7"),
    ]
    result = apply_search_replace(original, edits)
    assert "a = 9" in result and "c = 7" in result and "b = 2" in result
