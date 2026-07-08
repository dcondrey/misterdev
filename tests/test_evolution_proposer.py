import pytest

from misterdev.core.evolution import (
    Blame,
    LLMProposer,
    build_instruction,
    parse_paths,
    parse_tag,
)

_EDIT = (
    "tag: contract-extraction\n"
    "```python:misterdev/core/context/contracts/extraction.py\n"
    "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n```\n"
    "```rust:misterdev/other.rs\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```\n"
)


def test_parse_paths_extracts_fence_paths_in_order_without_dupes():
    dup = _EDIT + "```python:misterdev/core/context/contracts/extraction.py\nx\n```"
    assert parse_paths(dup) == [
        "misterdev/core/context/contracts/extraction.py",
        "misterdev/other.rs",
    ]


def test_parse_paths_empty_when_no_fences():
    assert parse_paths("just prose, no edits") == []


def test_parse_tag_reads_declared_kind():
    assert parse_tag(_EDIT) == "contract-extraction"
    assert parse_tag("no tag here\n```python:x.py\n```") is None


def test_build_instruction_targets_niche_and_shows_failures():
    blame = Blame(
        niche="rust/wrong_type", failures=8, total=10, examples=["E0308: mismatch"]
    )
    instr = build_instruction(blame, favored_kinds=["prompt"])
    assert "rust/wrong_type" in instr
    assert "8/10" in instr and "80%" in instr
    assert "E0308: mismatch" in instr
    assert "prompt" in instr  # prior steer included
    assert "tag:" in instr  # asks for the kind tag


def test_propose_normalizes_editor_response_into_a_mutation():
    prop = LLMProposer(generate=lambda instr: _EDIT)
    blame = Blame(niche="rust/wrong_type", failures=1, total=1)
    mut = prop.propose(blame)
    assert mut.target == "rust/wrong_type"
    assert mut.paths[0].endswith("extraction.py")
    assert mut.note == "contract-extraction"
    assert mut.patch == _EDIT


def test_propose_defaults_note_to_niche_when_untagged():
    edit = (
        "```python:misterdev/x.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```"
    )
    prop = LLMProposer(generate=lambda instr: edit)
    mut = prop.propose(Blame(niche="go/missing_symbol", failures=1, total=1))
    assert mut.note == "go/missing_symbol"


def test_propose_raises_on_uneditable_response():
    prop = LLMProposer(generate=lambda instr: "I could not find anything to change.")
    with pytest.raises(ValueError):
        prop.propose(Blame(niche="rust", failures=1, total=1))


def test_build_instruction_grounds_in_real_surfaces_and_biases_structural():
    instr = build_instruction(Blame(niche="rust/test_assertion", failures=2, total=2))
    # Grounds the editor in a real editable surface (prevents invented paths).
    assert "misterdev/core/context/guidance/" in instr
    assert "failure_view.py" in instr
    assert "do NOT invent paths" in instr
    # Steers toward a general structural fix, not a task-keyed (overfit) tweak.
    assert "GENERAL mechanism" in instr and "overfit" in instr
    # Tag examples are structural; the old "tag: prompt" nudge is gone.
    assert "tag: guard" in instr
    assert "tag: prompt" not in instr


def test_build_instruction_uses_classified_cause_to_steer_fix_kind():
    blame = Blame(niche="rust/test_assertion", failures=2, total=2)
    blame.cause = "artifact"
    blame.cause_evidence = "a guard rejected the candidate edit"
    instr = build_instruction(blame)
    assert "cause (classified): artifact" in instr.lower()
    assert "guard rejected the candidate edit" in instr
    # Steers at the mechanism for THIS cause (a wrongly-blocking guard/gate).
    assert "must never be rejected" in instr
    # A different cause routes to a different fix.
    blame.cause = "observation"
    blame.cause_evidence = "failure output was truncated"
    assert "observation seam" in build_instruction(blame)


def test_propose_rejects_invented_paths(tmp_path):
    # repo_root has a real package dir but not the invented one the editor names.
    (tmp_path / "misterdev" / "core").mkdir(parents=True)
    invented = (
        "```python:src/prompts/rust_test_assertion.md\n"
        "<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```"
    )
    prop = LLMProposer(generate=lambda instr: invented, repo_root=tmp_path)
    with pytest.raises(ValueError, match="invented paths"):
        prop.propose(Blame(niche="rust/test_assertion", failures=2, total=2))


def test_propose_keeps_grounded_paths_and_drops_invented(tmp_path):
    (tmp_path / "misterdev" / "core").mkdir(parents=True)
    (tmp_path / "misterdev" / "real.py").write_text("x = 1\n")
    mixed = (
        "tag: guard\n"
        "```python:misterdev/real.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n```\n"
        "```python:totally/invented/path.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```\n"
    )
    prop = LLMProposer(generate=lambda instr: mixed, repo_root=tmp_path)
    mut = prop.propose(Blame(niche="rust/test_assertion", failures=2, total=2))
    assert mut.paths == ["misterdev/real.py"]  # invented path dropped
    assert mut.note == "guard"
