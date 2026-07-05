import json
import tempfile
from pathlib import Path

from misterdev.utils.file_utils import atomic_write_json


def test_atomic_write_json_roundtrips_and_seeds_gitignore():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / ".orchestrator" / "state.json"
        atomic_write_json(path, {"b": 1, "a": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"b": 1, "a": 2}
        # ensure_artifact_dir seeds a self-ignoring .gitignore so the artifact
        # never dirties the working tree.
        assert (Path(td) / ".orchestrator" / ".gitignore").read_text() == "*\n"


def test_atomic_write_json_honors_indent_and_sort_keys():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "out.json"
        atomic_write_json(path, {"b": 1, "a": 2}, indent=2, sort_keys=True)
        text = path.read_text(encoding="utf-8")
        assert text == json.dumps({"a": 2, "b": 1}, indent=2, sort_keys=True)
        assert "\n" in text  # indent applied


def test_atomic_write_json_overwrites_atomically():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "out.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}
        # No leftover .tmp files from the temp-then-rename write.
        assert not any(p.name.endswith(".tmp") for p in Path(td).iterdir())
