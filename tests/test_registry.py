import json
import os
import tempfile
from pathlib import Path

from misterdev.core.execution.registry import ProjectRegistry


def _make_project_dir(root, name="test-proj", with_yaml=True):
    proj_dir = root / name
    proj_dir.mkdir(parents=True, exist_ok=True)
    if with_yaml:
        os.environ["_TEST_ORCH_KEY"] = "fake-key-for-testing"
        (proj_dir / "project.yaml").write_text(
            f"name: {name}\ndescription: A test project\n"
            f"llm:\n  provider: openrouter\n  model: test-model\n  api_key_env_var: _TEST_ORCH_KEY\n"
        )
    return proj_dir


def test_registry_init():
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "registry.json"
        reg = ProjectRegistry(state_file=state_file)
        assert reg.projects == {}


def test_register_project():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        state_file = td / "state" / "registry.json"
        proj_dir = _make_project_dir(td)
        reg = ProjectRegistry(state_file=state_file)
        project = reg.register_project(proj_dir)
        assert project.name == "test-proj"
        assert str(proj_dir.resolve()) in reg.projects


def test_register_idempotent():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        state_file = td / "state" / "registry.json"
        proj_dir = _make_project_dir(td)
        reg = ProjectRegistry(state_file=state_file)
        p1 = reg.register_project(proj_dir)
        p2 = reg.register_project(proj_dir)
        assert p1 is p2
        assert len(reg.projects) == 1


def test_get_project():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        state_file = td / "state" / "registry.json"
        proj_dir = _make_project_dir(td)
        reg = ProjectRegistry(state_file=state_file)
        reg.register_project(proj_dir)
        found = reg.get_project(proj_dir)
        assert found is not None
        assert found.name == "test-proj"


def test_get_project_not_found():
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "registry.json"
        reg = ProjectRegistry(state_file=state_file)
        assert reg.get_project("/nonexistent") is None


def test_list_projects():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        state_file = td / "state" / "registry.json"
        _make_project_dir(td, "proj-a")
        _make_project_dir(td, "proj-b")
        reg = ProjectRegistry(state_file=state_file)
        reg.register_project(td / "proj-a")
        reg.register_project(td / "proj-b")
        listing = reg.list_projects()
        assert len(listing) == 2
        names = {v["name"] for v in listing.values()}
        assert names == {"proj-a", "proj-b"}


def test_discover_projects():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        state_file = td / "state" / "registry.json"
        _make_project_dir(td, "sub/proj-a")
        _make_project_dir(td, "sub/proj-b")
        _make_project_dir(td, "sub/no-yaml", with_yaml=False)
        reg = ProjectRegistry(state_file=state_file)
        reg.discover_projects(td / "sub")
        assert len(reg.projects) == 2


def test_state_persistence():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        state_file = td / "state" / "registry.json"
        proj_dir = _make_project_dir(td)
        reg1 = ProjectRegistry(state_file=state_file)
        reg1.register_project(proj_dir)
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert len(data["registered_paths"]) == 1


def test_load_state_prunes_stale_entries():
    # A persisted path with no project.yaml (deleted/moved) must be pruned on
    # load — not reloaded with a warning — and dropped from the saved state.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        state_file = root / "registry.json"
        good = _make_project_dir(root, "good", with_yaml=True).resolve()
        gone = (root / "deleted-proj").resolve()  # no project.yaml
        state_file.write_text(
            json.dumps({"registered_paths": [str(good), str(gone)]})
        )
        reg = ProjectRegistry(state_file=state_file)
        assert str(good) in reg.projects
        assert str(gone) not in reg.projects  # pruned
        saved = json.loads(state_file.read_text())["registered_paths"]
        assert str(gone) not in saved and str(good) in saved  # re-saved clean
