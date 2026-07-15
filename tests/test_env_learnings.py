"""Per-project cross-run environment memory."""

import json
import tempfile
from pathlib import Path

from misterdev.core.execution.env_learnings import EnvLearnings
from misterdev.utils.file_utils import orchestrator_state_file


def test_read_write_round_trip():
    with tempfile.TemporaryDirectory() as td:
        e = EnvLearnings(
            worktree_setup_command="pnpm install --prefer-offline",
            worktree_healthcheck_command="npx --no-install tsc --version",
            max_workers=3,
            ordering_constraints=["dashboard build must precede server test"],
        )
        e.save(td)
        back = EnvLearnings.load(td)
        assert back == e
        # And the on-disk form is the documented tiny schema.
        on_disk = json.loads(
            orchestrator_state_file(td, "env_learnings.json").read_text()
        )
        assert on_disk["version"] == 1
        assert on_disk["max_workers"] == 3
        assert on_disk["ordering_constraints"] == [
            "dashboard build must precede server test"
        ]


def test_missing_file_loads_empty():
    with tempfile.TemporaryDirectory() as td:
        e = EnvLearnings.load(td)
        assert e == EnvLearnings()
        assert e.ordering_constraints == []


def test_corrupt_file_self_heals_to_empty():
    with tempfile.TemporaryDirectory() as td:
        orchestrator_state_file(td, "env_learnings.json").write_text("{not json")
        assert EnvLearnings.load(td) == EnvLearnings()


def test_apply_fills_unset_config():
    e = EnvLearnings(
        worktree_setup_command="npm ci",
        worktree_healthcheck_command="node -e 0",
        max_workers=2,
    )
    cfg = {}
    applied = e.apply_to_config(cfg)
    assert cfg["orchestrator"]["worktree_setup_command"] == "npm ci"
    assert cfg["orchestrator"]["worktree_healthcheck_command"] == "node -e 0"
    assert cfg["orchestrator"]["max_workers"] == 2
    assert len(applied) == 3


def test_explicit_config_always_wins():
    """A learned value never overrides an explicit project.yaml value; it only
    fills a key the user left unset."""
    e = EnvLearnings(worktree_setup_command="npm ci", max_workers=2)
    cfg = {"orchestrator": {"max_workers": 8}}  # user set max_workers explicitly
    applied = e.apply_to_config(cfg)
    assert cfg["orchestrator"]["max_workers"] == 8  # explicit wins
    assert cfg["orchestrator"]["worktree_setup_command"] == "npm ci"  # gap filled
    assert applied == ["orchestrator.worktree_setup_command='npm ci'"]


def test_none_learnings_apply_nothing():
    cfg = {"orchestrator": {}}
    assert EnvLearnings().apply_to_config(cfg) == []
    assert cfg == {"orchestrator": {}}  # untouched


def test_record_persists_reduced_workers_and_clears_on_recovery():
    """_record_env_learnings persists a contention-driven worker reduction, and a
    later run that held full concurrency clears it (no permanent pin)."""
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        orch = agent_mod.ProjectOrchestrator()
        project = MagicMock()
        project.path = Path(td)
        project.config = {}

        # Run 1 backed off from 4 to 2 workers -> persist the reduction.
        project.env_settled_workers = 2
        project.env_base_workers = 4
        orch._record_env_learnings(project)
        assert EnvLearnings.load(td).max_workers == 2

        # Run 2 held/recovered to the base -> clear the stale reduction.
        project.env_settled_workers = 4
        project.env_base_workers = 4
        orch._record_env_learnings(project)
        assert EnvLearnings.load(td).max_workers is None


def test_record_is_noop_without_settled_workers():
    """A run that never adapted (no settled value) leaves a prior learned
    max_workers untouched."""
    from unittest.mock import MagicMock
    import misterdev.agent as agent_mod

    with tempfile.TemporaryDirectory() as td:
        EnvLearnings(max_workers=2).save(td)
        orch = agent_mod.ProjectOrchestrator()
        project = MagicMock()
        project.path = Path(td)
        project.config = {}
        # No env_settled_workers/env_base_workers attributes present.
        del project.env_settled_workers
        del project.env_base_workers
        orch._record_env_learnings(project)
        assert EnvLearnings.load(td).max_workers == 2  # preserved


def test_ordering_constraints_persist_but_do_not_tune_config():
    """Ordering constraints round-trip in the ledger but are advisory — they are
    not written into the runtime config."""
    with tempfile.TemporaryDirectory() as td:
        EnvLearnings(ordering_constraints=["a before b"]).save(td)
        loaded = EnvLearnings.load(td)
        assert loaded.ordering_constraints == ["a before b"]
        cfg = {}
        loaded.apply_to_config(cfg)
        assert cfg == {}  # no config knob for ordering constraints
