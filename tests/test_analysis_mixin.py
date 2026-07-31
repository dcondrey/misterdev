"""Unit tests for AnalysisMixin."""

import pytest
from unittest.mock import MagicMock, patch, call

from misterdev.core.execution.analysis_mixin import AnalysisMixin


class _Orch(AnalysisMixin):
    pass


# ---------------------------------------------------------------------------
# _setup_env
# ---------------------------------------------------------------------------


def test_setup_env_no_env_manager():
    orch = _Orch()
    proj = MagicMock()
    proj.env_manager = None
    assert orch._setup_env(proj) is None


def test_setup_env_calls_setup_and_returns_prefix():
    orch = _Orch()
    proj = MagicMock()
    proj.env_manager.activate_command.return_value = "source .venv/bin/activate"
    result = orch._setup_env(proj)
    proj.env_manager.setup.assert_called_once()
    assert result == "source .venv/bin/activate"


# ---------------------------------------------------------------------------
# _container_engine
# ---------------------------------------------------------------------------


def test_container_engine_non_container_env_returns_none():
    orch = _Orch()
    proj = MagicMock()
    proj.env_manager = MagicMock()  # plain mock; isinstance check returns False
    result = orch._container_engine(proj)
    assert result is None


def test_container_engine_returns_engine():
    orch = _Orch()
    proj = MagicMock()
    from misterdev.environments.container_env import ContainerEnvironmentManager

    fake_engine = MagicMock()
    fake_mgr = MagicMock(spec=ContainerEnvironmentManager)
    fake_mgr.engine.return_value = fake_engine
    proj.env_manager = fake_mgr
    result = orch._container_engine(proj)
    assert result is fake_engine


# ---------------------------------------------------------------------------
# _project_file_map
# ---------------------------------------------------------------------------


def test_project_file_map_returns_cached():
    orch = _Orch()
    proj = MagicMock()
    proj._file_map_cache = "cached outline"
    assert orch._project_file_map(proj) == "cached outline"


def test_project_file_map_no_topography_returns_empty():
    orch = _Orch()
    proj = MagicMock(spec=[])  # no topography attribute
    assert orch._project_file_map(proj) == ""


def test_project_file_map_builds_and_caches():
    orch = _Orch()
    proj = MagicMock(spec=["topography"])
    proj.topography.get_project_outline.return_value = "file: symbol\n"
    result = orch._project_file_map(proj)
    proj.topography.initialize.assert_called_once()
    assert result == "file: symbol\n"
    assert proj._file_map_cache == "file: symbol\n"


def test_project_file_map_error_returns_empty():
    orch = _Orch()
    proj = MagicMock(spec=["topography"])
    proj.topography.initialize.side_effect = RuntimeError("topo failed")
    assert orch._project_file_map(proj) == ""


# ---------------------------------------------------------------------------
# _analyze
# ---------------------------------------------------------------------------


def test_analyze_passes_config_to_analyze_project():
    orch = _Orch()
    proj = MagicMock(spec=["topography", "path", "llm_client", "config"])
    proj.topography = None  # _project_file_map returns ""
    proj.config = {
        "build_command": "make build",
        "test_command": "pytest",
        "lint_command": None,
        "build": {
            "build_timeout": 60,
            "test_timeout": 90,
            "lint_timeout": 30,
            "parallel_analysis": True,
        },
    }
    with patch(
        "misterdev.core.execution.analysis_mixin.analyze_project"
    ) as mock_analyze:
        with patch(
            "misterdev.core.execution.analysis_mixin.get_setting",
            side_effect=lambda cfg, section, key: cfg.get(section, {}).get(key),
        ):
            orch._analyze(proj, "source .venv/bin/activate")
    mock_analyze.assert_called_once()
    kwargs = mock_analyze.call_args[1]
    assert kwargs["build_command"] == "make build"
    assert kwargs["test_command"] == "pytest"
    assert kwargs["env_activate"] == "source .venv/bin/activate"


def test_analyze_passes_file_map_when_available():
    orch = _Orch()
    proj = MagicMock(spec=["topography", "path", "llm_client", "config"])
    proj.topography.get_project_outline.return_value = "main.py: main\n"
    proj.config = {}
    with (
        patch(
            "misterdev.core.execution.analysis_mixin.analyze_project"
        ) as mock_analyze,
        patch("misterdev.core.execution.analysis_mixin.get_setting", return_value=None),
    ):
        orch._analyze(proj, None)
    kwargs = mock_analyze.call_args[1]
    assert kwargs["project_outline"] == "main.py: main\n"
