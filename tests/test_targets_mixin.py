"""Unit tests for TargetsMixin — pure-logic paths only (no LLM, no subprocess)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from misterdev.core.execution.targets_mixin import TargetsMixin


# ---------------------------------------------------------------------------
# _resolve_claim_file
# ---------------------------------------------------------------------------


def test_resolve_claim_file_direct_path(tmp_path):
    f = tmp_path / "src" / "foo.py"
    f.parent.mkdir()
    f.write_text("x = 1")
    result = TargetsMixin._resolve_claim_file(tmp_path, "src/foo.py", "")
    assert result == f


def test_resolve_claim_file_token_match(tmp_path):
    f = tmp_path / "parser_engine.py"
    f.write_text("class ParserEngine: pass")
    file_map = "parser_engine.py: ParserEngine, parse\n"
    result = TargetsMixin._resolve_claim_file(tmp_path, "ParserEngine", file_map)
    assert result == f


def test_resolve_claim_file_no_match_returns_none(tmp_path):
    result = TargetsMixin._resolve_claim_file(tmp_path, "NonExistent", "")
    assert result is None


def test_resolve_claim_file_short_tokens_skipped(tmp_path):
    # tokens under 5 chars with no CamelCase should not match
    f = tmp_path / "foo.py"
    f.write_text("x = 1")
    file_map = "foo.py: foo\n"
    result = TargetsMixin._resolve_claim_file(tmp_path, "foo", file_map)
    assert result is None


def test_resolve_claim_file_empty_label(tmp_path):
    result = TargetsMixin._resolve_claim_file(tmp_path, "", "")
    assert result is None


def test_resolve_claim_file_prefers_longest_token(tmp_path):
    short = tmp_path / "auth.py"
    long = tmp_path / "authenticate_user.py"
    short.write_text("x=1")
    long.write_text("x=1")
    file_map = "auth.py: auth\nauthenticate_user.py: authenticate_user\n"
    result = TargetsMixin._resolve_claim_file(tmp_path, "authenticate_user", file_map)
    assert result == long


# ---------------------------------------------------------------------------
# _resolve_targets
# ---------------------------------------------------------------------------


class _Orch(TargetsMixin):
    pass


def _make_project(config):
    p = MagicMock()
    p.config = config
    return p


def test_resolve_targets_returns_explicit(tmp_path):
    orch = _Orch()
    proj = _make_project({"targets": [{"name": "frontend", "path": "ui"}]})
    assert orch._resolve_targets(proj) == [{"name": "frontend", "path": "ui"}]


def test_resolve_targets_empty_when_disabled():
    orch = _Orch()
    proj = _make_project({"targets": [], "orchestrator": {"auto_targets": False}})
    assert orch._resolve_targets(proj) == []


def test_resolve_targets_auto_discovery(tmp_path):
    orch = _Orch()
    proj = _make_project(
        {
            "targets": [],
            "orchestrator": {"auto_targets": True},
            "name": "test",
        }
    )
    proj.path = tmp_path
    discovered = [{"name": "backend", "path": "api"}]
    with (
        patch(
            "misterdev.core.execution.targets_mixin.discover_targets",
            return_value=discovered,
        )
        if False
        else patch(
            "misterdev.core.planning.targets.discover_targets",
            return_value=discovered,
        )
    ):
        with patch(
            "misterdev.core.execution.targets_mixin.get_setting",
            side_effect=lambda cfg, *_: True,
        ):
            result = orch._resolve_targets(proj)
    assert result == discovered


# ---------------------------------------------------------------------------
# _run_target_runtime_gates — SKIP paths must not fail the target
# ---------------------------------------------------------------------------


def test_run_target_runtime_gates_skip_web_passes():
    orch = _Orch()
    proj = MagicMock()
    skip_result = MagicMock(status="skip", reason="no browser")
    with patch(
        "misterdev.core.verification.web_verify.run_web_gate",
        return_value=skip_result,
    ):
        ok, detail = orch._run_target_runtime_gates(
            proj, {"web": {"url": "http://x"}}, Path(".")
        )
    assert ok is True
    assert detail == "ok"


def test_run_target_runtime_gates_red_web_fails():
    orch = _Orch()
    proj = MagicMock()
    red_result = MagicMock(status="red", reason="h1 not found", evidence=None)
    with patch(
        "misterdev.core.verification.web_verify.run_web_gate",
        return_value=red_result,
    ):
        ok, detail = orch._run_target_runtime_gates(
            proj, {"web": {"url": "http://x"}}, Path(".")
        )
    assert ok is False
    assert "web verify failed" in detail


def test_run_target_runtime_gates_no_checks_passes():
    orch = _Orch()
    ok, detail = orch._run_target_runtime_gates(MagicMock(), {}, Path("."))
    assert ok is True
    assert detail == "ok"
