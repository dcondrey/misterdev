import time

from my_project_orchestrator.core.mutation_gate import (
    GREEN,
    RED,
    SKIP,
    MutationResult,
    parse_mutation_score,
    run_mutation_gate,
    _normalize_floor,
)


def _runner(output):
    """A gate runner seam that returns canned output (no subprocess)."""

    def run(cmd, timeout):
        return True, output

    return run


# --- SKIP semantics ---------------------------------------------------------


def test_skip_when_no_config(tmp_path):
    assert run_mutation_gate(tmp_path, None).status == SKIP
    assert run_mutation_gate(tmp_path, {}).status == SKIP


def test_skip_when_no_command(tmp_path):
    res = run_mutation_gate(tmp_path, {"min_score": 0.8})
    assert res.status == SKIP
    assert "no mutation config" in res.reason


def test_skip_when_score_unparseable(tmp_path):
    res = run_mutation_gate(
        tmp_path,
        {"command": "x", "min_score": 0.8, "timeout": 5},
        runner=_runner("ran 100 mutants, see the html report"),
    )
    assert res.status == SKIP
    assert "parse" in res.reason
    assert res.score is None


# --- GREEN / RED ------------------------------------------------------------


def test_green_at_or_above_floor(tmp_path):
    res = run_mutation_gate(
        tmp_path,
        {"command": "x", "min_score": 0.8, "timeout": 5},
        runner=_runner("Mutation score: 82.5%"),
    )
    assert res.status == GREEN
    assert res.passed
    assert abs(res.score - 0.825) < 1e-9


def test_green_exactly_at_floor(tmp_path):
    res = run_mutation_gate(
        tmp_path,
        {"command": "x", "min_score": 0.8, "timeout": 5},
        runner=_runner("score: 80%"),
    )
    assert res.status == GREEN


def test_red_below_floor(tmp_path):
    res = run_mutation_gate(
        tmp_path,
        {"command": "x", "min_score": 0.8, "timeout": 5},
        runner=_runner("Mutation score: 50%"),
    )
    assert res.status == RED
    assert not res.passed
    assert "below floor" in res.reason
    assert abs(res.score - 0.5) < 1e-9


def test_floor_accepts_percentage_form(tmp_path):
    # min_score given as 80 (a percentage) must mean 0.8, not 8000%.
    res = run_mutation_gate(
        tmp_path,
        {"command": "x", "min_score": 80, "timeout": 5},
        runner=_runner("score: 75%"),
    )
    assert res.status == RED


# --- never blocks / errors --------------------------------------------------


def test_hanging_runner_returns_within_timeout(tmp_path):
    def slow(cmd, timeout):
        time.sleep(3600)
        return True, "score: 90%"

    start = time.monotonic()
    res = run_mutation_gate(tmp_path, {"command": "x", "timeout": 0.3}, runner=slow)
    assert time.monotonic() - start < 10
    assert res.status == SKIP
    assert "timed out" in res.reason


def test_runner_error_is_skip_not_crash(tmp_path):
    def boom(cmd, timeout):
        raise RuntimeError("tool not installed")

    res = run_mutation_gate(tmp_path, {"command": "x", "timeout": 5}, runner=boom)
    assert res.status == SKIP
    assert "error" in res.reason


# --- score parsing ----------------------------------------------------------


def test_parse_labeled_percentage():
    assert abs(parse_mutation_score("Mutation score: 82.5%") - 0.825) < 1e-9
    assert abs(parse_mutation_score("score = 90 %") - 0.90) < 1e-9


def test_parse_labeled_fraction():
    assert abs(parse_mutation_score("mutation score: 0.82") - 0.82) < 1e-9


def test_parse_kill_ratio():
    assert abs(parse_mutation_score("killed 41/50 mutants") - 0.82) < 1e-9


def test_parse_bare_percentage_fallback():
    assert abs(parse_mutation_score("Result: 73.0%") - 0.73) < 1e-9


def test_parse_prefers_labeled_over_bare():
    # A labeled score must win over an unrelated trailing percentage.
    out = "coverage 99%\nmutation score: 60%\n"
    assert abs(parse_mutation_score(out) - 0.60) < 1e-9


def test_parse_none_when_no_number():
    assert parse_mutation_score("no score here") is None
    assert parse_mutation_score("") is None


def test_parse_rejects_out_of_range():
    # A ratio with zero total is skipped; a bare huge percent is normalized but
    # an impossible fraction like a lone "5.0" with no % is out of 0..1 -> None.
    assert parse_mutation_score("killed 5/0") is None


def test_normalize_floor():
    assert _normalize_floor(0.8) == 0.8
    assert _normalize_floor(80) == 0.8
    assert _normalize_floor("0.5") == 0.5
    assert _normalize_floor(-1) == 0.0
    assert _normalize_floor(150) == 1.0
    assert _normalize_floor("not a number") == 0.0


def test_result_repr_and_flags():
    r = MutationResult(GREEN, score=0.9)
    assert r.passed and not r.skipped
    assert "green" in repr(r)


# --- gatekeeper integration -------------------------------------------------


def _keeper(tmp_path, monkeypatch, *, enabled, status, score=0.5, reason=""):
    """Build a GateKeeper with the mutation runner monkeypatched."""
    from my_project_orchestrator.core.gatekeeper import GateKeeper
    import my_project_orchestrator.core.mutation_gate as mut_mod

    (tmp_path / "a.py").write_text("x = 1\n")

    def fake_run_mutation_gate(project_root, config, runner=None):
        return MutationResult(status, score=score, reason=reason)

    # The gatekeeper imports run_mutation_gate lazily inside run_gates from this
    # source module, so patching it here is what the gate will resolve.
    monkeypatch.setattr(mut_mod, "run_mutation_gate", fake_run_mutation_gate)

    return GateKeeper(
        tmp_path,
        mutation_gate=enabled,
        mutation_config={"command": "mut", "min_score": 0.8},
    )


def test_gatekeeper_skips_mutation_when_off(tmp_path, monkeypatch):
    keeper = _keeper(tmp_path, monkeypatch, enabled=False, status=RED, score=0.1)
    _success, issues, _ = keeper.run_gates({})
    assert not any("G3.6" in i for i in issues)


def test_gatekeeper_red_mutation_blocks_build(tmp_path, monkeypatch):
    keeper = _keeper(
        tmp_path, monkeypatch, enabled=True, status=RED, score=0.1, reason="too low"
    )
    success, issues, _ = keeper.run_gates({})
    assert not success
    assert any("G3.6" in i for i in issues)


def test_gatekeeper_green_mutation_passes(tmp_path, monkeypatch):
    keeper = _keeper(tmp_path, monkeypatch, enabled=True, status=GREEN, score=0.9)
    _success, issues, _ = keeper.run_gates({})
    assert not any("G3.6" in i for i in issues)


def test_gatekeeper_skip_mutation_does_not_block(tmp_path, monkeypatch):
    keeper = _keeper(tmp_path, monkeypatch, enabled=True, status=SKIP, score=None)
    _success, issues, _ = keeper.run_gates({})
    assert not any("G3.6" in i for i in issues)
