"""Interactive guided menu: prompt parsing + action dispatch, with scripted
input and a fake orchestrator (no real builds)."""

from types import SimpleNamespace

from misterdev.interactive import _confirm, _menu, run_interactive


def _script(monkeypatch, replies):
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda *a: next(it))


def test_menu_selects_index(monkeypatch):
    _script(monkeypatch, ["2"])
    assert _menu("pick", ["a", "b", "c"]) == 1


def test_menu_rejects_then_accepts(monkeypatch):
    _script(monkeypatch, ["9", "x", "1"])  # out of range, non-numeric, then valid
    assert _menu("pick", ["a", "b"]) == 0


def test_confirm_default_and_explicit(monkeypatch):
    _script(monkeypatch, ["", "n", "y"])
    assert _confirm("x", default=True) is True  # empty -> default
    assert _confirm("x", default=True) is False  # explicit no
    assert _confirm("x", default=False) is True  # explicit yes


def test_status_then_quit(monkeypatch):
    calls = {}

    def status(p):
        calls["path"] = p
        return {"name": "P", "path": p, "tasks": []}

    orch = SimpleNamespace(get_project_status=status)
    _script(monkeypatch, ["5", "/proj", "n"])  # Status, path, don't continue
    assert run_interactive(orch) == 0
    assert calls["path"] == "/proj"


def test_build_builds_goal_with_budget(monkeypatch):
    calls = {}

    def build(p, a):
        calls["args"] = (p, a)
        return "ok"

    orch = SimpleNamespace(build=build)
    # Build(1), path, goal, budget, confirm-yes, then stop
    _script(monkeypatch, ["1", "/proj", "add a feature", "50", "y", "n"])
    assert run_interactive(orch) == 0
    assert calls["args"][0] == "/proj"
    assert "add a feature --budget 50" in calls["args"][1]


def test_debug_shortcut_passes_debug_goal(monkeypatch):
    calls = {}
    orch = SimpleNamespace(build=lambda p, a: calls.setdefault("a", a))
    # Debug(3), path, budget, confirm-yes, stop
    _script(monkeypatch, ["3", "/proj", "100", "y", "n"])
    assert run_interactive(orch) == 0
    assert calls["a"].startswith("debug")


def test_run_previews_then_executes(monkeypatch):
    calls = []
    orch = SimpleNamespace(
        run_project=lambda p, dry_run=False, tasklist=None: calls.append(
            (p, dry_run, tasklist)
        )
    )
    # Run(2), path, tasklist file, execute-yes, stop
    _script(monkeypatch, ["2", "/proj", "PLAN.md", "y", "n"])
    assert run_interactive(orch) == 0
    assert calls[0] == ("/proj", True, "PLAN.md")  # dry-run preview first
    assert calls[1] == ("/proj", False, "PLAN.md")  # then execute


def test_quit_immediately(monkeypatch):
    _script(monkeypatch, ["7"])
    assert run_interactive(SimpleNamespace()) == 0


def test_eof_exits_cleanly(monkeypatch):
    def boom(*a):
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    assert run_interactive(SimpleNamespace()) == 0


def test_action_error_does_not_crash_menu(monkeypatch):
    def boom(p):
        raise RuntimeError("kaboom")

    orch = SimpleNamespace(get_project_status=boom)
    # Status(5) raises -> caught; then quit at "do something else?"
    _script(monkeypatch, ["5", "/proj", "n"])
    assert run_interactive(orch) == 0
