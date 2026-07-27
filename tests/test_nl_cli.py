"""Natural-language CLI: plain-English request -> resolved action."""

from misterdev import nl_cli


class _StubClient:
    def __init__(self, reply):
        self._reply = reply

    def generate_code(self, prompt, system):
        return self._reply


class _FakeOrch:
    last_build_succeeded = True
    calls: dict = {}

    def build(self, path, args):
        _FakeOrch.calls["build"] = (path, args)
        return "REPORT"

    def run_project(self, path, dry_run=False):
        _FakeOrch.calls["run"] = (path, dry_run)

    def scan_directory(self, directory):
        _FakeOrch.calls["scan"] = directory

    def get_project_status(self, path):
        return {"path": path}

    def list_projects(self):
        return {"projects": []}

    def interactive_plan(self, path, args):
        _FakeOrch.calls["plan"] = (path, args)


def _stub_client(monkeypatch, reply):
    _FakeOrch.calls = {}
    monkeypatch.setattr(nl_cli, "create_llm_client", lambda cfg: _StubClient(reply))


def test_parse_intent_extracts_json():
    client = _StubClient(
        'Sure:\n{"command": "build", "path": ".", "goal": "add X", "budget": 5}'
    )
    intent = nl_cli.parse_intent("add X", client)
    assert intent["command"] == "build"
    assert intent["goal"] == "add X"
    assert intent["budget"] == 5


def test_build_args_composition():
    args = nl_cli._build_args(
        {"goal": "add X", "budget": 5, "parallel": True, "max_tasks": 3}
    )
    assert args == "add X --budget 5 --parallel --max-tasks 3"


def test_preview_reads_naturally():
    assert nl_cli.preview({"command": "build", "path": ".", "goal": "x"}).startswith(
        "build ."
    )


def test_dispatch_build_routes_to_orchestrator():
    _FakeOrch.calls = {}
    rc = nl_cli._dispatch(
        {"command": "build", "path": "/r", "goal": "add X"}, _FakeOrch()
    )
    assert rc == 0
    assert _FakeOrch.calls["build"] == ("/r", "add X")


def test_route_confirms_then_runs_mutating(monkeypatch):
    _stub_client(monkeypatch, '{"command": "build", "path": ".", "goal": "add X"}')
    rc = nl_cli.route("add X", _FakeOrch(), confirm=lambda _p: "y")
    assert rc == 0
    assert "build" in _FakeOrch.calls


def test_route_cancel_skips_execution(monkeypatch):
    _stub_client(monkeypatch, '{"command": "build", "path": ".", "goal": "x"}')
    rc = nl_cli.route("x", _FakeOrch(), confirm=lambda _p: "n")
    assert rc == 0
    assert "build" not in _FakeOrch.calls


def test_route_readonly_does_not_confirm(monkeypatch):
    _stub_client(monkeypatch, '{"command": "status", "path": "."}')

    def _no_confirm(_p):
        raise AssertionError("read-only actions must not prompt for confirmation")

    rc = nl_cli.route("what is the status", _FakeOrch(), confirm=_no_confirm)
    assert rc == 0


def test_route_unmappable_request(monkeypatch):
    _stub_client(monkeypatch, '{"command": "nonsense"}')
    # "what" is a query word so it falls through to the LLM, which returns an
    # unknown command — the router should return 1.
    rc = nl_cli.route("what is blah blah", _FakeOrch(), confirm=lambda _p: "y")
    assert rc == 1


def test_fast_route_skips_llm_for_build_requests(monkeypatch):
    called = []
    monkeypatch.setattr(
        nl_cli, "parse_intent", lambda req, client: called.append(1) or {}
    )
    orch = _FakeOrch()
    _FakeOrch.calls = {}
    nl_cli.route("add caching to the API", orch, confirm=lambda _p: "y")
    assert not called, "LLM should not be called for an obvious build request"
    assert "build" in _FakeOrch.calls


def test_fast_route_falls_back_to_llm_for_queries(monkeypatch):
    called = []
    _stub_client(monkeypatch, '{"command": "status", "path": "."}')
    orig = nl_cli.parse_intent

    def spy(req, client):
        called.append(req)
        return orig(req, client)

    monkeypatch.setattr(nl_cli, "parse_intent", spy)
    nl_cli.route("what is the current status", _FakeOrch(), confirm=lambda _p: "n")
    assert called, "LLM should be called for a query-word request"
