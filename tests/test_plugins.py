from misterdev.plugins import Registry, TOOLS


class _FakeEP:
    def __init__(self, name, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


def _patch_eps(monkeypatch, eps):
    monkeypatch.setattr("misterdev.plugins._iter_entry_points", lambda group: eps)


def test_register_direct_and_decorator():
    reg = Registry("thing", "misterdev.things")
    reg.register("a", 1)

    @reg.register("b")
    def b_factory():
        return "b"

    assert reg.get("a") == 1
    assert reg.get("b") is b_factory
    assert reg.get("missing") is None
    assert reg.names() == ["a", "b"]


def test_builtin_tools_registered():
    import misterdev.tools  # noqa: F401 - registration side effect
    from misterdev.tools import CommandTool, FileIOTool, FormatterTool, GitTool

    assert TOOLS.get("command") is CommandTool
    assert TOOLS.get("test_runner") is CommandTool
    assert TOOLS.get("file_io") is FileIOTool
    assert TOOLS.get("formatter") is FormatterTool
    assert TOOLS.get("git") is GitTool


def test_tool_manager_resolves_via_registry():
    from misterdev.core.execution.project import ToolManager
    from misterdev.tools import CommandTool, FormatterTool

    tm = ToolManager(
        [
            {"name": "fmt", "type": "formatter"},
            {"name": "unknown", "type": "does_not_exist"},
        ]
    )
    assert isinstance(tm.get_tool("fmt"), FormatterTool)
    # Unknown type degrades to the command tool rather than raising.
    assert isinstance(tm.get_tool("unknown"), CommandTool)


def test_entry_point_plugin_is_discovered(monkeypatch):
    reg = Registry("tool", "misterdev.tools")

    class MyTool:
        pass

    _patch_eps(monkeypatch, [_FakeEP("mytool", lambda: MyTool)])
    assert reg.get("mytool") is MyTool
    assert "mytool" in reg.names()


def test_plugin_cannot_shadow_builtin(monkeypatch):
    reg = Registry("tool", "misterdev.tools")
    reg.register("command", str)  # built-in claims the name first
    _patch_eps(monkeypatch, [_FakeEP("command", lambda: int)])
    # The built-in wins; a plugin can't hijack a core name.
    assert reg.get("command") is str


def test_unloadable_plugin_is_skipped_not_fatal(monkeypatch):
    reg = Registry("tool", "misterdev.tools")

    def boom():
        raise ImportError("missing dependency")

    _patch_eps(monkeypatch, [_FakeEP("broken", boom), _FakeEP("ok", lambda: 42)])
    # A broken plugin is skipped; a good sibling still loads.
    assert reg.get("broken") is None
    assert reg.get("ok") == 42


def test_entry_points_loaded_once(monkeypatch):
    reg = Registry("tool", "misterdev.tools")
    calls = {"n": 0}

    def counting(group):
        calls["n"] += 1
        return [_FakeEP("x", lambda: 1)]

    monkeypatch.setattr("misterdev.plugins._iter_entry_points", counting)
    reg.get("x")
    reg.get("x")
    reg.names()
    assert calls["n"] == 1  # discovery happens exactly once
