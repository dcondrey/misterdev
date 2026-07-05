from misterdev.core.planning.targets import select_target, target_commands


_TARGETS = [
    {
        "name": "core",
        "path": "emathy-core",
        "build_command": "cargo build -p emathy-core",
        "test_command": "cargo test -p emathy-core --lib",
    },
    {
        "name": "web",
        "path": "clients/web",
        "build_command": "npm run typecheck",
    },
]


def test_select_target_routes_by_owning_files():
    t = select_target(_TARGETS, ["clients/web/src/main.ts"])
    assert t["name"] == "web"
    t = select_target(_TARGETS, ["emathy-core/src/engine/mod.rs"])
    assert t["name"] == "core"


def test_select_target_picks_majority():
    files = ["clients/web/src/a.ts", "clients/web/src/b.ts", "emathy-core/src/x.rs"]
    assert select_target(_TARGETS, files)["name"] == "web"


def test_select_target_none_when_no_match():
    assert select_target(_TARGETS, ["docs/readme.md"]) is None


def test_select_target_none_when_empty():
    assert select_target([], ["clients/web/src/a.ts"]) is None
    assert select_target(_TARGETS, []) is None


def test_select_target_prefers_more_specific_path_on_tie():
    targets = [
        {"name": "web", "path": "clients/web"},
        {"name": "websub", "path": "clients/web/sub"},
    ]
    t = select_target(targets, ["clients/web/sub/x.ts"])
    assert t["name"] == "websub"


def test_select_target_ignores_targets_without_path():
    targets = [{"name": "bad"}, {"name": "web", "path": "clients/web"}]
    assert select_target(targets, ["clients/web/x.ts"])["name"] == "web"


def test_target_commands_matched_target_is_self_contained():
    config = {
        "build_command": "cargo build --workspace",
        "test_command": "cargo test",
        "lint_command": "cargo clippy",
    }
    web = {"path": "clients/web", "build_command": "npm run typecheck"}
    cmds = target_commands(web, config)
    # Only the target's own commands apply; unspecified ones skip (NOT inherited
    # — a web task must not be gated by cargo test).
    assert cmds["build_command"] == "npm run typecheck"
    assert cmds["test_command"] is None
    assert cmds["lint_command"] is None


def test_target_commands_none_target_is_top_level():
    config = {"build_command": "cargo build", "test_command": "cargo test"}
    cmds = target_commands(None, config)
    assert cmds["build_command"] == "cargo build"
    assert cmds["test_command"] == "cargo test"
    assert cmds["lint_command"] is None


def _mk(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    # package.json needs a build/test script for a command to be detectable.
    content = '{"scripts": {"build": "tsc"}}\n' if p.name == "package.json" else "{}\n"
    p.write_text(content, encoding="utf-8")


def test_discover_targets_finds_distinct_subprojects(tmp_path):
    from misterdev.core.planning.targets import discover_targets

    _mk(tmp_path / "rust" / "Cargo.toml")
    _mk(tmp_path / "rust" / "emathy-core" / "Cargo.toml")  # nested crate
    _mk(tmp_path / "clients" / "web" / "package.json")
    targets = discover_targets(str(tmp_path))
    paths = sorted(t["path"] for t in targets)
    assert paths == ["clients/web", "rust"]  # nested crate NOT a separate target


def test_discover_targets_single_project_returns_empty(tmp_path):
    from misterdev.core.planning.targets import discover_targets

    _mk(tmp_path / "pyproject.toml")
    _mk(tmp_path / "pkg" / "mod.py")
    assert discover_targets(str(tmp_path)) == []  # <2 sub-projects -> unchanged


def test_discover_targets_skips_vendor_dirs(tmp_path):
    from misterdev.core.planning.targets import discover_targets

    _mk(tmp_path / "app" / "package.json")
    _mk(tmp_path / "node_modules" / "dep" / "package.json")  # must be skipped
    _mk(tmp_path / "svc" / "go.mod")
    paths = sorted(t["path"] for t in discover_targets(str(tmp_path)))
    assert paths == ["app", "svc"]
