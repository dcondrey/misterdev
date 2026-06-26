from my_project_orchestrator.core.targets import select_target, target_commands


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


def test_target_commands_uses_target_then_falls_back():
    config = {
        "build_command": "cargo build --workspace",
        "test_command": "cargo test",
        "lint_command": "cargo clippy",
    }
    web = {"path": "clients/web", "build_command": "npm run typecheck"}
    cmds = target_commands(web, config)
    # Overrides build; inherits test/lint from top-level config.
    assert cmds["build_command"] == "npm run typecheck"
    assert cmds["test_command"] == "cargo test"
    assert cmds["lint_command"] == "cargo clippy"


def test_target_commands_none_target_is_top_level():
    config = {"build_command": "cargo build", "test_command": "cargo test"}
    cmds = target_commands(None, config)
    assert cmds["build_command"] == "cargo build"
    assert cmds["test_command"] == "cargo test"
    assert cmds["lint_command"] is None
