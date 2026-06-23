import subprocess
import tempfile
from pathlib import Path

from my_project_orchestrator.core.gatekeeper import (
    GateKeeper,
    BANNED_MARKERS,
    SECRET_PATTERNS,
    CODE_EXTENSIONS,
    SKIP_DIRS,
)


def _make_project(files=None):
    """Create a temp project with optional source files."""
    td = tempfile.mkdtemp()
    root = Path(td)
    if files:
        for rel_path, content in files.items():
            p = root / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
    return root


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _make_git_project(committed=None, working=None):
    """Create a temp git repo: ``committed`` files form HEAD, then ``working``
    files are written on top (unstaged) to form a diff."""
    root = _make_project(committed)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    if committed:
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "init")
    if working:
        for rel_path, content in working.items():
            p = root / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
    return root


def test_no_commands_all_pass():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates({})
    assert success
    assert issues == []
    assert health.builds is True
    assert health.tests_pass is True


def test_banned_markers_detected():
    root = _make_project(
        {
            "src/main.py": "x = 1  # FIXME this is broken\n",
            "src/util.py": "y = 2  # HACK workaround\n",
        }
    )
    gk = GateKeeper(root)
    found = gk._scan_banned_markers()
    assert "FIXME" in found
    assert "HACK" in found


def test_banned_markers_clean():
    root = _make_project(
        {
            "src/main.py": "x = 1\ny = 2\n",
        }
    )
    gk = GateKeeper(root)
    found = gk._scan_banned_markers()
    assert found == []


def test_secrets_scan_detects():
    root = _make_project(
        {
            "src/config.py": 'API_KEY = "sk-abc123"\n',
        }
    )
    gk = GateKeeper(root)
    found = gk._scan_secrets()
    assert len(found) == 1
    assert "src/config.py" in found[0]


def test_secrets_scan_clean():
    root = _make_project(
        {
            "src/main.py": "def hello(): return 'world'\n",
        }
    )
    gk = GateKeeper(root)
    found = gk._scan_secrets()
    assert found == []


def test_skip_dirs_excluded():
    root = _make_project(
        {
            "src/main.py": "# FIXME real issue\n",
            "__pycache__/cached.py": "# FIXME should be ignored\n",
            ".git/hooks/pre-commit": "# FIXME should be ignored\n",
            "node_modules/pkg.js": "# FIXME should be ignored\n",
        }
    )
    gk = GateKeeper(root)
    found = gk._scan_banned_markers()
    assert "FIXME" in found
    source_files = list(gk._iter_source_files())
    assert all("__pycache__" not in str(f) for f in source_files)
    assert all(".git" not in str(f) for f in source_files)
    assert all("node_modules" not in str(f) for f in source_files)


def test_code_extensions_filter():
    root = _make_project(
        {
            "src/main.py": "# FIXME\n",
            "docs/readme.md": "# FIXME in docs\n",
            "data/input.csv": "FIXME,data\n",
        }
    )
    gk = GateKeeper(root)
    source_files = list(gk._iter_source_files())
    paths = [str(f) for f in source_files]
    assert any("main.py" in p for p in paths)
    assert not any("readme.md" in p for p in paths)
    assert not any("input.csv" in p for p in paths)


def test_build_failure_stops_gates():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates({"build_command": "false"})
    assert not success
    assert any("G1" in i for i in issues)
    assert not health.builds


def test_test_failure_stops_gates():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates(
        {
            "build_command": "true",
            "test_command": "false",
        }
    )
    assert not success
    assert any("G3" in i for i in issues)


def test_lint_failure_does_not_stop():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates(
        {
            "build_command": "true",
            "lint_command": "false",
            "test_command": "true",
        }
    )
    assert not success
    assert any("G2" in i for i in issues)


def test_typecheck_failure_stops_gates():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates(
        {
            "build_command": "true",
            "test_command": "true",
            "typecheck_command": "false",
        }
    )
    assert not success
    assert any("G4" in i for i in issues)


def test_typecheck_pass_does_not_block():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates(
        {
            "build_command": "true",
            "test_command": "true",
            "typecheck_command": "true",
        }
    )
    assert success
    assert not any("G4" in i for i in issues)


def test_no_typecheck_command_no_penalty():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates(
        {"build_command": "true", "test_command": "true"}
    )
    assert success
    assert not any("G4" in i for i in issues)


def test_banned_marker_preexisting_not_flagged():
    # FIXME lives in a committed file with no working changes: nothing to flag.
    root = _make_git_project(committed={"src/main.py": "x = 1  # FIXME old\n"})
    gk = GateKeeper(root)
    assert gk._scan_banned_markers() == []


def test_banned_marker_introduced_in_diff_flagged():
    root = _make_git_project(
        committed={"src/main.py": "x = 1\n"},
        working={"src/main.py": "x = 1\ny = 2  # FIXME new\n"},
    )
    gk = GateKeeper(root)
    assert "FIXME" in gk._scan_banned_markers()


def test_banned_marker_untouched_region_not_flagged():
    # A pre-existing FIXME stays out of the diff even when the file is edited
    # elsewhere; only the newly added clean line shows up.
    root = _make_git_project(
        committed={"src/main.py": "x = 1  # FIXME old\n"},
        working={"src/main.py": "x = 1  # FIXME old\nz = 3\n"},
    )
    gk = GateKeeper(root)
    assert gk._scan_banned_markers() == []


def test_secret_preexisting_not_flagged():
    root = _make_git_project(committed={"src/config.py": 'API_KEY = "sk-old1234"\n'})
    gk = GateKeeper(root)
    assert gk._scan_secrets() == []


def test_secret_introduced_in_diff_flagged():
    root = _make_git_project(
        committed={"src/config.py": "X = 1\n"},
        working={"src/config.py": 'X = 1\nAPI_KEY = "sk-new12345"\n'},
    )
    gk = GateKeeper(root)
    found = gk._scan_secrets()
    assert any("src/config.py" in f for f in found)


def test_constants():
    assert "FIXME" in BANNED_MARKERS
    assert "HACK" in BANNED_MARKERS
    assert "sk-" in SECRET_PATTERNS
    assert ".py" in CODE_EXTENSIONS
    assert ".rs" in CODE_EXTENSIONS
    assert "__pycache__" in SKIP_DIRS
    assert ".git" in SKIP_DIRS


def test_golden_command_blocks_when_failing():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates({"golden_command": "false"})
    assert not success
    assert any("G3.5" in i for i in issues)
    assert health.tests_pass is False


def test_golden_command_passes_when_succeeding():
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates({"golden_command": "true"})
    assert success
    assert issues == []


def test_golden_failure_blocks_even_when_visible_tests_pass():
    # The whole point: visible tests green, golden red -> gate must fail.
    root = _make_project()
    gk = GateKeeper(root)
    success, issues, health = gk.run_gates(
        {"test_command": "true", "golden_command": "false"}
    )
    assert not success
    assert any("G3.5" in i for i in issues)


def test_gatekeeper_honors_configured_timeouts(monkeypatch):
    # The gate runs on every wave/iteration; its build/test timeouts must come
    # from config so a slow compiler isn't falsely failed.
    import my_project_orchestrator.core.gatekeeper as gk

    calls = {}

    def fake_run_cmd(cmd, cwd, env_activate=None, timeout=180):
        calls[cmd] = timeout
        return True, "ok"

    monkeypatch.setattr(gk, "_run_cmd", fake_run_cmd)
    keeper = gk.GateKeeper(_make_project(), build_timeout=600, test_timeout=300)
    keeper.run_gates(
        {
            "build_command": "cargo build",
            "test_command": "cargo test",
            "lint_command": "cargo clippy",
            "golden_command": "cargo test --lib golden",
        }
    )
    assert calls["cargo build"] == 600
    assert calls["cargo test"] == 300
    assert calls["cargo clippy"] == 300
    assert calls["cargo test --lib golden"] == 300


def test_gatekeeper_explicit_lint_timeout(monkeypatch):
    import my_project_orchestrator.core.gatekeeper as gk

    calls = {}

    def fake_run_cmd(cmd, cwd, env_activate=None, timeout=180):
        calls[cmd] = timeout
        return True, "ok"

    monkeypatch.setattr(gk, "_run_cmd", fake_run_cmd)
    keeper = gk.GateKeeper(_make_project(), test_timeout=300, lint_timeout=240)
    keeper.run_gates({"lint_command": "clippy", "test_command": "test"})
    assert calls["clippy"] == 240
    assert calls["test"] == 300


def test_run_gates_surfaces_banned_and_secrets():
    # Non-git repo: scans all files. A banned marker and a planted secret must
    # surface as G5/G6 issues through the full gate run.
    root = _make_project(
        {
            "src/a.py": "x = 1  # FIXME broken\n",
            "src/conf.py": 'API_KEY = "sk-deadbeefdeadbeefdeadbeef"\n',
        }
    )
    gk = GateKeeper(root)
    success, issues, _ = gk.run_gates({})
    assert not success
    assert any("G5" in i for i in issues)
    assert any("G6" in i for i in issues)


def test_lsp_gate_blocks_on_errors(monkeypatch):
    import my_project_orchestrator.core.lsp as lspmod

    monkeypatch.setattr(
        lspmod, "find_source_files", lambda root, lang, cap=40: ["a.py"]
    )
    monkeypatch.setattr(
        lspmod,
        "collect_diagnostics",
        lambda root, lang, files, timeout=30: [
            {"file": "a.py", "line": 3, "message": "undefined name"}
        ],
    )
    gk = GateKeeper(_make_project(), lsp_diagnostics=True, lsp_language="python")
    success, issues, _ = gk.run_gates({})
    assert not success
    assert any("G4.5" in i and "undefined name" in i for i in issues)


def test_lsp_gate_skips_when_no_diagnostics(monkeypatch):
    import my_project_orchestrator.core.lsp as lspmod

    monkeypatch.setattr(
        lspmod, "find_source_files", lambda root, lang, cap=40: ["a.py"]
    )
    # None = server unavailable/slow -> skip, never fail.
    monkeypatch.setattr(
        lspmod, "collect_diagnostics", lambda root, lang, files, timeout=30: None
    )
    gk = GateKeeper(_make_project(), lsp_diagnostics=True, lsp_language="python")
    success, issues, _ = gk.run_gates({})
    assert success
    assert not any("G4.5" in i for i in issues)
