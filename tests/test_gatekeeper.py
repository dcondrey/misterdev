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
