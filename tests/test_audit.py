import json

from misterdev.core.audit import AuditTrail


def _lines(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_record_appends_wellformed_json(tmp_path):
    trail = AuditTrail(tmp_path)
    trail.record("command", command="cargo test", ok=True)
    entries = _lines(trail.path)
    assert len(entries) == 1
    e = entries[0]
    assert e["type"] == "command"
    assert e["command"] == "cargo test"
    assert e["ok"] is True
    assert "ts" in e and e["ts"].endswith("+00:00")


def test_record_is_append_only(tmp_path):
    trail = AuditTrail(tmp_path)
    trail.record("command", command="a", ok=True)
    trail.record("gate", reason="rm -rf", allowed=False)
    entries = _lines(trail.path)
    assert len(entries) == 2
    assert entries[0]["command"] == "a"
    assert entries[1]["type"] == "gate"


def test_creates_orchestrator_dir_lazily(tmp_path):
    trail = AuditTrail(tmp_path)
    assert not trail.path.parent.exists()
    trail.record("command", command="x", ok=True)
    assert trail.path.parent.name == ".orchestrator"
    assert trail.path.exists()


def test_disabled_writes_nothing(tmp_path):
    trail = AuditTrail(tmp_path, enabled=False)
    trail.record("command", command="x", ok=True)
    assert not trail.path.exists()


def test_unwritable_path_does_not_raise(tmp_path):
    # Make the project root a file so mkdir of .orchestrator fails; record must
    # swallow the OSError rather than break the caller.
    rootfile = tmp_path / "asfile"
    rootfile.write_text("x")
    trail = AuditTrail(rootfile)
    trail.record("command", command="x", ok=True)  # must not raise
    assert not trail.path.exists()


def test_nonserializable_detail_is_coerced(tmp_path):
    trail = AuditTrail(tmp_path)

    class Weird:
        def __repr__(self):
            return "<weird>"

    trail.record("tool", payload=Weird())
    entries = _lines(trail.path)
    assert entries[0]["payload"] == "<weird>"


def test_helpers_record_command_and_edit(tmp_path):
    trail = AuditTrail(tmp_path)
    trail.record_command("ls", ok=True, cwd="/repo")
    trail.record_edit("src/x.py", action="apply")
    entries = _lines(trail.path)
    assert entries[0]["type"] == "command" and entries[0]["cwd"] == "/repo"
    assert entries[1]["type"] == "edit" and entries[1]["path"] == "src/x.py"
