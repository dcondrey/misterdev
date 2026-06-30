from my_project_orchestrator.utils import process
from my_project_orchestrator.utils.process import kill_process_group


class _FakeProc:
    pid = 4321

    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True


def test_kill_process_group_uses_killpg(monkeypatch):
    calls = []
    monkeypatch.setattr(process.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        process.os, "killpg", lambda pgid, sig: calls.append((pgid, sig))
    )
    proc = _FakeProc()
    kill_process_group(proc)
    # The whole group is signalled and the per-child fallback is not used.
    assert calls == [(4321, process.signal.SIGKILL)]
    assert proc.killed is False


def test_kill_process_group_falls_back_to_child_kill(monkeypatch):
    # When killpg is unavailable/raises (Windows, or the group is already gone),
    # the direct child is killed instead so a timed-out process is never leaked.
    def _boom(pgid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(process.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(process.os, "killpg", _boom)
    proc = _FakeProc()
    kill_process_group(proc)
    assert proc.killed is True


def test_kill_process_group_no_killpg_platform(monkeypatch):
    # On a platform without os.killpg at all, fall straight through to child kill.
    monkeypatch.delattr(process.os, "killpg", raising=False)
    proc = _FakeProc()
    kill_process_group(proc)
    assert proc.killed is True
