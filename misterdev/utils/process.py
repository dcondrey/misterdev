import os
import signal
import subprocess


def kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL a timed-out command's whole process group so grandchildren
    (build/test workers like rustc or pytest, servers, a backgrounded ``cmd &``)
    don't outlive the caller and hold locks or keep running.

    Falls back to killing just the direct child when the platform has no process
    groups (Windows) or the group is already gone.
    """
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass
