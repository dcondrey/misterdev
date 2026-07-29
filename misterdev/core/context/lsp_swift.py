"""Minimal, self-contained sourcekit-lsp client for Swift diagnostics.

The multilspy-backed gate in :mod:`misterdev.core.context.lsp` has no Swift
language server: its ``_LANG_MAP`` intentionally omits swift/c/cpp because
multilspy ships no server for them, so Swift projects get zero semantic
diagnostics before a build. Apple's ``sourcekit-lsp`` speaks plain LSP
JSON-RPC over stdio, so this module drives it directly with the standard
library only — no multilspy dependency.

Like the multilspy gate this is strictly best-effort: :func:`swift_diagnostics`
is bounded and never raises. On any problem (binary absent, timeout, protocol
error) it returns ``None`` meaning "no opinion / gate skips", so it can never
block or fail a build.

The framing primitives (:func:`frame_message` / :func:`parse_frames`) are pure
and are the core of the unit tests; the subprocess driver is thin around them.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from misterdev.logging_setup import setup_logger

logger = setup_logger(__name__)

_LSP_SEVERITY_ERROR = 1  # LSP DiagnosticSeverity.Error
_HEADER_SEP = b"\r\n\r\n"
_CONTENT_LENGTH = b"Content-Length:"


def frame_message(payload: dict) -> bytes:
    """Encode ``payload`` as one LSP stdio frame.

    Produces ``Content-Length: N\\r\\n\\r\\n<json>`` where N is the byte length
    of the UTF-8 JSON body. The inverse of :func:`parse_frames`.
    """
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def parse_frames(buf: bytes) -> Tuple[List[dict], bytes]:
    """Parse as many complete LSP frames as ``buf`` contains.

    Returns ``(messages, remainder)`` where ``messages`` are the decoded JSON
    payloads for every frame whose full body was present, and ``remainder`` is
    the trailing bytes of an incomplete frame (empty if ``buf`` ended on a frame
    boundary). Safe to call repeatedly on a growing buffer: feed the returned
    remainder back in with the next chunk to reassemble a split frame.

    A frame with an unparseable header or a malformed Content-Length is skipped
    (its header consumed) so a single bad frame cannot wedge the stream.
    """
    messages: List[dict] = []
    while True:
        sep = buf.find(_HEADER_SEP)
        if sep == -1:
            # No complete header yet; keep everything for the next chunk.
            return messages, buf
        header = buf[:sep]
        content_length = _content_length(header)
        if content_length is None:
            # Malformed header (no valid Content-Length). Drop it and resync
            # past the separator rather than blocking forever.
            buf = buf[sep + len(_HEADER_SEP) :]
            continue
        body_start = sep + len(_HEADER_SEP)
        body_end = body_start + content_length
        if len(buf) < body_end:
            # Body not fully arrived; retain the whole frame for reassembly.
            return messages, buf
        body = buf[body_start:body_end]
        buf = buf[body_end:]
        try:
            messages.append(json.loads(body.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            # A bad body is skipped; the buffer has already advanced past it.
            continue


def _content_length(header: bytes) -> Optional[int]:
    """Extract a non-negative Content-Length from a frame header, or None."""
    for line in header.split(b"\r\n"):
        if line[: len(_CONTENT_LENGTH)].lower() == _CONTENT_LENGTH.lower():
            try:
                value = int(line[len(_CONTENT_LENGTH) :].strip())
            except ValueError:
                return None
            return value if value >= 0 else None
    return None


def _resolve_binary() -> Optional[List[str]]:
    """Command prefix for sourcekit-lsp, or None if it cannot be located.

    Prefers a ``sourcekit-lsp`` on PATH; falls back to ``xcrun sourcekit-lsp``
    on macOS where the toolchain binary is not directly on PATH.
    """
    direct = shutil.which("sourcekit-lsp")
    if direct:
        return [direct]
    xcrun = shutil.which("xcrun")
    if xcrun:
        try:
            found = subprocess.run(
                [xcrun, "--find", "sourcekit-lsp"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        path = found.stdout.strip()
        if found.returncode == 0 and path:
            return [path]
    return None


def _to_uri(path: Path) -> str:
    return path.resolve().as_uri()


class SourceKitSession:
    """One long-lived ``sourcekit-lsp`` server shared across many files.

    The per-file :func:`swift_diagnostics` pays the initialize/initialized
    handshake (and process spawn) on every call, which dominates wall time when
    a project touches dozens of Swift files. This session does that handshake
    exactly once in :meth:`__enter__`, then :meth:`diagnostics` reuses the same
    server for each ``textDocument/didOpen``, so N files cost one spawn instead
    of N.

    Best-effort like the one-shot path: construction and every method are
    bounded and never raise. If the binary is absent or the handshake fails the
    session is *inactive* and :meth:`diagnostics` returns None for every file.
    Use as a context manager so :meth:`__exit__` always shuts the server down.
    """

    def __init__(
        self,
        project_root: str,
        timeout: float = 20.0,
        handshake_timeout: Optional[float] = None,
    ):
        self._root = Path(project_root).resolve()
        # Per-file settle budget: how long to wait for a publish before giving
        # up on one file without wedging the whole batch.
        self._timeout = max(timeout, 1.0)
        # Handshake budget: initialize round-trip on a cold server can be slow.
        self._handshake_timeout = max(
            handshake_timeout if handshake_timeout is not None else timeout, 5.0
        )
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._inbox: List[dict] = []
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._active = False
        self._version = 0

    def __enter__(self) -> "SourceKitSession":
        try:
            self._active = self._start()
        except Exception as e:  # spawn/handshake failures are non-fatal
            logger.debug(f"sourcekit-lsp session unavailable: {e}")
            self._active = False
            _shutdown(self._proc)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        _shutdown(self._proc)
        return False  # never suppress a caller's exception

    def diagnostics(self, file_path: str) -> Optional[list]:
        """Error-severity diagnostics for one file on the shared server.

        Opens ``file_path`` via ``textDocument/didOpen`` and collects the first
        ``textDocument/publishDiagnostics`` for it, returning a list of
        ``{"line", "message", "severity"}`` for error-severity diagnostics.

        None means "no opinion": the session never came up, the file is missing
        or unreadable, or no publish arrived within the per-file timeout. Never
        raises, so one bad file cannot abort a batch.
        """
        try:
            return self._diagnostics(file_path)
        except Exception as e:  # protocol failures are non-fatal
            logger.debug(f"Swift LSP diagnostics unavailable ({file_path}): {e}")
            return None

    def _start(self) -> bool:
        cmd = _resolve_binary()
        if cmd is None:
            logger.debug("sourcekit-lsp not found; skipping Swift diagnostics")
            return False

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(self._root),
        )
        assert self._proc.stdout is not None
        stdout = self._proc.stdout

        # Read frames on a daemon thread so a stalled server can never block a
        # caller past its deadline; methods poll the shared inbox under a lock.
        def _read() -> None:
            buf = b""
            while not self._stop.is_set():
                try:
                    chunk = stdout.read(4096)
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                buf += chunk
                try:
                    msgs, buf = parse_frames(buf)
                except Exception:
                    break
                if msgs:
                    with self._lock:
                        self._inbox.extend(msgs)

        self._reader = threading.Thread(target=_read, daemon=True)
        self._reader.start()

        self._send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": os.getpid(),
                    "rootUri": _to_uri(self._root),
                    "capabilities": {
                        "textDocument": {
                            "publishDiagnostics": {"relatedInformation": False}
                        }
                    },
                },
            }
        )
        if not self._await_initialize():
            return False
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        return True

    def _await_initialize(self) -> bool:
        """Wait for the initialize result, or False on timeout/exit."""
        deadline = time.monotonic() + self._handshake_timeout
        while time.monotonic() < deadline:
            for msg in self._drain():
                if msg.get("id") == 1 and "result" in msg:
                    return True
            if self._proc is None or self._proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    def _diagnostics(self, file_path: str) -> Optional[list]:
        if not self._active or self._proc is None:
            return None

        target = Path(file_path)
        if not target.is_absolute():
            target = (self._root / target).resolve()
        else:
            target = target.resolve()
        if not target.is_file():
            logger.debug(f"Swift source not found for diagnostics: {target}")
            return None

        try:
            source = target.read_text(encoding="utf-8")
        except OSError as e:
            logger.debug(f"Cannot read Swift source {target}: {e}")
            return None

        target_uri = _to_uri(target)
        # Drop any stale frames buffered before this file was opened so a prior
        # file's publish can never be mistaken for this one's.
        self._drain()
        self._version += 1
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": target_uri,
                        "languageId": "swift",
                        "version": self._version,
                        "text": source,
                    }
                },
            }
        )

        result = None
        try:
            deadline = time.monotonic() + self._timeout
            while time.monotonic() < deadline:
                for msg in self._drain():
                    if msg.get("method") != "textDocument/publishDiagnostics":
                        continue
                    params = msg.get("params", {})
                    if params.get("uri") == target_uri:
                        result = _to_errors(params.get("diagnostics", []))
                        return result
                if self._proc.poll() is not None:
                    break
                time.sleep(0.05)
            return None
        finally:
            try:
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/didClose",
                        "params": {"textDocument": {"uri": target_uri}},
                    }
                )
            except Exception:
                pass

    def _send(self, payload: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("sourcekit-lsp process not running")
        with self._send_lock:
            self._proc.stdin.write(frame_message(payload))
            self._proc.stdin.flush()

    def _drain(self) -> List[dict]:
        with self._lock:
            pending = self._inbox[:]
            self._inbox.clear()
        return pending


def swift_diagnostics(
    project_root: str, file_path: str, timeout: float = 20.0
) -> Optional[list]:
    """Error-severity Swift diagnostics for ``file_path``, or None if skipped.

    Launches ``sourcekit-lsp`` over stdio, performs the LSP handshake
    (initialize -> initialized -> textDocument/didOpen) and collects
    ``textDocument/publishDiagnostics`` for the opened file, returning a list of
    ``{"line", "message", "severity"}`` for error-severity (severity == 1)
    diagnostics.

    A thin one-shot wrapper over :class:`SourceKitSession`: it opens a session,
    queries a single file, and closes it, so existing callers and tests keep the
    same signature and semantics. Prefer :func:`diagnostics_for` for many files.

    None means "no opinion": the binary is absent, the handshake failed, the
    server did not respond within ``timeout``, or any other error occurred.
    This function never raises — callers treat None as a no-op, never a
    pass/fail signal.
    """
    try:
        with SourceKitSession(project_root, timeout) as session:
            return session.diagnostics(file_path)
    except Exception as e:  # sourcekit-lsp / protocol failures are non-fatal
        logger.debug(f"Swift LSP diagnostics unavailable ({file_path}): {e}")
        return None


def diagnostics_for(
    project_root: str, file_paths: List[str], timeout: float = 30.0
) -> Dict[str, Optional[list]]:
    """Batch Swift diagnostics for many files over ONE shared server.

    Opens a single :class:`SourceKitSession` and queries every path against it,
    paying the spawn/handshake once for the whole batch instead of once per
    file. Returns ``{file_path: diagnostics}`` keyed by the caller's original
    string, where each value is the same list-or-None :func:`swift_diagnostics`
    yields (None = no opinion for that file).

    ``timeout`` is the total budget, split evenly across the files as a per-file
    settle bound so the batch can't run unbounded. Never raises: if the session
    fails to come up every entry is None.
    """
    if not file_paths:
        return {}
    per_file = max(timeout / len(file_paths), 1.0)
    results: Dict[str, Optional[list]] = {path: None for path in file_paths}
    try:
        with SourceKitSession(
            project_root, per_file, handshake_timeout=timeout
        ) as session:
            for path in file_paths:
                results[path] = session.diagnostics(path)
    except Exception as e:  # session/protocol failures are non-fatal
        logger.debug(f"Swift LSP batch diagnostics unavailable: {e}")
    return results


def _to_errors(diags: List[dict]) -> List[dict]:
    """Filter to error-severity diagnostics in the gate's output shape."""
    errors: List[dict] = []
    for diag in diags:
        if diag.get("severity") == _LSP_SEVERITY_ERROR:
            line = diag.get("range", {}).get("start", {}).get("line", 0) + 1
            errors.append(
                {
                    "line": line,
                    "message": diag.get("message", ""),
                    "severity": _LSP_SEVERITY_ERROR,
                }
            )
    return errors


def _shutdown(proc: Optional[subprocess.Popen]) -> None:
    """Terminate the server, escalating to kill, never raising."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
    except Exception:  # cleanup must not mask the primary result
        pass
    finally:
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
