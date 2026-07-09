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
from typing import List, Optional, Tuple

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


def swift_diagnostics(
    project_root: str, file_path: str, timeout: float = 20.0
) -> Optional[list]:
    """Error-severity Swift diagnostics for ``file_path``, or None if skipped.

    Launches ``sourcekit-lsp`` over stdio, performs the LSP handshake
    (initialize -> initialized -> textDocument/didOpen) and collects
    ``textDocument/publishDiagnostics`` for the opened file, returning a list of
    ``{"line", "message", "severity"}`` for error-severity (severity == 1)
    diagnostics.

    None means "no opinion": the binary is absent, the handshake failed, the
    server did not respond within ``timeout``, or any other error occurred.
    This function never raises — callers treat None as a no-op, never a
    pass/fail signal.
    """
    try:
        return _run(project_root, file_path, timeout)
    except Exception as e:  # sourcekit-lsp / protocol failures are non-fatal
        logger.debug(f"Swift LSP diagnostics unavailable ({file_path}): {e}")
        return None


def _run(project_root: str, file_path: str, timeout: float) -> Optional[list]:
    cmd = _resolve_binary()
    if cmd is None:
        logger.debug("sourcekit-lsp not found; skipping Swift diagnostics")
        return None

    root = Path(project_root).resolve()
    target = Path(file_path)
    if not target.is_absolute():
        target = (root / target).resolve()
    if not target.is_file():
        logger.debug(f"Swift source not found for diagnostics: {target}")
        return None

    try:
        source = target.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug(f"Cannot read Swift source {target}: {e}")
        return None

    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(root),
    )
    try:
        return _drive(proc, root, target, source, deadline)
    finally:
        _shutdown(proc)


def _drive(
    proc: subprocess.Popen,
    root: Path,
    target: Path,
    source: str,
    deadline: float,
) -> Optional[list]:
    assert proc.stdin is not None and proc.stdout is not None
    target_uri = _to_uri(target)

    def _send(payload: dict) -> None:
        proc.stdin.write(frame_message(payload))
        proc.stdin.flush()

    _send(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": os.getpid(),
                "rootUri": _to_uri(root),
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {"relatedInformation": False}
                    }
                },
            },
        }
    )

    # Read frames on a daemon thread so a stalled server can never block past
    # the deadline; the main loop polls the shared buffer under a lock.
    lock = threading.Lock()
    inbox: List[dict] = []
    stop = threading.Event()

    def _reader() -> None:
        buf = b""
        while not stop.is_set():
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            msgs, buf = parse_frames(buf)
            if msgs:
                with lock:
                    inbox.extend(msgs)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    initialized = False
    opened = False
    diagnostics: List[dict] = []

    while time.monotonic() < deadline:
        with lock:
            pending = inbox[:]
            inbox.clear()

        for msg in pending:
            if not initialized and msg.get("id") == 1 and "result" in msg:
                # Handshake: ack initialize, then open the document.
                _send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
                _send(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/didOpen",
                        "params": {
                            "textDocument": {
                                "uri": target_uri,
                                "languageId": "swift",
                                "version": 1,
                                "text": source,
                            }
                        },
                    }
                )
                initialized = True
                opened = True
            elif msg.get("method") == "textDocument/publishDiagnostics":
                params = msg.get("params", {})
                if params.get("uri") == target_uri:
                    diagnostics = _to_errors(params.get("diagnostics", []))
                    # First publish for our file is authoritative; done.
                    return diagnostics

        if proc.poll() is not None:
            break
        time.sleep(0.05)

    # Timed out or server exited before publishing for our file.
    if not opened:
        return None
    return diagnostics if diagnostics else None


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


def _shutdown(proc: subprocess.Popen) -> None:
    """Terminate the server, escalating to kill, never raising."""
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
