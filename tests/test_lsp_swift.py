"""Tests for the pure LSP framing primitives and the never-raising guard.

The subprocess-driven ``swift_diagnostics`` is exercised only for its
binary-absent path (monkeypatched); no real sourcekit-lsp server is launched.
"""

import shutil

from misterdev.core.context import lsp_swift
from misterdev.core.context.lsp_swift import (
    frame_message,
    parse_frames,
    swift_diagnostics,
)


def test_frame_message_round_trips_through_parse():
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    msgs, remainder = parse_frames(frame_message(payload))
    assert msgs == [payload]
    assert remainder == b""


def test_frame_message_uses_byte_length_not_char_length():
    # A multi-byte UTF-8 body must round-trip: Content-Length is in bytes.
    payload = {"message": "café — naïve ☃"}
    msgs, remainder = parse_frames(frame_message(payload))
    assert msgs == [payload]
    assert remainder == b""


def test_parse_frames_handles_multiple_concatenated_frames():
    a = {"id": 1}
    b = {"id": 2}
    msgs, remainder = parse_frames(frame_message(a) + frame_message(b))
    assert msgs == [a, b]
    assert remainder == b""


def test_parse_frames_reassembles_frame_split_across_two_chunks():
    payload = {"method": "textDocument/publishDiagnostics", "params": {"uri": "x"}}
    whole = frame_message(payload)
    split = len(whole) // 2

    # First chunk is incomplete: nothing parsed, everything retained.
    msgs, remainder = parse_frames(whole[:split])
    assert msgs == []
    assert remainder == whole[:split]

    # Feeding the remainder plus the rest reassembles the frame.
    msgs, remainder = parse_frames(remainder + whole[split:])
    assert msgs == [payload]
    assert remainder == b""


def test_parse_frames_split_inside_header_reassembles():
    payload = {"id": 7}
    whole = frame_message(payload)
    # Split partway through the header (before the \r\n\r\n separator).
    msgs, remainder = parse_frames(whole[:5])
    assert msgs == []
    assert remainder == whole[:5]
    msgs, remainder = parse_frames(remainder + whole[5:])
    assert msgs == [payload]
    assert remainder == b""


def test_parse_frames_retains_trailing_partial_after_complete_frame():
    complete = frame_message({"id": 1})
    partial = frame_message({"id": 2})[:6]
    msgs, remainder = parse_frames(complete + partial)
    assert msgs == [{"id": 1}]
    assert remainder == partial


def test_parse_frames_empty_buffer():
    msgs, remainder = parse_frames(b"")
    assert msgs == []
    assert remainder == b""


def test_swift_diagnostics_returns_none_when_binary_absent(monkeypatch, tmp_path):
    # No sourcekit-lsp and no xcrun on PATH -> "no opinion", never raises.
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(lsp_swift.shutil, "which", lambda name: None)
    src = tmp_path / "main.swift"
    src.write_text("let x = 1\n")
    assert swift_diagnostics(str(tmp_path), str(src)) is None


def test_swift_diagnostics_returns_none_for_missing_file(monkeypatch, tmp_path):
    # Even if a binary "exists", a non-existent source is a no-op, not a raise.
    monkeypatch.setattr(
        lsp_swift, "_resolve_binary", lambda: ["/nonexistent/sourcekit-lsp"]
    )
    missing = tmp_path / "does_not_exist.swift"
    assert swift_diagnostics(str(tmp_path), str(missing)) is None
