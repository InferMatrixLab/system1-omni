"""Length-prefixed frames between ``omni-clef`` and the Clef executor.

The Rust HTTP server is the only public socket. This stream is private to
one executor process. Every integer is big-endian.

A request frame is ``u32 length``, then ``u8 opcode``, then the payload.
Opcode 1 is one ``/v1/systemone`` body. Opcode 2 shuts the process down.

A response frame is ``u32 length``, then ``u16 status``, then the JSON
bytes that the HTTP server forwards unchanged. The first frame the
executor writes, before it reads a request, is ``200`` and the health
document. ``stdout`` of the executor process is this stream; logs go to
stderr.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from typing import Any, BinaryIO

MAX_FRAME = 16 * 1024 * 1024
OPCODE_DECIDE = 1
OPCODE_SHUTDOWN = 2
_LEN = struct.Struct(">I")
_STATUS = struct.Struct(">H")


class FrameError(RuntimeError):
    """A frame could not be read or written. The process should exit."""


def _read_exact(stream: BinaryIO, count: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if chunks:
                raise FrameError("truncated frame")
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO) -> bytes | None:
    """Return one payload, or ``None`` at a clean end of stream."""

    prefix = _read_exact(stream, 4)
    if prefix is None:
        return None
    (length,) = _LEN.unpack(prefix)
    if length < 1 or length > MAX_FRAME:
        raise FrameError(f"frame length {length} is outside 1..={MAX_FRAME}")
    payload = _read_exact(stream, length)
    if payload is None:
        raise FrameError("truncated frame")
    return payload


def write_frame(stream: BinaryIO, payload: bytes) -> None:
    if not payload or len(payload) > MAX_FRAME:
        raise FrameError(f"frame length {len(payload)} is outside 1..={MAX_FRAME}")
    stream.write(_LEN.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def encode_request(opcode: int, body: bytes = b"") -> bytes:
    """One complete request frame, including its length prefix."""

    if opcode < 0 or opcode > 255:
        raise FrameError(f"opcode {opcode} does not fit in one byte")
    payload = bytes((opcode,)) + body
    if len(payload) > MAX_FRAME:
        raise FrameError(f"frame length {len(payload)} is outside 1..={MAX_FRAME}")
    return _LEN.pack(len(payload)) + payload


def write_response(stream: BinaryIO, status: int, body: dict[str, Any]) -> None:
    raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if status < 0 or status > 0xFFFF:
        raise FrameError(f"status {status} does not fit in u16")
    write_frame(stream, _STATUS.pack(status) + raw)


def decode_response(payload: bytes) -> tuple[int, bytes]:
    if len(payload) < 2:
        raise FrameError("response frame is missing a status")
    (status,) = _STATUS.unpack(payload[:2])
    return status, payload[2:]


def claim_protocol_stdout() -> BinaryIO:
    """Point ``sys.stdout`` at stderr and return the original stdout.

    Checkpoint loaders print progress. Those bytes have to stay off the
    frame stream or the ready frame is no longer aligned. Call this before
    importing the snapshot.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    return os.fdopen(protocol_fd, "wb", buffering=0)
