"""The executor loop stays on the calling thread and keeps Clef's envelope."""

import io
import json
import threading
from threading import Thread

import pytest

from models.clef import ipc
from models.clef.ipc import OPCODE_DECIDE, OPCODE_SHUTDOWN, FrameError, decode_response, read_frame
from models.clef.worker import run_stdio


def three_kinds():
    return {
        "model": "clef-flash",
        "state": "Checkout has been failing for every customer.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        },
    }


class Guard(io.BytesIO):
    def __init__(self, raw: bytes):
        super().__init__(raw)
        self.allow = False

    def read(self, size: int = -1) -> bytes:
        if not self.allow:
            raise AssertionError("executor read a request before warmup")
        return super().read(size)


class Recording:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.events: list[str] = []
        self.threads: list[int] = []
        self.guard: Guard | None = None

    def warmup(self):
        self.events.append("warmup")
        self.threads.append(threading.get_ident())
        if self.guard is not None:
            self.guard.allow = True

    def health(self):
        return {
            "status": "ready",
            "model": "Cloudflare/clef-flash",
            "variant": "clef-flash",
            "prompt_mode": "non-thinking",
            "modalities": ["text"],
            "max_input_tokens": 16384,
            "linear_attention": "fla+causal_conv1d",
        }

    def serve(self, payload):
        self.events.append("serve")
        self.threads.append(threading.get_ident())
        if self.fail:
            raise RuntimeError("boom")
        return {
            "model": payload["model"],
            "answers": {"urgent": {"type": "noul", "noul": 0.25}},
            "usage": {"input_tokens": 3, "output_tokens": 0},
        }

    def close(self):
        self.events.append("close")


def _run(executor: Recording, frames: bytes) -> list[tuple[int, dict]]:
    guard = Guard(frames)
    executor.guard = guard
    stdout = io.BytesIO()
    run_stdio(executor, guard, stdout)
    stdout.seek(0)
    messages = []
    while True:
        payload = read_frame(stdout)
        if payload is None:
            return messages
        status, raw = decode_response(payload)
        messages.append((status, json.loads(raw)))


def test_requests_are_scored_on_the_calling_main_thread():
    raw = json.dumps(three_kinds()).encode()
    frames = ipc.encode_request(OPCODE_DECIDE, raw) + ipc.encode_request(OPCODE_SHUTDOWN)
    executor = Recording()
    messages = _run(executor, frames)
    assert executor.events == ["warmup", "serve", "close"]
    assert executor.threads
    assert set(executor.threads) == {threading.get_ident()}
    assert threading.current_thread() is threading.main_thread()
    assert messages[0][0] == 200
    assert messages[0][1]["prompt_mode"] == "non-thinking"
    assert messages[0][1]["modalities"] == ["text"]
    assert messages[1][0] == 200
    assert messages[1][1]["answers"]["urgent"]["noul"] == 0.25
    assert messages[1][1]["usage"]["output_tokens"] == 0


def test_request_error_status_is_forwarded_and_the_loop_continues():
    images = three_kinds()
    images["images"] = ["data:image/png;base64,AAAA"]
    frames = (
        ipc.encode_request(OPCODE_DECIDE, json.dumps(images).encode())
        + ipc.encode_request(OPCODE_DECIDE, json.dumps(three_kinds()).encode())
        + ipc.encode_request(OPCODE_SHUTDOWN)
    )
    messages = _run(Recording(), frames)
    assert messages[1][0] == 422
    assert "images are not served" in messages[1][1]["detail"]
    assert messages[2][0] == 200
    assert messages[2][1]["answers"]["urgent"]["noul"] == 0.25


def test_inference_failure_is_500_and_does_not_stop_the_loop():
    frames = ipc.encode_request(OPCODE_DECIDE, json.dumps(three_kinds()).encode())
    frames += ipc.encode_request(OPCODE_DECIDE, json.dumps(three_kinds()).encode())
    frames += ipc.encode_request(OPCODE_SHUTDOWN)
    messages = _run(Recording(fail=True), frames)
    assert messages[1] == (500, {"detail": "inference failed"})
    assert messages[2] == (500, {"detail": "inference failed"})


def test_run_stdio_rejects_a_worker_thread():
    errors: list[str] = []

    def target():
        try:
            run_stdio(Recording(), io.BytesIO(), io.BytesIO())
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = Thread(target=target)
    thread.start()
    thread.join()
    assert errors
    assert "main thread" in errors[0]


def test_frame_limits():
    import struct

    with pytest.raises(FrameError, match="truncated"):
        read_frame(io.BytesIO(struct.pack(">I", 4) + b"\x01"))
    with pytest.raises(FrameError, match="outside"):
        read_frame(io.BytesIO(struct.pack(">I", ipc.MAX_FRAME + 1)))
