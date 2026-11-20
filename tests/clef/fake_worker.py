"""Frame process used by the Rust serve tests.

No checkpoint and no CUDA. Every response names the process and the thread
that scored it, so a test can show that HTTP requests share one main thread.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time

from models.clef.ipc import (
    OPCODE_DECIDE,
    OPCODE_SHUTDOWN,
    claim_protocol_stdout,
    read_frame,
    write_response,
)


def identity() -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "cuda_thread_id": threading.get_ident(),
        "cuda_main_thread": threading.current_thread() is threading.main_thread(),
        "active_threads": threading.active_count(),
    }


def main() -> None:
    if os.environ.get("CLEF_FAKE_WARMUP_FAIL") == "1":
        print("warmup failed", file=sys.stderr)
        raise SystemExit(2)
    protocol = claim_protocol_stdout()
    time.sleep(float(os.environ.get("CLEF_FAKE_WARMUP", "0")))
    delay = float(os.environ.get("CLEF_FAKE_DELAY", "0"))
    fail = os.environ.get("CLEF_FAKE_FAIL") == "1"
    health = {
        "status": "ready",
        "model": "Cloudflare/clef-flash",
        "variant": "clef-flash",
        "revision": "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
        "max_input_tokens": 16384,
        "modalities": ["text"],
        "prompt_mode": "non-thinking",
        "linear_attention": "fla+causal_conv1d",
        **identity(),
    }
    write_response(protocol, 200, health)
    while True:
        frame = read_frame(sys.stdin.buffer)
        if frame is None:
            return
        opcode, payload = frame[0], frame[1:]
        if opcode == OPCODE_SHUTDOWN:
            return
        if opcode != OPCODE_DECIDE:
            write_response(protocol, 400, {"detail": "unknown worker opcode"})
            continue
        if trace := os.environ.get("CLEF_FAKE_TRACE"):
            state = json.loads(payload)["state"]
            with open(trace, "a") as events:
                events.write(f"start {state}\n")
            if state in ("block", "crash", "malformed"):
                while not Path(trace + ".release").exists():
                    time.sleep(0.005)
            if state == "crash":
                os._exit(7)
            if state == "malformed":
                protocol.write(b"\x00\x00\x00\x01x")
                protocol.flush()
                continue
        time.sleep(delay)
        if fail:
            write_response(protocol, 500, {"detail": "inference failed"})
            continue
        try:
            request = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            write_response(
                protocol,
                400,
                {"detail": "request body must contain valid JSON and UTF-8 text"},
            )
            continue
        if not isinstance(request, dict):
            write_response(protocol, 400, {"detail": "request must be a JSON object"})
            continue
        if trace:
            with open(trace, "a") as events:
                events.write(f"finish {state}\n")
        write_response(
            protocol,
            200,
            {
                "model": request.get("model", "clef-flash"),
                "answers": {"urgent": {"type": "noul", "noul": 0.25}},
                "usage": {"input_tokens": 3, "output_tokens": 0},
                **identity(),
            },
        )
        if os.environ.get("CLEF_FAKE_EXIT") == "1":
            os._exit(0)


if __name__ == "__main__":
    main()
