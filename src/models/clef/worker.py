"""Persistent Clef executor process.

``omni-clef`` spawns this module once. Startup, warmup, and every later
``serve`` run on this process's main thread, which is the thread that
first touches CUDA. The process does not listen for HTTP and does not
start a thread per request.

Request admission lives in the Rust server. This process only scores the
body it is given, using the same ``decide`` path the Python worker used.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Any, BinaryIO

from models.clef.engine import DEFAULT_VARIANT, FLA_PREFILL, GDN_PREFILL, VARIANTS, ClefExecutor
from models.clef.ipc import (
    OPCODE_DECIDE,
    OPCODE_SHUTDOWN,
    FrameError,
    claim_protocol_stdout,
    read_frame,
    write_response,
)
from models.clef.preprocess import validate_questions
from models.clef.protocol import RequestError, parse_body, parse_request

LOG = logging.getLogger(__name__)


def decide(raw: bytes, executor: Any) -> dict[str, Any]:
    """Admit one complete text request and run it if it fits intact."""

    request = parse_request(parse_body(raw))
    validate_questions(request)
    return executor.serve(request.payload)


def _error_body(exc: RequestError) -> dict[str, Any]:
    if exc.body is not None:
        return exc.body
    return {"detail": str(exc)}


def respond(stdout: BinaryIO, executor: Any, raw: bytes) -> None:
    try:
        write_response(stdout, 200, decide(raw, executor))
    except RequestError as exc:
        write_response(stdout, exc.status, _error_body(exc))
    except Exception as exc:
        LOG.error("clef inference failed: %s", type(exc).__name__)
        write_response(stdout, 500, {"detail": "inference failed"})


def run_stdio(executor: Any, stdin: BinaryIO, stdout: BinaryIO) -> None:
    """Warm up, publish health, then score requests on the calling thread.

    The calling thread must be the process main thread. A CUDA context
    created here stays on this thread for every later request.
    """

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Clef executor must run on the process main thread")
    executor.warmup()
    write_response(stdout, 200, executor.health())
    while True:
        try:
            frame = read_frame(stdin)
        except FrameError:
            LOG.error("clef executor frame was truncated")
            raise
        if frame is None:
            break
        opcode = frame[0]
        if opcode == OPCODE_SHUTDOWN:
            break
        if opcode != OPCODE_DECIDE:
            write_response(stdout, 400, {"detail": "unknown worker opcode"})
            continue
        respond(stdout, executor, frame[1:])
    executor.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Clef requests on this process's main thread. Not an HTTP server."
    )
    parser.add_argument("--model", required=True, help="snapshot directory of the pinned revision")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gdn-prefill",
        default=FLA_PREFILL,
        choices=(FLA_PREFILL, GDN_PREFILL),
        help=(
            "Linear-attention prefill. fla is the measured default and stays "
            "the default. flashinfer is opt-in for reported majors 9, 10, and "
            "12 with flashinfer-python==0.6.16.post3. Answer parity is "
            "recipe/clef/answer_parity.py; do not make flashinfer the default "
            "until that gate passes."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol_out = claim_protocol_stdout()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    executor = ClefExecutor(
        args.model,
        variant=args.variant,
        device=args.device,
        gdn_prefill=args.gdn_prefill,
    )
    try:
        run_stdio(executor, sys.stdin.buffer, protocol_out)
    except KeyboardInterrupt:
        pass
    finally:
        executor.close()


if __name__ == "__main__":
    main()
