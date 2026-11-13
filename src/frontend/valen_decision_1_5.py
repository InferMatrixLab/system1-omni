"""Python reference worker for Valen-Decision-1.5.

The Rust frontend remains the public transport layer. This worker owns only
HTTP lifecycle and request orchestration; preprocessing lives in
models.valen_decision_1_5.preprocess.ValenDecisionProcessor and model
execution lives in models.valen_decision_1_5.engine.ValenDecisionExecutor.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from models.valen_decision_1_5.engine import DTYPES, ExecutorOutput, ValenDecisionExecutor
from models.valen_decision_1_5.pinned import verify_checkpoint
from models.valen_decision_1_5.postprocess import answer, build_response
from models.valen_decision_1_5.preprocess import ValenDecisionProcessor, prepare_request
from models.valen_decision_1_5.protocol import (
    MAX_BODY,
    RequestError,
    parse_body,
    parse_request,
)

LOG = logging.getLogger(__name__)


def decide(
    raw: bytes, processor: ValenDecisionProcessor, executor: ValenDecisionExecutor
) -> dict:
    """Prepare, compile, execute, and finish one complete request."""

    request = parse_request(
        parse_body(raw), executor.model_name, special_tokens=processor.special_tokens
    )
    with TemporaryDirectory(prefix="system1-valen-decision-") as directory:
        prepared = prepare_request(request, Path(directory))
        try:
            compiled = processor.compile(prepared)
        except RequestError:
            raise
        except ValueError as exc:
            # The pinned compiler is the input-validation authority for what
            # the wire layer cannot foresee (token budget, reserved tokens);
            # its rejections are client errors, not worker failures.
            raise RequestError(
                f"request rejected by the pinned compiler: {exc}", 422
            ) from exc
        output = executor.execute(compiled)
        if not isinstance(output, ExecutorOutput):
            raise RuntimeError("Valen-Decision-1.5 executor returned an invalid output object")
        if len(output.logits) != len(compiled.response_context.questions):
            raise RuntimeError("Valen-Decision-1.5 executor returned an unexpected answer count")
        answers = {
            question.name: answer(question, logits)
            for question, logits in zip(compiled.response_context.questions, output.logits)
        }
        return build_response(
            compiled.response_context,
            answers,
            output.logical_tokens,
            output.compute_tokens,
        )


def _warmup_requests(model_name: str):
    """One image and one text request, each asking all three question types."""

    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(stream, "PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    questions = {
        "_choice": {
            "type": "choice",
            "instructions": "Choose the only candidate.",
            "criteria": {"ok": "OK"},
        },
        "_noul": {"type": "noul", "instructions": "This is a warmup request."},
        "_score": {
            "type": "score",
            "instructions": "Grade the warmup.",
            "criteria": ["Not warm.", "Warm."],
        },
    }
    for state in (
        {"image": f"data:image/png;base64,{encoded}"},
        "Warmup: answer every question.",
    ):
        yield parse_request(
            {"model": model_name, "state": state, "questions": questions}, model_name
        )


def warmup(processor: ValenDecisionProcessor, executor: ValenDecisionExecutor) -> None:
    for request in _warmup_requests(executor.model_name):
        with TemporaryDirectory(prefix="system1-valen-decision-warmup-") as directory:
            prepared = prepare_request(request, Path(directory))
            executor.warmup(processor.compile(prepared))


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, processor: ValenDecisionProcessor, executor: ValenDecisionExecutor):
        self.processor = processor
        self.executor = executor
        self.inference_lock = threading.Lock()
        self._executor_closed = False
        super().__init__(address, Handler)

    def server_close(self):
        super().server_close()
        if not self._executor_closed:
            self.executor.close()
            self._executor_closed = True


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        # Do not log paths, images, prompts, or arbitrary request headers.
        pass

    def send_json(self, status: int, value: dict):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "status": "ready",
                    "model": self.server.executor.model_name,
                    "modality": "multimodal",
                },
            )
        else:
            self.send_json(404, {"detail": "unknown route"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self.send_json(404, {"detail": "unknown route"})
            return
        if self.headers.get("Transfer-Encoding"):
            self.send_json(411, {"detail": "Content-Length is required; chunked requests are unsupported"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            self.send_json(411, {"detail": "one Content-Length is required"})
            return
        try:
            length = int(lengths[0])
        except ValueError:
            self.send_json(400, {"detail": "invalid Content-Length"})
            return
        if length < 0 or length > MAX_BODY:
            self.send_json(413, {"detail": "request exceeds body limit"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"detail": "Content-Type must be application/json"})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.send_json(503, {"detail": "worker busy"})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                self.send_json(400, {"detail": "incomplete body"})
                return
            self.send_json(200, decide(raw, self.server.processor, self.server.executor))
        except RequestError as exc:
            self.send_json(exc.status, {"detail": str(exc)})
        except (TimeoutError, socket.timeout):
            self.send_json(408, {"detail": "request body timed out"})
        except Exception as exc:
            LOG.error("Valen-Decision-1.5 inference failed: %s", type(exc).__name__)
            self.send_json(500, {"detail": "inference failed"})
        finally:
            self.server.inference_lock.release()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", required=True, help="pinned Valen-Decision-1.5 checkpoint directory"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument(
        "--no-verify-weights",
        action="store_true",
        help="skip hashing the weight shards; code, config and tokenizer are always verified",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        checkpoint = verify_checkpoint(args.checkpoint, weights=not args.no_verify_weights)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    processor = ValenDecisionProcessor(checkpoint)
    executor = ValenDecisionExecutor(checkpoint, args.device, args.dtype)
    server = None
    try:
        warmup(processor, executor)
        server = WorkerServer((args.host, args.port), processor, executor)
        LOG.info(
            "Valen-Decision-1.5 worker %s ready on %s:%s",
            executor.model_name,
            args.host,
            args.port,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        else:
            executor.close()


if __name__ == "__main__":
    main()
