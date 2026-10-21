"""Python reference worker for d3 (Decision 3.0).

The Rust frontend remains the public transport layer. This worker owns only
HTTP lifecycle and request orchestration; wire admission lives in
models.d3.protocol, prepared-error admission in models.d3.preprocess, the
pinned runtime and its manifest verification in models.d3.engine, and
response finishing in models.d3.postprocess.

The error surface has three layers: transport errors follow the shared
Python-worker discipline; a request with an invalid or over-limit question is
rejected whole with HTTP 422 before any execution (nothing is truncated); and
a question whose model output cannot be decoded keeps its HTTP 200 response
with the failure embedded in that question's answer — plus a worker warning,
because a silent partial failure is otherwise undetectable.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from models.d3.engine import VERIFY_MODES, D3Executor
from models.d3.postprocess import finish_response
from models.d3.preprocess import reject_pre_execution_errors
from models.d3.protocol import MAX_BODY, RequestError, parse_body, parse_request

LOG = logging.getLogger(__name__)


def decide(raw: bytes, executor: D3Executor) -> dict[str, Any]:
    """Admit, execute, and finish one complete request."""

    request = parse_request(parse_body(raw))
    questions = dict(request.questions)
    try:
        prepared = executor.prepare(request.state, questions)
    except ValueError as exc:
        # The pinned runtime is the validation authority for what the wire
        # layer cannot foresee (envelope-level question problems); its
        # rejections are client errors, not worker failures.
        raise RequestError(f"request rejected by the pinned runtime: {exc}", 422) from exc
    reject_pre_execution_errors(prepared)
    probabilities, tokens = executor.run(prepared)
    response = executor.respond(prepared, probabilities, tokens)
    return finish_response(tuple(questions), response)


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, executor: D3Executor):
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
        # Do not log paths, prompts, or arbitrary request headers.
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

    def _send_request_error(self, exc: RequestError):
        if exc.body is not None:
            self.send_json(exc.status, exc.body)
        else:
            self.send_json(exc.status, {"detail": str(exc)})

    def do_GET(self):
        if self.path == "/health":
            executor = self.server.executor
            self.send_json(
                200,
                {
                    "status": "ready",
                    "model": executor.model_name,
                    "max_input_tokens": executor.max_length,
                    "modalities": list(executor.modalities),
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
            self.send_json(200, decide(raw, self.server.executor))
        except RequestError as exc:
            self._send_request_error(exc)
        except (TimeoutError, socket.timeout):
            self.send_json(408, {"detail": "request body timed out"})
        except Exception as exc:
            LOG.error("d3 inference failed: %s", type(exc).__name__)
            self.send_json(500, {"detail": "inference failed"})
        finally:
            self.server.inference_lock.release()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="pinned d3 package directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--verify", default="fast", choices=VERIFY_MODES)
    parser.add_argument("--name", help="served model name (default: the package's model name)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    executor = D3Executor(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        verify=args.verify,
        model_name=args.name,
    )
    server = None
    try:
        executor.warmup()
        server = WorkerServer((args.host, args.port), executor)
        LOG.info("%s worker ready on %s:%s", executor.model_name, args.host, args.port)
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
