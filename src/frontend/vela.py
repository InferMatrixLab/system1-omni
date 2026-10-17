"""Python reference worker for Vela 2.0 0.3B.

The Rust frontend remains the public transport layer. This worker owns
only HTTP lifecycle and request orchestration; wire admission lives in
models.vela.protocol, the pinned-runtime error mapping in
models.vela.preprocess, the pinned runtime and its package gate in
models.vela.engine, and response finishing in models.vela.postprocess.

The error surface has two layers: transport errors follow the shared
Python-worker discipline, and every request rejection the pinned runtime
makes — an invalid field, an invalid question, a questions-overflow — is
the upstream whole-request 422 ``{"detail": [{"loc", "msg", "type"}]}``
(or 413 ``too_long``), passed through verbatim. There are no per-question
errors. Over-limit state text is truncated exactly as the pinned
reference truncates it (see the model contract); only questions plus
schema that cannot fit the 8,192-token window are refused.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from models.vela.engine import Vela2Executor
from models.vela.postprocess import finish_response
from models.vela.preprocess import map_runtime_error, to_request_error
from models.vela.protocol import MODEL_NAME, MAX_BODY, RequestError, parse_body, admit_request

LOG = logging.getLogger(__name__)

# Upstream vela2_serve.py /v1/models entry, kept verbatim so clients see
# the reference server's description and release date.
MODELS_ENTRY = {
    "name": None,  # the served model name, filled per request
    "release_date": "2026-09-29",
    "description": (
        "Vela 2.0 0.3B: SystemOne Choice / Noul / Score "
        "with span and set extensions, 8,192-token context"
    ),
}
UNAUTHORIZED_BODY = {"detail": "Missing or invalid API key"}


def decide(raw: bytes, executor: Vela2Executor) -> dict[str, Any]:
    """Admit, execute, and finish one complete request."""

    body = admit_request(parse_body(raw), executor.served)
    questions = body["questions"]
    try:
        response = executor.system_one(body)
    except executor.SystemOneError as exc:
        raise to_request_error(exc) from exc
    except ValueError as exc:
        # The pinned runtime is the validation authority for what the wire
        # layer cannot foresee (state rendering, question semantics, the
        # window budget); its rejections are client errors, not failures.
        raise map_runtime_error(exc) from exc
    return finish_response(questions, response)


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, executor: Vela2Executor, api_key: str | None = None):
        self.executor = executor
        self.api_key = api_key
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

    def _send_request_error(self, exc):
        if exc.body is not None:
            self.send_json(exc.status, exc.body)
        else:
            self.send_json(exc.status, {"detail": str(exc)})

    def _unauthorized(self) -> bool:
        """Upstream vela2_serve.py: no key configured means no gate; with
        VELA2_API_KEY set, exactly ``Authorization: Bearer <key>`` passes."""

        key = self.server.api_key
        if not key:
            return False
        got = self.headers.get("authorization", "")
        if hmac.compare_digest(got, f"Bearer {key}"):
            return False
        self.send_json(401, UNAUTHORIZED_BODY)
        return True

    def do_GET(self):
        path = self.path
        if path == "/health":
            # Upstream verbatim; the worker binds the port only after
            # loading and warmup, so answering at all means ready.
            self.send_json(200, {"status": "ok"})
        elif path == "/ready":
            self.send_json(200, {"ready": True})
        elif path == "/v1/models":
            if self._unauthorized():
                return
            entry = dict(MODELS_ENTRY)
            entry["name"] = self.server.executor.model_name
            self.send_json(200, {"models": [entry]})
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
            self.send_json(413, {"detail": "request exceeds the 8 MiB body limit"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"detail": "Content-Type must be application/json"})
            return
        if self._unauthorized():
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
            LOG.error("vela inference failed: %s", type(exc).__name__)
            self.send_json(500, {"detail": "inference failed"})
        finally:
            self.server.inference_lock.release()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Python reference worker for Vela 2.0 0.3B")
    parser.add_argument("--model", required=True, help="pinned Vela 2.0 0.3B package directory")
    parser.add_argument("--device", default="cpu", help="torch device (reference contract: cpu)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--name", default=os.environ.get("VELA2_MODEL_NAME", MODEL_NAME),
                        help="model name accepted in the request's model field")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    executor = Vela2Executor(
        args.model,
        device=args.device,
        model_name=args.name,
        batch_size=args.batch_size,
    )
    server = None
    try:
        executor.warmup()
        server = WorkerServer((args.host, args.port), executor,
                              api_key=os.environ.get("VELA2_API_KEY") or None)
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
