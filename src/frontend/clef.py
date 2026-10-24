"""Python reference worker for Cloudflare Clef and Clef-flash.

The Rust frontend remains the public transport layer. This worker owns HTTP
lifecycle and request orchestration. Wire admission lives in
``models.clef.protocol``, question and token admission in
``models.clef.preprocess``, and the pinned joint-schema checkpoint in
``models.clef.engine``.

Over-long state is rejected. The checkpoint encoder would otherwise drop the
tail of the state without an error. Image and video fields are rejected;
this slice does not serve them.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from models.clef.engine import DEFAULT_VARIANT, VARIANTS, ClefExecutor
from models.clef.preprocess import validate_questions
from models.clef.protocol import MAX_BODY, RequestError, parse_body, parse_request

LOG = logging.getLogger(__name__)


def decide(raw: bytes, executor: Any) -> dict[str, Any]:
    """Admit one complete text request and run it if it fits intact."""

    request = parse_request(parse_body(raw))
    validate_questions(request)
    return executor.serve(request.payload)


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, executor: Any):
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
            self.send_json(200, self.server.executor.health())
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
            LOG.error("clef inference failed: %s", type(exc).__name__)
            self.send_json(500, {"detail": "inference failed"})
        finally:
            self.server.inference_lock.release()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="snapshot directory of the pinned revision")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    executor = ClefExecutor(args.model, variant=args.variant, device=args.device)
    server = None
    try:
        server = WorkerServer((args.host, args.port), executor)
        LOG.info(
            "%s worker ready on %s:%s",
            executor.variant.repo_id,
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
