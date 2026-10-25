"""Python reference worker for pplx-decider-v1.1-27b.

The Rust frontend stays the public transport. This process admits the
checkpoint's ``/v1/systemone`` request, then calls the checkpoint's own
``DecisionModel``. Option order is kept. Over-long prompts and too many
options are rejected; they are not truncated.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from models.pplx_decider.contract import MAX_BODY, RequestError, parse_body, parse_request
from models.pplx_decider.engine import PplxExecutor

LOG = logging.getLogger("pplx_decider")


def decide(raw: bytes, executor: Any) -> dict[str, Any]:
    request = parse_request(parse_body(raw))
    try:
        return executor.evaluate(request)
    except ValueError as exc:
        raise RequestError(str(exc), 422) from exc


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, executor):
        self.executor = executor
        self.inference_lock = threading.Lock()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        return None

    def send_json(self, status: int, value: dict):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            return None

    def do_GET(self):
        if self.path != "/health":
            self.send_json(404, {"detail": "unknown route"})
            return
        executor = self.server.executor
        body = {
            "status": "ready",
            "model": executor.model_name,
            "max_input_tokens": executor.max_length,
            "modalities": list(executor.modalities),
            "attention_mode": executor.attention_mode,
        }
        layers = getattr(executor, "unpadded_full_attention_layers", None)
        if layers is not None:
            body["unpadded_full_attention_layers"] = layers
        conv = getattr(executor, "causal_conv1d_version", None)
        if conv is not None:
            body["causal_conv1d"] = conv
        self.send_json(200, body)

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
            self.send_json(413, {"detail": "request exceeds the 50 MiB body limit"})
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
            self.send_json(exc.status, {"detail": str(exc)})
        except (TimeoutError, socket.timeout):
            self.send_json(408, {"detail": "request body timed out"})
        except Exception as exc:
            LOG.error("inference failed: %s", type(exc).__name__)
            self.send_json(500, {"detail": "inference failed"})
        finally:
            self.server.inference_lock.release()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="directory of the pinned checkpoint")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-image-pixels", type=int, default=2048 * 2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    executor = PplxExecutor(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_image_pixels=args.max_image_pixels,
    )
    server = None
    try:
        server = WorkerServer((args.host, args.port), executor)
        LOG.info(
            "ready model=%s device=%s attention=%s unpadded_full_attention_layers=%s causal_conv1d=%s bind=%s:%s",
            executor.model_name,
            args.device,
            executor.attention_mode,
            executor.unpadded_full_attention_layers,
            executor.causal_conv1d_version,
            args.host,
            args.port,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        return None
    finally:
        if server is not None:
            server.server_close()
        executor.close()


if __name__ == "__main__":
    main()
