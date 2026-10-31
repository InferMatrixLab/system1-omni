"""Python reference worker for the Vela 2.0 members (0.3B, 0.8B hybrid).

The Rust frontend remains the public transport layer. This worker owns
only HTTP lifecycle and request orchestration; wire admission lives in
models.vela.protocol, the pinned-runtime error mapping in
models.vela.preprocess, the pinned runtimes and their package gates in
models.vela.engine (0.3B) and models.vela.hybrid (0.8B), and response
finishing in models.vela.postprocess. The executor is chosen by the
package's declared ``model_type``; one worker process serves one pinned
member.

The error surface has two layers: transport errors follow the shared
Python-worker discipline, and every request rejection the pinned runtime
makes — an invalid field, an invalid question, a questions-overflow — is
the upstream whole-request 422 ``{"detail": [{"loc", "msg", "type"}]}``
(or 413 ``too_long``), passed through verbatim. There are no per-question
errors. Over-limit state text is truncated exactly as the pinned
reference truncates it (see the model contract); only questions plus
schema that cannot fit the member's input window (8,192 or 16,384
tokens) are refused.
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
from pathlib import Path
from typing import Any

from models.vela.engine import Vela2Executor
from models.vela.hybrid import EXPECTED_MODEL_NAME as HYBRID_MODEL_NAME, Vela2HybridExecutor
from models.vela.postprocess import finish_response
from models.vela.preprocess import map_runtime_error, to_request_error
from models.vela.protocol import MODEL_NAME, MAX_BODY, RequestError, parse_body, admit_request

LOG = logging.getLogger(__name__)

# Upstream vela2_serve.py /v1/models entry for the 0.3B member, kept
# verbatim so clients see the reference server's description and release
# date; the hybrid members source their entry from config.json, exactly
# as their vela2_serve.py does.
MODELS_ENTRY = {
    "name": None,  # the served model name, filled per request
    "release_date": "2026-09-29",
    "description": (
        "Vela 2.0 0.3B: SystemOne Choice / Noul / Score "
        "with span and set extensions, 8,192-token context"
    ),
}
UNAUTHORIZED_BODY = {"detail": "Missing or invalid API key"}
# Upstream verbatim endpoint bodies; compare_reference.py pins them
# against the live reference app.
HEALTH_BODY = {"status": "ok"}
READY_BODY = {"ready": True}


def detect_member(package: str | Path) -> tuple[type, str]:
    """Pick the pinned executor for a package's declared member.

    Returns the executor class and the member's default served name; the
    structural gate inside the executor is the real authority — this
    only routes the right gate to the package.
    """

    config_path = Path(package) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"the package is missing config.json: {package}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"config.json in {package} is not readable JSON") from exc
    if not isinstance(config, dict):
        raise ValueError(f"config.json in {package} is not a JSON object")
    model_type = config.get("model_type")
    if model_type == "vela2-unified":
        return Vela2Executor, MODEL_NAME
    if model_type == "vela2-decoder":
        return Vela2HybridExecutor, HYBRID_MODEL_NAME
    raise ValueError(
        f"config.json declares model_type {model_type!r}; this worker serves the "
        "vela2-unified (0.3B) and vela2-decoder (0.8B) members of Vela 2.0"
    )


def models_entry(executor) -> dict[str, Any]:
    """The /v1/models entry, verbatim per member's reference server."""

    if getattr(executor, "is_hybrid", False):
        # Upstream hybrid vela2_serve.py: name/description/release_date
        # straight from config.json.
        config = getattr(executor, "engine", None)
        config = getattr(config, "config", None) if config is not None else None
        config = config or {}
        return {
            "name": executor.model_name,
            "release_date": config.get("release_date", ""),
            "description": config.get(
                "description",
                "Vela 2.0: SystemOne Choice / Noul / Score with span and set extensions",
            ),
        }
    entry = dict(MODELS_ENTRY)
    entry["name"] = executor.model_name
    return entry


def decide(raw: bytes, executor) -> dict[str, Any]:
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
    return finish_response(questions, response, hybrid=getattr(executor, "is_hybrid", False))


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, executor: Vela2Executor | Vela2HybridExecutor, api_key: str | None = None):
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
        try:
            authorized = hmac.compare_digest(got, f"Bearer {key}")
        except TypeError:
            # Header bytes outside ASCII (a raw 0xff, say) decode to
            # non-ASCII text, which compare_digest refuses to compare; a
            # value that cannot be compared cannot equal the key either.
            authorized = False
        if authorized:
            return False
        self.send_json(401, UNAUTHORIZED_BODY)
        return True

    def do_GET(self):
        path = self.path
        if path == "/health":
            # Upstream verbatim; the worker binds the port only after
            # loading and warmup, so answering at all means ready.
            self.send_json(200, HEALTH_BODY)
        elif path == "/ready":
            self.send_json(200, READY_BODY)
        elif path == "/v1/models":
            if self._unauthorized():
                return
            self.send_json(200, {"models": [models_entry(self.server.executor)]})
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
    parser = argparse.ArgumentParser(description="Python reference worker for Vela 2.0 members")
    parser.add_argument("--model", required=True,
                        help="pinned Vela 2.0 package directory (0.3B or the 0.8B hybrid)")
    parser.add_argument("--device", default="cpu",
                        help="torch device (reference contract: cpu; cuda runs the hybrid "
                             "backbone under bf16 autocast with FP32 heads)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--name", default=os.environ.get("VELA2_MODEL_NAME"),
                        help="model name accepted in the request's model field "
                             "(default: the pinned member's served name)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    executor_class, default_name = detect_member(args.model)
    executor = executor_class(
        args.model,
        device=args.device,
        model_name=args.name or default_name,
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
