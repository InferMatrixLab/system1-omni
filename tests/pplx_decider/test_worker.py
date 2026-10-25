"""HTTP tests for the reference worker. No weights and no GPU."""

from __future__ import annotations

import base64
import json
import threading
from http.client import HTTPConnection
from io import BytesIO
from pathlib import Path

from PIL import Image

from frontend.pplx_decider import WorkerServer, decide
from models.pplx_decider.contract import (
    MODEL_NAME,
    TOKEN_LIMIT_MESSAGE,
    RequestError,
    answer,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = json.loads((ROOT / "recipe/pplx_decider/example-request.json").read_text())


class Fake:
    model_name = MODEL_NAME
    max_length = 8192
    modalities = ("text", "image")
    attention_mode = "noncausal_full_attention"

    def __init__(self, probabilities=None, error=None):
        self.probabilities = probabilities or {}
        self.error = error
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        if self.error:
            raise ValueError(self.error)
        answers = {
            name: answer(question, self.probabilities[name])
            for name, question in request.questions
        }
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }

    def close(self):
        return None


def png(width=2, height=2):
    stream = BytesIO()
    Image.new("RGB", (width, height), "white").save(stream, "PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def test_example_request_preserves_order_and_probabilities():
    fake = Fake(
        {
            "refund": [0.2, 0.8],
            "urgent": [0.25, 0.75],
            "severity": [1.0, 2.0, 0.0, 0.0],
        }
    )
    raw = json.dumps(EXAMPLE).encode()
    body = decide(raw, fake)
    assert list(body["answers"]) == ["refund", "urgent", "severity"]
    assert list(body["answers"]["refund"]["probabilities"]) == ["refund", "wait"]
    assert body["answers"]["refund"]["choice"] == "wait"
    assert body["answers"]["urgent"] == {"type": "noul", "noul": 0.75}
    assert list(body["answers"]["severity"]["probabilities"]) == ["0", "1", "2", "3"]
    assert body["answers"]["severity"]["score"] == 2 / 3
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 0}
    assert fake.calls == 1


def test_over_limit_does_not_call_the_model():
    fake = Fake()
    raw = json.dumps(
        {
            "model": "jev-latest",
            "state": "s",
            "questions": {"q": {"type": "choice", "criteria": {str(i): "d" for i in range(256)}}},
        }
    ).encode()
    try:
        decide(raw, fake)
    except RequestError as exc:
        assert exc.status == 422
        assert "1 to 255" in str(exc)
    else:
        raise AssertionError("expected rejection")
    assert fake.calls == 0


def test_token_limit_is_a_client_error():
    fake = Fake(error=TOKEN_LIMIT_MESSAGE)
    raw = json.dumps(
        {"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul"}}}
    ).encode()
    try:
        decide(raw, fake)
    except RequestError as exc:
        assert exc.status == 422
        assert str(exc) == TOKEN_LIMIT_MESSAGE
    else:
        raise AssertionError("expected rejection")


def oversized_png():
    import struct
    import zlib

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 5000, 5000, 8, 2, 0, 0, 0)
    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def test_image_rules():
    request = {
        "model": "jev-latest",
        "state": "s",
        "questions": {"q": {"type": "noul"}},
        "images": [png()],
    }
    decide(json.dumps(request).encode(), Fake({"q": [0.4, 0.6]}))
    request["images"] = ["data:image/gif;base64,AAAA"]
    try:
        decide(json.dumps(request).encode(), Fake())
    except RequestError as exc:
        assert "PNG, JPEG, or WebP" in str(exc)
    else:
        raise AssertionError("expected rejection")
    request["images"] = [oversized_png()]
    try:
        decide(json.dumps(request).encode(), Fake())
    except RequestError as exc:
        assert "16 million pixels" in str(exc)
    else:
        raise AssertionError("expected rejection")
    request["images"] = ["x" * 12_000_001]
    try:
        decide(json.dumps(request).encode(), Fake())
    except RequestError as exc:
        assert "8 MB before base64" in str(exc)
    else:
        raise AssertionError("expected rejection")


def start(executor):
    server = WorkerServer(("127.0.0.1", 0), executor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stop(server):
    server.shutdown()
    server.server_close()


def post(server, payload, headers=None):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    header_list = list(headers or [])
    if not any(name.lower() == "content-type" for name, _ in header_list):
        header_list.append(("Content-Type", "application/json"))
    if not any(name.lower() == "content-length" for name, _ in header_list):
        header_list.append(("Content-Length", str(len(raw))))
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest("POST", "/v1/systemone", skip_accept_encoding=True)
    for name, value in header_list:
        conn.putheader(name, value)
    conn.endheaders(raw)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, body


def test_http_health_and_decision():
    server = start(Fake({"q": [0.1, 0.9]}))
    try:
        conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("GET", "/health")
        health = json.loads(conn.getresponse().read())
        conn.close()
        assert health["status"] == "ready"
        assert health["model"] == MODEL_NAME
        assert health["max_input_tokens"] == 8192
        assert "text" in health["modalities"]
        assert "unpadded_full_attention_layers" not in health
        server.executor.unpadded_full_attention_layers = 16
        conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("GET", "/health")
        health = json.loads(conn.getresponse().read())
        conn.close()
        assert health["unpadded_full_attention_layers"] == 16
        status, body = post(
            server,
            {"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul"}}},
        )
        assert status == 200
        payload = json.loads(body)
        assert payload["answers"]["q"]["noul"] == 0.9
        status, body = post(server, b"{", [("Content-Type", "application/json"), ("Content-Length", "1")])
        assert status == 400
        status, _ = post(
            server,
            b"{}",
            [("Content-Type", "text/plain"), ("Content-Length", "2")],
        )
        assert status == 415
    finally:
        stop(server)


def test_http_disabled_thinking_is_accepted_and_other_values_are_422():
    server = start(Fake({"q": [0.1, 0.9]}))
    try:
        status, body = post(
            server,
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "noul"}},
                "thinking": {"enabled": False},
            },
        )
        assert status == 200
        assert json.loads(body)["answers"]["q"]["noul"] == 0.9
        status, body = post(
            server,
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "noul"}},
                "thinking": {"enabled": True},
            },
        )
        assert status == 422
        assert b"thinking.enabled must be false" in body
    finally:
        stop(server)


def test_http_busy_is_503():
    server = start(Fake({"q": [0.5, 0.5]}))
    server.inference_lock.acquire()
    try:
        status, body = post(
            server,
            {"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul"}}},
        )
        assert status == 503
        assert json.loads(body)["detail"] == "worker busy"
    finally:
        server.inference_lock.release()
        stop(server)
