"""HTTP tests for the Vela 2.0 worker handler: status mapping and bodies.

Starts the real WorkerServer on an ephemeral port with the FakeVela2
from test_worker: no weights, no torch, no GPU. The 408 slow-client
timeout is not covered because the handler's 15 s connection timeout is
deliberately not configurable for tests.

    PYTHONPATH=src python -m pytest tests/vela -q
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import pytest

from frontend.vela import WorkerServer
from models.vela.protocol import MAX_BODY

from .test_protocol import body
from .test_worker import FakeVela2, FakeSystemOneError, five_kind_body


class FailingExecutor:
    model_name = "vllm-sr/Vela-2.0-0.3B"
    SystemOneError = FakeSystemOneError

    @property
    def served(self):
        return {self.model_name}

    def system_one(self, body):
        raise RuntimeError("boom")

    def close(self):
        pass


def start(executor, api_key=None):
    server = WorkerServer(("127.0.0.1", 0), executor, api_key=api_key)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stop(server):
    server.shutdown()
    server.server_close()


def request(server, method, path, headers=(), payload=b""):
    """One raw HTTP request with full header control."""

    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest(method, path, skip_accept_encoding=True)
    for name, value in headers:
        conn.putheader(name, value)
    conn.endheaders(payload)
    response = conn.getresponse()
    raw = response.read()
    status, ctype = response.status, response.getheader("Content-Type")
    conn.close()
    return status, ctype, raw


def post(server, payload, headers=()):
    """POST /v1/systemone with JSON defaults; explicit headers replace them."""

    defaults = {"Content-Type": "application/json", "Content-Length": str(len(payload))}
    extra = []
    for name, value in headers:
        defaults.pop(name, None)
        extra.append((name, value))
    return request(
        server,
        "POST",
        "/v1/systemone",
        headers=(*defaults.items(), *extra),
        payload=payload,
    )


@pytest.fixture()
def server():
    instance = start(FakeVela2())
    yield instance
    stop(instance)


def test_health_and_ready_answer_upstream_bodies(server):
    status, ctype, raw = request(server, "GET", "/health")
    assert status == 200 and ctype == "application/json"
    assert json.loads(raw) == {"status": "ok"}

    status, _, raw = request(server, "GET", "/ready")
    assert status == 200 and json.loads(raw) == {"ready": True}

    status, _, raw = request(server, "GET", "/nope")
    assert status == 404 and json.loads(raw) == {"detail": "unknown route"}


def test_models_lists_the_served_member_upstream_verbatim(server):
    status, _, raw = request(server, "GET", "/v1/models")
    assert status == 200
    assert json.loads(raw) == {
        "models": [
            {
                "name": "vllm-sr/Vela-2.0-0.3B",
                "release_date": "2026-09-29",
                "description": (
                    "Vela 2.0 0.3B: SystemOne Choice / Noul / Score "
                    "with span and set extensions, 8,192-token context"
                ),
            }
        ]
    }


def test_valid_five_kind_decision_returns_200(server):
    status, ctype, raw = post(server, json.dumps(five_kind_body()).encode("utf-8"))
    assert status == 200 and ctype == "application/json"
    result = json.loads(raw)
    assert result["model"] == "vllm-sr/Vela-2.0-0.3B"
    assert result["answers"]["polite"] == {"type": "noul", "noul": 0.75}
    assert result["sets"]["issues"]["selected"] == ["delivery", "login"]
    assert result["usage"]["output_tokens"] == 0


def test_unknown_post_route_is_404(server):
    status, _, _ = request(
        server,
        "POST",
        "/other",
        headers=(("Content-Type", "application/json"), ("Content-Length", "2")),
        payload=b"{}",
    )
    assert status == 404


def test_length_headers_are_enforced(server):
    payload = b"{}"
    base = {"Content-Type": "application/json"}

    status, _, raw = post(server, payload, headers=(("Transfer-Encoding", "chunked"),))
    assert status == 411 and "Content-Length" in json.loads(raw)["detail"]

    status, _, _ = request(
        server,
        "POST",
        "/v1/systemone",
        headers=[*base.items(), ("Content-Length", "2"), ("Content-Length", "2")],
        payload=payload,
    )
    assert status == 411

    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest("POST", "/v1/systemone", skip_accept_encoding=True)
    conn.putheader("Content-Type", "application/json")
    conn.endheaders(payload)
    response = conn.getresponse()
    response.read()
    assert response.status == 411
    conn.close()

    status, _, _ = post(server, payload, headers=(("Content-Length", "abc"),))
    assert status == 400


def test_body_limit_and_content_type_are_enforced(server):
    payload = json.dumps(body()).encode("utf-8")

    status, _, raw = post(server, payload, headers=(("Content-Length", str(MAX_BODY + 1)),))
    assert status == 413 and "body limit" in json.loads(raw)["detail"]

    status, _, raw = post(server, payload, headers=(("Content-Type", "text/plain"),))
    assert status == 415 and "application/json" in json.loads(raw)["detail"]


def test_transport_errors_map_to_their_status_and_detail(server):
    status, _, raw = post(server, b"{not json")
    assert status == 400 and "detail" in json.loads(raw)

    value = body(extra=True)
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422
    assert json.loads(raw) == {
        "detail": [
            {"loc": ["body", "extra"], "msg": "Extra inputs are not permitted",
             "type": "extra_forbidden"}
        ]
    }


def test_an_invalid_question_returns_the_upstream_422_detail(server):
    value = five_kind_body()
    value["questions"]["broken"] = {"type": "triangle"}
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422
    payload = json.loads(raw)
    assert set(payload) == {"detail"}
    entry = payload["detail"][0]
    assert entry["loc"] == ["body", "questions", "broken", "type"]
    assert "type must be one of" in entry["msg"]
    assert entry["type"] == "literal_error"


def test_questions_overflow_returns_the_upstream_413_detail():
    overflow = start(FakeVela2(too_long=True))
    try:
        status, _, raw = post(overflow, json.dumps(body()).encode("utf-8"))
    finally:
        stop(overflow)
    assert status == 413
    assert json.loads(raw) == {
        "detail": [
            {"loc": ["body", "questions"],
             "msg": "the questions alone do not fit the 8192-token window",
             "type": "too_long"}
        ]
    }


def test_busy_worker_returns_503(server):
    server.inference_lock.acquire()
    try:
        status, _, raw = post(server, json.dumps(body()).encode("utf-8"))
    finally:
        server.inference_lock.release()
    assert status == 503 and json.loads(raw) == {"detail": "worker busy"}


def test_executor_failure_returns_500_without_leaking_details():
    failing = start(FailingExecutor())
    try:
        status, _, raw = post(failing, json.dumps(body()).encode("utf-8"))
    finally:
        stop(failing)
    assert status == 500 and json.loads(raw) == {"detail": "inference failed"}


@pytest.fixture()
def keyed_server():
    instance = start(FakeVela2(), api_key="secret-key")
    yield instance
    stop(instance)


def test_a_missing_or_wrong_bearer_is_rejected_with_upstreams_401(keyed_server):
    status, _, raw = post(keyed_server, json.dumps(body()).encode("utf-8"))
    assert status == 401 and json.loads(raw) == {"detail": "Missing or invalid API key"}

    status, _, raw = post(
        keyed_server,
        json.dumps(body()).encode("utf-8"),
        headers=(("Authorization", "Bearer wrong"),),
    )
    assert status == 401 and json.loads(raw) == {"detail": "Missing or invalid API key"}

    status, _, raw = request(keyed_server, "GET", "/v1/models")
    assert status == 401 and json.loads(raw) == {"detail": "Missing or invalid API key"}

    status, _, raw = request(keyed_server, "GET", "/v1/models",
                             headers=(("Authorization", "Bearer secret-key"),))
    assert status == 200 and json.loads(raw)["models"][0]["name"] == "vllm-sr/Vela-2.0-0.3B"


def test_health_stays_open_behind_the_api_key(keyed_server):
    status, _, raw = request(keyed_server, "GET", "/health")
    assert status == 200 and json.loads(raw) == {"status": "ok"}
    status, _, raw = request(keyed_server, "GET", "/ready")
    assert status == 200 and json.loads(raw) == {"ready": True}


def test_the_correct_bearer_answers_normally(keyed_server):
    status, _, raw = post(
        keyed_server,
        json.dumps(body()).encode("utf-8"),
        headers=(("Authorization", "Bearer secret-key"),),
    )
    assert status == 200 and json.loads(raw)["answers"]["billing"] == {"type": "noul", "noul": 0.75}
