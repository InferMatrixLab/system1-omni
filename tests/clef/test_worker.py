import json
from http.client import HTTPConnection
from threading import Thread

import pytest

from frontend.clef import WorkerServer, decide
from models.clef.postprocess import systemone_answer
from models.clef.preprocess import enforce_limit, measure
from models.clef.protocol import MAX_BODY, Request, RequestError


def three_kinds():
    return {
        "model": "clef-flash",
        "state": "Checkout has been failing for every customer.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"technical": "Bugs", "billing": "Payments"},
            },
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "severity": {
                "type": "score",
                "instructions": "How severe is the impact?",
                "criteria": ["No impact", "Minor", "Major"],
            },
        },
    }


class FakeClef:
    def __init__(self, max_length=16384, fail=False):
        self.max_length = max_length
        self.fail = fail
        self.served = 0

    def count_tokens(self, text: str) -> int:
        return len(text)

    def serve(self, payload: dict):
        self.served += 1
        if self.fail:
            raise RuntimeError("boom")
        fixed, total = measure(self.count_tokens, Request(payload))
        enforce_limit(fixed, total, self.max_length)
        answers = {
            name: systemone_answer(
                question,
                _uniform(question),
            )
            for name, question in payload["questions"].items()
        }
        return {
            "model": payload["model"],
            "answers": answers,
            "usage": {"input_tokens": total, "output_tokens": 0},
        }

    def health(self):
        return {
            "status": "ready",
            "model": "Cloudflare/clef-flash",
            "variant": "clef-flash",
            "revision": "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
            "max_input_tokens": self.max_length,
            "modalities": ["text"],
            "prompt_mode": "non-thinking",
        }

    def close(self):
        pass


def _uniform(question):
    if question["type"] == "noul":
        return {"true": 0.25, "false": 0.75}
    if question["type"] == "choice":
        keys = list(question["criteria"])
    else:
        keys = [str(index) for index in range(len(question["criteria"]))]
    share = 1 / len(keys)
    return {key: share for key in keys}


def test_decide_returns_three_answer_types_in_request_order():
    raw = json.dumps(three_kinds()).encode("utf-8")
    response = decide(raw, FakeClef())
    assert list(response["answers"]) == ["team", "urgent", "severity"]
    assert response["answers"]["team"]["choice"] == "technical"
    assert response["answers"]["urgent"] == {"type": "noul", "noul": 0.25}
    assert response["answers"]["severity"]["legend"]["0"] == "No impact"
    assert response["usage"]["output_tokens"] == 0
    assert response["usage"]["input_tokens"] > 0


def test_decide_rejects_overflow_before_a_second_serve_step():
    payload = three_kinds()
    fixed, total = measure(lambda text: len(text), Request(payload))
    assert total > fixed
    executor = FakeClef(max_length=total - 1)
    with pytest.raises(RequestError, match="was not truncated") as excinfo:
        decide(json.dumps(payload).encode("utf-8"), executor)
    assert excinfo.value.status == 422
    assert executor.served == 1


def test_decide_rejects_images():
    payload = three_kinds()
    payload["images"] = ["data:image/png;base64,AAAA"]
    with pytest.raises(RequestError, match="images are not served"):
        decide(json.dumps(payload).encode("utf-8"), FakeClef())


def start(executor):
    server = WorkerServer(("127.0.0.1", 0), executor)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stop(server):
    server.shutdown()
    server.server_close()


def post(server, payload, headers=()):
    defaults = {"Content-Type": "application/json", "Content-Length": str(len(payload))}
    extra = []
    for name, value in headers:
        defaults.pop(name, None)
        extra.append((name, value))
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest("POST", "/v1/systemone", skip_accept_encoding=True)
    for name, value in (*defaults.items(), *extra):
        conn.putheader(name, value)
    conn.endheaders(payload)
    response = conn.getresponse()
    raw = response.read()
    status = response.status
    conn.close()
    return status, raw


@pytest.fixture()
def server():
    instance = start(FakeClef())
    yield instance
    stop(instance)


def test_health_and_post(server):
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.request("GET", "/health")
    response = conn.getresponse()
    health = json.loads(response.read())
    conn.close()
    assert response.status == 200
    assert health["variant"] == "clef-flash"
    assert health["prompt_mode"] == "non-thinking"
    assert health["modalities"] == ["text"]
    assert health["max_input_tokens"] == 16384

    status, raw = post(server, json.dumps(three_kinds()).encode("utf-8"))
    body = json.loads(raw)
    assert status == 200
    assert body["answers"]["urgent"]["noul"] == 0.25


def test_transport_errors(server):
    status, raw = post(server, b"{")
    assert status == 400 and "detail" in json.loads(raw)
    status, raw = post(
        server,
        json.dumps(three_kinds()).encode("utf-8"),
        headers=(("Content-Type", "text/plain"),),
    )
    assert status == 415
    status, raw = post(
        server,
        b"{}",
        headers=(("Content-Length", str(MAX_BODY + 1)),),
    )
    assert status == 413 and "8 MiB" in json.loads(raw)["detail"]
    status, raw = post(server, b"", headers=(("Transfer-Encoding", "chunked"),))
    assert status == 411
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.request("POST", "/missing", body=b"{}")
    response = conn.getresponse()
    assert response.status == 404
    response.read()
    conn.close()


def test_busy_and_inference_failure(server):
    server.inference_lock.acquire()
    try:
        status, raw = post(server, json.dumps(three_kinds()).encode("utf-8"))
    finally:
        server.inference_lock.release()
    assert status == 503

    failed = start(FakeClef(fail=True))
    try:
        status, raw = post(failed, json.dumps(three_kinds()).encode("utf-8"))
    finally:
        stop(failed)
    assert status == 500
    assert json.loads(raw) == {"detail": "inference failed"}
