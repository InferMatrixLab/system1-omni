import json

import pytest

from frontend.clef import decide
from models.clef.postprocess import systemone_answer
from models.clef.preprocess import enforce_limit, measure
from models.clef.protocol import Request, RequestError


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


def test_decide_accepts_thinking_enabled_false():
    payload = three_kinds()
    payload["thinking"] = {"enabled": False}
    response = decide(json.dumps(payload).encode("utf-8"), FakeClef())
    assert response["answers"]["urgent"]["noul"] == 0.25
    bare = decide(json.dumps(three_kinds()).encode("utf-8"), FakeClef())
    assert response["usage"]["input_tokens"] == bare["usage"]["input_tokens"]


def test_decide_rejects_images():
    payload = three_kinds()
    payload["images"] = ["data:image/png;base64,AAAA"]
    with pytest.raises(RequestError, match="images are not served"):
        decide(json.dumps(payload).encode("utf-8"), FakeClef())
