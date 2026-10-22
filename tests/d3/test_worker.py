"""End-to-end decide() tests with a CPU-only FakeD3: no weights, no torch.

FakeD3 mirrors the upstream runtime surface the worker drives —
prepare/run/respond plus model_name, max_length and modalities — with the
pinned question-validation rules reimplemented just far enough to produce
upstream-shaped per-question errors, and answer construction delegated to
models.d3.postprocess (which locks the pinned formulas).
"""

import json
import logging

import pytest

from frontend.d3 import decide
from models.d3.postprocess import describe, to_answer
from models.d3.protocol import MODEL_NAME, RequestError

from .test_protocol import body


class FakePrepared:
    def __init__(self, keys, questions, errors):
        self.keys = list(keys)
        self.questions = questions
        self.errors = errors


class FakeD3:
    """The pinned runtime's contract, driven without its code."""

    model_name = MODEL_NAME
    max_length = None
    modalities = ["text"]

    def __init__(self, probabilities=None, max_length=None):
        self.probabilities = dict(probabilities or {})
        self.max_length = max_length
        self.warmed = False
        self.closed = False

    # -- upstream normalize_question, reduced to what the tests exercise ----
    @staticmethod
    def _normalize(question):
        if not isinstance(question, dict):
            raise ValueError("a question is an object with a type")
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError("choice criteria must map 1 to 255 option keys to descriptions")
            if any(not isinstance(key, str) or not key for key in criteria):
                raise ValueError("choice option keys must be nonempty strings")
            return {"kind": "choice", "keys": list(criteria), "levels": None}
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
            ):
                raise ValueError('noul criteria may only describe "false" and "true"')
            return {"kind": "noul", "keys": ["false", "true"], "levels": None}
        if kind == "score":
            if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10:
                raise ValueError("score criteria must be an ordered list of 2 to 10 levels")
            levels = [describe(level) for level in criteria]
            if any(not text for text in levels) or len(set(levels)) != len(levels):
                raise ValueError("score levels must be distinct and nonempty")
            return {
                "kind": "score",
                "keys": [str(index) for index in range(len(criteria))],
                "levels": list(criteria),
            }
        raise ValueError(f"unsupported question type {kind!r} (choice, noul or score)")

    def _tokens(self, state, question):
        return 8 + len(json.dumps(state)) // 8 + len(json.dumps(question)) // 8

    def prepare(self, state, questions):
        errors = {}
        normalized = {}
        for key, question in questions.items():
            try:
                normalized[key] = self._normalize(question)
            except ValueError as exc:
                kind = question.get("type") if isinstance(question, dict) else None
                errors[key] = {"type": kind, "error": "invalid_question", "message": str(exc)}
        prepared = FakePrepared(questions, normalized, errors)
        if self.max_length is not None:
            for key, question in questions.items():
                if key in errors:
                    continue
                tokens = self._tokens(state, question)
                if tokens > self.max_length:
                    errors[key] = {
                        "type": normalized[key]["kind"],
                        "error": "max_length_exceeded",
                        "message": (
                            f"the question prompt has {tokens} tokens, over the maximum"
                            f" context length of {self.max_length} tokens; nothing was truncated"
                        ),
                    }
        return prepared

    def run(self, prepared):
        probabilities = {}
        tokens = 0
        for key in prepared.keys:
            if key in prepared.errors:
                continue
            probabilities[key] = self.probabilities.get(key) or [
                1.0 for _ in prepared.questions[key]["keys"]
            ]
            tokens += 8 + 4 * len(probabilities[key])
        return probabilities, tokens

    def respond(self, prepared, probabilities, tokens):
        answers = {}
        for key in prepared.keys:
            if key in prepared.errors:
                answers[key] = dict(prepared.errors[key])
                continue
            question = prepared.questions[key]
            try:
                answers[key] = to_answer(
                    question["kind"], question["keys"], question["levels"], probabilities[key]
                )
            except ValueError as exc:
                answers[key] = {
                    "type": question["kind"],
                    "error": "invalid_model_output",
                    "message": str(exc),
                }
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    def warmup(self):
        self.warmed = True

    def close(self):
        self.closed = True


def three_kind_body():
    return {
        "model": MODEL_NAME,
        "state": {"ticket": "double charge", "customer": "vip"},
        "questions": {
            "refund": {
                "type": "choice",
                "instructions": "Decide the refund action.",
                "criteria": {"refund": "Refund the duplicate charge", "wait": "Wait for review"},
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is this urgent for the customer?",
                "criteria": {"true": "Customer hardship"},
            },
            "severity": {
                "type": "score",
                "instructions": "Rate the severity.",
                "criteria": ["minor", "moderate", "major"],
            },
        },
    }


def test_decide_answers_all_three_kinds_in_one_request():
    executor = FakeD3(
        probabilities={
            "refund": [1.0, 3.0],
            "urgent": [1.0, 3.0],
            "severity": [1.0, 2.0, 1.0],
        }
    )
    result = decide(json.dumps(three_kind_body()).encode("utf-8"), executor)

    assert result["model"] == "d3"
    assert list(result["answers"]) == ["refund", "urgent", "severity"]
    assert result["answers"]["refund"]["choice"] == "wait"
    assert result["answers"]["refund"]["probabilities"] == {"refund": 0.25, "wait": 0.75}
    assert result["answers"]["urgent"] == {"type": "noul", "noul": 0.75}
    assert result["answers"]["severity"]["score"] == pytest.approx(1.0)
    assert result["answers"]["severity"]["legend"] == {"0": "minor", "1": "moderate", "2": "major"}
    assert result["usage"]["output_tokens"] == 0
    assert result["usage"]["input_tokens"] == 8 + 4 * 2 + 8 + 4 * 2 + 8 + 4 * 3


def test_decide_aggregates_invalid_questions_into_one_422():
    value = three_kind_body()
    value["questions"]["broken"] = {"type": "triangle"}
    value["questions"]["also_broken"] = {"type": "choice", "criteria": {}}
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(value).encode("utf-8"), FakeD3())

    assert excinfo.value.status == 422
    assert list(excinfo.value.body["invalid_questions"]) == ["broken", "also_broken"]
    assert "unsupported question type" in excinfo.value.body["invalid_questions"]["broken"]
    assert "choice criteria" in excinfo.value.body["invalid_questions"]["also_broken"]


def test_decide_rejects_over_limit_questions_without_truncating():
    value = three_kind_body()
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(value).encode("utf-8"), FakeD3(max_length=16))

    assert excinfo.value.status == 422
    assert excinfo.value.body is None
    assert "tokens, over the maximum context length of 16" in str(excinfo.value)
    assert "nothing was truncated" in str(excinfo.value)


def test_decide_keeps_partial_success_and_warns(caplog):
    executor = FakeD3(
        probabilities={"refund": [1.0, 3.0], "urgent": [0.0, 0.0], "severity": [1.0, 2.0, 1.0]}
    )
    with caplog.at_level(logging.WARNING, logger="models.d3.postprocess"):
        result = decide(json.dumps(three_kind_body()).encode("utf-8"), executor)

    broken = result["answers"]["urgent"]
    assert broken["type"] == "noul"
    assert broken["error"] == "invalid_model_output"
    assert "one finite non-negative probability" in broken["message"]
    assert result["answers"]["refund"]["choice"] == "wait"
    assert result["answers"]["severity"]["type"] == "score"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "urgent" in warnings[0].getMessage()
    assert "invalid_model_output" in warnings[0].getMessage()


def test_decide_maps_runtime_value_errors_to_422():
    class RejectingD3(FakeD3):
        def prepare(self, state, questions):
            raise ValueError("questions must be a nonempty mapping")

    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(body()).encode("utf-8"), RejectingD3())
    assert excinfo.value.status == 422
    assert "pinned runtime" in str(excinfo.value)
    assert "nonempty mapping" in str(excinfo.value)


def test_decide_rejects_responses_that_break_question_order():
    class ReorderedD3(FakeD3):
        def respond(self, prepared, probabilities, tokens):
            response = super().respond(prepared, probabilities, tokens)
            response["answers"] = dict(reversed(list(response["answers"].items())))
            return response

    with pytest.raises(RuntimeError, match="question order"):
        decide(json.dumps(three_kind_body()).encode("utf-8"), ReorderedD3())
