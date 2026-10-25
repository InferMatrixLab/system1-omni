"""CPU contract tests. No weights, no torch, no GPU.

Fixtures in fixtures/reference_cases.json were produced by executing the
pinned model.py prompt and answer functions. This file checks the port
against those outputs and the rejection boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from models.pplx_decider.contract import (
    CODES,
    MAX_OPTIONS,
    MODEL_NAME,
    TOKEN_LIMIT_MESSAGE,
    RequestError,
    answer,
    messages,
    parse_body,
    parse_request,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reference_cases.json"
PINNED = ROOT / "src/models/pplx_decider/pinned_decision_config.json"
CASES = json.loads(FIXTURES.read_text(encoding="utf-8"))


def test_pinned_decision_config_hash():
    digest = hashlib.sha256(PINNED.read_bytes()).hexdigest()
    assert digest == "68d4a4c704a0c42837f749b0287bd80f70d186ed3be4dc6a90802e78223925e8"
    assert len(CODES) == MAX_OPTIONS


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["name"])
def test_reference_prompts_and_answers(case):
    produced = messages(case["state"], case["question"], case["n_images"])
    assert produced == case["messages"]
    if "answer" in case:
        assert answer(case["question"], case["probabilities_in"]) == case["answer"]


def test_large_integer_is_not_rewritten():
    case = next(item for item in CASES["cases"] if item["name"] == "noul-default")
    text = case["messages"][1]["content"][0]["text"]
    assert "18446744073709551617" in text
    assert "e+" not in text
    request = parse_request(
        {
            "model": "jev-latest",
            "state": {"event_id": 18446744073709551617},
            "questions": {"q": {"type": "noul"}},
        }
    )
    assert request.state["event_id"] == 18446744073709551617
    assert isinstance(request.state["event_id"], int)


def test_falsy_admitted_prompts_use_the_checkpoint_fallbacks():
    """Empty instructions and noul criteria are admitted, then replaced.

    ``""``, ``[]``, and ``{}`` are falsy. The checkpoint ``or`` fallbacks
    turn them into the default question and the default true/false labels.
    ``noul-falsy-fallbacks`` is the case ``check_contract.py`` compares.
    """

    questions = [
        {"type": "noul", "instructions": [], "criteria": {"false": "", "true": ""}},
        {"type": "noul", "instructions": "", "criteria": {}},
        {"type": "noul", "instructions": [], "criteria": {}},
    ]
    for question in questions:
        admitted = parse_request(
            {"model": "jev-latest", "state": "s", "questions": {"q": question}}
        )
        assert admitted.questions[0][1]["instructions"] in ("", [])
        text = messages("s", question)[1]["content"][0]["text"]
        assert "Question:\nChoose the best matching option.\n" in text
        assert "A: No / false" in text
        assert "B: Yes / true" in text
    case = next(item for item in CASES["cases"] if item["name"] == "noul-falsy-fallbacks")
    assert case["question"]["instructions"] == []
    assert case["question"]["criteria"] == {"false": "", "true": ""}


def test_choice_tie_keeps_first_option_and_key_order():
    case = next(item for item in CASES["cases"] if item["name"] == "choice-tie")
    produced = answer(case["question"], case["probabilities_in"])
    assert list(produced["probabilities"]) == ["z", "a"]
    assert produced["choice"] == "z"


def test_two_hundred_fifty_five_options_are_not_truncated():
    criteria = {f"k{i:03d}": "d" for i in range(255)}
    question = {"type": "choice", "criteria": criteria}
    text = messages("s", question)[1]["content"][0]["text"]
    assert f"{CODES[0]}: k000: d" in text
    assert f"{CODES[254]}: k254: d" in text
    assert CODES[254] in text
    request = parse_request(
        {"model": MODEL_NAME, "state": "s", "questions": {"q": question}}
    )
    assert len(request.questions[0][1]["criteria"]) == 255


def test_question_order_follows_the_request():
    request = parse_request(
        {
            "model": "autojev",
            "state": "s",
            "questions": {
                "second": {"type": "score", "criteria": ["low", "high"]},
                "first": {"type": "noul"},
            },
        }
    )
    assert [name for name, _ in request.questions] == ["second", "first"]


@pytest.mark.parametrize(
    ("name", "body", "message"),
    [
        (
            "choice-256",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {
                    "q": {"type": "choice", "criteria": {str(i): "d" for i in range(256)}}
                },
            },
            "1 to 255",
        ),
        (
            "score-11",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}},
            },
            "2 to 10",
        ),
        (
            "score-1",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "score", "criteria": ["only"]}},
            },
            "2 to 10",
        ),
        (
            "empty-questions",
            {"model": "jev-latest", "state": "s", "questions": {}},
            "nonempty",
        ),
        (
            "numeric-state",
            {"model": "jev-latest", "state": 1, "questions": {"q": {"type": "noul"}}},
            "string, object, or array",
        ),
        (
            "unknown-field",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "noul"}},
                "extra": 1,
            },
            "unknown request fields",
        ),
        (
            "bad-model",
            {"model": "pplx", "state": "s", "questions": {"q": {"type": "noul"}}},
            "Unknown model",
        ),
        (
            "noul-bad-key",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "noul", "criteria": {"yes": "y"}}},
            },
            "true and false",
        ),
        (
            "choice-number",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "choice", "criteria": {"a": 1}}},
            },
            "string, object, array, or null",
        ),
        (
            "score-null",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "score", "criteria": ["a", None]}},
            },
            "string, object, or array",
        ),
        (
            "five-images",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "noul"}},
                "images": ["x"] * 5,
            },
            "at most 4",
        ),
        (
            "images-null",
            {
                "model": "jev-latest",
                "state": "s",
                "questions": {"q": {"type": "noul"}},
                "images": None,
            },
            "at most 4",
        ),
    ],
)
def test_rejections(name, body, message):
    with pytest.raises(RequestError, match=message):
        parse_request(body)


def test_disabled_thinking_is_ignored():
    base = {"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul"}}}
    admitted = parse_request({**base, "thinking": {"enabled": False}})
    assert admitted == parse_request(base)
    assert messages(admitted.state, admitted.questions[0][1]) == messages("s", {"type": "noul"})


@pytest.mark.parametrize(
    "thinking",
    [
        {"enabled": True},
        {"enabled": "false"},
        {"enabled": 0},
        {"enabled": 0.0},
        {"enabled": False, "budget": 0},
        {},
        False,
        "off",
        None,
    ],
)
def test_other_thinking_values_are_rejected(thinking):
    body = {
        "model": "jev-latest",
        "state": "s",
        "questions": {"q": {"type": "noul"}},
        "thinking": thinking,
    }
    with pytest.raises(RequestError, match="thinking.enabled must be false") as caught:
        parse_request(body)
    assert caught.value.status == 422


def test_question_thinking_stays_unknown():
    body = {
        "model": "jev-latest",
        "state": "s",
        "questions": {"q": {"type": "noul", "thinking": {"enabled": False}}},
    }
    with pytest.raises(RequestError, match="unknown fields"):
        parse_request(body)


def test_bad_json_is_400():
    with pytest.raises(RequestError) as caught:
        parse_body(b"{")
    assert caught.value.status == 400


def test_token_limit_message_is_the_checkpoint_sentence():
    assert "no input was truncated" in TOKEN_LIMIT_MESSAGE


def test_answer_rejects_bad_probabilities():
    question = {"type": "noul"}
    with pytest.raises(ValueError, match="positive mass"):
        answer(question, [0.0, 0.0])
    with pytest.raises(ValueError, match="Each option"):
        answer(question, [1.0])
