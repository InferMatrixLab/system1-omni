import json
from pathlib import Path

import pytest

from models.clef.preprocess import MAX_INPUT_TOKENS, enforce_limit, measure, validate_questions
from models.clef.protocol import MAX_BODY, MalformedJSON, RequestError, parse_body, parse_request


def body(**overrides):
    value = {
        "model": "clef-flash",
        "state": "Checkout has been failing.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments", "technical": "Bugs"},
            }
        },
    }
    value.update(overrides)
    return value


def test_parse_body_rejects_ambiguous_json_and_the_body_cap():
    with pytest.raises(MalformedJSON, match="duplicate"):
        parse_body(b'{"model":"a","model":"b"}')
    with pytest.raises(MalformedJSON, match="non-finite"):
        parse_body(b'{"state":NaN}')
    with pytest.raises(MalformedJSON):
        parse_body(b"[1]")
    with pytest.raises(RequestError) as excinfo:
        parse_body(b'{"state":"' + b"a" * MAX_BODY + b'"}')
    assert excinfo.value.status == 413


def test_thinking_enabled_false_is_admitted_and_omitted_from_the_prompt():
    from models.clef.prompt import prompt_pieces

    plain = body()
    marked = body(thinking={"enabled": False, "ignored": 1})
    parsed = parse_request(marked)
    assert "thinking" not in parsed.payload
    assert prompt_pieces(parsed.payload) == prompt_pieces(plain)
    with pytest.raises(RequestError, match="thinking.enabled must be false"):
        parse_request(body(thinking={"enabled": True}))
    with pytest.raises(RequestError, match="thinking.enabled must be false"):
        parse_request(body(thinking="off"))
    with pytest.raises(RequestError, match="thinking.enabled must be false"):
        parse_request(body(thinking={"budget": 0}))


def test_parse_requires_model_and_state_and_rejects_unknown_fields():
    with pytest.raises(RequestError, match="Unknown request fields"):
        parse_request(body(media_kwargs={"fps": 1}))
    with pytest.raises(RequestError, match="model and state are required"):
        parse_request(body(model=None))
    parsed = parse_request(body(state={"b": 1, "a": 2}))
    assert parsed.payload["state"] == {"b": 1, "a": 2}


def test_media_lists_are_refused_and_empty_lists_are_absent():
    assert parse_request(body(images=[], videos=[])).payload["images"] == []
    with pytest.raises(RequestError, match="images are not served"):
        parse_request(body(images=["data:image/png;base64,AAAA"]))
    with pytest.raises(RequestError, match="videos must be a list"):
        parse_request(body(videos="clip.mp4"))


def test_question_shapes_match_the_three_decision_types():
    payload = body()
    payload["questions"] = {
        "team": payload["questions"]["team"],
        "urgent": {"type": "noul"},
        "severity": {"type": "score", "criteria": ["No impact", "Critical"]},
    }
    validate_questions(parse_request(payload))
    with pytest.raises(RequestError, match="type must be noul, choice, or score"):
        validate_questions(parse_request(body(questions={"x": {"type": "rank"}})))
    with pytest.raises(RequestError, match="criteria must not be empty"):
        validate_questions(
            parse_request(body(questions={"x": {"type": "choice", "criteria": {}}}))
        )
    with pytest.raises(RequestError, match="criteria must not be empty"):
        validate_questions(
            parse_request(body(questions={"x": {"type": "score", "criteria": {"0": "low"}}}))
        )


def test_limits_reject_schema_overflow_and_state_overflow_without_slicing():
    request = parse_request(body())
    fixed, total = measure(lambda text: len(text), request)
    assert total > fixed > 0
    enforce_limit(fixed, total, total)
    with pytest.raises(RequestError, match="schema requires") as schema_error:
        enforce_limit(fixed, total, fixed - 1)
    assert schema_error.value.status == 422
    with pytest.raises(RequestError, match="was not truncated") as state_error:
        enforce_limit(fixed, total, total - 1)
    assert state_error.value.status == 422
    assert MAX_INPUT_TOKENS == 16384


def test_example_request_is_admitted():
    path = Path(__file__).resolve().parents[2] / "recipe" / "clef" / "example-request.json"
    request = parse_request(json.loads(path.read_text(encoding="utf-8")))
    validate_questions(request)
    fixed, total = measure(lambda text: 1, request)
    enforce_limit(fixed, total)


def test_smoke_manifest_requests_are_admitted():
    path = Path_smoke()
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        payload = {**case["request"], "model": "clef-flash"}
        request = parse_request(payload)
        validate_questions(request)
        fixed, total = measure(lambda text: 1, request)
        enforce_limit(fixed, total)


def Path_smoke():
    return Path(__file__).resolve().parents[2] / "benchmarks" / "smoke.jsonl"
