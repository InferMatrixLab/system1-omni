"""Envelope admission tests for the Vela 2.0 wire contract.

No weights, no torch, no package: what is under test is the worker's own
outer layer — the shared JSON transport discipline and the upstream
``system_one_request`` preamble with its exact ``detail`` triples. The
deeper validation (state rendering, question semantics, the 413) belongs
to the pinned runtime and is pinned by the author-side fixtures.
"""

from __future__ import annotations

import json

import pytest

from models.vela.protocol import (
    MAX_BODY,
    MODEL_NAME,
    MalformedJSON,
    RequestError,
    admit_request,
    envelope_error,
    parse_body,
)

SERVED = {MODEL_NAME}


def body(**overrides):
    value = {
        "model": MODEL_NAME,
        "state": "My card was charged twice.",
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this about a duplicate charge?",
            }
        },
    }
    value.update(overrides)
    return value


def detail(exc):
    return exc.value.body["detail"]


# ---------------------------------------------------------------- parse_body


def test_parse_body_decodes_a_json_object():
    assert parse_body(json.dumps(body()).encode("utf-8")) == body()


def test_parse_body_rejects_non_objects():
    for raw in (b"[1, 2]", b'"text"', b"3", b"null", b"true"):
        with pytest.raises(MalformedJSON, match="request must be a JSON object"):
            parse_body(raw)


def test_parse_body_rejects_invalid_json_and_non_utf8():
    with pytest.raises(MalformedJSON, match="valid JSON and UTF-8"):
        parse_body(b"{nope")
    with pytest.raises(MalformedJSON, match="valid JSON and UTF-8"):
        parse_bytes = b'{"state": "' + b"\xff\xfe" + b'"}'
        parse_body(parse_bytes)


def test_parse_body_rejects_duplicate_keys_and_nonfinite_numbers():
    raw = b'{"model": "a", "model": "b"}'
    with pytest.raises(MalformedJSON, match="duplicate JSON key 'model'"):
        parse_body(raw)
    with pytest.raises(MalformedJSON, match="non-finite JSON number"):
        parse_body(b'{"state": NaN}')


def test_parse_body_rejects_lone_surrogates():
    with pytest.raises(MalformedJSON, match="valid JSON and UTF-8"):
        parse_body('{"state": "ends with \\ud800"}'.encode("utf-8"))


def test_parse_body_rejects_oversize_bodies():
    padding = b"x" * (MAX_BODY + 1)
    with pytest.raises(RequestError) as excinfo:
        parse_body(b'{"state": "' + padding + b'"}')
    assert excinfo.value.status == 413


# ------------------------------------------------------------ admit_request


def test_admit_request_passes_the_envelope_through_unchanged():
    value = body()
    assert admit_request(value, SERVED) is value


@pytest.mark.parametrize(
    ("drop", "field"),
    [(("model",), "model"), (("state",), "state"), (("questions",), "questions")],
)
def test_missing_fields_report_upstreams_missing_triple(drop, field):
    value = body()
    for key in drop:
        del value[key]
    with pytest.raises(RequestError) as excinfo:
        admit_request(value, SERVED)
    assert excinfo.value.status == 422
    assert detail(excinfo) == [
        {"loc": ["body", field], "msg": "Field required", "type": "missing"}
    ]


def test_extra_fields_report_upstreams_extra_forbidden_triple():
    value = body(images=[])
    with pytest.raises(RequestError) as excinfo:
        admit_request(value, SERVED)
    assert detail(excinfo) == [
        {
            "loc": ["body", "images"],
            "msg": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        }
    ]


def test_the_first_sorted_extra_field_is_named():
    value = body(videos=1, images=2)
    with pytest.raises(RequestError) as excinfo:
        admit_request(value, SERVED)
    assert detail(excinfo)[0]["loc"] == ["body", "images"]


def test_missing_fields_are_reported_before_extras():
    value = body(extra=1)
    del value["state"]
    with pytest.raises(RequestError) as excinfo:
        admit_request(value, SERVED)
    assert detail(excinfo)[0]["type"] == "missing"


def test_a_blank_model_is_rejected_as_a_value_error():
    for model in ("", "   ", None, 7):
        with pytest.raises(RequestError) as excinfo:
            admit_request(body(model=model), SERVED)
        assert excinfo.value.status == 422
        assert detail(excinfo) == [
            {
                "loc": ["body", "model"],
                "msg": "model must not be empty or whitespace",
                "type": "value_error",
            }
        ]


def test_another_model_name_is_rejected_with_upstreams_model_type():
    with pytest.raises(RequestError) as excinfo:
        admit_request(body(model="vllm-sr/Vela-2.0-0.8B"), SERVED)
    assert detail(excinfo) == [
        {
            "loc": ["body", "model"],
            "msg": "Model is not served by this runtime",
            "type": "value_error.model",
        }
    ]


def test_a_custom_served_name_admits_its_own_model():
    served = {"vllm-sr/Vela-2.0-0.8B"}
    assert admit_request(body(model="vllm-sr/Vela-2.0-0.8B"), served)["model"] == (
        "vllm-sr/Vela-2.0-0.8B"
    )


def test_envelope_error_builds_the_upstream_detail_shape():
    exc = envelope_error(["questions", "qid", "criteria"], "boom", "literal_error")
    assert exc.status == 422
    assert exc.body == {
        "detail": [
            {
                "loc": ["body", "questions", "qid", "criteria"],
                "msg": "boom",
                "type": "literal_error",
            }
        ]
    }
    assert str(exc) == "questions.qid.criteria: boom"
