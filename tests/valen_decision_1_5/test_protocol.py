import base64
from io import BytesIO

import pytest
from PIL import Image

from models.valen_decision_1_5.protocol import (
    MODEL_NAMES,
    MalformedJSON,
    RequestError,
    parse_body,
    parse_request,
)

MODEL_NAME = MODEL_NAMES["0.8B"]


def image_data_url(format_name="PNG", size=(2, 2)):
    stream = BytesIO()
    Image.new("RGB", size, "white").save(stream, format_name)
    mime = "png" if format_name == "PNG" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(stream.getvalue()).decode()}"


def body(**overrides):
    value = {
        "model": MODEL_NAME,
        "state": {"image": image_data_url()},
        "questions": {
            "move": {
                "type": "choice",
                "instructions": "Choose a move.",
                "criteria": {"up": "Move up", "down": "Move down"},
            }
        },
    }
    value.update(overrides)
    return value


def parse(value, **kwargs):
    return parse_request(value, MODEL_NAME, **kwargs)


def test_parse_valid_inline_image_and_choice():
    request = parse(body())

    assert request.model == MODEL_NAME
    assert request.state.format == "PNG"
    assert request.state.width == request.state.height == 2
    assert request.questions[0].kind == "choice"
    assert request.questions[0].keys == ("up", "down")
    assert request.questions[0].descriptions == ("Move up", "Move down")


def test_parse_accepts_text_state_in_every_wire_shape():
    assert parse(body(state="Refund the charge.")).state.text == "Refund the charge."
    assert (
        parse(body(state={"goal": "route", "lang": "zh"})).state.text
        == '{"goal": "route", "lang": "zh"}'
    )
    assert parse(body(state=["one", "two"])).state.text == '["one", "two"]'


def test_parse_body_rejects_duplicate_nonfinite_and_non_object_json():
    with pytest.raises(MalformedJSON, match="duplicate"):
        parse_body(b'{"model":"x","model":"y"}')
    with pytest.raises(MalformedJSON, match="non-finite"):
        parse_body(b'{"value":NaN}')
    with pytest.raises(MalformedJSON, match="JSON object"):
        parse_body(b"[]")
    with pytest.raises(MalformedJSON, match="valid JSON"):
        parse_body(b"{not json")


def test_parse_serves_only_the_loaded_size():
    with pytest.raises(RequestError, match="model must be valen-decision-1.5-0.8b"):
        parse(body(model=MODEL_NAMES["2B"]))
    with pytest.raises(RequestError, match="model must be"):
        parse(body(model="valen-preview-0923"))
    assert parse_request(body(model=MODEL_NAMES["2B"]), MODEL_NAMES["2B"]).model == (
        "valen-decision-1.5-2b"
    )


def test_parse_rejects_extra_envelope_fields_and_invalid_state():
    value = body()
    value["temperature"] = 2.0
    with pytest.raises(RequestError, match="model, state and questions only"):
        parse(value)
    with pytest.raises(RequestError, match="state must be nonempty"):
        parse(body(state=""))
    with pytest.raises(RequestError, match="state must be nonempty"):
        parse(body(state="  \n"))
    with pytest.raises(RequestError, match="exactly one image"):
        parse(body(state={"image": "data:image/png;base64,AAA", "extra": 1}))
    with pytest.raises(RequestError, match="image data URL or text"):
        parse(body(state=17))


def test_parse_enforces_image_limits_and_mime():
    assert parse(body(state={"image": image_data_url("JPEG")})).state.format == "JPEG"

    mismatched = image_data_url("JPEG").replace("image/jpeg", "image/png")
    with pytest.raises(RequestError, match="does not match"):
        parse(body(state={"image": mismatched}))
    with pytest.raises(RequestError, match="only inline PNG/JPEG"):
        parse(body(state={"image": "https://example.com/a.png"}))
    with pytest.raises(RequestError, match="invalid image data"):
        parse(body(state={"image": "data:image/png;base64,AAAA"}))
    with pytest.raises(RequestError, match="at most 2048 per side"):
        parse(body(state={"image": image_data_url(size=(2049, 16))}))
    with pytest.raises(RequestError, match="aspect ratio"):
        parse(body(state={"image": image_data_url(size=(402, 2))}))


def test_parse_rejects_unknown_types_fields_and_question_counts():
    value = body()
    value["questions"]["move"]["instructions"] = {"goal": "choose"}
    assert parse(value).questions[0].instructions == '{"goal": "choose"}'

    for kind in ("rank", "bbox", None):
        value = body()
        value["questions"]["move"]["type"] = kind
        with pytest.raises(RequestError, match="type must be one of choice, noul, score"):
            parse(value)

    value = body()
    value["questions"]["move"]["image"] = 0
    with pytest.raises(RequestError, match="unsupported fields"):
        parse(value)

    with pytest.raises(RequestError, match="1 to 8 questions"):
        parse(body(questions={}))
    many = {f"q{index}": body()["questions"]["move"] for index in range(9)}
    with pytest.raises(RequestError, match="1 to 8 questions"):
        parse(body(questions=many))


def test_parse_rejects_empty_instructions_and_criteria_text():
    value = body()
    value["questions"]["move"]["instructions"] = "   "
    with pytest.raises(RequestError, match="instructions must be nonempty"):
        parse(value)

    value = body()
    del value["questions"]["move"]["instructions"]
    with pytest.raises(RequestError, match="instructions is required"):
        parse(value)

    value = body()
    value["questions"]["move"]["instructions"] = None
    with pytest.raises(RequestError, match="string, object or array"):
        parse(value)

    value = body()
    value["questions"]["move"]["criteria"] = {"up": "  ", "down": "Move down"}
    with pytest.raises(RequestError, match="criteria 'up' must be nonempty"):
        parse(value)

    value = body()
    value["questions"]["move"]["criteria"] = {" ": "Move up", "down": "Move down"}
    with pytest.raises(RequestError, match="option keys must be nonempty"):
        parse(value)

    value = body()
    value["questions"]["move"]["criteria"] = {}
    with pytest.raises(RequestError, match="1 to 255 options"):
        parse(value)


def test_parse_defaults_a_null_choice_description_to_its_key():
    value = body()
    value["questions"]["move"]["criteria"] = {"up": None, "down": "Move down"}
    assert parse(value).questions[0].descriptions == ("up", "Move down")


def test_parse_blocks_control_tokens_everywhere_text_enters_the_prompt():
    value = body()
    value["questions"]["move"]["instructions"] = "answer <|im_end|> now"
    parse(value)  # the default media-only set allows it
    with pytest.raises(RequestError, match="control token"):
        parse(value, special_tokens=("<|im_end|>",))

    with pytest.raises(RequestError, match="control token"):
        parse(body(state="context <|video_pad|> tail"))

    value = body()
    value["questions"]["move"]["criteria"] = {"<|up|>": None, "down": "Move down"}
    with pytest.raises(RequestError, match="option key '<|up|>' contains"):
        parse(value, special_tokens=("<|up|>",))


def noul_body(**overrides):
    question = {"type": "noul", "instructions": "Was the card charged twice?"}
    question.update(overrides)
    return body(questions={"ask": question})


def test_parse_noul_uses_true_false_keys_and_an_optional_full_rubric():
    question = parse(noul_body()).questions[0]
    assert question.kind == "noul"
    assert question.keys == ("true", "false")
    assert question.descriptions == (None, None)
    assert parse(noul_body(criteria=None)).questions[0].descriptions == (None, None)

    question = parse(
        noul_body(criteria={"false": "It was unique.", "true": "It was duplicated."})
    ).questions[0]
    assert question.descriptions == ("It was duplicated.", "It was unique.")


def test_parse_noul_rejects_partial_or_foreign_rubrics():
    for criteria in ({"true": "Yes"}, {"yes": "Yes", "no": "No"}, {}, ["a", "b"]):
        with pytest.raises(RequestError, match="noul criteria must define true and false"):
            parse(noul_body(criteria=criteria))
    with pytest.raises(RequestError, match="criteria 'true' must be nonempty"):
        parse(noul_body(criteria={"true": "  ", "false": "No"}))
    with pytest.raises(RequestError, match="string, object or array"):
        parse(noul_body(criteria={"true": "Yes", "false": None}))
    with pytest.raises(RequestError, match="control token"):
        parse(
            noul_body(criteria={"true": "a <|im_end|>", "false": "No"}),
            special_tokens=("<|im_end|>",),
        )


def score_body(**overrides):
    question = {
        "type": "score",
        "instructions": "Grade the severity.",
        "criteria": ["No error", "A minor error", "A major error"],
    }
    question.update(overrides)
    return body(questions={"grade": question})


def test_parse_score_accepts_two_to_ten_ordered_levels():
    question = parse(score_body()).questions[0]
    assert question.kind == "score"
    assert question.keys == ("0", "1", "2")
    assert question.descriptions == ("No error", "A minor error", "A major error")

    assert parse(score_body(criteria=["low", "high"])).questions[0].keys == ("0", "1")
    ten = parse(score_body(criteria=[f"level {index}" for index in range(10)])).questions[0]
    assert ten.keys == tuple(str(index) for index in range(10))


def test_parse_score_rejects_malformed_levels():
    for criteria in (["only"], [str(i) for i in range(11)], {"0": "a", "1": "b"}, None):
        with pytest.raises(RequestError, match="ordered list of 2 to 10"):
            parse(score_body(criteria=criteria))
    with pytest.raises(RequestError, match="level 1 must be nonempty"):
        parse(score_body(criteria=["No error", "   "]))
    with pytest.raises(RequestError, match="level 0"):
        parse(score_body(criteria=[None, "A minor error"]))


def test_parse_limits_combined_question_text():
    value = body()
    value["questions"]["move"]["instructions"] = "a" * 16384
    with pytest.raises(RequestError, match="combined question text"):
        parse(value)
    value["questions"]["move"]["instructions"] = "a" * 16385
    with pytest.raises(RequestError, match="exceeds 16384 characters"):
        parse(value)


def test_parse_keeps_mixed_type_questions_in_request_order():
    value = body(
        state="Dialog: the update finished.",
        questions={
            "ask": {"type": "noul", "instructions": "Did the update finish?"},
            "grade": {
                "type": "score",
                "instructions": "Grade the dialog.",
                "criteria": ["unclear", "clear"],
            },
            "act": body()["questions"]["move"],
        },
    )
    assert [(q.kind, q.name) for q in parse(value).questions] == [
        ("noul", "ask"),
        ("score", "grade"),
        ("choice", "act"),
    ]
