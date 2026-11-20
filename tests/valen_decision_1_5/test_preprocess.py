import hashlib

from models.valen_decision_1_5.preprocess import prepare_request, question_record
from models.valen_decision_1_5.protocol import parse_request

from .test_protocol import MODEL_NAME, body, noul_body, score_body


def prepare(value, tmp_path):
    return prepare_request(parse_request(value, MODEL_NAME), tmp_path)


def test_prepare_materializes_request_scoped_image_and_record(tmp_path):
    prepared = prepare(body(), tmp_path)

    data = (tmp_path / "input.png").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert prepared.image_path == tmp_path / "input.png"
    assert prepared.image_sha256 == digest
    assert prepared.media_root == tmp_path
    assert prepared.record == {
        "request": {
            "state": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "input.png"}}],
                    }
                ]
            },
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Choose a move.",
                    "criteria": {"up": "Move up", "down": "Move down"},
                }
            },
        },
        "assets": [{"path": "input.png", "sha256": digest}],
    }
    assert prepared.response_context.model == MODEL_NAME
    assert [q.name for q in prepared.response_context.questions] == ["move"]


def test_prepare_names_jpeg_media_by_format(tmp_path):
    from .test_protocol import image_data_url

    prepared = prepare(body(state={"image": image_data_url("JPEG")}), tmp_path)
    assert prepared.image_path == tmp_path / "input.jpg"


def test_prepare_compiles_text_state_without_media(tmp_path):
    prepared = prepare(body(state="Dialog: the update finished."), tmp_path)

    assert prepared.image_path is None
    assert prepared.image_sha256 is None
    assert prepared.record["assets"] == []
    assert prepared.record["request"]["state"] == {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Dialog: the update finished."}],
            }
        ]
    }
    assert list(tmp_path.iterdir()) == []


def question(value):
    return question_record(parse_request(value, MODEL_NAME).questions[0])


def test_noul_record_passes_a_rubric_through_and_omits_an_absent_one():
    assert question(noul_body()) == {
        "type": "noul",
        "instructions": "Was the card charged twice?",
    }
    assert question(noul_body(criteria={"false": "Unique.", "true": "Duplicated."})) == {
        "type": "noul",
        "instructions": "Was the card charged twice?",
        "criteria": {"true": "Duplicated.", "false": "Unique."},
    }


def test_score_record_keeps_the_ordered_level_list():
    assert question(score_body()) == {
        "type": "score",
        "instructions": "Grade the severity.",
        "criteria": ["No error", "A minor error", "A major error"],
    }


def test_choice_record_fills_null_descriptions_with_their_keys():
    value = body()
    value["questions"]["move"]["criteria"] = {"up": None, "down": "Move down"}
    assert question(value)["criteria"] == {"up": "up", "down": "Move down"}
