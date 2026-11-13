"""Tests for recipe/valen_decision_1_5/compare_reference.py.

Checks the pure helpers without weights, torch or a GPU: the reference
request builder and the PASS/FAIL verdicts of compare_case.
"""

import base64
import copy
import json
import sys
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "recipe" / "valen_decision_1_5"
sys.path.insert(0, str(RECIPE))

import compare_reference as tool  # noqa: E402

from models.valen_decision_1_5.protocol import parse_request  # noqa: E402


def response(answers, tokens=17, model="valen-decision-1.5-0.8b"):
    return {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": 0},
        "internal_usage": {"compute_tokens": tokens},
    }


ANSWERS = {
    "move": {
        "type": "choice",
        "probabilities": {"up": 0.25, "down": 0.75},
        "choice": "down",
        "confidence": 0.5,
    },
    "ask": {"type": "noul", "noul": 0.75},
    "grade": {
        "type": "score",
        "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
        "score": 0.4,
        "legend": {"0": "none", "1": "minor", "2": "major"},
        "confidence": 0.4,
    },
}


def verdict(worker_answers, reference_answers=ANSWERS, **kwargs):
    lines, ok = tool.compare_case(
        "case", response(worker_answers, **kwargs), response(reference_answers, model="Valen")
    )
    return "\n".join(lines), ok


def changed(qid, field, value):
    answers = copy.deepcopy(ANSWERS)
    answers[qid][field] = value
    return answers


def test_identical_responses_pass_with_zero_deltas():
    report, ok = verdict(copy.deepcopy(ANSWERS))

    assert ok
    assert "reference model: Valen" in report
    assert "worker model: valen-decision-1.5-0.8b" in report
    assert "input_tokens: 17 == 17" in report
    assert "max probability delta: 0.0000000000e+00" in report
    assert "max answer delta: 0.0000000000e+00" in report
    assert "decision agreement: PASS" in report
    assert "token accounting: PASS" in report


def test_differences_inside_the_tolerance_pass_and_are_reported():
    answers = changed("ask", "noul", 0.75 + 4e-7)
    report, ok = verdict(answers)

    assert ok
    assert "max probability delta: 4.0000000" in report
    assert "probability tolerance <= 1e-06: PASS" in report


@pytest.mark.parametrize(
    "qid, field, value, line",
    [
        ("ask", "noul", 0.75 + 2e-6, "probability tolerance <= 1e-06: FAIL"),
        (
            "move",
            "probabilities",
            {"up": 0.25 + 2e-6, "down": 0.75},
            "probability tolerance <= 1e-06: FAIL",
        ),
        ("grade", "score", 0.4 + 2e-6, "answer tolerance <= 1e-06: FAIL"),
        ("grade", "confidence", 0.4 - 2e-6, "confidence tolerance <= 1e-06: FAIL"),
    ],
)
def test_a_delta_over_the_tolerance_fails(qid, field, value, line):
    report, ok = verdict(changed(qid, field, value))

    assert not ok
    assert line in report


@pytest.mark.parametrize(
    "qid, field, value",
    [
        ("move", "choice", "up"),
        ("move", "type", "score"),
        ("move", "probabilities", {"down": 0.75, "up": 0.25}),
        ("grade", "legend", {"0": "none", "1": "minor", "2": "severe"}),
        ("ask", "noul", float("nan")),
        ("grade", "confidence", float("inf")),
        ("ask", "noul", True),
    ],
)
def test_discrete_or_non_finite_differences_fail(qid, field, value):
    report, ok = verdict(changed(qid, field, value))

    assert not ok
    assert "decision agreement: FAIL" in report


def test_a_non_finite_reference_fails_too():
    _, ok = verdict(copy.deepcopy(ANSWERS), changed("ask", "noul", float("nan")))
    assert not ok


def test_extra_missing_or_reordered_fields_and_questions_fail():
    extra = copy.deepcopy(ANSWERS)
    extra["ask"]["confidence"] = 0.5
    assert not verdict(extra)[1]

    missing = copy.deepcopy(ANSWERS)
    del missing["grade"]["legend"]
    assert not verdict(missing)[1]

    reordered = dict(reversed(ANSWERS.items()))
    report, ok = verdict(reordered)
    assert not ok and "question order" in report


def test_token_accounting_must_match_exactly():
    report, ok = verdict(copy.deepcopy(ANSWERS), tokens=18)

    assert not ok
    assert "input_tokens: 18 != 17" in report
    assert "compute_tokens: 18 != 17" in report
    assert "token accounting: FAIL" in report


def test_reference_request_keeps_text_and_questions_verbatim(tmp_path):
    body = json.loads((RECIPE / "example-noul-request.json").read_text(encoding="utf-8"))
    request = tool.reference_request(body, tmp_path)

    assert request == {"state": body["state"], "questions": body["questions"]}
    assert request["questions"] is not body["questions"]
    assert list(tmp_path.iterdir()) == []


def test_reference_request_materializes_the_inline_image(tmp_path):
    body = json.loads((RECIPE / "example-request.json").read_text(encoding="utf-8"))
    request = tool.reference_request(body, tmp_path)

    encoded = body["state"]["image"].partition(",")[2]
    assert (tmp_path / "reference.png").read_bytes() == base64.b64decode(encoded)
    assert request["state"] == {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "reference.png"}}],
            }
        ]
    }
    with pytest.raises(ValueError, match="plain-text or inline-image"):
        tool.reference_request({"state": {"goal": "x"}, "questions": {}}, tmp_path)


@pytest.mark.parametrize("name", sorted(tool.CASES))
def test_recipe_fixtures_are_valid_wire_requests(name):
    body = json.loads((RECIPE / tool.CASES[name]).read_text(encoding="utf-8"))
    request = parse_request(body, "valen-decision-1.5-0.8b")

    assert request.questions


def test_fixtures_cover_every_question_type_for_text_and_image_state():
    covered = set()
    for fixture in tool.CASES.values():
        body = json.loads((RECIPE / fixture).read_text(encoding="utf-8"))
        state = "image" if isinstance(body["state"], dict) else "text"
        covered |= {(state, question["type"]) for question in body["questions"].values()}

    assert covered == {
        (state, kind) for state in ("image", "text") for kind in ("choice", "noul", "score")
    }
