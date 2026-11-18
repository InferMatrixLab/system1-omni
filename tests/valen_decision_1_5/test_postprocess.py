import math

import pytest

from models.valen_decision_1_5.postprocess import answer, build_response
from models.valen_decision_1_5.preprocess import ResponseContext
from models.valen_decision_1_5.protocol import Question

CHOICE = Question("move", "choice", "Choose.", ("up", "down"), ("Move up", "Move down"))
NOUL = Question("ask", "noul", "Was it duplicated?", ("true", "false"), (None, None))
SCORE = Question("grade", "score", "Grade.", ("0", "1", "2"), ("none", "minor", "major"))


def test_choice_answer_follows_the_runtime_formula_and_field_order():
    result = answer(CHOICE, [0.0, math.log(3.0)])

    assert list(result) == ["type", "probabilities", "choice", "confidence"]
    assert result["type"] == "choice"
    assert result["choice"] == "down"
    assert result["probabilities"]["up"] == pytest.approx(0.25)
    assert result["probabilities"]["down"] == pytest.approx(0.75)
    assert result["confidence"] == pytest.approx(0.5)


def test_choice_ties_follow_candidate_order_and_one_option_is_confident():
    assert answer(CHOICE, [1.0, 1.0])["choice"] == "up"
    assert answer(CHOICE, [1.0, 1.0])["confidence"] == 0.0
    one = Question("only", "choice", "Pick.", ("ok",), ("OK",))
    result = answer(one, [0.0])
    assert result["confidence"] == 1.0
    assert result["probabilities"] == {"ok": 1.0}


def test_noul_answer_is_the_true_probability_only():
    result = answer(NOUL, [math.log(3.0), 0.0])

    assert list(result) == ["type", "noul"]
    assert result["noul"] == pytest.approx(0.75)


def test_score_answer_reports_expected_level_legend_and_concentration():
    result = answer(SCORE, [math.log(7.0), math.log(2.0), 0.0])

    assert list(result) == ["type", "probabilities", "score", "legend", "confidence"]
    assert result["probabilities"] == pytest.approx({"0": 0.7, "1": 0.2, "2": 0.1})
    assert result["score"] == pytest.approx(0.4)
    assert result["legend"] == {"0": "none", "1": "minor", "2": "major"}
    # distance from the mode = 0.2 * 1 + 0.1 * 2; uniform mean deviation = 2/3.
    assert result["confidence"] == pytest.approx(1 - 0.4 / (2 / 3))
    assert answer(SCORE, [0.0, 0.0, 0.0])["confidence"] == 0.0


def test_probabilities_sum_to_one_for_large_logits():
    result = answer(CHOICE, [1000.0, 999.0])
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["choice"] == "up"


def test_answer_rejects_invalid_logits():
    with pytest.raises(ValueError, match="one logit"):
        answer(CHOICE, [0.0])
    with pytest.raises(ValueError, match="invalid logits"):
        answer(CHOICE, [float("nan"), 0.0])
    with pytest.raises(ValueError, match="invalid logits"):
        answer(NOUL, [float("inf"), 0.0])


def test_build_response_preserves_question_order_and_usage():
    context = ResponseContext("valen-decision-1.5-0.8b", (CHOICE, NOUL))
    answers = {"move": answer(CHOICE, [0.0, 1.0]), "ask": answer(NOUL, [0.0, 1.0])}

    assert build_response(context, answers, 17, 17) == {
        "model": "valen-decision-1.5-0.8b",
        "answers": answers,
        "usage": {"input_tokens": 17, "output_tokens": 0},
        "internal_usage": {"compute_tokens": 17},
    }
    with pytest.raises(ValueError, match="question order"):
        build_response(context, dict(reversed(answers.items())), 17, 17)
    with pytest.raises(ValueError, match="non-negative"):
        build_response(context, answers, -1, 17)
