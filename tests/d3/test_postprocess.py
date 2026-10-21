import logging
import math

import pytest

from models.d3.postprocess import (
    canonical,
    choice_answer,
    confidence,
    describe,
    finish_response,
    noul_answer,
    score_answer,
    to_answer,
)


def test_confidence_is_one_minus_normalized_entropy():
    # H(0.25, 0.75) = 0.562335..., ln 2 = 0.693147...
    assert confidence([0.25, 0.75]) == pytest.approx(1 - 0.5623351446188083 / math.log(2))
    assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert confidence([0.5]) == 1.0
    # The pinned checkpoint's Decision 2.0 measure, not max-p.
    assert confidence([0.25, 0.75]) == pytest.approx(0.18872187554086717)


def test_describe_and_canonical_mirror_the_upstream_renderers():
    assert describe("plain") == "plain"
    assert describe({"x": 1}) == '{"x": 1}'
    assert canonical("plain") == "plain"
    assert canonical({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_choice_answer_renormalizes_and_breaks_ties_by_order():
    result = choice_answer(("up", "down", "left"), [1.0, 2.0, 1.0])
    assert result["type"] == "choice"
    assert result["choice"] == "down"
    assert result["probabilities"] == {"up": 0.25, "down": 0.5, "left": 0.25}
    assert result["confidence"] == pytest.approx(confidence([0.25, 0.5, 0.25]))

    tied = choice_answer(("first", "second"), [0.5, 0.5])
    assert tied["choice"] == "first"
    assert tied["confidence"] == pytest.approx(0.0)


def test_noul_answer_reports_only_the_true_probability():
    assert noul_answer([1.0, 3.0]) == {"type": "noul", "noul": pytest.approx(0.75)}
    assert set(noul_answer([0.2, 0.8])) == {"type", "noul"}


def test_score_answer_uses_string_index_keys_and_a_canonical_legend():
    result = score_answer(["minor", {"kind": "major"}, "critical"], [1.0, 2.0, 1.0])
    values = [0.25, 0.5, 0.25]
    assert result["type"] == "score"
    assert result["score"] == pytest.approx(sum(i * p for i, p in enumerate(values)))
    assert result["probabilities"] == {"0": 0.25, "1": 0.5, "2": 0.25}
    assert result["confidence"] == pytest.approx(confidence(values))
    assert result["legend"] == {
        "0": "minor",
        "1": '{"kind":"major"}',
        "2": "critical",
    }


def test_to_answer_rejects_invalid_probabilities():
    for bad in ([0.5], [0.5, -0.1], [0.0, 0.0], [float("nan"), 1.0], [0.5, float("inf")]):
        with pytest.raises(ValueError, match="one finite non-negative probability"):
            to_answer("choice", ("a", "b"), None, bad)
    with pytest.raises(ValueError, match="one finite non-negative probability"):
        to_answer("score", ("a", "b"), ["a", "b"], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="unsupported question type"):
        to_answer("triangle", ("a",), None, [1.0])


def _response(answers, input_tokens=17):
    return {
        "model": "d3",
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


def test_finish_response_passes_a_valid_response_through(caplog):
    answers = {"q": choice_answer(("a", "b"), [0.25, 0.75])}
    response = _response(answers)
    assert finish_response(("q",), response) is response
    assert not caplog.records


def test_finish_response_warns_on_embedded_invalid_model_output(caplog):
    bad = {"type": "noul", "error": "invalid_model_output", "message": "need one finite"}
    good = choice_answer(("a", "b"), [0.25, 0.75])
    response = _response({"bad": bad, "good": good})

    with caplog.at_level(logging.WARNING, logger="models.d3.postprocess"):
        finished = finish_response(("bad", "good"), response)

    assert finished is response
    assert finished["answers"]["bad"] is bad
    assert finished["answers"]["good"] is good
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "bad" in warnings[0].getMessage()
    assert "invalid_model_output" in warnings[0].getMessage()
    assert "need one finite" in warnings[0].getMessage()


def test_finish_response_rejects_broken_contracts():
    with pytest.raises(RuntimeError, match="question order"):
        finish_response(("a",), _response({"b": {"type": "noul", "noul": 0.5}}))
    with pytest.raises(RuntimeError, match="envelope"):
        finish_response(("a",), {"model": "d3", "answers": {}})
    with pytest.raises(RuntimeError, match="usage"):
        finish_response(("a",), _response({"a": {}}, input_tokens=-1))
    with pytest.raises(RuntimeError, match="usage"):
        bad_usage = _response({"a": {}})
        bad_usage["usage"]["output_tokens"] = 3
        finish_response(("a",), bad_usage)
    with pytest.raises(RuntimeError, match="pre-execution error"):
        embedded = {"type": None, "error": "invalid_question", "message": "bad"}
        finish_response(("a",), _response({"a": embedded}))
