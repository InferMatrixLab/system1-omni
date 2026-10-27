"""Tests for the mirrored answer formulas and response finishing.

The formulas are pure mirrors of the pinned ``system_one_response``
mathematics — the top-two choice margin, the normalized-variance score
confidence, the expected zero-based level — and ``finish_response`` is
the worker's own response-contract gate. No weights, no torch.
"""

from __future__ import annotations

import pytest

from models.vela.postprocess import (
    choice_confidence,
    expected_answer_order,
    expected_level,
    finish_response,
    score_confidence,
)

FIVE_KIND_QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this about billing?",
                "criteria": {"true": "money", "false": "no money"}},
    "area": {"type": "choice", "instructions": "Which area?",
             "criteria": {"health": "medicine", "math": "arithmetic"}},
    "severity": {"type": "score", "instructions": "How severe?",
                 "criteria": ["minor", "moderate", "major"]},
    "issues": {"type": "set", "instructions": "Which issues apply?",
               "criteria": {"delivery": "parcels", "login": "accounts"}},
    "pii": {"type": "span", "instructions": "Which spans are personal?",
            "criteria": {"PERSON": "a person's name"}},
}


def five_kind_response():
    return {
        "model": "vllm-sr/Vela-2.0-0.3B",
        "answers": {
            "billing": {"type": "noul", "noul": 0.75},
            "area": {"type": "choice", "choice": "health", "confidence": 0.5,
                     "probabilities": {"health": 0.75, "math": 0.25}},
            "severity": {"type": "score", "score": 1.25, "confidence": 0.5,
                         "legend": {"0": "minor", "1": "moderate", "2": "major"},
                         "probabilities": {"0": 0.125, "1": 0.25, "2": 0.625}},
            "issues.delivery": {"type": "noul", "noul": 0.9},
            "issues.login": {"type": "noul", "noul": 0.1},
            "pii": {"type": "noul", "noul": 0.5},
        },
        "usage": {"input_tokens": 128, "output_tokens": 0},
        "spans": {"pii": [{"label": "PERSON", "start": 8, "end": 17,
                           "text": "Tom Baker", "probability": 1.0}]},
        "sets": {"issues": {"selected": ["delivery"],
                            "probabilities": {"delivery": 0.9, "login": 0.1}}},
        "thresholds": {"issues": 0.3, "pii": 0.75},
    }


# ------------------------------------------------------------------ formulas


def test_choice_confidence_is_the_top_two_margin():
    assert choice_confidence([0.75, 0.25]) == pytest.approx(0.5)
    assert choice_confidence([0.25, 0.75, 0.0]) == pytest.approx(0.5)
    assert choice_confidence([1.0]) == 1.0


def test_choice_confidence_clamps_to_the_unit_interval():
    assert choice_confidence([1.5, 0.25]) == 1.0
    assert choice_confidence([0.25, 0.75, 1.25]) == pytest.approx(0.5)


def test_score_confidence_is_one_minus_the_normalized_variance():
    # Point mass on one level: zero variance, confidence 1.
    assert score_confidence([0.0, 1.0, 0.0]) == 1.0
    assert score_confidence([1.0]) == 1.0
    # Uniform over n levels has exactly the uniform variance: confidence 0.
    assert score_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=1e-12)
    # Half mass on each of two adjacent levels: variance 0.25 of the
    # uniform 2/3 for n=3, so confidence is 1 - 0.375.
    assert score_confidence([0.5, 0.5, 0.0]) == pytest.approx(1 - 0.375)


def test_expected_level_is_the_zero_based_expectation():
    assert expected_level([0.125, 0.25, 0.625]) == pytest.approx(1.5)
    assert expected_level([1.0, 0.0]) == pytest.approx(0.0)


# ------------------------------------------------------------ answer ordering


def test_set_questions_contribute_label_views_in_criteria_order():
    assert expected_answer_order(FIVE_KIND_QUESTIONS) == [
        "billing", "area", "severity",
        "issues.delivery", "issues.login", "pii",
    ]


def test_plain_questions_contribute_their_own_id():
    assert expected_answer_order({"a": {"type": "noul", "instructions": "x"}}) == ["a"]


# ---------------------------------------------------------- finish_response


def test_finish_response_passes_a_complete_response_through():
    response = five_kind_response()
    assert finish_response(FIVE_KIND_QUESTIONS, response) is response


def test_finish_response_matches_span_and_set_key_sets():
    questions = {"pii": FIVE_KIND_QUESTIONS["pii"], "issues": FIVE_KIND_QUESTIONS["issues"]}
    response = {
        "model": "m", "answers": {"pii": {"type": "noul", "noul": 0.5},
                                  "issues.delivery": {"type": "noul", "noul": 0.9},
                                  "issues.login": {"type": "noul", "noul": 0.1}},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "spans": {"pii": []}, "sets": {"issues": {"selected": [], "probabilities": {}}},
        "thresholds": {"pii": 0.5, "issues": 0.3},
    }
    assert finish_response(questions, response) is response


@pytest.mark.parametrize(
    "mutate",
    [
        # Envelope violations.
        lambda r: r.pop("model"),
        lambda r: r.update(extra=1),
        # Hybrid-only field on the 0.3B member.
        lambda r: r.update(span_heads={"pii": "router"}),
        # Answer-order violations.
        lambda r: r["answers"].update({"other": {"type": "noul", "noul": 0.5}}),
        lambda r: r["answers"].pop("pii"),
        lambda r: r["answers"].update({"billing": r["answers"].pop("billing")}),  # moved last
        # Extension key-set violations.
        lambda r: r["spans"].pop("pii"),
        lambda r: r["sets"].pop("issues"),
        lambda r: r["thresholds"].pop("pii"),
        # Usage violations.
        lambda r: r.update(usage={"input_tokens": 5}),
        lambda r: r.update(usage={"input_tokens": -1, "output_tokens": 0}),
        lambda r: r.update(usage={"input_tokens": 10, "output_tokens": 3}),
        # Answer-shape violations.
        lambda r: r["answers"]["billing"].update(probabilities={"yes": 0.75}),
    ],
)
def test_finish_response_rejects_contract_violations(mutate):
    response = five_kind_response()
    mutate(response)
    with pytest.raises(RuntimeError):
        finish_response(FIVE_KIND_QUESTIONS, response)


def test_extension_fields_without_their_questions_are_rejected():
    response = {
        "model": "m",
        "answers": {"a": {"type": "noul", "noul": 0.5}},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "spans": {"a": []},
    }
    with pytest.raises(RuntimeError, match="spans"):
        finish_response({"a": {"type": "noul", "instructions": "x"}}, response)


def test_a_non_object_response_is_rejected():
    with pytest.raises(RuntimeError, match="non-object"):
        finish_response({"a": {"type": "noul", "instructions": "x"}}, [1, 2])
