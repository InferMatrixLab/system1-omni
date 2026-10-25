from models.clef.postprocess import systemone_answer


def test_choice_keeps_request_order_and_breaks_ties_that_way():
    question = {
        "type": "choice",
        "criteria": {"technical": "Bugs", "billing": "Payments"},
    }
    tied = systemone_answer(question, {"technical": 0.5, "billing": 0.5})
    assert list(tied["probabilities"]) == ["technical", "billing"]
    assert tied["choice"] == "technical"
    assert tied["confidence"] == 0.5

    later = systemone_answer(question, {"technical": 0.25, "billing": 0.75})
    assert later["choice"] == "billing"
    assert later["probabilities"] == {"technical": 0.25, "billing": 0.75}


def test_choice_is_selected_before_four_decimal_rounding():
    question = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
    answer = systemone_answer(question, {"a": 0.50004, "b": 0.49996})
    assert answer["choice"] == "a"
    assert answer["probabilities"] == {"a": 0.5, "b": 0.5}
    assert answer["confidence"] == 0.5


def test_noul_reports_only_rounded_true_probability():
    assert systemone_answer({"type": "noul"}, {"true": 1 / 3, "false": 2 / 3}) == {
        "type": "noul",
        "noul": round(1 / 3, 4),
    }


def test_score_is_the_rounded_expected_index():
    question = {"type": "score", "criteria": ["low", {"kind": "high"}]}
    answer = systemone_answer(question, {"0": 0.25, "1": 0.75})
    assert answer["score"] == round(0.75, 4)
    assert answer["confidence"] == 0.75
    assert answer["legend"] == {"0": "low", "1": {"kind": "high"}}
    assert list(answer["probabilities"]) == ["0", "1"]
