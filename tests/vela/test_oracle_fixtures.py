"""Consume the frozen 0.3B CPU FP32 oracles without weights or torch.

The Unicode case was recorded through compare_reference.py from the pinned
d6f03aa9 package, with torch2.11.0+cpu, Python3.12.13, and the recipe dependency
pins. Generation and the distinction from native/device evidence are documented
in recipe/vela/reference.md. The original four oracle files are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.vela.postprocess import finish_response

ROOT = Path(__file__).resolve().parents[2]
CASES = ["example-request", "example-request-cjk", "example-request-set-threshold",
         "example-request-score", "example-request-unicode"]


def load_oracle(case):
    return json.loads((ROOT / "tests/vela/fixtures" / f"oracle-{case}.json")
                      .read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES)
def test_oracle_request_is_the_maintained_recipe_example(case):
    recorded = load_oracle(case)
    request = json.loads((ROOT / "recipe/vela" / f"{case}.json").read_text(encoding="utf-8"))
    assert recorded["request"] == request
    assert list(recorded["request"]) == list(request)
    assert Path(recorded["case"]).stem == case


@pytest.mark.parametrize("case", CASES)
def test_reference_response_passes_the_production_envelope_gate(case):
    recorded = load_oracle(case)
    response = recorded["system_one"]
    assert response["model"] == recorded["request"]["model"]
    assert finish_response(recorded["request"]["questions"], response) is response


def test_unicode_oracle_has_positive_spans_after_non_bmp_text():
    recorded = load_oracle("example-request-unicode")
    response = recorded["system_one"]
    question = recorded["request"]["questions"]["pii"]
    field = recorded["request"]["state"][question["over"]]
    spans = response["spans"]["pii"]
    assert spans, "an empty oracle cannot test offset units"
    email = next(span for span in spans if span["label"] == "EMAIL_ADDRESS")
    assert (email["start"], email["end"], email["text"]) == (10, 30, "zhang.wei@example.cn")
    for span in spans:
        start, end = span["start"], span["end"]
        assert field[start:end] == span["text"]
        assert any(ord(char) > 0xffff for char in field[:start])
        assert len(field[:start].encode("utf-8")) != start
        assert len(field[:start].encode("utf-16-le")) // 2 != start
    assert response["thresholds"]["pii"] == question["threshold"] == 0.05
    assert "span_heads" not in response  # the 0.3B has one span head


def test_unicode_set_probabilities_remain_independent_and_keep_noul_views():
    recorded = load_oracle("example-request-unicode")
    response = recorded["system_one"]
    result = response["sets"]["topics"]
    probabilities = result["probabilities"]
    assert list(probabilities) == ["contact", "order"]
    assert sum(probabilities.values()) > 1.3  # cannot pass after cross-label softmax
    threshold = response["thresholds"]["topics"]
    assert threshold == recorded["request"]["questions"]["topics"]["threshold"] == 0.5
    assert result["selected"] == [label for label, p in probabilities.items() if p > threshold]
    for label, probability in probabilities.items():
        assert response["answers"][f"topics.{label}"] == {"type": "noul", "noul": probability}
    assert response["usage"] == {"input_tokens": 155, "output_tokens": 0}
