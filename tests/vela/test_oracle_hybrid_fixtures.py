"""Ties the frozen hybrid oracle fixtures to the recipe examples they came from.

The 0.8B hybrid fixtures recorded by ``recipe/vela/compare_reference.py``
under the ``oracle-hybrid-<member>-`` prefix keep the recipe's example requests
(with the ``model`` field rewritten to the hybrid's served name) and their
recorded ``system_one`` responses, which must pass the production
``finish_response`` gate on the hybrid setting; the 0.3B oracles have no
equivalent consumer yet. No weights, no torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.vela.hybrid import EXPECTED_MODEL_NAME
from models.vela.postprocess import finish_response

TESTS = Path(__file__).parent
RECIPE = TESTS.parent.parent / "recipe" / "vela"

CASES = [
    ("oracle-hybrid-Vela-2.0-0.8B-example-request", "example-request.json"),
    ("oracle-hybrid-Vela-2.0-0.8B-example-request-cjk", "example-request-cjk.json"),
    ("oracle-hybrid-Vela-2.0-0.8B-example-request-set-threshold", "example-request-set-threshold.json"),
    ("oracle-hybrid-Vela-2.0-0.8B-example-request-score", "example-request-score.json"),
]


def oracle(case: str) -> dict:
    return json.loads((TESTS / "fixtures" / f"{case}.json").read_text())


def recipe_body(example: str) -> dict:
    body = json.loads((RECIPE / example).read_text())
    # The tool's load_cases rewrite for file cases: the examples are
    # member-agnostic apart from the model field.
    return {**body, "model": EXPECTED_MODEL_NAME}


@pytest.mark.parametrize("case,example", CASES)
def test_the_oracle_request_is_the_recipe_example(case: str, example: str):
    recorded = oracle(case)
    served = recipe_body(example)
    assert recorded["request"] == served
    # Same insertion order, so byte-for-byte serialization holds too.
    assert list(recorded["request"]) == list(served)


@pytest.mark.parametrize("case,example", CASES)
def test_the_recorded_response_passes_the_worker_gate(case: str, example: str):
    recorded = oracle(case)
    response = recorded["system_one"]
    assert response["model"] == EXPECTED_MODEL_NAME
    assert finish_response(recorded["request"]["questions"], response, hybrid=True) is response
    if any(q.get("type") == "span" for q in recorded["request"]["questions"].values()):
        assert set(response["span_heads"]) == {
            qid for qid, q in recorded["request"]["questions"].items() if q.get("type") == "span"
        }
    # The frozen case label still names the recipe file it belongs to.
    assert recorded["case"] == example.removesuffix(".json")
