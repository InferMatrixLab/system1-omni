"""Ties the frozen hybrid oracle fixtures to the recipe examples they came from.

The decoder-member fixtures recorded by ``recipe/vela/compare_reference.py``
under the ``oracle-hybrid-<member>-`` prefix keep the recipe's example requests
(with the ``model`` field rewritten to the member's served name) and their
recorded ``system_one`` responses, which must pass the production
``finish_response`` gate on the hybrid setting; the 0.3B oracles have no
equivalent consumer yet. Every member of the registry must have its four
oracles frozen here. No weights, no torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.vela.hybrid import MEMBERS
from models.vela.postprocess import finish_response

TESTS = Path(__file__).parent
RECIPE = TESTS.parent.parent / "recipe" / "vela"

EXAMPLES = [
    "example-request",
    "example-request-cjk",
    "example-request-set-threshold",
    "example-request-score",
]

# (fixture stem, example file, served model name) for every pinned
# member's frozen oracles.
CASES = [
    (f"oracle-hybrid-{member.label}-{case}", f"{case}.json", member.model_name)
    for member in MEMBERS.values()
    for case in EXAMPLES
]


def oracle(case: str) -> dict:
    return json.loads((TESTS / "fixtures" / f"{case}.json").read_text())


def recipe_body(example: str, model_name: str) -> dict:
    body = json.loads((RECIPE / example).read_text())
    # The tool's load_cases rewrite for file cases: the examples are
    # member-agnostic apart from the model field.
    return {**body, "model": model_name}


@pytest.mark.parametrize("case,example,served", CASES)
def test_the_oracle_request_is_the_recipe_example(case: str, example: str, served: str):
    recorded = oracle(case)
    expected = recipe_body(example, served)
    assert recorded["request"] == expected
    # Same insertion order, so byte-for-byte serialization holds too.
    assert list(recorded["request"]) == list(expected)


@pytest.mark.parametrize("case,example,served", CASES)
def test_the_recorded_response_passes_the_worker_gate(case: str, example: str, served: str):
    recorded = oracle(case)
    response = recorded["system_one"]
    assert response["model"] == served
    assert finish_response(recorded["request"]["questions"], response, hybrid=True) is response
    if any(q.get("type") == "span" for q in recorded["request"]["questions"].values()):
        assert set(response["span_heads"]) == {
            qid for qid, q in recorded["request"]["questions"].items() if q.get("type") == "span"
        }
    # The frozen case label still names the recipe file it belongs to.
    assert recorded["case"] == example.removesuffix(".json")
