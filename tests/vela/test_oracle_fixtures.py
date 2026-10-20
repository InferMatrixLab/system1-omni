"""Ties the frozen oracle fixtures to the recipe examples they came from.

The three oracle fixtures under ``tests/vela/fixtures`` are the frozen
comparison baseline for the later native encoder slices: each one's
``request`` is the recipe's example request, and its recorded
``system_one`` response passes the worker's own response gate. This
keeps the tie mechanical instead of by-convention. No weights, no
torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.vela.postprocess import finish_response
from models.vela.protocol import MODEL_NAME

TESTS = Path(__file__).parent
RECIPE = TESTS.parent.parent / "recipe" / "vela"

CASES = [
    ("oracle-example-request", "example-request.json"),
    ("oracle-example-request-cjk", "example-request-cjk.json"),
    ("oracle-example-request-set-threshold", "example-request-set-threshold.json"),
]


def oracle(case: str) -> dict:
    return json.loads((TESTS / "fixtures" / f"{case}.json").read_text())


@pytest.mark.parametrize("case,example", CASES)
def test_the_oracle_request_is_the_recipe_example(case: str, example: str):
    recorded = oracle(case)
    served = json.loads((RECIPE / example).read_text())
    assert recorded["request"] == served
    # Same insertion order, so byte-for-byte serialization holds too.
    assert list(recorded["request"]) == list(served)
    assert recorded["request"]["model"] == MODEL_NAME


@pytest.mark.parametrize("case,example", CASES)
def test_the_recorded_response_passes_the_worker_gate(case: str, example: str):
    recorded = oracle(case)
    response = recorded["system_one"]
    assert finish_response(recorded["request"]["questions"], response) is response
    # The frozen case label still names the recipe file it belongs to.
    assert recorded["case"] == example.removesuffix(".json")
