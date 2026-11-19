"""Opt-in contract checks against the hash-pinned 0.3B runtime.

These exercise upstream validation, response assembly with controlled model
outputs, and the actual token-budget function. They do not run a model forward
or establish learned-head/native parity. Ordinary CPU suites skip them without
VELA_CONTRACT_PACKAGE; no package is downloaded by a test.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def pinned_runtime():
    package = os.environ.get("VELA_CONTRACT_PACKAGE")
    if not package:
        pytest.skip("set VELA_CONTRACT_PACKAGE to the pinned 0.3B package")
    from models.vela.engine import (
        put_first_on_sys_path,
        verify_no_foreign_runtime,
        verify_package,
    )

    root = Path(package).resolve()
    verify_package(root)
    verify_no_foreign_runtime(root)
    previous_path = sys.path[:]
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        put_first_on_sys_path(root)
        runtime = importlib.import_module("vela2_inference")
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    return runtime, config


@pytest.mark.parametrize(
    "state,questions,location",
    [
        (None, {}, ["state"]),
        ("valid", {}, ["questions"]),
        ("valid", {"bad": {"type": "span", "instructions": "Find.",
                           "criteria": ["PERSON"]}}, ["questions", "bad", "criteria"]),
        ("valid", {"bad": {"type": "set", "instructions": "Select.",
                           "criteria": []}}, ["questions", "bad", "criteria"]),
        ("valid", {"bad": {"type": "set", "instructions": "Select.",
                           "criteria": {"x": "x"}, "threshold": True}},
         ["questions", "bad", "threshold"]),
        ("valid", {"bad": {"type": "span", "instructions": "Find.",
                           "criteria": [], "over": "missing"}},
         ["questions", "bad", "over"]),
        ("valid", {"first": {"type": "set", "instructions": "Select.", "criteria": []},
                   "second": {"type": "span", "instructions": "Find.", "criteria": []}},
         ["questions", "first", "criteria"]),
        ("valid", {"labels": {"type": "set", "instructions": "Select.",
                              "criteria": {"a": "a label"}},
                   "labels.a": {"type": "noul", "instructions": "Yes?"}},
         ["questions", "labels.a"]),
    ],
)
def test_upstream_validation_and_error_precedence(pinned_runtime, state, questions, location):
    runtime, _ = pinned_runtime
    with pytest.raises(runtime.SystemOneError) as error:
        runtime.system_one_to_predict(state, questions)
    assert error.value.detail()["loc"] == ["body", *location]


def test_upstream_span_offsets_are_codepoints_into_the_named_field(pinned_runtime):
    runtime, _ = pinned_runtime
    state = {"prompt": "前文🚀", "request": "😀 张伟 来自上海"}
    questions = {"pii": {"type": "span", "instructions": "Find names.",
                         "criteria": {"PERSON": "a person"}, "over": "request"}}
    # Two aliased fields share a user part. This expected prefix is independent
    # of system_one_state's calculated field map.
    prefix = "prompt:\n前文🚀\n\nrequest:\n"
    absolute_start = len(prefix) + 2

    def predict(request, span_limits):
        assert request["parts"] == [{"type": "user", "text": prefix + state["request"]}]
        assert span_limits == {"pii": (len(prefix), len(prefix) + len(state["request"]))}
        return ({"answers": [{"id": "pii", "threshold": 0.5, "spans": [
            {"label": "PERSON", "start": absolute_start, "end": absolute_start + 2,
             "text": "张伟", "probability": 0.9},
        ]}]}, {"pii": 0.95}, 37)

    response = runtime.system_one_response(SimpleNamespace(_predict=predict), state, questions)
    assert response["spans"]["pii"] == [
        {"label": "PERSON", "start": 2, "end": 4, "text": "张伟", "probability": 0.9},
    ]
    assert state["request"][2:4] == "张伟"
    assert len(state["request"][:2].encode("utf-8")) != 2
    assert len(state["request"][:2].encode("utf-16-le")) // 2 != 2
    assert response["answers"]["pii"] == {"type": "noul", "noul": 0.95}
    assert response["thresholds"] == {"pii": 0.5}
    assert response["usage"] == {"input_tokens": 37, "output_tokens": 0}


def assemble(runtime, config, segments, *, protected=(), shrink="user", instructions=1):
    # Valid two-option question with controlled token IDs: exact sequence
    # geometry, independent of tokenizer/model arithmetic.
    token = config["bos_token_id"]
    question = {"qid": "q", "type": "choice", "over": "user",
                "qtoks": [token] * instructions,
                "opts": [("a", [token]), ("b", [token])],
                "abstain": False, "len": 5 + instructions}
    return runtime.assemble(config["marker_ids"], token, config["eos_token_id"],
                            [question], [], segments, config["max_length"],
                            set(protected), shrink)


def test_upstream_exact_window_boundary_truncates_state(pinned_runtime):
    runtime, config = pinned_runtime
    limit = config["max_length"]
    assert limit == 8192
    # BOS/EOS + SEP + one segment marker + six question tokens.
    capacity = limit - 10
    exact = assemble(runtime, config, [("user", [7] * capacity)])
    cut = assemble(runtime, config, [("user", [7] * (capacity + 1))])
    assert len(exact["ids"]) == len(cut["ids"]) == limit
    assert exact["trunc"] == {"shrink_cut": 0, "protected_cut": 0}
    assert cut["trunc"] == {"shrink_cut": 1, "protected_cut": 0}
    assert cut["seg_lens"] == {"user": capacity}


def test_upstream_unprotected_floor_precedes_protected_cut(pinned_runtime):
    runtime, config = pinned_runtime
    result = assemble(runtime, config,
                      [("context", [7] * 65), ("user", [8] * 8181)],
                      protected=("user",), shrink="context")
    assert len(result["ids"]) == 8192
    assert result["seg_lens"] == {"context": 1, "user": 8180}
    assert result["trunc"] == {"shrink_cut": 64, "protected_cut": 1}


def test_upstream_schema_alone_overflow_is_distinct(pinned_runtime):
    runtime, config = pinned_runtime
    with pytest.raises(ValueError, match="schema alone does not fit"):
        assemble(runtime, config, [("user", [7])], instructions=8192)
