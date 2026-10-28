"""Offline checks for the opt-in turbo oracle/fixture comparison tool."""

import copy

import pytest


def response():
    return {
        "model": "d3-turbo",
        "answers": {
            "z": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.7, "wait": 0.3}},
            "a": {"type": "noul", "noul": 0.5000001},
        },
        "usage": {"input_tokens": 97, "output_tokens": 0},
    }


def test_response_comparison_checks_threshold_flips_within_numeric_tolerance():
    from .turbo_reference import compare_response

    expected = response()
    assert compare_response(expected, copy.deepcopy(expected)) == 0
    actual = copy.deepcopy(expected)
    actual["answers"]["a"]["noul"] = 0.4999999
    with pytest.raises(AssertionError, match="decision"):
        compare_response(expected, actual)


def test_response_comparison_keeps_usage_order_and_errors_exact():
    from .turbo_reference import compare_response

    expected = response()
    for change in ("usage", "order", "error"):
        actual = copy.deepcopy(expected)
        if change == "usage":
            actual["usage"]["input_tokens"] += 1
        elif change == "order":
            actual["answers"] = dict(reversed(list(actual["answers"].items())))
        else:
            actual["answers"]["z"] = {"type": "choice", "error": "invalid_question", "message": "invalid"}
        with pytest.raises(AssertionError):
            compare_response(expected, actual)


def test_response_comparison_rejects_nonfinite_or_large_drift():
    from .turbo_reference import compare_response

    for value in (float("nan"), float("inf"), 0.5001):
        actual = response()
        actual["answers"]["a"]["noul"] = value
        with pytest.raises(AssertionError):
            compare_response(response(), actual)


def test_snapshot_comparison_distinguishes_exact_preparation_from_probabilities():
    from .turbo_reference import compare_fixture

    expected = {"preparation": {"state_end": 63, "token_ids": [1, 2]},
                "probabilities": {"q": [0.25, 0.75]}, "response": response()}
    actual = copy.deepcopy(expected)
    actual["probabilities"]["q"][0] += 1e-7
    compare_fixture(expected, actual)
    actual["preparation"]["state_end"] = 64
    with pytest.raises(AssertionError, match="preparation"):
        compare_fixture(expected, actual)


def test_snapshot_comparison_rejects_changed_request_even_with_same_error():
    from .turbo_reference import compare_fixture

    expected = {"request": {"state_repeat": {"text": "token ", "count": 32769}},
                "preparation": {}, "probabilities": {}, "response": response()}
    actual = copy.deepcopy(expected)
    actual["request"]["state_repeat"]["text"] = "other "
    with pytest.raises(AssertionError, match="request"):
        compare_fixture(expected, actual)


def test_freeze_check_reads_shared_storage_without_numpy_or_accelerator(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from models.d3 import turbo
    from . import turbo_reference as reference
    from .test_turbo_fixture_storage import document
    from .turbo_fixture_storage import write_fixture

    expected = document()
    expected["metadata"].update({"package": "pinned", "revision": "a" * 40,
        "manifest_sha256": "b" * 64, "attention_mode": "prefix_bidir"})
    path = tmp_path / "state.json"
    write_fixture(path, expected, b"tensors")
    closed = []
    executor = SimpleNamespace(model=object(), warmup=lambda: None, close=lambda: closed.append(True))
    monkeypatch.setattr(turbo, "D3TurboExecutor", lambda *args, **kwargs: executor)
    monkeypatch.setattr(reference, "metadata", lambda model: copy.deepcopy(expected["metadata"]))
    monkeypatch.setattr(reference, "fixture_requests", lambda: {"text": {}})
    monkeypatch.setattr(reference, "snapshot", lambda *args: copy.deepcopy(expected["cases"]["text"]))
    reference.freeze(SimpleNamespace(model="unused", device="cpu", verify="full", source_revision="c" * 40,
                                     check=str(path), output=None))
    assert closed == [True]
