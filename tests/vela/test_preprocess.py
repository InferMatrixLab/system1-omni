"""Tests for the pinned-runtime error mapping.

Uses a stub with the pinned ``SystemOneError`` shape (``loc``, ``msg``,
``type``, ``detail()``): no package, no torch. What is under test is the
worker's mapping — 422 for every whole-request rejection, 413 only for
``too_long``, with the detail entry passed through verbatim.
"""

from __future__ import annotations

import pytest

from models.vela.preprocess import map_runtime_error, to_request_error
from models.vela.protocol import RequestError


class StubSystemOneError(ValueError):
    """The pinned runtime's error shape, mirrored for these tests."""

    def __init__(self, loc, msg, type_="value_error"):
        super().__init__(f"{'.'.join(str(x) for x in loc)}: {msg}")
        self.loc, self.msg, self.type = list(loc), msg, type_

    def detail(self):
        return {"loc": ["body"] + self.loc, "msg": self.msg, "type": self.type}


def test_a_system_one_error_maps_to_a_422_with_its_own_detail():
    exc = StubSystemOneError(["questions", "billing", "criteria"],
                             "choice criteria must be an object with 2 to 255 entries")
    error = to_request_error(exc)

    assert isinstance(error, RequestError)
    assert error.status == 422
    assert error.body == {
        "detail": [
            {
                "loc": ["body", "questions", "billing", "criteria"],
                "msg": "choice criteria must be an object with 2 to 255 entries",
                "type": "value_error",
            }
        ]
    }
    assert str(error) == "questions.billing.criteria: choice criteria must be an object with 2 to 255 entries"


@pytest.mark.parametrize(
    ("type_", "expected"),
    [("value_error", 422), ("missing", 422), ("extra_forbidden", 422),
     ("literal_error", 422), ("value_error.model", 422), ("too_long", 413)],
)
def test_only_too_long_maps_to_413(type_, expected):
    error = to_request_error(StubSystemOneError(["questions"], "boom", type_))
    assert error.status == expected
    assert error.body["detail"][0]["type"] == type_


def test_a_runtime_value_error_is_a_client_error():
    error = map_runtime_error(ValueError("state must be a non-empty string, JSON object or JSON array"))
    assert error.status == 422
    assert error.body is None
    assert "request rejected by the pinned runtime" in str(error)
    assert "non-empty string" in str(error)
