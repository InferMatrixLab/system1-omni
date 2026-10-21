import pytest

from models.d3.preprocess import reject_pre_execution_errors
from models.d3.protocol import RequestError


class FakePrepared:
    def __init__(self, errors):
        self.keys = list(errors)
        self.errors = dict(errors)


def invalid(message="choice criteria must map 1 to 255 option keys to descriptions"):
    return {"type": "choice", "error": "invalid_question", "message": message}


def over(limit=8192, tokens=9000):
    return {
        "type": "choice",
        "error": "max_length_exceeded",
        "message": (
            f"the question prompt has {tokens} tokens, over the maximum context"
            f" length of {limit} tokens; nothing was truncated"
        ),
    }


def test_clean_prepared_admits_the_request():
    assert reject_pre_execution_errors(FakePrepared({})) is None


def test_invalid_questions_reject_with_an_ordered_map_body():
    prepared = FakePrepared({"b": invalid("b is bad"), "a": invalid("a is bad")})
    with pytest.raises(RequestError) as excinfo:
        reject_pre_execution_errors(prepared)

    assert excinfo.value.status == 422
    assert excinfo.value.body == {
        "invalid_questions": {"b": "b is bad", "a": "a is bad"}
    }
    assert list(excinfo.value.body["invalid_questions"]) == ["b", "a"]


def test_max_length_rejects_with_the_plain_message_and_wins():
    prepared = FakePrepared({"long": over(), "bad": invalid()})
    with pytest.raises(RequestError) as excinfo:
        reject_pre_execution_errors(prepared)

    assert excinfo.value.status == 422
    assert excinfo.value.body is None
    assert str(excinfo.value) == prepared.errors["long"]["message"]
    assert "9000 tokens" in str(excinfo.value)
    assert "nothing was truncated" in str(excinfo.value)


def test_runtime_without_an_error_map_is_a_worker_bug():
    class Bare:
        errors = None

    with pytest.raises(RuntimeError, match="error map"):
        reject_pre_execution_errors(Bare())
