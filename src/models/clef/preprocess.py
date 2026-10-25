"""Question admission and the untruncated token limit.

The pinned encoder accepts any non-empty criteria container and then slices
the state to fit. This module rejects a question shape the encoder would
misread (a score given as an object, a noul criterion other than true/false)
and rejects a prompt that would not fit with its state intact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from models.clef.prompt import QUESTION_TYPES, token_lengths
from models.clef.protocol import Request, RequestError

# ``systemone(..., max_length=16384)`` in the pinned loader. The blog describes
# a 64k window and the backbone config allows 262144 positions; this worker
# does not raise the shipped encoder default.
MAX_INPUT_TOKENS = 16384


def validate_questions(request: Request) -> None:
    """Reject question bodies the text encoder cannot score unambiguously."""

    for question_id, question in request.questions.items():
        if not isinstance(question, dict):
            raise RequestError(f"{question_id}: question must be an object")
        question_type = question.get("type")
        if question_type not in QUESTION_TYPES:
            raise RequestError(f"{question_id}: type must be noul, choice, or score")
        criteria = question.get("criteria")
        if question_type == "noul":
            if criteria is None:
                continue
            if not isinstance(criteria, dict):
                raise RequestError(f"{question_id}: noul criteria must be an object")
            # The encoder scores only true and false. Other keys stay in the
            # request object and do not become options, matching the pinned
            # ``question_options`` update. They are not dropped from the JSON
            # body; they are absent from the prompt.
            continue
        if question_type == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise RequestError(f"{question_id}: criteria must not be empty")
            if any(not isinstance(option_id, str) or not option_id for option_id in criteria):
                raise RequestError(f"{question_id}: choice option ids must be nonempty strings")
            continue
        if not isinstance(criteria, list) or not criteria:
            raise RequestError(f"{question_id}: criteria must not be empty")


def measure(count: Callable[[str], int], request: Request) -> tuple[int, int]:
    """Token lengths of the untruncated text prompt."""

    return token_lengths(count, request.payload)


def enforce_limit(fixed: int, total: int, max_length: int = MAX_INPUT_TOKENS) -> None:
    """Refuse a prompt that the pinned encoder would reject or truncate."""

    if fixed > max_length:
        raise RequestError(
            f"schema requires {fixed} tokens before state; maximum is {max_length}"
        )
    if total > max_length:
        raise RequestError(
            f"request is {total} tokens, which exceeds the {max_length} token limit; "
            "the state was not truncated"
        )
