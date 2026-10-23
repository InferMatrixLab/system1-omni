"""Wire contract for the d3 reference worker.

This module deliberately has no torch dependency and validates only the
request envelope: the five-field System One request, unknown-field rejection,
and the slice-scoped media refusal. Per-question validation belongs to the
checkpoint's pinned runtime, whose aggregated rejections ``preprocess`` maps
to HTTP 422.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MODEL_NAME = "d3"
# Slice-scoped transport cap. Upstream d3_server.py sets no total-body limit
# (its budgets are per image and per video); stage 2 re-aligns this when the
# media paths open.
MAX_BODY = 8 * 1024 * 1024
REQUEST_FIELDS = {"model", "state", "questions", "images", "videos"}


class RequestError(ValueError):
    """A request cannot be evaluated under the reference contract."""

    def __init__(self, message: str, status: int = 422, body: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        # A full JSON error body ({"invalid_questions": {...}}); None sends
        # the default {"detail": message} shape.
        self.body = body


class MalformedJSON(RequestError):
    """The body is not a usable UTF-8 JSON object."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 400)


@dataclass(frozen=True)
class Request:
    """One admitted request: the raw upstream fields, in request order.

    ``model`` is accepted and ignored (upstream semantics). ``state`` passes
    through unchanged; the pinned runtime renders it. ``questions`` keeps the
    request order for response reconstruction.
    """

    model: Any
    state: Any
    questions: tuple[tuple[str, Any], ...]


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MalformedJSON(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise MalformedJSON(f"non-finite JSON number {value} is not supported")


def parse_body(raw: bytes) -> dict[str, Any]:
    """Decode a bounded JSON body and reject ambiguous JSON values."""

    if len(raw) > MAX_BODY:
        raise RequestError("request body exceeds 8 MiB", 413)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=_nonfinite,
        )
        # Reject lone surrogates and values that cannot be emitted as UTF-8.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except MalformedJSON:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise MalformedJSON("request body must contain valid JSON and UTF-8 text") from exc
    if not isinstance(value, dict):
        raise MalformedJSON("request must be a JSON object")
    return value


def _media(value: dict[str, Any], field: str) -> None:
    """Admit an absent, null or empty media list; refuse anything else."""

    media = value.get(field)
    if media is None:
        return
    if not isinstance(media, list):
        raise RequestError(f"{field} must be a list of base64 data URLs")
    if media:
        raise RequestError(
            f"this reference worker serves text state only; non-empty {field}"
            " are not supported yet (stage 2)"
        )


def parse_request(value: dict[str, Any]) -> Request:
    """Validate the system1-omni wire request envelope.

    Unknown fields are rejected with upstream's message; ``model`` is not
    checked (upstream ignores it); per-question content validation is left
    to the pinned runtime.
    """

    unknown = set(value) - REQUEST_FIELDS
    if unknown:
        raise RequestError(f"Unknown request fields: {sorted(unknown)}")

    for field in ("images", "videos"):
        _media(value, field)

    questions = value.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a nonempty mapping of question IDs to questions")
    if any(not isinstance(name, str) or not name for name in questions):
        raise RequestError("question IDs must be nonempty strings")

    return Request(value.get("model"), value.get("state"), tuple(questions.items()))
