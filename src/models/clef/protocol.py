"""Wire contract for the Clef reference worker.

Validation here is the HTTP envelope only: a JSON object, the SystemOne
fields, and rejection of image or video payloads. Question shape and the
token limit live in ``preprocess`` so a prompt change stays next to the
encoder pieces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Text-only cap. The pinned ``systemone`` function has no HTTP server and no
# body limit; 8 MiB is enough for a request that still has to be refused at
# the 16384-token admission limit, and it refuses larger bodies explicitly.
MAX_BODY = 8 * 1024 * 1024
REQUEST_FIELDS = {"model", "state", "questions", "images", "videos", "thinking"}
MEDIA_UNSUPPORTED = (
    "{field} are not served by this integration; image and video inputs remain unsupported"
)


class RequestError(ValueError):
    """A request cannot be evaluated under the reference contract."""

    def __init__(self, message: str, status: int = 422, body: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class MalformedJSON(RequestError):
    """The body is not a usable UTF-8 JSON object."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 400)


@dataclass(frozen=True)
class Request:
    """One admitted envelope. ``payload`` keeps the parsed object and its key order."""

    payload: dict[str, Any]

    @property
    def questions(self) -> dict[str, Any]:
        return self.payload["questions"]


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
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except MalformedJSON:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise MalformedJSON("request body must contain valid JSON and UTF-8 text") from exc
    if not isinstance(value, dict):
        raise MalformedJSON("request must be a JSON object")
    return value


def _reject_media(value: dict[str, Any], field: str) -> None:
    if field not in value or value[field] is None or value[field] == []:
        return
    if not isinstance(value[field], list):
        raise RequestError(f"{field} must be a list")
    raise RequestError(MEDIA_UNSUPPORTED.format(field=field))


def _accept_thinking(value: dict[str, Any]) -> None:
    """Allow the stock Public231 ``thinking`` object only when it is off.

    The pinned encoder always writes the empty think block. ``enabled: true``
    is rejected so a thinking request is not scored on that prompt. Other keys
    on the object are not interpreted. The key is removed before encoding, so
    the text matches a request that never had the field.
    """

    if "thinking" not in value:
        return
    thinking = value["thinking"]
    if not isinstance(thinking, dict) or thinking.get("enabled") is not False:
        raise RequestError(
            "thinking.enabled must be false; this worker only serves the non-thinking prompt"
        )


def parse_request(value: dict[str, Any]) -> Request:
    """Validate the systemone envelope and refuse media inputs."""

    unknown = sorted(set(value) - REQUEST_FIELDS)
    if unknown:
        raise RequestError(f"Unknown request fields: {unknown}")
    _reject_media(value, "images")
    _reject_media(value, "videos")
    _accept_thinking(value)
    if not isinstance(value.get("model"), str) or "state" not in value:
        raise RequestError("model and state are required")
    questions = value.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("at least one question is required")
    if any(not isinstance(name, str) or not name for name in questions):
        raise RequestError("question IDs must be nonempty strings")
    payload = {key: item for key, item in value.items() if key != "thinking"}
    return Request(payload)
