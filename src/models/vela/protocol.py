"""Wire contract for the Vela 2.0 reference workers.

This module deliberately has no torch dependency and validates only the
request envelope's outer layer: JSON body admission under the shared
Python-worker transport discipline, and the field-set and model-name
checks of the upstream ``system_one_request`` preamble — with upstream's
exact ``{"detail": [{"loc", "msg", "type"}]}`` triples so clients cannot
tell this layer from the pinned runtime.

Everything deeper is the pinned runtime's own validation, verbatim:
state rendering to typed parts, the per-question walk (type, criteria
limits, ``over`` resolution, ``threshold``), the set-label Noul-view
collision check, and the 413 questions-overflow. Their error shapes are
pinned by the author-side fixtures and the reference comparison, not
reimplemented here.
"""

from __future__ import annotations

import json
from typing import Any

# The served name of the 0.3B member (upstream DEFAULT_MODEL_NAME); the
# hybrid members serve their config.json "model_name" instead.
MODEL_NAME = "vllm-sr/Vela-2.0-0.3B"
# Slice-scoped transport cap. Upstream vela2_serve.py sets no total-body
# limit (its only size boundary is the member's token input window); a
# text request bounded by that window is orders of magnitude smaller, so
# this cap never binds a valid request. A deliberate difference recorded
# with the integration.
MAX_BODY = 8 * 1024 * 1024
# The envelope fields, in upstream's missing-field check order.
REQUEST_FIELDS = ("state", "model", "questions")


class RequestError(ValueError):
    """A request cannot be evaluated under the reference contract."""

    def __init__(self, message: str, status: int = 422, body: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        # A full JSON error body ({"detail": [...]}); None sends the
        # default {"detail": message} shape.
        self.body = body


class MalformedJSON(RequestError):
    """The body is not a usable UTF-8 JSON object."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 400)


def envelope_error(loc: list[str], msg: str, type_: str = "value_error") -> RequestError:
    """One upstream SystemOneError as a request error.

    ``loc`` is the path below ``["body"]``; the status is 422 for every
    envelope violation (the only 413, ``too_long``, is raised by the
    pinned runtime during execution).
    """

    return RequestError(
        f"{'.'.join(str(part) for part in loc)}: {msg}",
        422,
        body={"detail": [{"loc": ["body", *loc], "msg": msg, "type": type_}]},
    )


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
        raise RequestError("request exceeds the 8 MiB body limit", 413)
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


def admit_request(value: dict[str, Any], served: set[str]) -> dict[str, Any]:
    """Admit the envelope exactly as upstream's request preamble does.

    Checks, in upstream's order: the three required fields are present;
    no extra fields; ``model`` is non-blank and equals the served name.
    Returns the body unchanged — the pinned runtime re-runs this preamble
    and then owns state rendering and question validation, so this layer
    never rejects anything the runtime would not reject the same way.
    """

    for key in REQUEST_FIELDS:
        if key not in value:
            raise envelope_error([key], "Field required", "missing")
    extra = sorted(set(value) - set(REQUEST_FIELDS))
    if extra:
        raise envelope_error([extra[0]], "Extra inputs are not permitted", "extra_forbidden")
    model = value["model"]
    if not isinstance(model, str) or not model.strip():
        raise envelope_error(["model"], "model must not be empty or whitespace")
    if model not in served:
        raise envelope_error(["model"], "Model is not served by this runtime", "value_error.model")
    return value
