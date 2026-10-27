"""Mapping of the pinned runtime's request errors to HTTP request errors.

The Vela 2.0 runtime validates the whole request itself and raises its
``SystemOneError`` — a whole-request 422 with a FastAPI-style
``{"detail": [{"loc", "msg", "type"}]}`` entry, or a 413 when the
questions plus schema alone exceed the context window (``too_long``).
This module owns that mapping for the reference workers; there are no
per-question errors anywhere on this path.
"""

from __future__ import annotations

from typing import Any

from .protocol import RequestError

# The only SystemOneError type that is not a 422.
TOO_LONG = "too_long"


def to_request_error(exc: Any) -> RequestError:
    """One pinned-runtime SystemOneError as a request error.

    Accepts anything with the runtime's ``detail()``/``type`` shape, so
    the CPU tests can drive it with a stub instead of the package. The
    body is the runtime's own detail entry, verbatim.
    """

    status = 413 if getattr(exc, "type", None) == TOO_LONG else 422
    return RequestError(str(exc), status, body={"detail": [exc.detail()]})


def map_runtime_error(exc: ValueError) -> RequestError:
    """A non-SystemOneError rejection from the pinned runtime.

    The runtime is the validation authority for everything the wire
    layer cannot foresee; answering its ValueError rejections as client
    errors is this worker's policy — the upstream server would answer
    them with 500.
    """

    return RequestError(f"request rejected by the pinned runtime: {exc}", 422)
