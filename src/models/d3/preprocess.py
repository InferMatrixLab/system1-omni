"""Admission of the pinned runtime's prepared-request errors.

``D3.prepare`` validates every question and reports failures per question in
``prepared.errors`` instead of raising. The upstream server turns that map
into HTTP 422 before any model execution; this module owns the same mapping
for the reference worker: a ``max_length_exceeded`` question rejects the whole
request with its plain message (nothing is ever truncated), and invalid
questions reject it with an ``invalid_questions`` body preserving request
order.
"""

from __future__ import annotations

from typing import Any

from .protocol import RequestError

MAX_LENGTH_EXCEEDED = "max_length_exceeded"


def reject_pre_execution_errors(prepared: Any) -> None:
    """Turn ``prepared.errors`` into request errors before any execution.

    ``max_length_exceeded`` wins over invalid questions, mirroring the
    upstream server: its message names the actual token count and limit.
    """

    errors = getattr(prepared, "errors", None)
    if not isinstance(errors, dict):
        raise RuntimeError("the pinned runtime returned no per-question error map")

    over = [
        entry
        for entry in errors.values()
        if isinstance(entry, dict) and entry.get("error") == MAX_LENGTH_EXCEEDED
    ]
    if over:
        message = str(over[0].get("message") or "a question prompt exceeds the maximum context length")
        raise RequestError(message, 422)
    invalid: dict[str, str] = {}
    for name, entry in errors.items():
        # The pinned runtime always reports a string message; anything else
        # degrades to a stable placeholder instead of a repr leaking into
        # the JSON body.
        message = entry.get("message") if isinstance(entry, dict) else None
        invalid[name] = message if isinstance(message, str) and message else "invalid question"
    if invalid:
        # Insertion order of the runtime's map is the request order.
        raise RequestError(
            f"request contains {len(invalid)} invalid question(s)", 422,
            body={"invalid_questions": invalid},
        )
