"""Pure Valen-Decision-1.5 response reconstruction from executor logits.

The formulas and the answer field order follow the checkpoint's own
``responses.answer`` at temperature 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .preprocess import ResponseContext
from .protocol import Question


def _probabilities(logits: Sequence[float]) -> list[float]:
    if not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("model returned invalid logits")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def answer(question: Question, logits: Sequence[float]) -> dict[str, Any]:
    """Decode one question's candidate logits into its typed answer."""

    if len(logits) != len(question.keys):
        raise ValueError("model must return one logit per candidate")
    probabilities = _probabilities(logits)
    if question.kind == "noul":
        return {"type": "noul", "noul": probabilities[question.keys.index("true")]}
    count = len(probabilities)
    mode = max(range(count), key=probabilities.__getitem__)
    result: dict[str, Any] = {
        "type": question.kind,
        "probabilities": dict(zip(question.keys, probabilities)),
    }
    if question.kind == "choice":
        result["choice"] = question.keys[mode]
        result["confidence"] = (
            1.0 if count == 1 else max(0.0, (probabilities[mode] - 1 / count) / (1 - 1 / count))
        )
    else:
        result["score"] = sum(i * p for i, p in enumerate(probabilities))
        result["legend"] = dict(zip(question.keys, question.descriptions))
        distance = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
        uniform_mad = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
        result["confidence"] = max(0.0, 1.0 - distance / uniform_mad)
    return result


def build_response(
    context: ResponseContext,
    answers: Mapping[str, dict[str, Any]],
    logical_tokens: int,
    compute_tokens: int,
) -> dict[str, Any]:
    """Reconstruct the public response and the internal usage fields."""

    expected = tuple(question.name for question in context.questions)
    if tuple(answers) != expected:
        raise ValueError("answers do not match request question order")
    if logical_tokens < 0 or compute_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return {
        "model": context.model,
        "answers": dict(answers),
        "usage": {"input_tokens": logical_tokens, "output_tokens": 0},
        "internal_usage": {"compute_tokens": compute_tokens},
    }
