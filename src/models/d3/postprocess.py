"""Pure d3 answer reconstruction and response finishing.

The runtime's ``respond`` builds each answer from per-option probabilities.
Its formulas are mirrored here as pure functions — first-max tie-breaking,
renormalization, the entropy confidence, the score expectation and the legend
canonicalization — so the CPU contract tests can lock them without weights
and the recipe oracle can re-derive them on a GPU host. ``finish_response``
is the runtime-facing half: it passes the response through unchanged, warns
on every embedded ``invalid_model_output`` answer (a silent partial failure
is undetectable otherwise; the log never contains request content) and
rejects anything that breaks the response contract.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from typing import Any

LOG = logging.getLogger(__name__)

INVALID_MODEL_OUTPUT = "invalid_model_output"


def describe(value: Any) -> str:
    """Upstream's option-text rendering: strings pass, other JSON is dumped."""

    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def canonical(value: Any) -> str:
    """Upstream's legend canonicalization (sorted keys, compact separators)."""

    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def confidence(values: Sequence[float]) -> float:
    """One minus the normalized entropy, clipped to [0, 1].

    Upstream docstring: "the Decision 2.0 measure". Not max-p.
    """

    if len(values) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))


def renormalized(probabilities: Sequence[float]) -> list[float]:
    """Validate and renormalize one probability per option (upstream to_answer)."""

    values = [float(value) for value in probabilities]
    if (
        len(values) < 1
        or any(not math.isfinite(value) or value < 0 for value in values)
        or sum(values) <= 0
    ):
        raise ValueError("need one finite non-negative probability per option")
    total = sum(values)
    return [value / total for value in values]


def choice_answer(keys: Sequence[str], probabilities: Sequence[float]) -> dict[str, Any]:
    """``{"type", "choice", "probabilities", "confidence"}``.

    Ties go to the earliest option in order (upstream's max over indices).
    """

    values = renormalized(probabilities)
    if len(values) != len(keys):
        raise ValueError("need one finite non-negative probability per option")
    best = max(range(len(values)), key=values.__getitem__)
    return {
        "type": "choice",
        "choice": keys[best],
        "probabilities": dict(zip(keys, values)),
        "confidence": confidence(values),
    }


def noul_answer(probabilities: Sequence[float]) -> dict[str, Any]:
    """``{"type": "noul", "noul": p_true}`` — no probabilities or confidence."""

    values = renormalized(probabilities)
    if len(values) != 2:
        raise ValueError("need one finite non-negative probability per option")
    return {"type": "noul", "noul": values[1]}


def score_answer(levels: Sequence[Any], probabilities: Sequence[float]) -> dict[str, Any]:
    """``{"type", "score", "probabilities", "confidence", "legend"}``.

    The answer keys are the stringified level indices; ``score`` is the
    0-based expectation and ``legend`` maps each key to its canonical level.
    """

    values = renormalized(probabilities)
    if len(values) != len(levels):
        raise ValueError("need one finite non-negative probability per level")
    keys = [str(index) for index in range(len(levels))]
    return {
        "type": "score",
        "score": sum(index * value for index, value in enumerate(values)),
        "probabilities": dict(zip(keys, values)),
        "confidence": confidence(values),
        "legend": {key: canonical(level) for key, level in zip(keys, levels)},
    }


def to_answer(
    kind: str,
    keys: Sequence[str],
    levels: Sequence[Any] | None,
    probabilities: Sequence[float],
) -> dict[str, Any]:
    """Dispatch on the question kind (upstream product_answer shapes)."""

    if kind == "noul":
        return noul_answer(probabilities)
    if kind == "choice":
        return choice_answer(keys, probabilities)
    if kind == "score":
        return score_answer(levels or (), probabilities)
    raise ValueError(f"unsupported question type {kind!r}")


def finish_response(expected_names: Sequence[str], response: dict[str, Any]) -> dict[str, Any]:
    """Pass the runtime response through, warning on embedded model failures.

    ``invalid_model_output`` answers stay in the HTTP 200 response (partial
    success; the other questions are still answered) but each one logs a
    warning naming the question and the failure — never request content.
    Any other embedded error means admission failed to intercept it and is a
    worker bug, not a client error.
    """

    if not isinstance(response, dict) or set(response) != {"model", "answers", "usage"}:
        raise RuntimeError("the pinned runtime returned an unexpected response envelope")
    answers = response["answers"]
    if not isinstance(answers, dict) or tuple(answers) != tuple(expected_names):
        raise RuntimeError("answers do not match request question order")
    usage = response["usage"]
    if (
        not isinstance(usage, dict)
        or set(usage) != {"input_tokens", "output_tokens"}
        or not isinstance(usage["input_tokens"], int)
        or usage["input_tokens"] < 0
        or usage["output_tokens"] != 0
    ):
        raise RuntimeError("the pinned runtime returned invalid usage accounting")
    for name, answer in answers.items():
        if not isinstance(answer, dict) or "error" not in answer:
            continue
        if answer.get("error") == INVALID_MODEL_OUTPUT:
            LOG.warning(
                "question %s was answered with invalid_model_output: %s",
                name,
                answer.get("message"),
            )
        else:
            raise RuntimeError(
                f"the pinned runtime embedded a pre-execution error for question {name!r}"
            )
    return response
