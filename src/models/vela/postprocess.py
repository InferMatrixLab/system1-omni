"""Pure Vela 2.0 answer formulas and response finishing.

The pinned runtime builds the whole response envelope itself; this
worker passes it through verbatim. What lives here is the runtime-facing
half — ``finish_response`` checks the response against the admitted
questions (answer-key order including the set-label Noul views, the
``spans``/``sets``/``thresholds`` key sets, usage accounting) and
rejects anything that breaks the response contract — and, as pure
functions mirrored from the pinned ``system_one_response``, the
confidence and score formulas (top-two probability margin; one minus
the normalized level variance; the expected zero-based level), so the
CPU contract tests can lock them without weights and the recipe oracle
can re-derive them on a host with the checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

TOP_LEVEL_FIELDS = {"model", "answers", "usage", "spans", "sets", "thresholds"}
REQUIRED_FIELDS = ("model", "answers", "usage")
# Emitted by the hybrid members only; a 0.3B response carrying it means
# the wrong member's runtime answered.
HYBRID_ONLY_FIELDS = {"span_heads"}
# The exact answer field sets of the pinned runtime's
# ``system_one_response``: equality, not subset — a missing field is as
# broken as a foreign one. Set and span questions answer through their
# top-level ``sets``/``spans``/``thresholds`` payloads plus Noul views.
ANSWER_FIELDS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "confidence", "probabilities"},
    "score": {"type", "score", "confidence", "legend", "probabilities"},
}
SET_FIELDS = {"selected", "probabilities"}
SPAN_FIELDS = {"label", "start", "end", "text", "probability"}


def choice_confidence(probabilities: Sequence[float]) -> float:
    """The top-two probability margin, clamped to [0, 1].

    Upstream ``_choice_confidence``: 1.0 for a single option. Not
    max-p, and not an entropy.
    """

    values = sorted((float(p) for p in probabilities), reverse=True)
    if len(values) == 1:
        return 1.0
    return min(1.0, max(0.0, values[0] - values[1]))


def score_confidence(probabilities: Sequence[float]) -> float:
    """One minus the normalized level variance.

    Upstream ``_score_confidence``: the variance of the level
    distribution around its own mean, divided by the uniform
    distribution's variance ``(n^2 - 1) / 12``. A statistic of the
    distribution, not a calibrated probability of being right; 1.0 for
    a single level and 0.0 for the uniform distribution.
    """

    values = [float(p) for p in probabilities]
    n = len(values)
    if n == 1:
        return 1.0
    mean = math.fsum(i * x for i, x in enumerate(values))
    var = math.fsum(x * (i - mean) ** 2 for i, x in enumerate(values))
    return min(1.0, max(0.0, 1.0 - var / ((n * n - 1) / 12)))


def expected_level(probabilities: Sequence[float]) -> float:
    """The SystemOne ``score``: the expected zero-based level, fsum order."""

    return math.fsum(i * float(p) for i, p in enumerate(probabilities))


def _labels(question: Any) -> list[str]:
    criteria = question.get("criteria") if isinstance(question, dict) else None
    return [str(key) for key in criteria] if isinstance(criteria, Mapping) else []


def expected_answer_order(questions: Mapping[str, Any]) -> list[str]:
    """The answer keys in response order: each question's id, except that a
    set question contributes one ``"<qid>.<label>"`` Noul view per label
    (criteria order) and no entry of its own."""

    order: list[str] = []
    for qid, question in questions.items():
        if isinstance(question, dict) and question.get("type") == "set":
            order.extend(f"{qid}.{label}" for label in _labels(question))
        else:
            order.append(str(qid))
    return order


def _question_ids_of(questions: Mapping[str, Any], *kinds: str) -> list[str]:
    return [
        str(qid)
        for qid, question in questions.items()
        if isinstance(question, dict) and question.get("type") in kinds
    ]


def finish_response(questions: Mapping[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Check the runtime response against the admitted questions and pass it through.

    The pinned runtime owns every value in this response; these checks
    only refuse a response that could not be reconstructed for the
    request that produced it (wrong key set or order, a hybrid-only
    field, broken usage accounting, an answer view that is not an
    object or carries fields its kind does not allow).
    """

    if not isinstance(response, dict):
        raise RuntimeError("the pinned runtime returned a non-object response")
    keys = set(response)
    foreign = keys & HYBRID_ONLY_FIELDS
    if foreign:
        raise RuntimeError(
            f"the pinned runtime returned hybrid-only field(s) {sorted(foreign)}; "
            "this worker serves the 0.3B member"
        )
    missing = [field for field in REQUIRED_FIELDS if field not in response]
    if missing or not keys <= TOP_LEVEL_FIELDS:
        raise RuntimeError("the pinned runtime returned an unexpected response envelope")
    answers = response["answers"]
    expected = expected_answer_order(questions)
    if not isinstance(answers, dict) or list(answers) != expected:
        raise RuntimeError("answers do not match the request's question and set-view order")
    for field, ids in (
        ("spans", _question_ids_of(questions, "span")),
        ("sets", _question_ids_of(questions, "set")),
        ("thresholds", _question_ids_of(questions, "span", "set")),
    ):
        if ids:
            if not isinstance(response.get(field), dict) or list(response[field]) != ids:
                raise RuntimeError(f"{field} do not match the request's {field} questions")
        elif field in response:
            raise RuntimeError(f"the pinned runtime returned {field} without {field} questions")
    for qid, payload in (response.get("sets") or {}).items():
        if not isinstance(payload, dict) or set(payload) != SET_FIELDS:
            raise RuntimeError(f"set {qid!r} does not carry exactly {sorted(SET_FIELDS)}")
    for qid, spans in (response.get("spans") or {}).items():
        if not isinstance(spans, list):
            raise RuntimeError(f"spans {qid!r} is not a list")
        for span in spans:
            if not isinstance(span, dict) or set(span) != SPAN_FIELDS:
                raise RuntimeError(
                    f"a span of {qid!r} does not carry exactly {sorted(SPAN_FIELDS)}"
                )
    for qid, threshold in (response.get("thresholds") or {}).items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise RuntimeError(f"threshold {qid!r} is not a number")
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
        if not isinstance(answer, dict):
            raise RuntimeError(f"answer {name!r} is not an object")
        kind = answer.get("type")
        shape = ANSWER_FIELDS.get(kind) if isinstance(kind, str) else None
        if shape is None:
            raise RuntimeError(f"answer {name!r} has unknown type {kind!r}")
        if set(answer) != shape:
            raise RuntimeError(
                f"answer {name!r} does not carry exactly {sorted(shape)} "
                f"(got {sorted(answer)})"
            )
    return response
