"""Text prompt pieces for the pinned Clef ``encode_record`` path.

``joint_schema_model.py`` tokenizes each piece on its own and concatenates the
ids. The strings here follow that concatenation order (prefix, rendered state,
schema, non-thinking suffix). Image and video placeholders are not emitted;
this integration rejects media before encoding.

The shipped encoder silently slices ``state`` once the schema and fixed
affixes fill ``max_length``. Callers that need a limit must compare
``token_lengths`` with the full state and refuse the request. This module
does not slice.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
# The decision encoder hardcodes the empty think block. That is the non-thinking
# path of the Qwen chat template (``enable_thinking`` false). The template's
# default open ``<think>`` is not used by ``systemone``.
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
NOUL_DEFAULT_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
QUESTION_TYPES = ("noul", "choice", "score")


def render(value: Any) -> str:
    """Render one JSON value the way the pinned encoder does.

    Strings are inserted unchanged. Every other value is compact JSON with
    sorted keys, so object key order in the request does not change the prompt.
    """

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return ``(option_id, description)`` pairs in encoder order.

    Choice ids are sorted by their string form. That order is the order the
    joint head scores, which is not the order of keys in the response
    ``probabilities`` object. Score ids are the decimal indices of ``criteria``.
    Noul ids are always ``true`` then ``false``.
    """

    question_type = question["type"]
    if question_type == "noul":
        criteria = dict(NOUL_DEFAULT_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


def prompt_pieces(record: dict[str, Any]) -> list[str]:
    """Return the text pieces whose token ids are concatenated for one record."""

    schema: list[str] = ["\n\nSCHEMA FIELDS:\n"]
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema.append(
            f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: "
        )
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema.append(render(instructions))
        schema.append("\nALLOWED OPTIONS:\n")
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema.append(f"OPTION {option_index + 1}: ")
            semantics: dict[str, Any] = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema.append(render(semantics))
            schema.append("\n")
        schema.append("END FIELD\n")
    return [PREFIX, render(record["state"]), *schema, SUFFIX]


def token_lengths(count: Callable[[str], int], record: dict[str, Any]) -> tuple[int, int]:
    """Return ``(fixed_length, total_length)`` for an untruncated record.

    ``fixed_length`` excludes the rendered state, matching the pinned
    encoder's overflow check. ``total_length`` counts every state token.
    ``count`` must be the length of one piece tokenized alone
    (``add_special_tokens=False``); the pieces are not retokenized as one string.
    """

    lengths = [count(piece) for piece in prompt_pieces(record)]
    state_tokens = lengths[1]
    fixed = sum(lengths) - state_tokens
    return fixed, fixed + state_tokens
