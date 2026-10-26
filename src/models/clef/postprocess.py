"""SystemOne answers for one Clef question.

This is the pinned ``systemone_answer`` formula from
``joint_schema_model.py``: four-decimal rounding, choice ties broken by the
request's criteria order, and score as the expected index. The live worker
still calls the hashed checkpoint function. These mirrors exist so the
contract can be tested without the checkpoint.
"""

from __future__ import annotations

from typing import Any


def choice_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    options = [str(option) for option in question["criteria"]]
    choice = max(options, key=probabilities.__getitem__)
    return {
        "type": "choice",
        "choice": choice,
        "confidence": round(probabilities[choice], 4),
        "probabilities": {option: round(probabilities[option], 4) for option in options},
    }


def noul_answer(probabilities: dict[str, float]) -> dict[str, Any]:
    return {"type": "noul", "noul": round(probabilities["true"], 4)}


def score_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


def systemone_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    if question["type"] == "noul":
        return noul_answer(probabilities)
    if question["type"] == "choice":
        return choice_answer(question, probabilities)
    return score_answer(question, probabilities)
