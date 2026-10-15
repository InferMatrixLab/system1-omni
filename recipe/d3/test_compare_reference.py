#!/usr/bin/env python3
"""Regression tests for compare_reference.py.

Checks the pure comparison and export helpers without the pinned d3 package,
weights or a GPU: the PASS/FAIL verdicts of compare_case, embedded-error
shape agreement, the oracle export gate, and the oracle export structure.

    python recipe/d3/test_compare_reference.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare_reference as tool


def worker_response(answers, input_tokens=17):
    return {
        "model": "d3",
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


CHOICE = {
    "type": "choice",
    "choice": "wait",
    "probabilities": {"refund": 0.25, "wait": 0.75},
    "confidence": 0.18872187554086717,
}
NOUL = {"type": "noul", "noul": 0.75}
SCORE = {
    "type": "score",
    "score": 1.0,
    "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25},
    "confidence": 0.18872187554086717,
    "legend": {"0": "minor", "1": "moderate", "2": "major"},
}


def reference():
    return worker_response({"refund": dict(CHOICE), "urgent": dict(NOUL), "severity": dict(SCORE)})


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    # 1. An exactly matching worker response passes every verdict.
    lines, ok = tool.compare_case("fixture", worker_response(
        {"refund": dict(CHOICE), "urgent": dict(NOUL), "severity": dict(SCORE)}
    ), reference())
    report = "\n".join(lines)
    check(
        "exact match passes",
        ok
        and "decision agreement: PASS" in report
        and "numeric tolerance <= 1e-06: PASS" in report
        and "token accounting: PASS" in report,
        report,
    )

    # 2. A tiny delta stays inside the declared tolerance.
    drifted = dict(CHOICE)
    drifted["probabilities"] = {"refund": 0.25 + 1e-9, "wait": 0.75 - 1e-9}
    drifted["confidence"] = CHOICE["confidence"] + 1e-9
    lines, ok = tool.compare_case(
        "fixture", worker_response({"refund": drifted, "urgent": dict(NOUL), "severity": dict(SCORE)}),
        reference(),
    )
    check("delta within tolerance passes", ok, "\n".join(lines))

    # 3. Each failure mode fails on its own.
    flipped = dict(CHOICE)
    flipped["choice"] = "refund"
    lines, ok = tool.compare_case(
        "fixture", worker_response({"refund": flipped, "urgent": dict(NOUL), "severity": dict(SCORE)}),
        reference(),
    )
    check(
        "decision mismatch fails",
        not ok and "decision agreement: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    lines, ok = tool.compare_case(
        "fixture", worker_response(
            {"refund": dict(CHOICE), "urgent": dict(NOUL), "severity": dict(SCORE)},
            input_tokens=18,
        ), reference(),
    )
    check(
        "token mismatch fails",
        not ok and "token accounting: FAIL" in "\n".join(lines)
        and "input_tokens: 18 != 17" in "\n".join(lines),
        "\n".join(lines),
    )

    reordered = {"urgent": dict(NOUL), "refund": dict(CHOICE), "severity": dict(SCORE)}
    lines, ok = tool.compare_case("fixture", worker_response(reordered), reference())
    check(
        "answer order mismatch fails",
        not ok and "answer order differs" in "\n".join(lines),
        "\n".join(lines),
    )

    # 4. Embedded invalid_model_output shapes must agree between the paths.
    bad = {
        "type": "noul",
        "error": "invalid_model_output",
        "message": "need one finite non-negative probability per option",
    }
    lines, ok = tool.compare_case(
        "fixture", worker_response({"refund": dict(bad), "urgent": dict(NOUL), "severity": dict(SCORE)}),
        worker_response({"refund": dict(bad), "urgent": dict(NOUL), "severity": dict(SCORE)}),
    )
    check("matching embedded errors pass", ok, "\n".join(lines))

    other_message = dict(bad)
    other_message["message"] = "different failure"
    lines, ok = tool.compare_case(
        "fixture", worker_response({"refund": dict(other_message), "urgent": dict(NOUL), "severity": dict(SCORE)}),
        worker_response({"refund": dict(bad), "urgent": dict(NOUL), "severity": dict(SCORE)}),
    )
    check(
        "embedded error mismatch fails",
        not ok and "embedded error shape differs" in "\n".join(lines),
        "\n".join(lines),
    )

    # 5. Rejection agreement: matching 422s pass, mismatches fail.
    lines, ok = tool.compare_case("fixture", worker_response({}), ("422", "too long"))
    check(
        "worker answering a rejected request fails",
        not ok and "reference rejected the request: too long" in "\n".join(lines),
        "\n".join(lines),
    )

    rejected = worker_response({})
    rejected["worker rejected the request"] = {
        "status": 422,
        "body": {"invalid_questions": {"bad": "unsupported question type"}},
    }
    lines, ok = tool.compare_case(
        "fixture", rejected, ("422", {"invalid_questions": {"bad": "unsupported question type"}})
    )
    check(
        "matching 422 rejections pass",
        ok and "rejection agreement: PASS" in "\n".join(lines),
        "\n".join(lines),
    )

    wrong_body = dict(rejected)
    wrong_body["worker rejected the request"] = {
        "status": 422,
        "body": {"detail": "different"},
    }
    lines, ok = tool.compare_case(
        "fixture", wrong_body, ("422", {"invalid_questions": {"bad": "unsupported question type"}})
    )
    check(
        "mismatched rejection bodies fail",
        not ok and "rejection agreement: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    non_422 = dict(rejected)
    non_422["worker rejected the request"] = {"status": 500, "body": {"detail": "x"}}
    lines, ok = tool.compare_case("fixture", non_422, ("422", "too long"))
    check(
        "non-422 worker rejection fails",
        not ok and "rejection agreement: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    # 6. The oracle export freezes the full stage-3 comparison surface.
    body = {"model": "d3", "state": "s", "questions": {"q": {"type": "noul"}}}

    class Prepared:
        sequences = {"q": [1, 2, 3]}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "oracle.json"
        oracle = tool.export_oracle(
            body,
            Prepared(),
            {"q": [0.25, 0.75]},
            worker_response({"q": dict(NOUL)}),
            path,
        )
        reread = json.loads(path.read_text(encoding="utf-8"))
        check(
            "oracle export round-trips",
            oracle == reread
            and oracle["token_ids"] == {"q": [1, 2, 3]}
            and oracle["probabilities"] == {"q": [0.25, 0.75]}
            and oracle["answers"]["q"]["noul"] == 0.75
            and oracle["usage"] == {"input_tokens": 17, "output_tokens": 0},
            json.dumps(oracle)[:200],
        )

    # 7. The oracle is frozen only for a passing, non-rejected comparison.
    served = worker_response({"q": dict(NOUL)})
    check(
        "export allowed for a passing 200 case",
        tool.export_allowed(True, served, worker_response({"q": dict(NOUL)})),
    )
    check(
        "export refused for a failed comparison",
        not tool.export_allowed(False, served, worker_response({"q": dict(NOUL)})),
    )
    mutual_422 = worker_response({})
    mutual_422["worker rejected the request"] = {"status": 422, "body": {"detail": "too long"}}
    check(
        "export refused for an agreed 422 (nothing to freeze)",
        not tool.export_allowed(True, mutual_422, ("422", "too long")),
    )
    check(
        "export refused when only the worker rejected",
        not tool.export_allowed(True, mutual_422, worker_response({"q": dict(NOUL)})),
    )

    failed = 0
    for name, passed, detail in results:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        if not passed:
            failed += 1
            print("       " + detail.strip().replace("\n", "\n       ")[:400])
    print(
        "compare_reference tests: ok"
        if not failed
        else f"compare_reference tests: {failed} FAILED"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
