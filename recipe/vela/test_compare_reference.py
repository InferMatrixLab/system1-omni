#!/usr/bin/env python3
"""Regression tests for recipe/vela/compare_reference.py.

Checks the pure comparison, case-loading and export helpers without the
pinned package, weights, torch or fastapi: exact-match verdicts
including key order, rejection agreement, the default fixture set's
shape (the invalid-span 422 and over-limit 413 synthetic cases), the
oracle export gate, and the oracle export structure.

    python recipe/vela/test_compare_reference.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare_reference as tool


def response(answers=None, input_tokens=17):
    value = {
        "model": tool.MODEL_NAME,
        "answers": answers if answers is not None else {"q": {"type": "noul", "noul": 0.75}},
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
    if answers is None:
        value["thresholds"] = {"q": 0.3}
    return value


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    # 1. Matching 200 responses pass; every divergence fails.
    reference = response()
    lines, ok = tool.compare_case("fixture", (200, response()), (200, reference))
    report = "\n".join(lines)
    check(
        "exact 200 match passes",
        ok and "exact match (status, body, key order): PASS" in report,
        report,
    )

    lines, ok = tool.compare_case("fixture", (200, response(input_tokens=18)), (200, reference))
    check("token divergence fails", not ok, "\n".join(lines))

    reordered = {"q": {"type": "noul", "noul": 0.75}}
    other_first = {"other": {"type": "noul", "noul": 0.5}, "q": {"type": "noul", "noul": 0.75}}
    lines, ok = tool.compare_case("fixture", (200, response(other_first)), (200, response(reordered)))
    check("answer-set divergence fails", not ok, "\n".join(lines))

    drifted = {"q": {"type": "noul", "noul": 0.7500000001}}
    lines, ok = tool.compare_case("fixture", (200, response(drifted)), (200, reference))
    check("numeric drift fails (exact-match criterion)", not ok, "\n".join(lines))

    # 2. Rejections agree only on both status and body.
    rejection = {"detail": [{"loc": ["body", "questions", "bad", "criteria"],
                             "msg": "span criteria must be an object with 1 to 255 entries",
                             "type": "value_error"}]}
    lines, ok = tool.compare_case("fixture", (422, rejection), (422, rejection))
    check("matching 422 rejections pass", ok, "\n".join(lines))

    lines, ok = tool.compare_case("fixture", (413, rejection), (422, rejection))
    check("status mismatch fails", not ok, "\n".join(lines))

    other = {"detail": [{"loc": ["body", "questions"], "msg": "different", "type": "value_error"}]}
    lines, ok = tool.compare_case("fixture", (422, other), (422, rejection))
    check("body mismatch fails", not ok, "\n".join(lines))

    lines, ok = tool.compare_case("fixture", (200, response()), (422, rejection))
    check("worker answering a rejected request fails", not ok, "\n".join(lines))

    # 3. The default case set carries the charter's fixture list shape.
    cases = tool.default_cases()
    names = [name for name, _ in cases]
    check(
        "default cases are the three examples plus five rejections",
        names == ["example-request", "example-request-set-threshold", "example-request-cjk",
                  "invalid-span-422", "over-limit-413",
                  "missing-model-422", "extra-field-422", "wrong-model-422"],
        str(names),
    )
    missing = dict(cases[5][1])
    check(
        "the missing-model case is an envelope without the model field",
        set(missing) == {"state", "questions"},
        json.dumps(missing)[:120],
    )
    extra = dict(cases[6][1])
    check(
        "the extra-field case carries a field the preamble forbids",
        "images" in extra and set(extra) == {"model", "state", "questions", "images"},
        json.dumps(extra)[:120],
    )
    wrong = dict(cases[7][1])
    check(
        "the wrong-model case names the 0.8B member",
        wrong["model"] == "vllm-sr/Vela-2.0-0.8B",
        wrong["model"],
    )
    invalid = dict(cases[3][1])
    check(
        "the invalid-span case is a span question with list criteria",
        invalid["questions"]["bad"]["type"] == "span"
        and isinstance(invalid["questions"]["bad"]["criteria"], list),
        json.dumps(invalid)[:200],
    )
    over = dict(cases[4][1])
    check(
        "the over-limit case is questions+schema overflow by construction",
        len(over["questions"]["big"]["criteria"]) == 255,
        json.dumps(over)[:120],
    )

    # 4. The oracle export freezes the full native-comparison surface.
    body = {"model": tool.MODEL_NAME, "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "x"}}}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "oracle.json"
        oracle = tool.export_oracle(
            "case", body, response(), {"answers": [{"id": "q", "type": "choice"}]}, path
        )
        reread = json.loads(path.read_text(encoding="utf-8"))
        check(
            "oracle export round-trips",
            oracle == reread
            and oracle["request"] == body
            and oracle["system_one"]["answers"]["q"]["noul"] == 0.75
            and oracle["predict"]["answers"][0]["id"] == "q",
            json.dumps(oracle)[:200],
        )

    check("export allowed for a passing 200 case",
          tool.export_allowed(True, (200, response())))
    check("export refused for a failed comparison",
          not tool.export_allowed(False, (200, response())))
    check("export refused for an agreed rejection",
          not tool.export_allowed(True, (422, rejection)))

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
