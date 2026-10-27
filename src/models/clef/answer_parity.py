"""Answer parity between eager FLA Clef and opt-in piecewise backbone.

The gate compares two ``benchmarks/bench.py run`` directories that replayed
the same manifest. It does not start a server, does not change the default
backbone, and does not record a latency KEEP. A GPU run of this gate is the
blocker before ``--backbone piecewise`` can be treated as more than a draft.

Predeclared tolerances, fixed in this module. Do not widen them:

- maximum probability drift ``0.002``
- maximum score drift ``0.002``
- decision flips ``0``

Choice values must match. Noul flips when the two sides disagree about
``value >= 0.5``. Score is the continuous drift above. Both runs must be
the FLA prefill (``linear_attention`` ``fla+causal_conv1d``). The eager
``/health`` ``backbone`` must be ``eager`` with ``backbone_graphs`` 0. The
piecewise side must be ``piecewise+inductor`` with ``backbone_graphs`` at
least 1. A wrong bind fails even when the answers match.

``--device`` and ``--compute-capability`` are copied into the report when
the operator supplies them. The script does not invent either field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from models.clef.piecewise import BACKBONE_HEALTH_EAGER, BACKBONE_HEALTH_PIECEWISE

FLA_LINEAR_ATTENTION = "fla+causal_conv1d"
MAX_PROBABILITY_DRIFT = 0.002
MAX_SCORE_DRIFT = 0.002
MAX_FLIPS = 0


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError(f"{path} must contain nonempty, unique request IDs")
    for case in cases:
        questions = case["request"]["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError(f"{path} case {case['id']!r} needs questions")
        for qid, question in questions.items():
            if question.get("type") not in ("choice", "noul", "score"):
                raise ValueError(f"{path} question {qid!r} has an unsupported type")
    return cases


def load_responses(path: Path) -> dict[str, dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = record["id"]
        if case_id in by_id:
            raise ValueError(f"{path} repeats request id {case_id!r}")
        by_id[case_id] = record
    return by_id


def load_health(path: Path) -> dict[str, Any]:
    health = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(health, dict):
        raise ValueError(f"{path} health must be a JSON object")
    return health


def _probabilities(answer: dict[str, Any], kind: str) -> dict[str, float]:
    if kind == "noul" and "probabilities" not in answer:
        value = answer["value"]
        if not _finite(value):
            raise ValueError("noul value must be finite")
        return {"false": 1 - float(value), "true": float(value)}
    probabilities = answer["probabilities"]
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError("answer probabilities must be a nonempty object")
    parsed: dict[str, float] = {}
    for key, value in probabilities.items():
        if not _finite(value):
            raise ValueError(f"probability {key!r} must be finite")
        parsed[str(key)] = float(value)
    return parsed


def _require_fla(health: dict[str, Any], label: str, failures: list[str]) -> None:
    linear = health.get("linear_attention")
    if health.get("status") != "ready" or linear != FLA_LINEAR_ATTENTION:
        failures.append(
            f"{label} /health linear_attention is {linear!r}; expected {FLA_LINEAR_ATTENTION!r}"
        )


def evaluate(
    eager_run: Path,
    piecewise_run: Path,
    eager_health_path: Path,
    piecewise_health_path: Path,
    *,
    device: str | None = None,
    compute_capability: str | None = None,
) -> dict[str, Any]:
    """Compare one eager FLA run with one piecewise run. Do not relax the gate."""

    failures: list[str] = []
    eager_health = load_health(eager_health_path)
    piecewise_health = load_health(piecewise_health_path)
    _require_fla(eager_health, "eager", failures)
    _require_fla(piecewise_health, "piecewise", failures)
    eager_backbone = eager_health.get("backbone")
    piecewise_backbone = piecewise_health.get("backbone")
    eager_graphs = eager_health.get("backbone_graphs")
    piecewise_graphs = piecewise_health.get("backbone_graphs")
    if eager_backbone != BACKBONE_HEALTH_EAGER or eager_graphs != 0:
        failures.append(
            f"eager /health backbone is {eager_backbone!r} graphs {eager_graphs!r}; "
            f"expected {BACKBONE_HEALTH_EAGER!r} and 0"
        )
    if (
        piecewise_backbone != BACKBONE_HEALTH_PIECEWISE
        or not isinstance(piecewise_graphs, int)
        or isinstance(piecewise_graphs, bool)
        or piecewise_graphs < 1
    ):
        failures.append(
            "piecewise /health backbone is "
            f"{piecewise_backbone!r} graphs {piecewise_graphs!r}; "
            f"expected {BACKBONE_HEALTH_PIECEWISE!r} and a graph count of at least 1"
        )

    eager_manifest = eager_run / "requests.jsonl"
    piecewise_manifest = piecewise_run / "requests.jsonl"
    eager_bytes = eager_manifest.read_bytes()
    piecewise_bytes = piecewise_manifest.read_bytes()
    digest = hashlib.sha256(eager_bytes).hexdigest()
    if eager_bytes != piecewise_bytes:
        failures.append("the two runs did not replay the same requests.jsonl bytes")
        digest = None

    cases = load_cases(eager_manifest)
    eager_responses = load_responses(eager_run / "responses.jsonl")
    piecewise_responses = load_responses(piecewise_run / "responses.jsonl")
    expected_ids = [case["id"] for case in cases]
    if set(eager_responses) != set(expected_ids) or set(piecewise_responses) != set(expected_ids):
        failures.append("response IDs do not cover the manifest on both runs")

    compared = 0
    failed_pairs = 0
    flips = 0
    max_probability = 0.0
    max_score = 0.0
    for case in cases:
        case_id = case["id"]
        left = eager_responses.get(case_id)
        right = piecewise_responses.get(case_id)
        if left is None or right is None:
            failed_pairs += 1
            continue
        if left.get("error") or right.get("error"):
            failed_pairs += 1
            failures.append(f"{case_id}: a request failed")
            continue
        for qid, question in case["request"]["questions"].items():
            kind = question["type"]
            try:
                left_answer = left["answers"][qid]
                right_answer = right["answers"][qid]
                probs_left = _probabilities(left_answer, kind)
                probs_right = _probabilities(right_answer, kind)
            except (KeyError, TypeError, ValueError) as exc:
                failed_pairs += 1
                failures.append(f"{case_id}/{qid}: {exc}")
                continue
            if set(probs_left) != set(probs_right):
                failures.append(f"{case_id}/{qid}: probability labels differ")
                continue
            drift = max(abs(probs_left[key] - probs_right[key]) for key in probs_left)
            max_probability = max(max_probability, drift)
            if kind == "score":
                if not _finite(left_answer.get("value")) or not _finite(right_answer.get("value")):
                    failures.append(f"{case_id}/{qid}: score value must be finite")
                    continue
                max_score = max(
                    max_score, abs(float(left_answer["value"]) - float(right_answer["value"]))
                )
            elif kind == "noul":
                if not _finite(left_answer.get("value")) or not _finite(right_answer.get("value")):
                    failures.append(f"{case_id}/{qid}: noul value must be finite")
                    continue
                flips += int(
                    (float(left_answer["value"]) >= 0.5) != (float(right_answer["value"]) >= 0.5)
                )
            else:
                flips += int(left_answer.get("value") != right_answer.get("value"))
            compared += 1

    if compared == 0:
        failures.append("no decisions were compared")
    if max_probability > MAX_PROBABILITY_DRIFT:
        failures.append(f"probability drift {max_probability} exceeds {MAX_PROBABILITY_DRIFT}")
    if max_score > MAX_SCORE_DRIFT:
        failures.append(f"score drift {max_score} exceeds {MAX_SCORE_DRIFT}")
    if flips > MAX_FLIPS:
        failures.append(f"decision flips {flips} exceed {MAX_FLIPS}")
    if failed_pairs:
        failures.append(f"failed request pairs: {failed_pairs}")

    return {
        "kind": "clef-piecewise-answer-parity",
        "passed": not failures,
        "default_remains": "eager",
        "prefill_remains": "fla",
        "latency_keep": False,
        "manifest_sha256": digest,
        "eager_backbone": eager_backbone,
        "piecewise_backbone": piecewise_backbone,
        "eager_backbone_graphs": eager_graphs,
        "piecewise_backbone_graphs": piecewise_graphs,
        "eager_linear_attention": eager_health.get("linear_attention"),
        "piecewise_linear_attention": piecewise_health.get("linear_attention"),
        "compared_decisions": compared,
        "failed_request_pairs": failed_pairs,
        "decision_flips": flips,
        "max_probability_drift": max_probability,
        "max_score_drift": max_score,
        "gate": {
            "max_probability_drift": MAX_PROBABILITY_DRIFT,
            "max_score_drift": MAX_SCORE_DRIFT,
            "max_flips": MAX_FLIPS,
            "linear_attention": FLA_LINEAR_ATTENTION,
            "eager_backbone": BACKBONE_HEALTH_EAGER,
            "piecewise_backbone": BACKBONE_HEALTH_PIECEWISE,
        },
        "device": device,
        "compute_capability": compute_capability,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-run", type=Path, required=True)
    parser.add_argument("--piecewise-run", type=Path, required=True)
    parser.add_argument("--eager-health", type=Path, required=True)
    parser.add_argument("--piecewise-health", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="nvidia-smi name string. Recorded in the report. Not invented.",
    )
    parser.add_argument(
        "--compute-capability",
        default=None,
        help="Capability the worker reported. Recorded in the report. Not invented.",
    )
    args = parser.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}", file=sys.stderr)
        return 2
    try:
        report = evaluate(
            args.eager_run,
            args.piecewise_run,
            args.eager_health,
            args.piecewise_health,
            device=args.device,
            compute_capability=args.compute_capability,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(args.report)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
