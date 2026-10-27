"""Answer parity between Clef FLA prefill and opt-in FlashInfer prefill.

The gate compares two ``benchmarks/bench.py run`` directories that replayed
the same manifest. It does not start a server, does not change the default
prefill, and does not record a latency KEEP. A GPU run of this gate is the
blocker before ``--gdn-prefill flashinfer`` becomes the default.

Predeclared tolerances, fixed in this module:

- maximum probability drift ``0.002``
- maximum score drift ``0.002``
- decision flips ``0``

Choice values must match. Noul flips when the two sides disagree about
``value >= 0.5``. Score is the continuous drift above. ``/health``
``linear_attention`` must be ``fla+causal_conv1d`` on the FLA run and
``flashinfer+fla+causal_conv1d`` on the FlashInfer run. A wrong bind fails
even when the answers match.

``--device`` and ``--compute-capability`` are copied into the report when
the operator supplies them. The script does not invent either field. A
report that omits them is an incomplete campaign paste.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

FLA_LINEAR_ATTENTION = "fla+causal_conv1d"
FLASHINFER_LINEAR_ATTENTION = "flashinfer+fla+causal_conv1d"
MAX_PROBABILITY_DRIFT = 0.002
MAX_SCORE_DRIFT = 0.002
MAX_FLIPS = 0


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def evaluate(
    fla_run: Path,
    flashinfer_run: Path,
    fla_health_path: Path,
    flashinfer_health_path: Path,
    *,
    device: str | None = None,
    compute_capability: str | None = None,
) -> dict[str, Any]:
    """Compare one FLA run with one FlashInfer run. Do not relax the gate."""

    failures: list[str] = []
    fla_health = load_health(fla_health_path)
    flashinfer_health = load_health(flashinfer_health_path)
    fla_linear = fla_health.get("linear_attention")
    flashinfer_linear = flashinfer_health.get("linear_attention")
    if fla_health.get("status") != "ready" or fla_linear != FLA_LINEAR_ATTENTION:
        failures.append(
            f"FLA /health linear_attention is {fla_linear!r}; expected {FLA_LINEAR_ATTENTION!r}"
        )
    if (
        flashinfer_health.get("status") != "ready"
        or flashinfer_linear != FLASHINFER_LINEAR_ATTENTION
    ):
        failures.append(
            "FlashInfer /health linear_attention is "
            f"{flashinfer_linear!r}; expected {FLASHINFER_LINEAR_ATTENTION!r}"
        )

    fla_manifest = fla_run / "requests.jsonl"
    flashinfer_manifest = flashinfer_run / "requests.jsonl"
    fla_bytes = fla_manifest.read_bytes()
    flashinfer_bytes = flashinfer_manifest.read_bytes()
    digest = hashlib.sha256(fla_bytes).hexdigest()
    if fla_bytes != flashinfer_bytes:
        failures.append("the two runs did not replay the same requests.jsonl bytes")
        digest = None

    cases = load_cases(fla_manifest)
    fla_responses = load_responses(fla_run / "responses.jsonl")
    flashinfer_responses = load_responses(flashinfer_run / "responses.jsonl")
    expected_ids = [case["id"] for case in cases]
    if set(fla_responses) != set(expected_ids) or set(flashinfer_responses) != set(expected_ids):
        failures.append("response IDs do not cover the manifest on both runs")

    compared = 0
    failed_pairs = 0
    flips = 0
    max_probability = 0.0
    max_score = 0.0
    for case in cases:
        case_id = case["id"]
        left = fla_responses.get(case_id)
        right = flashinfer_responses.get(case_id)
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
                a = left["answers"][qid]
                b = right["answers"][qid]
                probs_a = _probabilities(a, kind)
                probs_b = _probabilities(b, kind)
            except (KeyError, TypeError, ValueError) as exc:
                failed_pairs += 1
                failures.append(f"{case_id}/{qid}: {exc}")
                continue
            if set(probs_a) != set(probs_b):
                failures.append(f"{case_id}/{qid}: probability labels differ")
                continue
            drift = max(abs(probs_a[key] - probs_b[key]) for key in probs_a)
            max_probability = max(max_probability, drift)
            if kind == "score":
                if not _finite(a.get("value")) or not _finite(b.get("value")):
                    failures.append(f"{case_id}/{qid}: score value must be finite")
                    continue
                max_score = max(max_score, abs(float(a["value"]) - float(b["value"])))
            elif kind == "noul":
                if not _finite(a.get("value")) or not _finite(b.get("value")):
                    failures.append(f"{case_id}/{qid}: noul value must be finite")
                    continue
                flips += int((float(a["value"]) >= 0.5) != (float(b["value"]) >= 0.5))
            else:
                flips += int(a.get("value") != b.get("value"))
            compared += 1

    if compared == 0:
        failures.append("no decisions were compared")
    if max_probability > MAX_PROBABILITY_DRIFT:
        failures.append(
            f"probability drift {max_probability} exceeds {MAX_PROBABILITY_DRIFT}"
        )
    if max_score > MAX_SCORE_DRIFT:
        failures.append(f"score drift {max_score} exceeds {MAX_SCORE_DRIFT}")
    if flips > MAX_FLIPS:
        failures.append(f"decision flips {flips} exceed {MAX_FLIPS}")
    if failed_pairs:
        failures.append(f"failed request pairs: {failed_pairs}")

    passed = not failures
    return {
        "kind": "clef-gdn-prefill-answer-parity",
        "passed": passed,
        "default_remains": "fla",
        "latency_keep": False,
        "manifest_sha256": digest,
        "fla_linear_attention": fla_linear,
        "flashinfer_linear_attention": flashinfer_linear,
        "compared_decisions": compared,
        "failed_request_pairs": failed_pairs,
        "decision_flips": flips,
        "max_probability_drift": max_probability,
        "max_score_drift": max_score,
        "gate": {
            "max_probability_drift": MAX_PROBABILITY_DRIFT,
            "max_score_drift": MAX_SCORE_DRIFT,
            "max_flips": MAX_FLIPS,
            "fla_linear_attention": FLA_LINEAR_ATTENTION,
            "flashinfer_linear_attention": FLASHINFER_LINEAR_ATTENTION,
        },
        "device": device,
        "compute_capability": compute_capability,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fla-run", type=Path, required=True)
    parser.add_argument("--flashinfer-run", type=Path, required=True)
    parser.add_argument("--fla-health", type=Path, required=True)
    parser.add_argument("--flashinfer-health", type=Path, required=True)
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
            args.fla_run,
            args.flashinfer_run,
            args.fla_health,
            args.flashinfer_health,
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
