"""Offline contract for eager FLA versus piecewise answer parity."""

import json
from pathlib import Path

from models.clef.answer_parity import (
    FLA_LINEAR_ATTENTION,
    evaluate,
    main,
)
from models.clef.piecewise import BACKBONE_HEALTH_EAGER, BACKBONE_HEALTH_PIECEWISE


def _case(case_id: str, kind: str) -> dict:
    if kind == "choice":
        question = {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "pay", "technical": "outage"},
        }
        expected = "billing"
    elif kind == "noul":
        question = {"type": "noul", "instructions": "Urgent?"}
        expected = True
    else:
        question = {
            "type": "score",
            "instructions": "How severe?",
            "criteria": ["none", "some", "all"],
        }
        expected = 1
    return {
        "id": case_id,
        "request": {"state": "checkout is down", "questions": {case_id: question}},
        "expected": {case_id: expected},
    }


def _answer(kind: str, value, probabilities: dict[str, float]) -> dict:
    del kind
    return {"value": value, "probabilities": probabilities}


def _write_run(path: Path, cases: list[dict], answers: dict[str, dict]) -> None:
    path.mkdir(parents=True)
    (path / "requests.jsonl").write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )
    records = []
    for case in cases:
        records.append(
            {"id": case["id"], "error": None, "answers": {case["id"]: answers[case["id"]]}}
        )
    (path / "responses.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _health(path: Path, backbone: str, graphs: int, linear: str = FLA_LINEAR_ATTENTION) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ready",
                "linear_attention": linear,
                "backbone": backbone,
                "backbone_graphs": graphs,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _pair(tmp_path: Path, eager_answers: dict, piecewise_answers: dict, *, graphs=1, backbone=None):
    cases = [_case("team", "choice"), _case("urgent", "noul"), _case("severity", "score")]
    eager = tmp_path / "eager"
    piecewise = tmp_path / "piecewise"
    _write_run(eager, cases, eager_answers)
    _write_run(piecewise, cases, piecewise_answers)
    eager_health = tmp_path / "eager-health.json"
    piecewise_health = tmp_path / "piecewise-health.json"
    _health(eager_health, BACKBONE_HEALTH_EAGER, 0)
    _health(piecewise_health, backbone or BACKBONE_HEALTH_PIECEWISE, graphs)
    return eager, piecewise, eager_health, piecewise_health


def _matching():
    return {
        "team": _answer("choice", "billing", {"billing": 0.8, "technical": 0.2}),
        "urgent": _answer("noul", 0.75, {"false": 0.25, "true": 0.75}),
        "severity": _answer("score", 1.25, {"0": 0.25, "1": 0.25, "2": 0.5}),
    }


def test_matching_answers_pass_and_keep_the_default(tmp_path):
    answers = _matching()
    eager, piecewise, eager_health, piecewise_health = _pair(tmp_path, answers, answers)
    report_path = tmp_path / "out" / "PARITY.json"
    code = main(
        [
            "--eager-run",
            str(eager),
            "--piecewise-run",
            str(piecewise),
            "--eager-health",
            str(eager_health),
            "--piecewise-health",
            str(piecewise_health),
            "--report",
            str(report_path),
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["default_remains"] == "eager"
    assert report["prefill_remains"] == "fla"
    assert report["latency_keep"] is False
    assert report["decision_flips"] == 0
    assert report["device"] is None
    assert report["compute_capability"] is None
    assert report["compared_decisions"] == 3
    assert report["gate"]["max_probability_drift"] == 0.002
    assert report["gate"]["max_score_drift"] == 0.002
    assert report["gate"]["max_flips"] == 0
    assert main(
        [
            "--eager-run",
            str(eager),
            "--piecewise-run",
            str(piecewise),
            "--eager-health",
            str(eager_health),
            "--piecewise-health",
            str(piecewise_health),
            "--report",
            str(report_path),
        ]
    ) == 2


def test_a_choice_flip_or_a_fake_piecewise_bind_fails_the_tip(tmp_path):
    eager_answers = _matching()
    flipped = _matching()
    flipped["team"] = _answer("choice", "technical", {"billing": 0.8, "technical": 0.2})
    eager, piecewise, eager_health, piecewise_health = _pair(tmp_path, eager_answers, flipped)
    report = evaluate(eager, piecewise, eager_health, piecewise_health)
    assert report["passed"] is False
    assert report["decision_flips"] == 1
    assert report["default_remains"] == "eager"

    same = tmp_path / "same"
    eager_b, piecewise_b, eager_health_b, piecewise_health_b = _pair(
        same, eager_answers, eager_answers, graphs=0, backbone=BACKBONE_HEALTH_EAGER
    )
    bound = evaluate(
        eager_b, piecewise_b, eager_health_b, piecewise_health_b, device="NVIDIA L20X"
    )
    assert bound["passed"] is False
    assert bound["device"] == "NVIDIA L20X"
    assert bound["piecewise_backbone"] == BACKBONE_HEALTH_EAGER
    assert any("backbone" in item for item in bound["failures"])


def test_drift_inside_the_gate_passes_and_outside_fails(tmp_path):
    base = _matching()
    near = _matching()
    near["severity"] = _answer("score", 1.251, {"0": 0.251, "1": 0.25, "2": 0.499})
    eager, piecewise, eager_health, piecewise_health = _pair(tmp_path, base, near)
    assert evaluate(eager, piecewise, eager_health, piecewise_health)["passed"] is True

    far = tmp_path / "far"
    moved = _matching()
    moved["urgent"] = _answer("noul", 0.501, {"false": 0.499, "true": 0.501})
    base_urgent = _matching()
    base_urgent["urgent"] = _answer("noul", 0.499, {"false": 0.501, "true": 0.499})
    eager_b, piecewise_b, eager_health_b, piecewise_health_b = _pair(far, base_urgent, moved)
    report = evaluate(eager_b, piecewise_b, eager_health_b, piecewise_health_b)
    assert report["passed"] is False
    assert report["decision_flips"] == 1


def test_manifest_mismatch_fails(tmp_path):
    answers = _matching()
    eager, piecewise, eager_health, piecewise_health = _pair(tmp_path, answers, answers)
    (piecewise / "requests.jsonl").write_text(
        (eager / "requests.jsonl").read_text(encoding="utf-8").replace("checkout", "billing"),
        encoding="utf-8",
    )
    report = evaluate(eager, piecewise, eager_health, piecewise_health)
    assert report["passed"] is False
    assert report["manifest_sha256"] is None
