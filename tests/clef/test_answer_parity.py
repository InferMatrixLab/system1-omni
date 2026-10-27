"""Offline contract for the FLA versus FlashInfer answer-parity gate."""

import json
from pathlib import Path

from models.clef.answer_parity import (
    FLASHINFER_LINEAR_ATTENTION,
    FLA_LINEAR_ATTENTION,
    evaluate,
    main,
)


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


def _health(path: Path, linear: str) -> None:
    path.write_text(
        json.dumps({"status": "ready", "linear_attention": linear}) + "\n",
        encoding="utf-8",
    )


def _pair(tmp_path: Path, fla_answers: dict, flashinfer_answers: dict, *, fi_linear=None):
    cases = [_case("team", "choice"), _case("urgent", "noul"), _case("severity", "score")]
    fla = tmp_path / "fla"
    flash = tmp_path / "flashinfer"
    _write_run(fla, cases, fla_answers)
    _write_run(flash, cases, flashinfer_answers)
    fla_health = tmp_path / "fla-health.json"
    flash_health = tmp_path / "flash-health.json"
    _health(fla_health, FLA_LINEAR_ATTENTION)
    _health(flash_health, fi_linear or FLASHINFER_LINEAR_ATTENTION)
    return fla, flash, fla_health, flash_health


def _matching():
    return {
        "team": _answer("choice", "billing", {"billing": 0.8, "technical": 0.2}),
        "urgent": _answer("noul", 0.75, {"false": 0.25, "true": 0.75}),
        "severity": _answer("score", 1.25, {"0": 0.25, "1": 0.25, "2": 0.5}),
    }


def test_matching_answers_pass_and_keep_the_default(tmp_path):
    answers = _matching()
    fla, flash, fla_health, flash_health = _pair(tmp_path, answers, answers)
    report_path = tmp_path / "out" / "PARITY.json"
    code = main(
        [
            "--fla-run",
            str(fla),
            "--flashinfer-run",
            str(flash),
            "--fla-health",
            str(fla_health),
            "--flashinfer-health",
            str(flash_health),
            "--report",
            str(report_path),
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["default_remains"] == "fla"
    assert report["latency_keep"] is False
    assert report["decision_flips"] == 0
    assert report["device"] is None
    assert report["compute_capability"] is None
    assert report["compared_decisions"] == 3
    assert main(
        [
            "--fla-run",
            str(fla),
            "--flashinfer-run",
            str(flash),
            "--fla-health",
            str(fla_health),
            "--flashinfer-health",
            str(flash_health),
            "--report",
            str(report_path),
        ]
    ) == 2


def test_a_choice_flip_or_the_wrong_bind_fails_the_tip(tmp_path):
    fla_answers = _matching()
    flipped = _matching()
    flipped["team"] = _answer("choice", "technical", {"billing": 0.8, "technical": 0.2})
    fla, flash, fla_health, flash_health = _pair(tmp_path, fla_answers, flipped)
    report = evaluate(fla, flash, fla_health, flash_health)
    assert report["passed"] is False
    assert report["decision_flips"] == 1
    assert report["default_remains"] == "fla"

    same = tmp_path / "same"
    fla_b, flash_b, fla_health_b, flash_health_b = _pair(
        same, fla_answers, fla_answers, fi_linear=FLA_LINEAR_ATTENTION
    )
    bound = evaluate(fla_b, flash_b, fla_health_b, flash_health_b, device="NVIDIA L20X")
    assert bound["passed"] is False
    assert bound["device"] == "NVIDIA L20X"
    assert bound["flashinfer_linear_attention"] == FLA_LINEAR_ATTENTION
    assert any("linear_attention" in item for item in bound["failures"])


def test_drift_inside_the_gate_passes_and_outside_fails(tmp_path):
    base = _matching()
    near = _matching()
    near["severity"] = _answer("score", 1.251, {"0": 0.251, "1": 0.25, "2": 0.499})
    fla, flash, fla_health, flash_health = _pair(tmp_path, base, near)
    assert evaluate(fla, flash, fla_health, flash_health)["passed"] is True

    far = tmp_path / "far"
    moved = _matching()
    moved["urgent"] = _answer("noul", 0.501, {"false": 0.499, "true": 0.501})
    base_urgent = _matching()
    base_urgent["urgent"] = _answer("noul", 0.499, {"false": 0.501, "true": 0.499})
    fla_b, flash_b, fla_health_b, flash_health_b = _pair(far, base_urgent, moved)
    report = evaluate(fla_b, flash_b, fla_health_b, flash_health_b)
    assert report["passed"] is False
    assert report["decision_flips"] == 1


def test_manifest_mismatch_fails(tmp_path):
    answers = _matching()
    fla, flash, fla_health, flash_health = _pair(tmp_path, answers, answers)
    (flash / "requests.jsonl").write_text(
        (fla / "requests.jsonl").read_text(encoding="utf-8").replace("checkout", "billing"),
        encoding="utf-8",
    )
    report = evaluate(fla, flash, fla_health, flash_health)
    assert report["passed"] is False
    assert report["manifest_sha256"] is None
