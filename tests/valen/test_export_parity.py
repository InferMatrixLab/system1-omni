"""Parity-gate tests for recipe/valen/verify_export_parity.py (spec 02, E2).

CPU-only checks of the export-parity tool's observable contract: the
merge-drift measurement (PEFT unmerged vs merged BF16 semantics, the
research-001 method) on hand-computed tensors, the drift aggregation, the
original-vs-exported case comparison verdicts, and the manifest parity-slot
filling. The full parity gate against the real pinned artifacts is env-gated
on VALEN_CKPT / VALEN_BASE / VALEN_SRC plus CUDA, following the test_export.py
precedent.

PYTHONPATH=src python -m pytest tests/valen -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RECIPE_VALEN = HERE.parents[1] / "recipe" / "valen"
sys.path.insert(0, str(RECIPE_VALEN))

import verify_export_parity as tool  # noqa: E402


# --- the BF16 ulp ruler (research 001 reports spacing at max |y|) ---

def test_bf16_ulp_at_known_magnitudes():
    # bf16 keeps 8 significant bits, so values in [2^e, 2^(e+1)) sit on a
    # 2^(e-7) grid: spacing 2^-7 at 1.0, 2^-6 on [2, 4), 2^-5 on [4, 8).
    assert tool.bf16_ulp_at(1.0) == 2.0**-7
    assert tool.bf16_ulp_at(1.0078125) == 2.0**-7
    assert tool.bf16_ulp_at(3.015625) == 2.0**-6
    assert tool.bf16_ulp_at(6.0) == 2.0**-5
    assert tool.bf16_ulp_at(0.25) == 2.0**-9
    assert tool.bf16_ulp_at(16.0) == 2.0**-3


# --- per-module output drift: unmerged (PEFT) vs merged BF16 GEMM ---

def test_module_output_drift_zero_b_adapter_is_bit_exact():
    """An untrained LoRA (zero B) leaves both outputs bit-identical."""
    import torch

    torch.manual_seed(0)
    base = torch.randn(6, 8).to(torch.bfloat16)
    merged = tool.merge_lora_weight(base, torch.randn(4, 8), torch.zeros(6, 4), 2.0)
    probe = torch.randn(16, 8).to(torch.bfloat16)
    metrics = tool.module_output_drift(
        base, torch.randn(4, 8), torch.zeros(6, 4), merged, 2.0, probe
    )
    assert metrics["max_abs_diff"] == 0.0
    assert metrics["bitexact_fraction"] == 1.0
    assert metrics["max_rel_diff"] == 0.0


def test_module_output_drift_hand_computed_one_ulp():
    """The rounding-order split, hand-computed on a 1x1 module.

    base W = 1.0, probe x = 3.0 (both bf16), delta weight 2*(B@A) = 0.0038:

    - merged: W' = bf16(1.0038) = 1.0 (below the 1.00390625 midpoint), so
      y_merged = 3.0 exactly;
    - unmerged (PEFT): y = bf16(3.0 + 3*0.0038) = bf16(3.0114) = 3.015625
      (above the 3.0078125 midpoint of the [2, 4) grid).

    So the outputs differ by exactly one bf16 ulp of the output.
    """
    import torch

    base = torch.tensor([[1.0]], dtype=torch.bfloat16)
    lora_a = torch.tensor([[0.0019]], dtype=torch.float32)
    lora_b = torch.tensor([[1.0]], dtype=torch.float32)
    merged = tool.merge_lora_weight(base, lora_a, lora_b, 2.0)
    assert merged.item() == 1.0  # the weight rounds back to the base

    probe = torch.tensor([[3.0]], dtype=torch.bfloat16)
    metrics = tool.module_output_drift(base, lora_a, lora_b, merged, 2.0, probe)
    assert metrics["max_abs_diff"] == pytest.approx(0.015625)
    assert metrics["max_output_abs"] == pytest.approx(3.015625)
    assert metrics["bf16_ulp"] == pytest.approx(0.015625)
    assert metrics["ulp_ratio"] == pytest.approx(1.0)
    assert metrics["max_rel_diff"] == pytest.approx(0.015625 / 3.015625)
    assert metrics["bitexact_fraction"] == 0.0


def test_module_output_drift_matches_peft_layer_forward():
    """The unmerged arm must reproduce peft 0.18.1's Linear.forward on the
    same tensors: bf16 base GEMM, FP32 LoRA path, one final rounding."""

    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(3)
    linear = nn.Linear(8, 6, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(6, 8).to(torch.bfloat16))

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = linear

        def forward(self, x):
            return self.lin(x)

    model = get_peft_model(
        Wrapped(),
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["lin"], bias="none"),
    )
    lora_linear = model.base_model.model.lin
    with torch.no_grad():
        lora_linear.lora_A["default"].weight.copy_(torch.randn(4, 8))
        lora_linear.lora_B["default"].weight.copy_(torch.randn(6, 4) * 0.05)
    scaling = lora_linear.scaling["default"]
    base_weight = linear.weight.detach().clone()
    lora_a = lora_linear.lora_A["default"].weight.detach().clone()
    lora_b = lora_linear.lora_B["default"].weight.detach().clone()
    merged = tool.merge_lora_weight(base_weight, lora_a, lora_b, scaling)
    probe = torch.randn(32, 8).to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        peft_unmerged = model(probe).detach()

    metrics = tool.module_output_drift(base_weight, lora_a, lora_b, merged, scaling, probe)
    assert metrics["max_abs_diff"] > 0.0  # the merge really does drift
    assert metrics["bitexact_fraction"] < 1.0
    # Recomputing the unmerged arm by hand must agree with peft bit-exactly:
    # the drift tool's semantics are the reference's, not a re-invention.
    y_base = torch.matmul(probe, base_weight.t())
    delta = scaling * ((probe.float() @ lora_a.t()) @ lora_b.t())
    y_unmerged = (y_base.float() + delta).to(torch.bfloat16)
    assert torch.equal(y_unmerged, peft_unmerged.to(torch.bfloat16))


# --- drift aggregation across the adapted modules ---

def drift_metrics(max_abs, max_out, ulp, bitexact):
    return {
        "max_abs_diff": max_abs,
        "max_output_abs": max_out,
        "max_rel_diff": max_abs / max_out,
        "bf16_ulp": ulp,
        "ulp_ratio": max_abs / ulp,
        "bitexact_fraction": bitexact,
    }


def test_aggregate_drift_reports_max_mean_and_worst_modules():
    records = [
        ("layers.0.mlp.down_proj.weight", drift_metrics(0.03125, 6.0, 0.03125, 0.5)),
        ("layers.0.mlp.gate_proj.weight", drift_metrics(0.015625, 3.0, 0.015625, 0.6)),
        ("layers.3.self_attn.q_proj.weight", drift_metrics(0.03125, 8.0, 0.0625, 0.4)),
    ]
    aggregate = tool.aggregate_drift(records)
    assert aggregate["modules"] == 3
    assert aggregate["max_abs_diff"] == pytest.approx(0.03125)
    assert aggregate["mean_abs_diff"] == pytest.approx((0.03125 + 0.015625 + 0.03125) / 3)
    assert aggregate["max_rel_diff"] == pytest.approx(0.03125 / 6.0)
    assert aggregate["mean_rel_diff"] == pytest.approx(
        (0.03125 / 6.0 + 0.015625 / 3.0 + 0.03125 / 8.0) / 3
    )
    assert aggregate["max_ulp_ratio"] == pytest.approx(1.0)
    assert aggregate["mean_bitexact_fraction"] == pytest.approx(0.5)
    # Ties break deterministically by module name.
    assert aggregate["worst_abs_module"] == "layers.0.mlp.down_proj.weight"
    assert aggregate["worst_rel_module"] == "layers.0.mlp.down_proj.weight"


def test_aggregate_drift_requires_records():
    with pytest.raises(ValueError, match="no drift records"):
        tool.aggregate_drift([])


# --- the original-vs-exported case comparison (the gate's verdict) ---

def case_side(probs, keys=("refund", "wait"), logical=48, compute=48):
    return {
        "questions": {"refund": {"keys": list(keys), "probabilities": probs}},
        "logical_tokens": logical,
        "compute_tokens": compute,
    }


def test_compare_case_identical_stacks_pass():
    original = case_side([0.5159631, 0.4840369])
    verdict = tool.compare_case("text", original, case_side([0.5159631, 0.4840369]))
    assert verdict["exact_answer_agreement"] is True
    assert verdict["max_probability_delta"] == 0.0
    assert verdict["probability_tolerance_pass"] is True
    assert verdict["tokens_agree"] is True
    assert verdict["passed"] is True
    assert verdict["problems"] == []


def test_compare_case_delta_within_tolerance_passes():
    original = case_side([0.5159631, 0.4840369])
    drifted = case_side([0.5159631 + 5e-7, 0.4840369 - 5e-7])
    verdict = tool.compare_case("text", original, drifted)
    assert verdict["passed"] is True
    assert verdict["max_probability_delta"] == pytest.approx(5e-7)
    assert verdict["exact_answer_agreement"] is True


def test_compare_case_delta_beyond_tolerance_fails():
    original = case_side([0.5159631, 0.4840369])
    drifted = case_side([0.5159631 + 2e-6, 0.4840369 - 2e-6])
    verdict = tool.compare_case("text", original, drifted)
    assert verdict["exact_answer_agreement"] is True  # argmax still agrees
    assert verdict["probability_tolerance_pass"] is False
    assert verdict["passed"] is False
    assert any("tolerance" in problem for problem in verdict["problems"])


def test_compare_case_flipped_answer_fails():
    original = case_side([0.5159631, 0.4840369])
    flipped = case_side([0.4840369, 0.5159631])
    verdict = tool.compare_case("text", original, flipped)
    assert verdict["exact_answer_agreement"] is False
    assert verdict["passed"] is False


def test_compare_case_candidate_order_mismatch_fails():
    original = case_side([0.5159631, 0.4840369])
    reordered = case_side([0.4840369, 0.5159631], keys=("wait", "refund"))
    verdict = tool.compare_case("text", original, reordered)
    assert verdict["passed"] is False
    assert any("candidate order" in problem for problem in verdict["problems"])


def test_compare_case_missing_question_fails():
    original = case_side([0.5159631, 0.4840369])
    emptied = case_side([0.5159631, 0.4840369])
    emptied["questions"] = {}
    verdict = tool.compare_case("text", original, emptied)
    assert verdict["passed"] is False


def test_compare_case_token_accounting_must_agree():
    original = case_side([0.5159631, 0.4840369])
    retokenized = case_side([0.5159631, 0.4840369], compute=49)
    verdict = tool.compare_case("text", original, retokenized)
    assert verdict["tokens_agree"] is False
    assert verdict["passed"] is False


# --- the manifest parity slot ---

def test_build_parity_record_covers_the_documented_slot_shape():
    record = tool.build_parity_record(
        status="passed",
        device="cuda:0",
        hardware="NVIDIA GeForce RTX 5060 Laptop GPU (sm_120)",
        cases=[{"case": "text", "passed": True}],
        exact_answer_agreement=True,
        max_probability_delta=2.7e-08,
        tolerance=1e-6,
        drift={"modules": 186, "max_abs_diff": 0.03125},
        evidence="recipe/valen/reference.md export-parity block",
    )
    # Every key the manifest's parity_slot_shape documents must be present.
    assert set(record) >= {
        "status", "device", "cases", "exact_answer_agreement",
        "max_probability_delta", "merge_drift", "evidence",
    }
    assert record["status"] == "passed"
    assert record["max_probability_delta"] == 2.7e-08
    assert record["tolerance"] == 1e-6
    assert record["merge_drift"] == {"modules": 186, "max_abs_diff": 0.03125}


def test_build_parity_record_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        tool.build_parity_record(
            status="maybe",
            device="cuda:0",
            hardware="gpu",
            cases=[],
            exact_answer_agreement=True,
            max_probability_delta=0.0,
            tolerance=1e-6,
            drift={},
            evidence="",
        )


def minimal_manifest(path, parity=None):
    manifest = {
        "format": "valen-checkpoint-export/1",
        "parity": parity,
        "parity_slot_shape": {"status": "passed | failed"},
    }
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parity_record(status="passed", delta=1e-8):
    return tool.build_parity_record(
        status=status,
        device="cuda:0",
        hardware="gpu",
        cases=[{"case": "text", "passed": status == "passed"}],
        exact_answer_agreement=status == "passed",
        max_probability_delta=delta,
        tolerance=1e-6,
        drift={"modules": 186},
        evidence="recipe/valen/reference.md",
    )


def test_fill_parity_slot_replaces_null(tmp_path):
    manifest_path = tmp_path / "valen_export.json"
    minimal_manifest(manifest_path)
    record = parity_record()
    updated = tool.fill_parity_slot(manifest_path, record)
    assert updated["parity"] == record
    reread = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert reread["parity"] == record
    # The documented formatting (sorted, indent 2, trailing newline) is kept.
    assert manifest_path.read_text(encoding="utf-8").endswith("}\n")


def test_fill_parity_slot_is_idempotent_for_the_same_record(tmp_path):
    manifest_path = tmp_path / "valen_export.json"
    minimal_manifest(manifest_path)
    record = parity_record()
    tool.fill_parity_slot(manifest_path, record)
    before = manifest_path.read_text(encoding="utf-8")
    tool.fill_parity_slot(manifest_path, record)
    assert manifest_path.read_text(encoding="utf-8") == before


def test_fill_parity_slot_refuses_to_silently_overwrite(tmp_path):
    manifest_path = tmp_path / "valen_export.json"
    minimal_manifest(manifest_path)
    tool.fill_parity_slot(manifest_path, parity_record(delta=1e-8))
    with pytest.raises(ValueError, match="parity slot already holds"):
        tool.fill_parity_slot(manifest_path, parity_record(delta=5e-3))
    # The earlier record survives the refusal.
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["parity"][
        "max_probability_delta"
    ] == 1e-8


# --- the full parity gate against the real pinned artifacts ---
# Set VALEN_CKPT, VALEN_BASE and VALEN_SRC (the recipe reference.md artifact
# exports) on a CUDA host to run: it exports a fresh artifact set, then runs
# the two serial model loads (original checkpoint stack, exported stack),
# the 186-module merge-drift measurement and the manifest parity fill.
#
# The gate's exit status is evidence, not a regression invariant: on the
# pinned artifacts the declared 1e-6 tolerance is exceeded by the measured
# merge drift (recorded honestly in recipe/valen/reference.md), so the test
# asserts the verdict's internal consistency — the recorded status, deltas,
# agreement and drift must match what the tool actually measured, and the
# exact-answer agreement of the pinned artifacts must hold.

def test_real_artifact_export_parity_gate(tmp_path):
    import subprocess
    import torch

    env_vars = ("VALEN_CKPT", "VALEN_BASE", "VALEN_SRC")
    if not all(os.environ.get(name) for name in env_vars):
        pytest.skip("set VALEN_CKPT, VALEN_BASE and VALEN_SRC to run")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    import export_checkpoint as exporter

    export_dir = tmp_path / "export"
    exporter.export(
        Path(os.environ["VALEN_CKPT"]), Path(os.environ["VALEN_BASE"]), export_dir
    )

    result = subprocess.run(
        [
            sys.executable,
            str(RECIPE_VALEN / "verify_export_parity.py"),
            "--valen-source", os.environ["VALEN_SRC"],
            "--checkpoint", os.environ["VALEN_CKPT"],
            "--base", os.environ["VALEN_BASE"],
            "--export", str(export_dir),
            "--device", "cuda",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode in (0, 1), result.stdout[-4000:] + result.stderr[-4000:]

    manifest = exporter.validate_export(export_dir)
    parity = manifest["parity"]
    assert parity["device"].startswith("cuda")
    assert parity["tolerance"] == 1e-6
    assert {case["case"] for case in parity["cases"]} == {"text", "image"}
    # The pinned artifacts keep agreeing on the discrete answers.
    assert parity["exact_answer_agreement"] is True
    assert all(case["exact_answer_agreement"] for case in parity["cases"])
    # The recorded verdict is internally consistent with the measurements.
    assert parity["max_probability_delta"] == max(
        case["max_probability_delta"] for case in parity["cases"]
    )
    expected_status = (
        "passed"
        if all(case["passed"] for case in parity["cases"])
        else "failed"
    )
    assert parity["status"] == expected_status
    for case in parity["cases"]:
        case_passes = (
            case["exact_answer_agreement"]
            and case["max_probability_delta"] <= parity["tolerance"]
        )
        assert case["passed"] is case_passes
        if not case_passes:
            assert any("tolerance" in problem for problem in case["problems"])
    assert parity["merge_drift"]["modules"] == 186
    aggregate = parity["merge_drift"]["aggregate"]
    assert aggregate["modules"] == 186
    assert 0.0 < aggregate["max_abs_diff"]
    assert 0.0 < aggregate["max_ulp_ratio"] <= 1.0
