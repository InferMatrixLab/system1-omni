"""CPU checks for the fused gate-up plan. No weights and no torch."""

from __future__ import annotations

import pytest

from models.pplx_decider.mlp import mlp_preference, prepare_fused_gate_up


class _Proj:
    def __init__(self, in_features, out_features, bias=None):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias


class _MLP:
    def __init__(self):
        self.gate_proj = _Proj(5120, 17408)
        self.up_proj = _Proj(5120, 17408)
        self.down_proj = _Proj(17408, 5120)
        self.intermediate_size = 17408
        self.act_fn = object()


def test_cuda_fuses_gate_up_unless_the_primary_mlp_is_requested():
    assert mlp_preference("cuda", None) == "fused-gate-up"
    assert mlp_preference("cuda", "fused-gate-up") == "fused-gate-up"
    assert mlp_preference("cuda", "default") == "default"
    assert mlp_preference("cpu", None) == "default"
    assert mlp_preference("cpu", "fused-gate-up") == "default"
    with pytest.raises(RuntimeError, match="fused-gate-up or default"):
        mlp_preference("cuda", "compile")


def test_prepare_accepts_the_unbiased_text_mlp():
    assert prepare_fused_gate_up(_MLP()) == 17408


def test_prepare_rejects_a_second_fusion_and_a_biased_projection():
    mlp = _MLP()
    mlp.fused_gate_up = object()
    with pytest.raises(RuntimeError, match="already fused"):
        prepare_fused_gate_up(mlp)
    biased = _MLP()
    biased.gate_proj.bias = object()
    with pytest.raises(RuntimeError, match="unbiased"):
        prepare_fused_gate_up(biased)


def test_prepare_rejects_a_shape_mismatch():
    mlp = _MLP()
    mlp.up_proj = _Proj(5120, 2048)
    with pytest.raises(RuntimeError, match="differ"):
        prepare_fused_gate_up(mlp)
    missing = _MLP()
    missing.down_proj = None
    with pytest.raises(RuntimeError, match="down_proj"):
        prepare_fused_gate_up(missing)
