"""CPU checks for the residual RMSNorm plan. No weights and no torch."""

from __future__ import annotations

import pytest

from models.pplx_decider.rmsnorm import (
    FLA_RMS_LENGTH_STABLE_FEATURES,
    FLA_RMS_MIN_TOKENS,
    dispatch_residual_rms_norm,
    fused_residual_rms_norm,
    prepare_residual_rms_norm,
    residual_rms_norm_modules,
)


class _Weight:
    def __init__(self, features: int):
        self.ndim = 1
        self.shape = (features,)


class Qwen3_5RMSNorm:
    def __init__(self, features: int = 5120, eps: float = 1e-6):
        self.weight = _Weight(features)
        self.eps = eps


class Qwen3_5RMSNormGated(Qwen3_5RMSNorm):
    pass


class _Layer:
    def __init__(self, features: int = 5120):
        self.input_layernorm = Qwen3_5RMSNorm(features)
        self.post_attention_layernorm = Qwen3_5RMSNorm(features)
        self.q_norm = Qwen3_5RMSNorm(256)
        self.linear_attn_norm = Qwen3_5RMSNormGated(128)


class _Model:
    def __init__(self, layers: int = 2, features: int = 5120):
        self.layers = [_Layer(features) for _ in range(layers)]
        self.norm = Qwen3_5RMSNorm(features)


class _Tensor:
    def __init__(self, shape, contiguous=True):
        self.shape = tuple(shape)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return _Tensor(self.shape, True)

    def reshape(self, *spec):
        if len(spec) == 1 and isinstance(spec[0], tuple):
            spec = spec[0]
        if len(spec) == 2 and spec[0] == -1:
            lead = 1
            for dim in self.shape[:-1]:
                lead *= dim
            return _Tensor((lead, spec[1]), True)
        return _Tensor(tuple(spec), True)


def test_prepare_accepts_the_wide_zero_centered_norm():
    assert prepare_residual_rms_norm(Qwen3_5RMSNorm()) == 5120
    assert FLA_RMS_LENGTH_STABLE_FEATURES == 512


def test_prepare_rejects_a_short_norm_a_gated_norm_and_a_second_fusion():
    with pytest.raises(RuntimeError, match="re-autotunes with length"):
        prepare_residual_rms_norm(Qwen3_5RMSNorm(256))
    with pytest.raises(RuntimeError, match="only replaces Qwen3_5RMSNorm"):
        prepare_residual_rms_norm(Qwen3_5RMSNormGated())
    fused = Qwen3_5RMSNorm()
    fused._pplx_rms_scale = object()
    with pytest.raises(RuntimeError, match="already fused"):
        prepare_residual_rms_norm(fused)
    bad_eps = Qwen3_5RMSNorm()
    bad_eps.eps = 0.0
    with pytest.raises(RuntimeError, match="positive float"):
        prepare_residual_rms_norm(bad_eps)


def test_collection_skips_query_and_gated_norms():
    model = _Model(layers=2)
    found = residual_rms_norm_modules(model)
    assert found == [
        model.layers[0].input_layernorm,
        model.layers[0].post_attention_layernorm,
        model.layers[1].input_layernorm,
        model.layers[1].post_attention_layernorm,
        model.norm,
    ]
    assert model.layers[0].q_norm not in found
    assert model.layers[0].linear_attn_norm not in found


def test_collection_rejects_a_missing_final_norm():
    model = _Model(layers=1)
    model.norm = None
    with pytest.raises(RuntimeError, match="final RMSNorm"):
        residual_rms_norm_modules(model)


def test_forward_calls_the_length_stable_kernel_and_restores_the_shape():
    norm = Qwen3_5RMSNorm()
    norm._pplx_rms_scale = object()
    seen = {}

    def layer_norm_fwd(flat, scale, bias, eps, is_rms_norm):
        seen["flat"] = flat.shape
        seen["scale"] = scale
        seen["bias"] = bias
        seen["eps"] = eps
        seen["is_rms_norm"] = is_rms_norm
        return _Tensor(flat.shape), None, None, None

    x = _Tensor((1, 136, 5120), contiguous=False)
    y = fused_residual_rms_norm(norm, x, layer_norm_fwd)
    assert seen == {
        "flat": (136, 5120),
        "scale": norm._pplx_rms_scale,
        "bias": None,
        "eps": 1e-6,
        "is_rms_norm": True,
    }
    assert y.shape == (1, 136, 5120)


def test_median_lengths_keep_the_original_forward():
    assert FLA_RMS_MIN_TOKENS == 512
    norm = Qwen3_5RMSNorm()
    norm._pplx_rms_scale = object()
    eager = []

    def original(x):
        eager.append(x.shape[-2])
        return x

    norm._pplx_rms_original = original
    fused = {"calls": 0}

    def layer_norm_fwd(flat, scale, bias, eps, is_rms_norm):
        fused["calls"] += 1
        return _Tensor(flat.shape), None, None, None

    for tokens in (136, 512):
        y = dispatch_residual_rms_norm(
            norm, _Tensor((1, tokens, 5120)), layer_norm_fwd, FLA_RMS_MIN_TOKENS
        )
        assert y.shape == (1, tokens, 5120)
    assert eager == [136, 512]
    assert fused["calls"] == 0
    for tokens in (513, 2676):
        y = dispatch_residual_rms_norm(
            norm, _Tensor((1, tokens, 5120)), layer_norm_fwd, FLA_RMS_MIN_TOKENS
        )
        assert y.shape == (1, tokens, 5120)
    assert fused["calls"] == 2


def test_short_path_refuses_a_missing_original_forward():
    norm = Qwen3_5RMSNorm()
    with pytest.raises(RuntimeError, match="eager forward was not saved"):
        dispatch_residual_rms_norm(norm, _Tensor((1, 136, 5120)), None, FLA_RMS_MIN_TOKENS)
