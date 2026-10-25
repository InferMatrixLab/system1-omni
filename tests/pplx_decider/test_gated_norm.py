"""CPU checks for the linear gated-norm lever. No CUDA extension."""

from __future__ import annotations

import inspect

import pytest
from types import SimpleNamespace

from models.pplx_decider.gated_norm import (
    GATED_NORM_FLA,
    GATED_NORM_WARM_ROWS,
    fused_gated_rms_forward,
    gated_norm_nb,
    install_fused_gated_norm,
)
from models.pplx_decider.gdn import LINEAR_ATTENTION_LAYERS, gva_gated_delta_forward


class Qwen3_5RMSNormGated:
    activation = "silu"
    variance_epsilon = 1e-6

    def forward(self, hidden_states, gate):
        return hidden_states


class Qwen3_5RMSNorm:
    def forward(self, hidden_states):
        return hidden_states


def _language_model():
    layers = []
    for index in range(64):
        block = "full_attention" if index % 4 == 0 else "linear_attention"
        residual = Qwen3_5RMSNorm()
        gated = Qwen3_5RMSNormGated()
        attn = SimpleNamespace(norm=gated, input_layernorm=residual)
        layers.append(SimpleNamespace(block_type=block, linear_attn=attn))
    return SimpleNamespace(layers=layers)


def test_fla_buckets_cover_the_profiled_lengths_and_the_cap():
    assert gated_norm_nb(136 * 48) == 1
    assert gated_norm_nb(2676 * 48) == 2
    assert gated_norm_nb(8192 * 48) == 6
    assert [gated_norm_nb(rows) for rows in GATED_NORM_WARM_ROWS] == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="positive"):
        gated_norm_nb(0)


def test_fused_gated_norm_replaces_48_eager_forwards_and_not_residual():
    language_model = _language_model()
    assert install_fused_gated_norm(language_model) == LINEAR_ATTENTION_LAYERS
    linear = language_model.layers[1].linear_attn
    assert linear.norm.forward.__func__ is fused_gated_rms_forward
    assert linear.norm.pplx_gated_norm == GATED_NORM_FLA
    assert linear.input_layernorm.forward.__func__ is Qwen3_5RMSNorm.forward
    assert language_model.layers[0].linear_attn.norm.forward.__func__ is Qwen3_5RMSNormGated.forward
    source = inspect.getsource(fused_gated_rms_forward)
    assert "layer_norm_gated_fwd(" in source
    assert "is_rms_norm=True" in source
    assert "rms_norm_gated(" not in source
    assert "self.norm(" in inspect.getsource(gva_gated_delta_forward)
    with pytest.raises(RuntimeError, match="not the eager forward"):
        install_fused_gated_norm(language_model)


def test_fused_gated_norm_rejects_a_short_stack():
    layer = SimpleNamespace(
        block_type="linear_attention",
        linear_attn=SimpleNamespace(norm=Qwen3_5RMSNormGated()),
    )
    with pytest.raises(RuntimeError, match="48 linear"):
        install_fused_gated_norm(SimpleNamespace(layers=[layer]))
