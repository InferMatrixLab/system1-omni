"""CPU checks for the text-MLP SwiGLU lever. No CUDA extension."""

from __future__ import annotations

import inspect

import pytest
from types import SimpleNamespace

from models.pplx_decider.engine import PplxExecutor
from models.pplx_decider.swiglu import (
    MEDIAN_SWIGLU_TOKENS,
    MLP_SWIGLU,
    TEXT_MLP_LAYERS,
    install_swiglu_mlp,
    swiglu_mlp_forward,
)


class Qwen3_5MLP:
    def __init__(self) -> None:
        self.gate_proj = SimpleNamespace()
        self.up_proj = SimpleNamespace()
        self.down_proj = SimpleNamespace()

    def forward(self, x):
        return x


def _language_model(count: int = TEXT_MLP_LAYERS):
    return SimpleNamespace(layers=[SimpleNamespace(mlp=Qwen3_5MLP()) for _ in range(count)])


def test_swiglu_replaces_every_text_mlp_and_not_a_second_time():
    language_model = _language_model()
    assert install_swiglu_mlp(language_model) == TEXT_MLP_LAYERS
    mlp = language_model.layers[0].mlp
    assert mlp.forward.__func__ is swiglu_mlp_forward
    assert mlp.pplx_mlp == MLP_SWIGLU
    source = inspect.getsource(swiglu_mlp_forward)
    assert "swiglu_fwd(" in source
    assert "self.down_proj(" in source
    assert "self.gate_proj(" in source
    assert "self.up_proj(" in source
    assert "swiglu_linear(" not in source
    assert "SwiGLUFunction" not in source
    assert "torch.cat" not in source
    assert MEDIAN_SWIGLU_TOKENS == 136
    with pytest.raises(RuntimeError, match="not the eager forward"):
        install_swiglu_mlp(language_model)


def test_swiglu_rejects_a_short_stack():
    with pytest.raises(RuntimeError, match="64 text"):
        install_swiglu_mlp(_language_model(1))


def test_engine_does_not_install_the_rejected_gated_norm():
    source = inspect.getsource(PplxExecutor.__init__)
    assert "install_fused_gated_norm(" not in source
    assert "warm_fused_gated_norm(" not in source
    assert "install_swiglu_mlp(" in source
    assert "warm_swiglu_mlp(" in source
    assert "GATED_NORM_FLA" not in source
