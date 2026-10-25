"""CPU checks for the dual linear-attention bind. No CUDA extension."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.pplx_decider.gdn import (
    FLA_LINEAR_ATTENTION,
    FLA_VERSION,
    FLASHINFER_LINEAR_ATTENTION,
    FI_VERSION,
    READY_LINEAR_ATTENTION,
    bound_linear_attention,
    flashinfer_chunk_gated_delta_rule,
    install_flashinfer_gdn_prefill,
    require_bound_linear_attention,
    uses_flashinfer_prefill,
    version_base,
)


def test_versions_and_ready_names():
    assert FLA_VERSION == "0.5.2"
    assert FI_VERSION == "0.6.16.post3"
    assert version_base("0.5.2+cu128") == "0.5.2"
    assert FLA_LINEAR_ATTENTION == "fla-chunk"
    assert FLASHINFER_LINEAR_ATTENTION == "flashinfer+fla+causal_conv1d"
    assert READY_LINEAR_ATTENTION == {FLA_LINEAR_ATTENTION, FLASHINFER_LINEAR_ATTENTION}


def _fla_chunk():
    return None


def _other_chunk():
    return None


def test_capability_8_9_stays_on_the_fla_chunk():
    assert uses_flashinfer_prefill((9, 0))
    assert not uses_flashinfer_prefill((8, 9))
    modeling = SimpleNamespace(
        torch_chunk_gated_delta_rule=_fla_chunk,
        causal_conv1d_fn=object(),
    )
    fla_chunk = _fla_chunk
    assert bound_linear_attention(modeling, fla_chunk=fla_chunk) == FLA_LINEAR_ATTENTION
    assert require_bound_linear_attention(modeling, fla_chunk=fla_chunk) == FLA_LINEAR_ATTENTION


def test_capability_9_0_flashinfer_bind_is_ready():
    modeling = SimpleNamespace(
        torch_chunk_gated_delta_rule=object(),
        causal_conv1d_fn=object(),
        chunk_gated_delta_rule=object(),
    )
    install_flashinfer_gdn_prefill(modeling)
    assert modeling.torch_chunk_gated_delta_rule is flashinfer_chunk_gated_delta_rule
    assert modeling.chunk_gated_delta_rule is flashinfer_chunk_gated_delta_rule
    assert require_bound_linear_attention(modeling) == FLASHINFER_LINEAR_ATTENTION


def test_flashinfer_install_refuses_a_missing_conv():
    modeling = SimpleNamespace(torch_chunk_gated_delta_rule=object(), causal_conv1d_fn=None)
    with pytest.raises(RuntimeError, match="causal_conv1d_fn"):
        install_flashinfer_gdn_prefill(modeling)


def test_torch_chunk_is_not_ready():
    modeling = SimpleNamespace(torch_chunk_gated_delta_rule=object(), causal_conv1d_fn=object())
    with pytest.raises(RuntimeError, match="not ready"):
        require_bound_linear_attention(modeling, fla_chunk=_other_chunk)


def test_flashinfer_wrapper_rejects_a_batch_and_a_state():
    class Batch:
        shape = (2, 4, 8, 16)

    with pytest.raises(RuntimeError, match="one sequence"):
        flashinfer_chunk_gated_delta_rule(Batch(), Batch(), Batch())

    class One:
        shape = (1, 4, 8, 16)

    with pytest.raises(RuntimeError, match="recurrent state"):
        flashinfer_chunk_gated_delta_rule(One(), One(), One(), initial_state=object())
