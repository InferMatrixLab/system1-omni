"""CPU checks for the dual linear-attention bind. No CUDA extension."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.pplx_decider.gdn import (
    FLA_LINEAR_ATTENTION,
    FLA_VERSION,
    FLASHINFER_LINEAR_ATTENTION,
    LINEAR_ATTENTION_LAYERS,
    SHORT_FLASHINFER_LINEAR_ATTENTION,
    SHORT_FLASHINFER_TOKENS,
    FI_PIN,
    FI_VERSION,
    READY_LINEAR_ATTENTION,
    bound_linear_attention,
    flashinfer_chunk_gated_delta_rule,
    install_flashinfer_gdn_prefill,
    install_short_flashinfer_prefill,
    require_bound_linear_attention,
    GDN_USE_CP,
    cu_seqlens_key,
    cp_heuristic_matches,
    gdn_prefill_mode,
    reuse_cu_seqlens_slot,
    resolve_flashinfer,
    use_short_flashinfer,
    uses_flashinfer_prefill,
    version_base,
)


def test_versions_and_ready_names():
    assert FLA_VERSION == "0.5.2"
    assert FI_VERSION == "0.6.16.post3"
    assert FI_PIN == "flashinfer-python==0.6.16.post3"
    assert version_base("0.5.2+cu128") == "0.5.2"
    assert FLA_LINEAR_ATTENTION == "fla-chunk"
    assert FLASHINFER_LINEAR_ATTENTION == "flashinfer+fla+causal_conv1d"
    assert READY_LINEAR_ATTENTION == {
        FLA_LINEAR_ATTENTION,
        FLASHINFER_LINEAR_ATTENTION,
        SHORT_FLASHINFER_LINEAR_ATTENTION,
    }


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


def test_capability_9_0_stays_ready_when_flashinfer_is_absent():
    version, status = resolve_flashinfer(None, imported=False)
    assert version is None
    assert status == "absent"
    version, status = resolve_flashinfer(FI_VERSION, imported=False)
    assert version is None
    assert status == "import-failed"
    version, status = resolve_flashinfer(FI_VERSION, imported=True)
    assert version == FI_VERSION
    assert status == "bound"
    with pytest.raises(RuntimeError, match="0.6.16.post3"):
        resolve_flashinfer("0.6.15", imported=True)


def test_gdn_prefill_defaults_to_all_sequence_flashinfer():
    assert gdn_prefill_mode(None) == "flashinfer"
    assert gdn_prefill_mode("flashinfer") == "flashinfer"
    assert gdn_prefill_mode("fla") == "fla"
    assert gdn_prefill_mode("short") == "short"
    with pytest.raises(RuntimeError, match="flashinfer, fla, or short"):
        gdn_prefill_mode("512")


def test_cu_seqlens_ring_is_one_slot_per_linear_layer():
    assert LINEAR_ATTENTION_LAYERS == 48
    assert cu_seqlens_key(None, 136) == (-1, 136)
    assert cu_seqlens_key(0, 2676) == (0, 2676)
    pool: dict[tuple[int, int], list[object]] = {}
    cursors: dict[tuple[int, int], int] = {}
    calls: list[int] = []

    def allocate() -> object:
        calls.append(1)
        return object()

    first_round = [
        reuse_cu_seqlens_slot(pool, cursors, (0, 136), LINEAR_ATTENTION_LAYERS, allocate)
        for _ in range(LINEAR_ATTENTION_LAYERS)
    ]
    assert len({id(item) for item in first_round}) == LINEAR_ATTENTION_LAYERS
    assert len(calls) == LINEAR_ATTENTION_LAYERS
    wrapped = reuse_cu_seqlens_slot(pool, cursors, (0, 136), LINEAR_ATTENTION_LAYERS, allocate)
    assert wrapped is first_round[0]
    assert len(calls) == LINEAR_ATTENTION_LAYERS
    other = reuse_cu_seqlens_slot(pool, cursors, (0, 2676), LINEAR_ATTENTION_LAYERS, allocate)
    assert other not in first_round
    assert len(calls) == LINEAR_ATTENTION_LAYERS * 2


def test_short_flashinfer_keeps_the_tail_and_a_batch_on_fla():
    assert SHORT_FLASHINFER_TOKENS == 256
    assert use_short_flashinfer(1, 136)
    assert use_short_flashinfer(1, 256)
    assert not use_short_flashinfer(1, 257)
    assert not use_short_flashinfer(1, 2676)
    assert not use_short_flashinfer(3, 136)
    modeling = SimpleNamespace(
        torch_chunk_gated_delta_rule=object(),
        causal_conv1d_fn=object(),
        chunk_gated_delta_rule=object(),
    )
    install_short_flashinfer_prefill(modeling)
    assert require_bound_linear_attention(modeling) == SHORT_FLASHINFER_LINEAR_ATTENTION


def test_flashinfer_wrapper_rejects_a_batch_and_a_state():
    class Batch:
        shape = (2, 4, 8, 16)

    with pytest.raises(RuntimeError, match="one sequence"):
        flashinfer_chunk_gated_delta_rule(Batch(), Batch(), Batch())

    class One:
        shape = (1, 4, 8, 16)

    with pytest.raises(RuntimeError, match="recurrent state"):
        flashinfer_chunk_gated_delta_rule(One(), One(), One(), initial_state=object())


def test_flashinfer_use_cp_is_auto_for_this_head_count_on_h200():
    assert GDN_USE_CP == "auto"
    # 48 value heads, one sequence. H200 has 132 SMs. HBM threshold is work * 2 < sms.
    assert cp_heuristic_matches(48, 132, "NVIDIA L20X")
    assert cp_heuristic_matches(48, 132, "NVIDIA H200")
    assert not cp_heuristic_matches(48, 96, "NVIDIA L20X")
    assert not cp_heuristic_matches(48, 132, "NVIDIA GeForce RTX 4090")
    with pytest.raises(ValueError, match="positive"):
        cp_heuristic_matches(0, 132, "NVIDIA L20X")
