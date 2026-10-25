"""CPU checks for the dual linear-attention bind. No CUDA extension."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from models.pplx_decider.engine import PplxExecutor

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
    GDN_LAYOUT_GVA,
    GDN_USE_CP,
    pack_qk_for_l2norm,
    cu_seqlens_key,
    cp_heuristic_matches,
    gva_gated_delta_forward,
    gva_group_size,
    gdn_prefill_mode,
    install_gva_prefill,
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


def test_unknown_gdn_prefill_fails_closed_on_cpu(monkeypatch, tmp_path):
    monkeypatch.setenv("PPLX_GDN_PREFILL", "typo")
    with pytest.raises(RuntimeError, match="flashinfer, fla, or short"):
        PplxExecutor(tmp_path, device="cpu")


def test_known_gdn_prefill_reaches_checkpoint_checks_on_cpu(monkeypatch, tmp_path):
    monkeypatch.setenv("PPLX_GDN_PREFILL", "fla")
    with pytest.raises(ValueError, match="checkpoint directory does not exist"):
        PplxExecutor(tmp_path / "absent", device="cpu")


def test_gdn_prefill_is_validated_before_the_capability_branch():
    source = inspect.getsource(PplxExecutor.__init__)
    assert source.index("gdn_prefill_mode(") < source.index("uses_flashinfer_prefill(")


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


def test_use_cp_stays_false_after_the_auto_miss():
    assert GDN_USE_CP is False
    # 48 value heads, one sequence, would have selected CP on an H200. That path missed.
    assert cp_heuristic_matches(48, 132, "NVIDIA L20X")
    assert not cp_heuristic_matches(48, 96, "NVIDIA L20X")
    assert not cp_heuristic_matches(48, 132, "NVIDIA GeForce RTX 4090")
    with pytest.raises(ValueError, match="positive"):
        cp_heuristic_matches(0, 132, "NVIDIA L20X")


def test_unrepeated_qk_are_packed_before_the_l2norm_view():
    import inspect

    class Split:
        def __init__(self):
            self.packed = False

        def contiguous(self):
            packed = Split()
            packed.packed = True
            return packed

        def view(self, *shape):
            if not self.packed:
                raise RuntimeError("view size is not compatible")
            return ("view", shape)

    query, key = pack_qk_for_l2norm(Split(), Split())
    assert query.view(-1, 128) == ("view", (-1, 128))
    assert key.view(-1, 128) == ("view", (-1, 128))
    source = inspect.getsource(flashinfer_chunk_gated_delta_rule)
    assert source.index("pack_qk_for_l2norm(q, k)") < source.index("l2norm_fwd(q)")


def test_gva_prefill_skips_the_q_repeat_on_48_linear_layers():
    import inspect

    assert gva_group_size(16, 48) == 3
    with pytest.raises(ValueError, match="value heads"):
        gva_group_size(48, 48)
    assert "repeat_interleave" not in inspect.getsource(gva_gated_delta_forward)
    layers = []
    for index in range(64):
        block = "full_attention" if index % 4 == 0 else "linear_attention"
        attn = SimpleNamespace(num_k_heads=16, num_v_heads=48)
        layers.append(SimpleNamespace(block_type=block, linear_attn=attn))
    assert install_gva_prefill(SimpleNamespace(layers=layers)) == LINEAR_ATTENTION_LAYERS
    linear = layers[1].linear_attn
    assert linear.pplx_gdn_layout == GDN_LAYOUT_GVA
    assert linear.forward.__func__ is gva_gated_delta_forward
    assert not hasattr(layers[0].linear_attn, "pplx_gdn_layout")
