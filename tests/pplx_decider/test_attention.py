"""CPU tests for the unpadded full-attention mask policy. No weights, no torch."""

from __future__ import annotations

import pytest

from models.pplx_decider.attention import (
    install_unpadded_full_attention,
    mark_full_attention_noncausal,
    mask_inputs_unpadded,
    noncausal_attention_mask,
    omit_explicit_full_attention_mask,
)


def mask_inputs(module, args, kwargs):
    return args, kwargs


class _Attn:
    def __init__(self, causal=True):
        self.is_causal = causal


class _Layer:
    def __init__(self, kind, causal=True):
        self.block_type = kind
        if kind == "full_attention":
            self.self_attn = _Attn(causal)


class _Config:
    _attn_implementation = "sdpa"


class _LanguageModel:
    def __init__(self, layers, hooks=None, with_kwargs=None, implementation="sdpa"):
        self.config = _Config()
        self.config._attn_implementation = implementation
        self.layers = layers
        self._forward_pre_hooks = {7: mask_inputs} if hooks is None else hooks
        self._forward_pre_hooks_with_kwargs = {7: True} if with_kwargs is None else with_kwargs


def _loaded():
    return _LanguageModel(
        [
            _Layer("linear_attention"),
            _Layer("full_attention"),
            _Layer("linear_attention"),
            _Layer("full_attention"),
        ]
    )


def test_all_true_padding_omits_the_explicit_mask():
    assert omit_explicit_full_attention_mask([True, True, True]) is True
    assert omit_explicit_full_attention_mask([True]) is True


def test_any_false_padding_bit_keeps_the_explicit_mask():
    assert omit_explicit_full_attention_mask([True, False]) is False
    assert omit_explicit_full_attention_mask([False]) is False
    assert omit_explicit_full_attention_mask([False, True, True]) is False


def test_empty_padding_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        omit_explicit_full_attention_mask([])


def test_mask_dict_drops_only_the_full_attention_mask():
    linear = object()
    kept = object()
    omitted = noncausal_attention_mask(True, kept, linear)
    assert omitted == {"full_attention": None, "linear_attention": linear}
    assert omitted["linear_attention"] is linear
    present = noncausal_attention_mask(False, kept, linear)
    assert present["full_attention"] is kept
    assert present["linear_attention"] is linear


def test_mask_dict_rejects_a_numpy_style_bool():
    with pytest.raises(TypeError):
        noncausal_attention_mask(1, None, None)  # type: ignore[arg-type]


def test_install_marks_only_full_attention_and_replaces_the_checkpoint_hook():
    model = _loaded()
    changed = install_unpadded_full_attention(model)
    assert changed == 2
    assert model.layers[0].block_type == "linear_attention"
    assert not hasattr(model.layers[0], "self_attn")
    assert model.layers[1].self_attn.is_causal is False
    assert model.layers[3].self_attn.is_causal is False
    assert model._forward_pre_hooks[7] is mask_inputs_unpadded
    assert 7 in model._forward_pre_hooks_with_kwargs


def test_mark_writes_nothing_when_a_full_attention_module_is_already_noncausal():
    model = _loaded()
    model.layers[3].self_attn.is_causal = False
    with pytest.raises(RuntimeError, match="not causal"):
        mark_full_attention_noncausal(model)
    assert model.layers[1].self_attn.is_causal is True


def test_install_leaves_modules_causal_when_the_checkpoint_hook_is_absent():
    model = _loaded()
    model._forward_pre_hooks = {7: lambda *args: args}
    with pytest.raises(RuntimeError, match="pre-hook"):
        install_unpadded_full_attention(model)
    assert model.layers[1].self_attn.is_causal is True
    assert model.layers[3].self_attn.is_causal is True


def test_install_rejects_a_hook_that_does_not_take_keyword_inputs():
    model = _loaded()
    model._forward_pre_hooks_with_kwargs = {}
    with pytest.raises(RuntimeError, match="keyword"):
        install_unpadded_full_attention(model)
    assert model.layers[1].self_attn.is_causal is True
    assert model._forward_pre_hooks[7] is mask_inputs


def test_install_rejects_non_sdpa_before_changing_modules():
    model = _loaded()
    model.config._attn_implementation = "eager"
    with pytest.raises(RuntimeError, match="SDPA"):
        install_unpadded_full_attention(model)
    assert model.layers[1].self_attn.is_causal is True
    assert model._forward_pre_hooks[7] is mask_inputs


def test_second_install_fails_closed():
    model = _loaded()
    install_unpadded_full_attention(model)
    with pytest.raises(RuntimeError, match="pre-hook"):
        install_unpadded_full_attention(model)
