"""CPU checks for the long-sequence compile switch. No weights and no torch."""

from __future__ import annotations

import pytest

from models.pplx_decider.compile_forward import (
    MEDIAN_PROOF_TOKENS,
    TAIL_MIN_TOKENS,
    TAIL_WARMUP_LENGTHS,
    compile_median,
    compile_min_tokens,
    compile_preference,
    forward_sequence_length,
    require_median_route,
    use_compiled_forward,
    warmup_lengths,
)


class _Tensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


def test_cuda_compiles_only_the_tail_unless_asked_otherwise():
    assert compile_preference("cuda", None) == "dynamic-tail"
    assert compile_preference("cuda", "dynamic-tail") == "dynamic-tail"
    assert compile_preference("cuda", "dynamic") == "dynamic"
    assert compile_preference("cuda", "default") == "default"
    assert compile_preference("cpu", None) == "default"
    assert compile_preference("cpu", "dynamic-tail") == "default"
    with pytest.raises(RuntimeError, match="dynamic-tail, dynamic, or default"):
        compile_preference("cuda", "reduce-overhead")


def test_median_stays_eager_and_the_tail_is_compiled():
    assert TAIL_MIN_TOKENS == 512
    assert MEDIAN_PROOF_TOKENS == 136
    assert compile_median("dynamic-tail") == "isolated"
    assert compile_median("dynamic") == "compiled"
    assert compile_median("default") == "eager"
    assert compile_min_tokens("dynamic-tail") == 512
    assert compile_min_tokens("dynamic") == 0
    assert compile_min_tokens("default") == 0
    assert use_compiled_forward("dynamic-tail", 136) is False
    assert use_compiled_forward("dynamic-tail", 512) is False
    assert use_compiled_forward("dynamic-tail", 513) is True
    assert use_compiled_forward("dynamic-tail", 2676) is True
    assert use_compiled_forward("dynamic-tail", None) is False
    assert use_compiled_forward("dynamic", 32) is True
    assert use_compiled_forward("default", 2676) is False


def test_median_proof_rejects_a_compiled_or_misread_short_forward():
    require_median_route(136, 136, "eager")
    with pytest.raises(RuntimeError, match="detected length 5120"):
        require_median_route(136, 5120, "compiled")
    with pytest.raises(RuntimeError, match="route compiled"):
        require_median_route(136, 136, "compiled")
    with pytest.raises(RuntimeError, match="detected length None"):
        require_median_route(136, None, "eager")


def test_sequence_length_comes_from_embeds_then_ids():
    embeds = _Tensor((1, 136, 5120))
    ids = _Tensor((1, 2676))
    assert forward_sequence_length((), {"inputs_embeds": embeds, "input_ids": None}) == 136
    assert forward_sequence_length((), {"input_ids": ids}) == 2676
    assert forward_sequence_length((ids,), {}) == 2676
    assert forward_sequence_length((), {}) is None


def test_tail_warmup_lengths_take_the_compiled_path():
    assert warmup_lengths("dynamic-tail") == TAIL_WARMUP_LENGTHS
    assert TAIL_WARMUP_LENGTHS == (768, 1024, 2048)
    assert all(length > TAIL_MIN_TOKENS for length in TAIL_WARMUP_LENGTHS)
    assert warmup_lengths("dynamic") == (32, 256, 1024)
    with pytest.raises(RuntimeError, match="dynamic-tail or dynamic"):
        warmup_lengths("default")
