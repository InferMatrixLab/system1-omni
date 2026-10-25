"""CPU checks for the dynamic-compile switch. No weights and no torch."""

from __future__ import annotations

import pytest

from models.pplx_decider.compile_forward import WARMUP_LENGTHS, compile_preference


def test_cuda_compiles_unless_the_primary_eager_path_is_requested():
    assert compile_preference("cuda", None) == "dynamic"
    assert compile_preference("cuda", "dynamic") == "dynamic"
    assert compile_preference("cuda", "default") == "default"
    assert compile_preference("cpu", None) == "default"
    assert compile_preference("cpu", "dynamic") == "default"
    with pytest.raises(RuntimeError, match="dynamic or default"):
        compile_preference("cuda", "reduce-overhead")


def test_warmup_lengths_are_text_only_and_increasing():
    assert WARMUP_LENGTHS == (32, 256, 1024)
    assert list(WARMUP_LENGTHS) == sorted(WARMUP_LENGTHS)
