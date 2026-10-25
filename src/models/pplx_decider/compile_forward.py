"""Compile the language-model forward before the worker is ready.

The kept primary's time is the eager 64-layer forward. GDN kernel choice,
SDPA order, and a wider gate-up GEMM did not beat both percentiles, and the
wider GEMM was slower, so this path does not change those kernels. It
compiles ``language_model.forward`` with a dynamic sequence length and runs
a few text-only forwards before ``/health`` so the measured requests do not
pay for the first compilation.

``PPLX_COMPILE=default`` leaves the forward eager. Inductor is not
bit-exact with eager. CUDA graphs are not selected.
"""

from __future__ import annotations

from typing import Any

COMPILE_DYNAMIC = "dynamic"
COMPILE_DEFAULT = "default"
# One short length and two longer ones. dynamic=True should cover the
# Public231 lengths after this; the extra forwards happen before ready.
WARMUP_LENGTHS = (32, 256, 1024)


def compile_preference(device: str, selected: str | None = None) -> str:
    """CUDA startup compiles the language-model forward. Other devices stay eager.

    ``selected`` is ``PPLX_COMPILE``. ``default`` reproduces the primary.
    """

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if selected is not None and selected not in {COMPILE_DYNAMIC, COMPILE_DEFAULT}:
        raise RuntimeError("PPLX_COMPILE must be dynamic or default")
    if device != "cuda":
        return COMPILE_DEFAULT
    if selected is None:
        return COMPILE_DYNAMIC
    return selected


def _compiled_frames() -> int:
    import torch._dynamo.utils as utils

    return int(utils.counters["frames"]["ok"])


def install_dynamic_compile(language_model: Any) -> str:
    """Replace the language-model forward with a dynamic ``torch.compile`` wrapper."""

    import torch
    import torch._dynamo as dynamo

    if getattr(language_model, "pplx_compile", None) == COMPILE_DYNAMIC:
        raise RuntimeError("language model forward is already compiled")
    dynamo.config.suppress_errors = False
    language_model.forward = torch.compile(
        language_model.forward,
        dynamic=True,
        fullgraph=False,
    )
    language_model.pplx_compile = COMPILE_DYNAMIC
    return COMPILE_DYNAMIC


def warmup_dynamic_compile(backbone: Any, device: str) -> int:
    """Run text-only forwards so compilation finishes before ``/health``.

    Returns the number of warmup lengths. Raises when Dynamo records no
    compiled frame, which would otherwise leave the measured run on eager.
    """

    import torch

    before = _compiled_frames()
    with torch.inference_mode():
        for length in WARMUP_LENGTHS:
            input_ids = torch.zeros(1, length, dtype=torch.long, device=device)
            attention_mask = torch.ones(1, length, dtype=torch.long, device=device)
            backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    after = _compiled_frames()
    if after <= before:
        raise RuntimeError("dynamic compile produced no compiled frames")
    return len(WARMUP_LENGTHS)
