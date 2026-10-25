"""Compile only the long language-model forwards.

Always-on ``dynamic`` compile, measured on ``e36b32ed``, cut p95 and raised
p50. Sequences at or below ``TAIL_MIN_TOKENS`` stay on the eager forward,
which is the primary median path. Longer sequences use the compiled
forward. 512 sits between the profiled median of about 136 tokens and the
profiled tail example of about 2676 tokens. It is not a fitted cutoff.

``PPLX_COMPILE=default`` leaves every length eager.
``PPLX_COMPILE=dynamic`` is the rejected always-on compile.
Inductor is not bit-exact with eager. CUDA graphs are not selected.
"""

from __future__ import annotations

from typing import Any

COMPILE_TAIL = "dynamic-tail"
COMPILE_DYNAMIC = "dynamic"
COMPILE_DEFAULT = "default"
COMPILE_MODES = frozenset({COMPILE_TAIL, COMPILE_DYNAMIC, COMPILE_DEFAULT})
TAIL_MIN_TOKENS = 512
# Lengths that take the compiled path, run before /health.
TAIL_WARMUP_LENGTHS = (768, 1024, 2048)
# Warmup for the rejected always-on compile.
DYNAMIC_WARMUP_LENGTHS = (32, 256, 1024)


def compile_preference(device: str, selected: str | None = None) -> str:
    """CUDA startup compiles long sequences only. Other devices stay eager.

    ``selected`` is ``PPLX_COMPILE``. Unset on CUDA selects ``dynamic-tail``.
    ``default`` reproduces the primary. ``dynamic`` is the rejected
    always-on compile.
    """

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if selected is not None and selected not in COMPILE_MODES:
        raise RuntimeError("PPLX_COMPILE must be dynamic-tail, dynamic, or default")
    if device != "cuda":
        return COMPILE_DEFAULT
    if selected is None:
        return COMPILE_TAIL
    return selected


def compile_min_tokens(mode: str) -> int:
    """Token length above which the compiled forward is used. Zero means all."""

    if mode == COMPILE_TAIL:
        return TAIL_MIN_TOKENS
    if mode in {COMPILE_DYNAMIC, COMPILE_DEFAULT}:
        return 0
    raise RuntimeError("PPLX_COMPILE must be dynamic-tail, dynamic, or default")


def warmup_lengths(mode: str) -> tuple[int, ...]:
    if mode == COMPILE_TAIL:
        return TAIL_WARMUP_LENGTHS
    if mode == COMPILE_DYNAMIC:
        return DYNAMIC_WARMUP_LENGTHS
    raise RuntimeError("PPLX_COMPILE must be dynamic-tail or dynamic")


def use_compiled_forward(mode: str, length: int | None) -> bool:
    """Return whether this forward should take the compiled language model."""

    if mode == COMPILE_DEFAULT:
        return False
    if mode == COMPILE_DYNAMIC:
        return True
    if mode == COMPILE_TAIL:
        return length is not None and length > TAIL_MIN_TOKENS
    raise RuntimeError("PPLX_COMPILE must be dynamic-tail, dynamic, or default")


def forward_sequence_length(args: tuple, kwargs: dict) -> int | None:
    """Read the sequence length from the language-model forward inputs."""

    embeds = kwargs.get("inputs_embeds")
    length = _sequence_length(embeds)
    if length is not None:
        return length
    length = _sequence_length(kwargs.get("input_ids"))
    if length is not None:
        return length
    if args:
        return _sequence_length(args[0])
    return None


def _sequence_length(tensor: Any) -> int | None:
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    shape = tuple(tensor.shape)
    if len(shape) < 2:
        return None
    return int(shape[1])


def _compiled_frames() -> int:
    import torch._dynamo.utils as utils

    return int(utils.counters["frames"]["ok"])


def install_compiled_forward(language_model: Any, mode: str) -> str:
    """Install eager, always-on compile, or the long-sequence compile."""

    import torch
    import torch._dynamo as dynamo

    if mode == COMPILE_DEFAULT:
        return COMPILE_DEFAULT
    if mode not in {COMPILE_DYNAMIC, COMPILE_TAIL}:
        raise RuntimeError("PPLX_COMPILE must be dynamic-tail, dynamic, or default")
    if getattr(language_model, "pplx_compile", None) is not None:
        raise RuntimeError("language model forward is already compiled")
    dynamo.config.suppress_errors = False
    eager = language_model.forward
    compiled = torch.compile(eager, dynamic=True, fullgraph=False)
    if mode == COMPILE_DYNAMIC:
        language_model.forward = compiled
    else:
        def dispatch(*args: Any, **kwargs: Any) -> Any:
            length = forward_sequence_length(args, kwargs)
            if use_compiled_forward(mode, length):
                return compiled(*args, **kwargs)
            return eager(*args, **kwargs)

        language_model.forward = dispatch
    language_model.pplx_compile = mode
    language_model.pplx_compile_min_tokens = compile_min_tokens(mode)
    return mode


def warmup_compiled_forward(backbone: Any, device: str, mode: str) -> int:
    """Run text-only forwards so compilation finishes before ``/health``.

    Returns the number of warmup lengths. Raises when Dynamo records no
    compiled frame.
    """

    import torch

    lengths = warmup_lengths(mode)
    before = _compiled_frames()
    with torch.inference_mode():
        for length in lengths:
            input_ids = torch.zeros(1, length, dtype=torch.long, device=device)
            attention_mask = torch.ones(1, length, dtype=torch.long, device=device)
            backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    after = _compiled_frames()
    if after <= before:
        raise RuntimeError("dynamic compile produced no compiled frames")
    return len(lengths)
