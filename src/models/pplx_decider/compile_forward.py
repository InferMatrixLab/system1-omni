"""Compile only the long language-model forwards.

Always-on ``dynamic`` compile, measured on ``e36b32ed``, cut p95 and raised
p50. The 512-token split on ``7e3b1be`` cut p95 further and raised p50
again, so a short forward was still paying for compile. Sequences at or
below ``TAIL_MIN_TOKENS`` now call the original eager function with the
Dynamo frame hook suspended and the CUDA flags captured before
``torch.compile`` restored. Longer sequences use the compiled forward.
512 sits between the profiled median of about 136 tokens and the profiled
tail example of about 2676 tokens. It is not a fitted cutoff.

Measured compile tips ``e36b32ed``, ``7e3b1be``, and ``34ba5de`` all
raised p50, including after the 136-token probe proved the median route
was eager. Unset ``PPLX_COMPILE`` now stays eager.
``PPLX_COMPILE=dynamic`` and ``PPLX_COMPILE=dynamic-tail`` are rejected
opt-ins. Inductor is not bit-exact with eager. CUDA graphs are not
selected.
"""

from __future__ import annotations

from typing import Any

COMPILE_TAIL = "dynamic-tail"
COMPILE_DYNAMIC = "dynamic"
COMPILE_DEFAULT = "default"
COMPILE_MODES = frozenset({COMPILE_TAIL, COMPILE_DYNAMIC, COMPILE_DEFAULT})
TAIL_MIN_TOKENS = 512
# Profiled median length. Startup runs one forward of this length and
# requires the dispatcher to see that length on the eager route.
MEDIAN_PROOF_TOKENS = 136
# Lengths that take the compiled path, run before /health.
TAIL_WARMUP_LENGTHS = (768, 1024, 2048)
# Warmup for the rejected always-on compile.
DYNAMIC_WARMUP_LENGTHS = (32, 256, 1024)


def compile_preference(device: str, selected: str | None = None) -> str:
    """Startup leaves the language-model forward eager.

    ``selected`` is ``PPLX_COMPILE``. Unset and ``default`` reproduce the
    primary. ``dynamic`` and ``dynamic-tail`` are rejected opt-ins. CPU
    always stays eager.
    """

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if selected is not None and selected not in COMPILE_MODES:
        raise RuntimeError("PPLX_COMPILE must be dynamic-tail, dynamic, or default")
    if device != "cuda":
        return COMPILE_DEFAULT
    if selected is None:
        return COMPILE_DEFAULT
    return selected


def compile_median(mode: str) -> str:
    """How the median-length forward is run."""

    if mode == COMPILE_TAIL:
        return "isolated"
    if mode == COMPILE_DYNAMIC:
        return "compiled"
    if mode == COMPILE_DEFAULT:
        return "eager"
    raise RuntimeError("PPLX_COMPILE must be dynamic-tail, dynamic, or default")


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


def require_median_route(length: int, detected: int | None, route: str | None) -> None:
    """Fail startup when the profiled median length is not the eager route."""

    if detected != length or route != "eager":
        raise RuntimeError(
            f"median forward of {length} tokens took route {route} with detected length {detected}"
        )


def capture_cuda_flags() -> dict[str, Any]:
    """Read the CUDA math flags that compile can change for later eager calls."""

    import torch

    flags: dict[str, Any] = {}
    matmul = torch.backends.cuda.matmul
    for name in (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if hasattr(matmul, name):
            flags[f"matmul.{name}"] = getattr(matmul, name)
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        flags["cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
    if hasattr(torch.backends.cudnn, "benchmark"):
        flags["cudnn.benchmark"] = torch.backends.cudnn.benchmark
    getter = getattr(torch, "get_float32_matmul_precision", None)
    if getter is not None:
        flags["float32_matmul_precision"] = getter()
    cuda = torch.backends.cuda
    for name in (
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "cudnn_sdp_enabled",
    ):
        reader = getattr(cuda, name, None)
        if reader is not None:
            flags[name] = reader()
    return flags


def apply_cuda_flags(flags: dict[str, Any]) -> None:
    """Put captured CUDA math flags back. Unchanged flags are left alone."""

    import torch

    matmul = torch.backends.cuda.matmul
    for name in (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        key = f"matmul.{name}"
        if key in flags and getattr(matmul, name, None) != flags[key]:
            setattr(matmul, name, flags[key])
    if "cudnn.allow_tf32" in flags and torch.backends.cudnn.allow_tf32 != flags["cudnn.allow_tf32"]:
        torch.backends.cudnn.allow_tf32 = flags["cudnn.allow_tf32"]
    if "cudnn.benchmark" in flags and torch.backends.cudnn.benchmark != flags["cudnn.benchmark"]:
        torch.backends.cudnn.benchmark = flags["cudnn.benchmark"]
    if "float32_matmul_precision" in flags:
        current = torch.get_float32_matmul_precision()
        if current != flags["float32_matmul_precision"]:
            torch.set_float32_matmul_precision(flags["float32_matmul_precision"])
    cuda = torch.backends.cuda
    setters = {
        "flash_sdp_enabled": "enable_flash_sdp",
        "mem_efficient_sdp_enabled": "enable_mem_efficient_sdp",
        "math_sdp_enabled": "enable_math_sdp",
        "cudnn_sdp_enabled": "enable_cudnn_sdp",
    }
    for reader_name, writer_name in setters.items():
        if reader_name not in flags:
            continue
        reader = getattr(cuda, reader_name, None)
        writer = getattr(cuda, writer_name, None)
        if reader is None or writer is None:
            continue
        if reader() != flags[reader_name]:
            writer(flags[reader_name])


def run_isolated_eager(eager: Any, flags: dict[str, Any], args: tuple, kwargs: dict) -> Any:
    """Run the original forward outside Dynamo, with the pre-compile flags."""

    import torch._dynamo.eval_frame as eval_frame

    prior = eval_frame.set_eval_frame(None)
    try:
        apply_cuda_flags(flags)
        return eager(*args, **kwargs)
    finally:
        eval_frame.set_eval_frame(prior)


def _compiled_frames() -> int:
    import torch._dynamo.utils as utils

    return int(utils.counters["frames"]["ok"])


def install_compiled_forward(language_model: Any, mode: str) -> str:
    """Install eager, always-on compile, or the long-sequence compile."""

    import torch
    import torch._dynamo as dynamo
    import torch._dynamo.eval_frame as eval_frame

    if mode == COMPILE_DEFAULT:
        return COMPILE_DEFAULT
    if mode not in {COMPILE_DYNAMIC, COMPILE_TAIL}:
        raise RuntimeError("PPLX_COMPILE must be dynamic-tail, dynamic, or default")
    if getattr(language_model, "pplx_compile", None) is not None:
        raise RuntimeError("language model forward is already compiled")
    dynamo.config.suppress_errors = False
    if not hasattr(eval_frame, "set_eval_frame"):
        raise RuntimeError("torch dynamo eval frame cannot be suspended")
    eager = language_model.forward
    flags = capture_cuda_flags()
    compiled = torch.compile(eager, dynamic=True, fullgraph=False)
    language_model.pplx_compile_eager_forwards = 0
    language_model.pplx_compile_compiled_forwards = 0
    language_model.pplx_last_length = None
    language_model.pplx_last_route = None
    language_model.pplx_median_frame_delta = None
    if mode == COMPILE_DYNAMIC:
        language_model.forward = compiled
    else:
        def dispatch(*args: Any, **kwargs: Any) -> Any:
            length = forward_sequence_length(args, kwargs)
            language_model.pplx_last_length = length
            if use_compiled_forward(mode, length):
                language_model.pplx_compile_compiled_forwards += 1
                language_model.pplx_last_route = "compiled"
                return compiled(*args, **kwargs)
            language_model.pplx_compile_eager_forwards += 1
            language_model.pplx_last_route = "eager"
            return run_isolated_eager(eager, flags, args, kwargs)

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
            _text_forward(backbone, device, length)
        after = _compiled_frames()
        if after <= before:
            raise RuntimeError("dynamic compile produced no compiled frames")
        if mode == COMPILE_TAIL:
            language_model = backbone.language_model
            proof_before = _compiled_frames()
            _text_forward(backbone, device, MEDIAN_PROOF_TOKENS)
            detected = getattr(language_model, "pplx_last_length", None)
            route = getattr(language_model, "pplx_last_route", None)
            require_median_route(MEDIAN_PROOF_TOKENS, detected, route)
            language_model.pplx_median_frame_delta = _compiled_frames() - proof_before
    return len(lengths)


def _text_forward(backbone: Any, device: str, length: int) -> None:
    import torch

    input_ids = torch.zeros(1, length, dtype=torch.long, device=device)
    attention_mask = torch.ones(1, length, dtype=torch.long, device=device)
    backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
