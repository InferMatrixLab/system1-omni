"""Bind the Qwen chunk GDN prefill to flash-linear-attention.

Transformers 5.17 resolves ``chunk_gated_delta_rule`` from package ``fla``
when ``modeling_qwen3_5`` is imported. If ``flash-linear-attention`` is
missing, the first forward warns and runs the PyTorch reference. This
checkpoint pins ``flash-linear-attention==0.5.2``. That package's chunk
kernel is the Triton path.

vLLM's auto GDN prefill on sm_89 is Triton/FLA. SGLang's CUDA linear
attention uses the same class of chunk kernel unless a Hopper or
Blackwell FlashInfer domain is selected. This classification prefill does
not select FlashInfer or CuteDSL, and it does not capture a CUDA graph.
The forward has no cache, so it calls the chunk prefill rather than the
single-token recurrent update.
"""

from __future__ import annotations

from .conv import binds_package

FLA_VERSION = "0.5.2"
LINEAR_ATTENTION = "fla-chunk"


def version_base(installed: str) -> str:
    return installed.split("+", 1)[0]


def fla_chunk_function():
    import fla.ops.gated_delta_rule as gated

    return gated.chunk_gated_delta_rule


def prepare_fla(*, required: bool) -> str | None:
    """Import FLA before Qwen3.5 modeling freezes the torch chunk reference."""

    try:
        import fla
        from importlib.metadata import PackageNotFoundError, version

        chunk = fla_chunk_function()
        try:
            installed = version("flash-linear-attention")
        except PackageNotFoundError:
            installed = getattr(fla, "__version__", "")
    except Exception as exc:
        if not required:
            return None
        raise RuntimeError(
            "flash-linear-attention is not installed, so chunk_gated_delta_rule "
            "would fall back to its reference PyTorch implementation. Install "
            "the checkpoint requirements before starting the CUDA worker."
        ) from exc
    if chunk is None:
        raise RuntimeError("fla does not export chunk_gated_delta_rule")
    if not isinstance(installed, str) or version_base(installed) != FLA_VERSION:
        raise RuntimeError(
            f"flash-linear-attention {installed!r} is loaded; expected {FLA_VERSION}"
        )
    return version_base(installed)


def assert_qwen_chunk_uses_fla() -> None:
    """Fail if modeling still closed over the PyTorch chunk reference."""

    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    if not binds_package(getattr(modeling, "torch_chunk_gated_delta_rule"), fla_chunk_function()):
        raise RuntimeError(
            "Transformers still has the PyTorch chunk_gated_delta_rule fallback. "
            "The load warning is `falling back to its reference PyTorch "
            "implementation`. Restart after installing flash-linear-attention "
            "0.5.2 into a fresh process."
        )
