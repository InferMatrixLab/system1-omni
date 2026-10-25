"""Bind the Qwen chunk GDN prefill the way vLLM and SGLang do on Hopper.

Transformers 5.17 resolves ``chunk_gated_delta_rule`` from package ``fla``
when ``modeling_qwen3_5`` is imported. If ``flash-linear-attention`` is
missing, the first forward warns and runs the PyTorch reference. This
checkpoint pins ``flash-linear-attention==0.5.2``, whose chunk kernel is
the Triton/FLA path.

On SM90, vLLM's auto GDN prefill is FlashInfer, and SGLang's validated
default is FlashInfer when the linear head dim is 128 and the prefill
kernel imports. This checkpoint's linear key and value dims are both 128.
After the FLA bind, an SM90 worker replaces the modeling global with
``flashinfer.gdn_prefill.chunk_gated_delta_rule``. CuteDSL is the SM100
opt-in and is not selected. Nothing here captures a CUDA graph.

The classification forward has no cache, so it calls the chunk prefill,
not the single-token recurrent update.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from .conv import binds_package

FLA_VERSION = "0.5.2"
HOPPER = (9, 0)
GDN_HEAD_DIM = 128
FLA_CHUNK = "fla-chunk"
FLASHINFER = "flashinfer"


def version_base(installed: str) -> str:
    return installed.split("+", 1)[0]


def select_gdn_prefill(
    capability: tuple[int, int],
    key_dim: int | None,
    value_dim: int | None,
    *,
    flashinfer_available: bool,
) -> str:
    """SM90 FlashInfer when both stacks use it; otherwise the FLA chunk path."""

    if (
        flashinfer_available
        and capability == HOPPER
        and key_dim == GDN_HEAD_DIM
        and value_dim == GDN_HEAD_DIM
    ):
        return FLASHINFER
    return FLA_CHUNK


def linear_head_dims(config: dict[str, Any]) -> tuple[int | None, int | None]:
    text = config.get("text_config")
    if not isinstance(text, dict):
        return None, None
    key = text.get("linear_key_head_dim")
    value = text.get("linear_value_head_dim")
    if type(key) is not int or type(value) is not int:
        return None, None
    return key, value


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


def flashinfer_prefill_available() -> bool:
    """Whether ``flashinfer.gdn_prefill.chunk_gated_delta_rule`` imports.

    SGLang sets ``FLASHINFER_DISABLE_VERSION_CHECK`` before this import.
    A missing package is the FLA partial, not a crash.
    """

    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
    except Exception:
        return False
    return callable(chunk_gated_delta_rule)


def flashinfer_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version_base(version("flashinfer"))
    except PackageNotFoundError:
        return "installed"


def bind_flashinfer_prefill() -> str:
    """Point the modeling chunk global at FlashInfer's SM90 prefill."""

    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    modeling.torch_chunk_gated_delta_rule = _flashinfer_chunk(chunk_gated_delta_rule)
    return flashinfer_version()


def _l2norm(tensor):
    import torch

    values = tensor.float()
    normed = values * torch.rsqrt((values * values).sum(dim=-1, keepdim=True) + 1e-6)
    return normed.to(dtype=tensor.dtype)


def _flashinfer_chunk(prefill):
    """HF ``[batch, seq, heads, dim]`` call, FlashInfer varlen prefill body.

    vLLM and SGLang exp the log-space gate, cast beta and state to float32,
    and run L2 norm before the kernel with ``use_qk_l2norm_in_kernel`` left
    false. Equal-length rows, including left padding the checkpoint already
    zeroed, become ``cu_seqlens`` of stride ``seq``. A caller-supplied
    ``cu_seqlens`` is used only for the packed batch-1 shape those servers
    pass.
    """

    parameters = inspect.signature(prefill).parameters

    def flashinfer_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size: int = 64,
        initial_state=None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens=None,
        **kwargs,
    ):
        import torch

        if chunk_size != 64:
            raise RuntimeError("FlashInfer GDN prefill is chunk size 64")
        if query.dim() != 4:
            raise RuntimeError("FlashInfer GDN prefill expects [batch, seq, heads, dim]")
        batch, seqlen, heads, dim = query.shape
        if cu_seqlens is None:
            query_fi = query.reshape(batch * seqlen, heads, dim).contiguous()
            key_fi = key.reshape(batch * seqlen, heads, dim).contiguous()
            value_fi = value.reshape(batch * seqlen, heads, dim).contiguous()
            g_fi = g.reshape(batch * seqlen, heads).contiguous()
            beta_fi = beta.reshape(batch * seqlen, heads).contiguous()
            lengths = torch.arange(
                0, (batch + 1) * seqlen, seqlen, device=query.device, dtype=torch.int64
            )
        elif batch == 1:
            query_fi = query[0].contiguous()
            key_fi = key[0].contiguous()
            value_fi = value[0].contiguous()
            g_fi = g[0].contiguous()
            beta_fi = beta[0].contiguous()
            lengths = cu_seqlens.to(device=query.device, dtype=torch.int64)
        else:
            raise RuntimeError("packed cu_seqlens requires batch 1")
        if use_qk_l2norm_in_kernel:
            query_fi = _l2norm(query_fi)
            key_fi = _l2norm(key_fi)
        passed = {
            "q": query_fi,
            "k": key_fi,
            "v": value_fi,
            "g": torch.exp(g_fi.float()),
            "beta": beta_fi.float(),
            "initial_state": None if initial_state is None else initial_state.float(),
            "output_final_state": output_final_state,
            "cu_seqlens": lengths,
            "use_qk_l2norm_in_kernel": False,
        }
        names = set(parameters)
        var_keyword = any(
            item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        )
        if "backend" in names or var_keyword:
            passed["backend"] = "flashinfer"
        if "scale" in names:
            passed["scale"] = None
        if output_final_state and "output_state" in names:
            nseq = int(lengths.numel() - 1)
            passed["output_state"] = torch.empty(
                nseq, heads, dim, dim, device=query.device, dtype=torch.float32
            )
        if var_keyword:
            result = prefill(**passed)
        else:
            result = prefill(**{key: value for key, value in passed.items() if key in names})
        if output_final_state:
            output, final_state = result
        else:
            output, final_state = result, None
        return output.reshape(batch, seqlen, heads, dim), final_state

    return flashinfer_chunk_gated_delta_rule
