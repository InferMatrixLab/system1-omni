"""Bind Qwen chunk GDN prefill for the lab L20X-labeled GPU.

That label is the H200 target. The device name is not a refusal.
Compute capability 9.0 selects ``flashinfer.gdn_prefill.chunk_gated_delta_rule``
(Clef tip ``4e430e18``) after ``causal_conv1d`` and FLA are imported.
Capability 8.9, and a 9.0 device whose FlashInfer pin is absent, stay on
FLA's chunk kernel. Both binds can become ready. Ready is not limited to
capability 9.0.

Import ``causal_conv1d``, then ``fla.ops.gated_delta_rule``, then
``flashinfer.gdn_prefill`` only when capability 9.0 will use it, and only
then Qwen3.5 modeling. Transformers 5.17 calls
``torch_chunk_gated_delta_rule``. The FlashInfer path replaces that global
before the checkpoint constructs layers. The call is one sequence, no
recurrent state, ``use_cp`` false, log-space ``g`` exponentiated to
float32, and Q/K L2 norm from ``fla.modules.l2norm.l2norm_fwd`` with
FlashInfer's own flag left false. CuteDSL, CUDA graphs, and
``torch.compile`` are not selected.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

from .conv import binds_package

FLA_DISTRIBUTION = "flash-linear-attention"
FLA_VERSION = "0.5.2"
FLA_MODULE = "fla.ops.gated_delta_rule"
FI_DISTRIBUTION = "flashinfer-python"
FI_VERSION = "0.6.16.post3"
FI_PIN = f"{FI_DISTRIBUTION}=={FI_VERSION}"
FI_MODULE = "flashinfer.gdn_prefill"
GDN_PREFILL = "flashinfer"
HOPPER = (9, 0)
FLA_LINEAR_ATTENTION = "fla-chunk"
# Qwen3.5 27B text stack: 64 layers, 16 full attention, so 48 linear
# attention layers. One forward calls the GDN kernel once per linear
# layer. This is that architecture count, not a fitted cutoff.
LINEAR_ATTENTION_LAYERS = 48
# One ring of that many CUDA cu_seqlens tensors per device and length.
# Building a CUDA tensor from a Python list syncs. Sharing one tensor
# across the in-flight linear layers is the hypothesis for the median
# giveback on the single-buffer primary. The first forward of a length
# allocates the ring. Later forwards of that length do not.
_CU_SEQLENS_POOL: dict[tuple[int, int], list[Any]] = {}
_CU_SEQLENS_CURSOR: dict[tuple[int, int], int] = {}
FLASHINFER_LINEAR_ATTENTION = "flashinfer+fla+causal_conv1d"
SHORT_FLASHINFER_LINEAR_ATTENTION = "flashinfer-short+fla-chunk"
# Profile anchors on this workload: median about 136 tokens, tail example
# about 2676 tokens. 256 sits between those recorded lengths. It is the
# candidate boundary, not a fitted cutoff.
SHORT_FLASHINFER_TOKENS = 256
READY_LINEAR_ATTENTION = frozenset({
    FLA_LINEAR_ATTENTION,
    FLASHINFER_LINEAR_ATTENTION,
    SHORT_FLASHINFER_LINEAR_ATTENTION,
})
MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
_CHUNK = "torch_chunk_gated_delta_rule"


def version_base(installed: str) -> str:
    return installed.split("+", 1)[0]


def uses_flashinfer_prefill(capability: tuple[int, int]) -> bool:
    """Capability 9.0 is the FlashInfer prefill. The device name is ignored."""

    return capability == HOPPER


def gdn_prefill_mode(selected: str | None) -> str:
    """Default GDN prefill is all-sequence FlashInfer.

    ``fla`` reproduces the prior ``f968e576`` chunk. ``short`` is the
    rejected length dispatch. The kept primary reuses one ``cu_seqlens``.
    This candidate keeps a ring of one buffer per linear layer.
    """

    chosen = "flashinfer" if selected is None else selected
    if chosen not in {"flashinfer", "fla", "short"}:
        raise RuntimeError("PPLX_GDN_PREFILL must be flashinfer, fla, or short")
    return chosen


def fla_chunk_function():
    import fla.ops.gated_delta_rule as gated

    return gated.chunk_gated_delta_rule


def prepare_fla(*, required: bool) -> str | None:
    """Import FLA before Qwen3.5 modeling freezes the torch chunk reference."""

    try:
        import fla

        chunk = fla_chunk_function()
        try:
            installed = importlib.metadata.version(FLA_DISTRIBUTION)
        except importlib.metadata.PackageNotFoundError:
            installed = getattr(fla, "__version__", "")
    except Exception as exc:
        if not required:
            return None
        raise RuntimeError(
            f"{FLA_DISTRIBUTION} is not installed, so chunk_gated_delta_rule would "
            "fall back to its reference PyTorch implementation. "
            "torch_chunk_gated_delta_rule is not served."
        ) from exc
    if chunk is None:
        raise RuntimeError("fla does not export chunk_gated_delta_rule")
    if not isinstance(installed, str) or version_base(installed) != FLA_VERSION:
        raise RuntimeError(
            f"{FLA_DISTRIBUTION} {installed!r} is loaded; expected {FLA_VERSION}"
        )
    return version_base(installed)


def resolve_flashinfer(installed: str | None, *, imported: bool) -> tuple[str | None, str]:
    """Choose the FlashInfer pin without refusing capability 9.0.

    ``absent`` and ``import-failed`` leave the FLA chunk bind in place.
    A different installed version is a startup error.
    """

    if installed is None:
        return None, "absent"
    if version_base(installed) != FI_VERSION:
        raise RuntimeError(f"{FI_DISTRIBUTION} {installed!r} is loaded; expected {FI_VERSION}")
    if not imported:
        return None, "import-failed"
    return version_base(installed), "bound"


def prepare_flashinfer() -> tuple[str | None, str]:
    """Import the pinned FlashInfer GDN prefill when this environment has it."""

    try:
        installed = importlib.metadata.version(FI_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    imported = False
    if installed is not None and version_base(installed) == FI_VERSION:
        try:
            importlib.import_module(FI_MODULE)
        except (ImportError, OSError):
            imported = False
        else:
            imported = True
    return resolve_flashinfer(installed, imported=imported)


def assert_qwen_chunk_uses_fla() -> None:
    """Fail if modeling still closed over the PyTorch chunk reference."""

    modeling = importlib.import_module(MODELING)
    if not binds_package(getattr(modeling, _CHUNK, None), fla_chunk_function()):
        raise RuntimeError(
            "Transformers still has the PyTorch chunk_gated_delta_rule fallback. "
            "The load warning is `falling back to its reference PyTorch "
            "implementation`. torch_chunk_gated_delta_rule is not served."
        )


def cu_seqlens_key(device_index: int | None, length: int) -> tuple[int, int]:
    """Cache identity for a device and a sequence length."""

    return (-1 if device_index is None else int(device_index), int(length))


def reuse_cu_seqlens_slot(
    pool: dict[tuple[int, int], list[Any]],
    cursors: dict[tuple[int, int], int],
    key: tuple[int, int],
    slots: int,
    allocate: Any,
) -> Any:
    """Return the next buffer in this key's ring.

    The first call allocates ``slots`` tensors. Later calls rotate and
    do not allocate. ``slots`` is the linear-layer count, so one forward
    does not hand the same tensor to two layers.
    """

    found = pool.get(key)
    if found is None:
        found = [allocate() for _ in range(slots)]
        pool[key] = found
        cursors[key] = 0
    cursor = cursors[key]
    chosen = found[cursor]
    cursors[key] = (cursor + 1) % len(found)
    return chosen


def cached_cu_seqlens(q: Any) -> Any:
    """Rotate a per-layer ``[0, length]`` instead of allocating every layer."""

    import torch

    length = int(q.shape[0])
    index = q.device.index
    key = cu_seqlens_key(index, length)

    def allocate() -> Any:
        return torch.tensor([0, length], dtype=torch.int64, device=q.device)

    return reuse_cu_seqlens_slot(
        _CU_SEQLENS_POOL,
        _CU_SEQLENS_CURSOR,
        key,
        LINEAR_ATTENTION_LAYERS,
        allocate,
    )


def flashinfer_chunk_gated_delta_rule(
    q: Any,
    k: Any,
    v: Any,
    g: Any = None,
    beta: Any = None,
    scale: float | None = None,
    initial_state: Any = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Any = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """One-sequence GDN prefill through FlashInfer's SM90 kernel.

    Transformers 5.17 calls this with FLA's ``[B, T, H, D]`` layout and a
    log-space decay ``g``. FlashInfer 0.6.16 wants ``[T, H, D]``, a float32
    multiplicative gate, and a non-null ``cu_seqlens``. ``use_cp`` is false.
    """

    del kwargs
    if initial_state is not None or output_final_state:
        raise RuntimeError(
            "pplx-decider FlashInfer GDN prefill does not continue or return a "
            "recurrent state. This worker serves one full-prompt prefill."
        )
    if getattr(q, "shape", (0,))[0] != 1:
        raise RuntimeError(
            "pplx-decider FlashInfer GDN prefill serves one sequence, "
            f"got batch {getattr(q, 'shape', None)}"
        )

    import torch
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as chunk_gated_delta_rule_fi
    from fla.modules.l2norm import l2norm_fwd

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()
    fi_g = None if g is None else torch.exp(g.squeeze(0).contiguous().to(torch.float32))
    fi_beta = None if beta is None else beta.squeeze(0).contiguous().to(torch.float32)
    if cu_seqlens is None:
        cu_seqlens = cached_cu_seqlens(q)
    else:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int64)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=fi_g,
        beta=fi_beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        use_cp=False,
    )
    return result.unsqueeze(0), None


def use_short_flashinfer(batch: int, tokens: int) -> bool:
    """FlashInfer only for one sequence at or under the candidate boundary.

    The all-sequence bind improved p50 and missed p95. A longer sequence,
    or more than one sequence in the forward, stays on FLA chunk.
    """

    return batch == 1 and 0 < tokens <= SHORT_FLASHINFER_TOKENS


def dispatch_chunk_gated_delta_rule(
    q: Any,
    k: Any,
    v: Any,
    g: Any = None,
    beta: Any = None,
    scale: float | None = None,
    initial_state: Any = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Any = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """One short sequence uses FlashInfer. Every other shape uses FLA chunk."""

    shape = getattr(q, "shape", ())
    batch = shape[0] if len(shape) >= 1 else 0
    tokens = shape[1] if len(shape) >= 2 else 0
    if use_short_flashinfer(batch, tokens):
        return flashinfer_chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
    return fla_chunk_function()(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )


def install_flashinfer_gdn_prefill(modeling: Any) -> None:
    """Replace the chunk global before the checkpoint builds Qwen layers."""

    if getattr(modeling, "causal_conv1d_fn", None) is None:
        raise RuntimeError(
            "causal_conv1d_fn is not bound. Import causal_conv1d before "
            f"{MODELING}. F.conv1d is not served."
        )
    if getattr(modeling, _CHUNK, None) is None:
        raise RuntimeError(
            f"{_CHUNK} is not bound. Import {FLA_DISTRIBUTION}=={FLA_VERSION} "
            f"before {MODELING}. torch_chunk_gated_delta_rule is not served."
        )
    setattr(modeling, _CHUNK, flashinfer_chunk_gated_delta_rule)
    if getattr(modeling, "chunk_gated_delta_rule", None) is not None:
        modeling.chunk_gated_delta_rule = flashinfer_chunk_gated_delta_rule
    modeling.pplx_gdn_prefill = GDN_PREFILL


def install_short_flashinfer_prefill(modeling: Any) -> None:
    """Install length dispatch. This is not the all-sequence FlashInfer bind."""

    if getattr(modeling, "causal_conv1d_fn", None) is None:
        raise RuntimeError(
            "causal_conv1d_fn is not bound. Import causal_conv1d before "
            f"{MODELING}. F.conv1d is not served."
        )
    if getattr(modeling, _CHUNK, None) is None:
        raise RuntimeError(
            f"{_CHUNK} is not bound. Import {FLA_DISTRIBUTION}=={FLA_VERSION} "
            f"before {MODELING}. torch_chunk_gated_delta_rule is not served."
        )
    setattr(modeling, _CHUNK, dispatch_chunk_gated_delta_rule)
    if getattr(modeling, "chunk_gated_delta_rule", None) is not None:
        modeling.chunk_gated_delta_rule = dispatch_chunk_gated_delta_rule
    modeling.pplx_gdn_prefill = "flashinfer-short"


def bound_linear_attention(modeling: Any, *, fla_chunk: Any = None) -> str:
    """Name the linear-attention implementation the modeling module exposes."""

    if modeling is None:
        return "unloaded"
    chunk = getattr(modeling, _CHUNK, None)
    conv = getattr(modeling, "causal_conv1d_fn", None) is not None
    if chunk is dispatch_chunk_gated_delta_rule and conv:
        return SHORT_FLASHINFER_LINEAR_ATTENTION
    if chunk is flashinfer_chunk_gated_delta_rule and conv:
        return FLASHINFER_LINEAR_ATTENTION
    reference = fla_chunk
    if reference is None and chunk is not None and chunk is not flashinfer_chunk_gated_delta_rule:
        reference = fla_chunk_function()
    fla_bound = (
        chunk is not None and reference is not None and binds_package(chunk, reference)
    )
    if fla_bound and conv:
        return FLA_LINEAR_ATTENTION
    if fla_bound:
        return "fla"
    if chunk is dispatch_chunk_gated_delta_rule:
        return "flashinfer-short"
    if chunk is flashinfer_chunk_gated_delta_rule:
        return GDN_PREFILL
    if conv:
        return "causal_conv1d"
    return "torch_chunk"


def require_bound_linear_attention(modeling: Any, *, fla_chunk: Any = None) -> str:
    """Accept FLA chunk or the FlashInfer prefill. Capability 9.0 is not required."""

    bound = bound_linear_attention(modeling, fla_chunk=fla_chunk)
    if bound in READY_LINEAR_ATTENTION:
        return bound
    raise RuntimeError(
        f"pplx-decider linear attention is {bound}, so the worker is not ready. "
        f"Ready accepts {FLA_LINEAR_ATTENTION}, {SHORT_FLASHINFER_LINEAR_ATTENTION}, "
        f"or {FLASHINFER_LINEAR_ATTENTION}. "
        "Capability 8.9 stays on the FLA chunk. Capability 9.0 with the "
        "FlashInfer pin uses the all-sequence bind."
    )
