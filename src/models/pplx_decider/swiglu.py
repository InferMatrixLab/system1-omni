"""FLA SwiGLU on the 64 text MLPs. Not a norm fusion.

``801ef7b`` replaced the 48 linear gated RMSNorms with FLA
``layer_norm_gated_fwd`` and missed the median: p50 89.172 ms, p95
274.790 ms against ``fcf5e2b`` 87.782 / 291.183. That kernel stays
unwired. ``3afb472`` and ``705ae089`` missed the same way on the
residual norm. This lever does not replace a norm.

The checkpoint MLP is ``silu(gate_proj(x)) * up_proj(x)`` and then
``down_proj``. Those two GEMMs stay separate. Stacking them
(``8b4ee326``, N=34816) missed both percentiles. FLA 0.5.2
``swiglu_fwd`` reads the two activations and writes the product in one
elementwise kernel. Its autotune key is ``D``, the intermediate size,
not the token count, so Public231 lengths do not rebenchmark. Startup
warms one forward at 136 tokens so the chosen block is the median
shape. ``swiglu`` and ``swiglu_linear`` are autograd wrappers and raise
under ``inference_mode``. They are not called. ``down_proj`` stays
``F.linear``.

``/health`` reports ``mlp=swiglu`` and ``fused_swiglu_layers``.
``fused_gate_up_layers`` stays 0. ``gated_norm`` stays ``eager``.
"""

from __future__ import annotations

from typing import Any

MLP_SWIGLU = "swiglu"
# Qwen3.5 27B text stack. Every decoder layer has an MLP, including the
# 16 full-attention layers. This is that architecture count.
TEXT_MLP_LAYERS = 64
# Profiled median length. The autotune key ignores it, so the first
# warmup shape is the one whose block size is kept for every length.
MEDIAN_SWIGLU_TOKENS = 136
_MLP_CLASS = "Qwen3_5MLP"


def swiglu_mlp_forward(self: Any, x: Any) -> Any:
    """``silu(gate) * up`` in one kernel, then the checkpoint down GEMM."""

    from fla.modules.activations import swiglu_fwd

    return self.down_proj(swiglu_fwd(self.gate_proj(x), self.up_proj(x)))


def install_swiglu_mlp(language_model: Any) -> int:
    """Replace the eager SwiGLU product on every text MLP. Returns the count.

    Valid only on the all-sequence GVA bind. Gate, up, and down weights
    are not stacked. A forward that is already not the class eager
    forward is a different kernel, and stacking another one is not this
    lever.
    """

    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("language model has no layers")
    count = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if type(mlp).__name__ != _MLP_CLASS:
            raise RuntimeError("text MLP is not Qwen3_5MLP")
        eager = getattr(type(mlp), "forward", None)
        current = getattr(getattr(mlp, "forward", None), "__func__", None)
        if current is not eager:
            raise RuntimeError("text MLP is not the eager forward")
        if getattr(mlp, "gate_proj", None) is None or getattr(mlp, "up_proj", None) is None:
            raise RuntimeError("text MLP is missing a gate or up projection")
        if getattr(mlp, "down_proj", None) is None:
            raise RuntimeError("text MLP is missing a down projection")
        mlp.forward = swiglu_mlp_forward.__get__(mlp, type(mlp))
        mlp.pplx_mlp = MLP_SWIGLU
        count += 1
    if count != TEXT_MLP_LAYERS:
        raise RuntimeError(
            f"SwiGLU MLP expected {TEXT_MLP_LAYERS} text layers, found {count}"
        )
    return count


def warm_swiglu_mlp(language_model: Any) -> None:
    """Autotune ``swiglu_fwd`` at the median length before the first request.

    The key is the intermediate size. One median-shaped call selects the
    block. Later lengths do not benchmark again.
    """

    import torch

    layers = getattr(language_model, "layers", None)
    mlp = None
    for layer in layers or ():
        candidate = getattr(layer, "mlp", None)
        if getattr(candidate, "pplx_mlp", None) == MLP_SWIGLU:
            mlp = candidate
            break
    if mlp is None:
        raise RuntimeError("SwiGLU MLP was not installed")
    weight = mlp.gate_proj.weight
    if weight.device.type != "cuda":
        raise RuntimeError("SwiGLU MLP warmup requires CUDA")
    from fla.modules.activations import swiglu_fwd

    intermediate = int(weight.shape[0])
    gate = torch.zeros(
        1,
        MEDIAN_SWIGLU_TOKENS,
        intermediate,
        device=weight.device,
        dtype=weight.dtype,
    )
    swiglu_fwd(gate, torch.zeros_like(gate))
    torch.cuda.synchronize()
