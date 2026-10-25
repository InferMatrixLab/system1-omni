"""FLA gated RMSNorm on the 48 linear-attention output norms.

``801ef7b`` installed this on the all-sequence path and missed the
median: p50 89.172 ms (+1.390 ms) and p95 274.790 ms (−16.393 ms)
against ``fcf5e2b`` 87.782 / 291.183. The engine does not call
``install_fused_gated_norm``. Do not length-gate it. Always-on residual
fusion ``3afb472`` and the 512-token gate ``705ae089`` missed the same
way. This module stays as the rejected implementation.

Residual RMSNorm stays eager. The checkpoint does not pass
``use_kernels=True``, and kernels 0.16 only annotates
``Qwen3_5RMSNormGated``, so the kept primary still runs the eager
chain: fp32 variance, a cast, the weight multiply, and SiLU(gate).

FLA 0.5.2 ``layer_norm_gated_fwd`` is one kernel. For ``D<=512`` its
autotune key uses ``cdiv(rows, 65536)``, not the token count. The GVA
forward passes ``[seq * 48, 128]``, so about 136 tokens is bucket 1 and
about 2676 tokens is bucket 2. The worker max of 8192 tokens is bucket
6. ``rms_norm_gated`` is an autograd wrapper and raises under
``inference_mode``, so the forward calls ``layer_norm_gated_fwd``.

``/health`` reports ``gated_norm`` and ``fused_gated_norm_layers``.
``rms_norm`` stays ``eager``. That is not ``rms_norm=fla``.
"""

from __future__ import annotations

from typing import Any

from .gdn import LINEAR_ATTENTION_LAYERS

GATED_NORM_EAGER = "eager"
GATED_NORM_FLA = "fla"
_GATED_NORM_CLASS = "Qwen3_5RMSNormGated"
_GATED_ACTIVATIONS = frozenset({"silu", "swish", "sigmoid"})
# FLA 0.5.2 layer_norm_gated_fwd: NB = cdiv(rows, 2048 * 32) when D <= 512.
_FLA_GATED_NB_ROWS = 2048 * 32
# One warmup row count per bucket the 8192-token cap can reach.
# 136*48 and 2676*48 are the profiled median and tail, not a fitted cutoff.
GATED_NORM_WARM_ROWS = (136 * 48, 2676 * 48, 4096 * 48, 5461 * 48, 6826 * 48, 8192 * 48)


def gated_norm_nb(rows: int) -> int:
    """FLA autotune bucket for a flattened gated-norm row count."""

    if type(rows) is not int or rows < 1:
        raise ValueError("gated-norm rows must be a positive int")
    return (rows + _FLA_GATED_NB_ROWS - 1) // _FLA_GATED_NB_ROWS


def fused_gated_rms_forward(self: Any, hidden_states: Any, gate: Any) -> Any:
    """One FLA RMSNorm+SiLU. ``hidden_states`` and ``gate`` share a shape."""

    from fla.modules.fused_norm_gate import layer_norm_gated_fwd

    if self.activation not in _GATED_ACTIVATIONS:
        raise RuntimeError("gated RMSNorm activation is not silu, swish, or sigmoid")
    if tuple(hidden_states.shape) != tuple(gate.shape):
        raise RuntimeError("gated RMSNorm expected the gate to match the attention output")
    rows = hidden_states.shape[-1]
    flat_x = hidden_states.reshape(-1, rows)
    flat_g = gate.reshape(-1, rows)
    if not flat_x.is_contiguous():
        flat_x = flat_x.contiguous()
    if not flat_g.is_contiguous():
        flat_g = flat_g.contiguous()
    y, _, _, _ = layer_norm_gated_fwd(
        flat_x,
        flat_g,
        self.weight,
        None,
        activation=self.activation,
        eps=self.variance_epsilon,
        is_rms_norm=True,
    )
    return y.reshape(hidden_states.shape)


def install_fused_gated_norm(language_model: Any) -> int:
    """Replace the eager gated norm on every linear layer. Returns the count.

    Valid only on the all-sequence GVA bind. Residual norms are left alone.
    A forward that is already not the class eager forward is a different
    kernel, and stacking another one is not this lever.
    """

    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("language model has no layers")
    count = 0
    for layer in layers:
        if getattr(layer, "block_type", None) != "linear_attention":
            continue
        attn = getattr(layer, "linear_attn", None)
        norm = getattr(attn, "norm", None)
        if type(norm).__name__ != _GATED_NORM_CLASS:
            raise RuntimeError("linear attention norm is not Qwen3_5RMSNormGated")
        eager = getattr(type(norm), "forward", None)
        current = getattr(getattr(norm, "forward", None), "__func__", None)
        if current is not eager:
            raise RuntimeError("linear gated norm is not the eager forward")
        norm.forward = fused_gated_rms_forward.__get__(norm, type(norm))
        norm.pplx_gated_norm = GATED_NORM_FLA
        count += 1
    if count != LINEAR_ATTENTION_LAYERS:
        raise RuntimeError(
            f"fused gated norm expected {LINEAR_ATTENTION_LAYERS} linear layers, found {count}"
        )
    return count


def warm_fused_gated_norm(language_model: Any) -> None:
    """Autotune each FLA bucket before the first measured request.

    The key ignores the token count inside a bucket. Warming a bucket at
    the profiled length keeps that choice. Later requests in the same
    bucket do not benchmark again.
    """

    import torch

    layers = getattr(language_model, "layers", None)
    norm = None
    for layer in layers or ():
        attn = getattr(layer, "linear_attn", None)
        candidate = getattr(attn, "norm", None)
        if getattr(candidate, "pplx_gated_norm", None) == GATED_NORM_FLA:
            norm = candidate
            break
    if norm is None:
        raise RuntimeError("fused gated norm was not installed")
    weight = norm.weight
    if weight.device.type != "cuda":
        raise RuntimeError("fused gated norm warmup requires CUDA")
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd

    features = int(weight.shape[0])
    for rows in GATED_NORM_WARM_ROWS:
        flat_x = torch.zeros(rows, features, device=weight.device, dtype=weight.dtype)
        flat_g = torch.zeros_like(flat_x)
        layer_norm_gated_fwd(
            flat_x,
            flat_g,
            weight,
            None,
            activation=norm.activation,
            eps=norm.variance_epsilon,
            is_rms_norm=True,
        )
    torch.cuda.synchronize()
