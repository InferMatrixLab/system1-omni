"""Rejected length-gated FLA residual RMSNorm. The engine does not install it.

Always-on fusion ``3afb472`` was p50 93.935 ms and p95 274.902 ms.
The 512-token gate ``705ae089`` was p50 94.604 ms and p95 274.835 ms.
Both missed the median against ``4b99b73`` at 88.942 / 295.797. Do not
wire this module back without a new hypothesis that keeps p50 under
88.942 ms.

``Qwen3_5RMSNorm`` computes ``rms(x.float()) * (1 + weight.float())`` and
casts back. Eager runs that as a kernel chain, twice per text layer plus
once on the final hidden state. FLA 0.5.2 ``layer_norm_fwd`` is one
kernel. Its autotune key ignores sequence length when the normalized
dimension is above 512. This checkpoint's hidden size is 5120. Query
and key norms are head_dim 256 and stay eager. The gated RMSNorm stays
eager. The functions below are the rejected implementation, kept so the
CPU contract of that attempt stays tested.

``evaluate`` runs under ``torch.inference_mode``. FLA's autograd
``rms_norm`` calls ``save_for_backward`` and would raise, so the long
path calls ``layer_norm_fwd`` directly. The fp32 scale is ``1 + weight``
and is built once at install.
"""

from __future__ import annotations

from typing import Any

RMS_NORM_EAGER = "eager"
# Always-on ``rms_norm=fla`` is the rejected ``3afb472`` tip.
RMS_NORM_FLA_LONG = "fla-long"
# FLA 0.5.2 ``layer_norm_fwd`` uses the length-independent kernel only
# above this feature count. At or below it, the autotune key includes
# ``cdiv(T, 2048)`` and Public231 lengths would rebenchmark.
FLA_RMS_LENGTH_STABLE_FEATURES = 512
# Eager at or below this many tokens. FLA above it. Between the profiled
# median of about 136 tokens and the profiled tail of about 2676.
FLA_RMS_MIN_TOKENS = 512
_RESIDUAL_NORM_NAMES = ("input_layernorm", "post_attention_layernorm")
_LAYER_NORM_FWD: Any = None


def fused_residual_rms_norm(norm: Any, x: Any, layer_norm_fwd: Any) -> Any:
    """One FLA RMSNorm. ``x`` may be rank 2 or higher; the last dim is features."""

    rows = x.shape[-1]
    flat = x.reshape(-1, rows)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    y, _, _, _ = layer_norm_fwd(
        flat,
        norm._pplx_rms_scale,
        None,
        norm.eps,
        is_rms_norm=True,
    )
    return y.reshape(x.shape)


def sequence_tokens(x: Any) -> int:
    """Sequence length of a residual RMSNorm input. The last dim is features."""

    shape = getattr(x, "shape", None)
    if shape is None or len(shape) < 2:
        raise RuntimeError("residual RMSNorm expects a batch and a sequence")
    return int(shape[-2])


def dispatch_residual_rms_norm(
    norm: Any,
    x: Any,
    layer_norm_fwd: Any,
    min_tokens: int = FLA_RMS_MIN_TOKENS,
) -> Any:
    """Eager at or below ``min_tokens``. FLA above it."""

    if sequence_tokens(x) <= min_tokens:
        original = getattr(norm, "_pplx_rms_original", None)
        if original is None:
            raise RuntimeError("residual RMSNorm eager forward was not saved")
        return original(x)
    return fused_residual_rms_norm(norm, x, layer_norm_fwd)


def _fused_forward(self: Any, x: Any) -> Any:
    if _LAYER_NORM_FWD is None:
        raise RuntimeError("fused residual RMSNorm is not installed")
    return dispatch_residual_rms_norm(self, x, _LAYER_NORM_FWD)


def prepare_residual_rms_norm(norm: Any) -> int:
    """Check one zero-centered residual norm. Returns its feature count.

    Raises before any forward is replaced when the module is not the wide
    ``Qwen3_5RMSNorm`` this fusion knows.
    """

    if type(norm).__name__ != "Qwen3_5RMSNorm":
        raise RuntimeError("residual RMSNorm fusion only replaces Qwen3_5RMSNorm")
    if getattr(norm, "_pplx_rms_scale", None) is not None:
        raise RuntimeError("residual RMSNorm is already fused")
    weight = getattr(norm, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 1:
        raise RuntimeError("residual RMSNorm weight must be a vector")
    features = int(weight.shape[0])
    if features <= FLA_RMS_LENGTH_STABLE_FEATURES:
        raise RuntimeError(
            "residual RMSNorm fusion refuses a feature count at or below "
            f"{FLA_RMS_LENGTH_STABLE_FEATURES}; that FLA kernel re-autotunes with length"
        )
    eps = getattr(norm, "eps", None)
    if type(eps) is not float or not eps > 0.0:
        raise RuntimeError("residual RMSNorm eps must be a positive float")
    return features


def residual_rms_norm_modules(language_model: Any) -> list[Any]:
    """Input norm, post-attention norm, and the final norm. Query and key norms stay out."""

    layers = getattr(language_model, "layers", None)
    if not layers:
        raise RuntimeError("language model has no layers")
    found: list[Any] = []
    for layer in layers:
        for name in _RESIDUAL_NORM_NAMES:
            norm = getattr(layer, name, None)
            if norm is None:
                raise RuntimeError(f"text layer is missing {name}")
            prepare_residual_rms_norm(norm)
            found.append(norm)
    final = getattr(language_model, "norm", None)
    if final is None:
        raise RuntimeError("language model is missing the final RMSNorm")
    prepare_residual_rms_norm(final)
    found.append(final)
    return found


def install_fused_residual_rms_norm(language_model: Any) -> int:
    """Replace every residual RMSNorm forward. Returns the number of modules changed."""

    global _LAYER_NORM_FWD
    import torch
    from fla.modules.layernorm import layer_norm_fwd

    norms = residual_rms_norm_modules(language_model)
    _LAYER_NORM_FWD = layer_norm_fwd
    for norm in norms:
        scale = (1.0 + norm.weight.detach()).to(dtype=torch.float32)
        if scale.ndim != 1 or int(scale.shape[0]) != int(norm.weight.shape[0]):
            raise RuntimeError("zero-centered RMSNorm scale does not match the weight")
        if scale.device != norm.weight.device:
            raise RuntimeError("zero-centered RMSNorm scale is on the wrong device")
        norm._pplx_rms_original = norm.forward
        norm._pplx_rms_scale = scale
        norm.forward = _fused_forward.__get__(norm, type(norm))
    return len(norms)
