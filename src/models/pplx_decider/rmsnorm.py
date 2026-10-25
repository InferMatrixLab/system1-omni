"""Fuse the checkpoint's zero-centered residual RMSNorms.

``Qwen3_5RMSNorm`` computes ``rms(x.float()) * (1 + weight.float())`` and
casts back. Eager runs that as a separate kernel chain, twice per text
layer plus once on the final hidden state. At the median those
activations are small, so the chain is launch-bound. At the tail the
same chain also rewrites the fp32 activation several times.

FLA 0.5.2 ``layer_norm_fwd`` is one kernel. Its autotune key ignores
sequence length when the normalized dimension is above 512, which is
this checkpoint's hidden size (5120). Query and key norms are head_dim
256 and stay eager: that kernel's autotune key changes with length.
The gated RMSNorm on linear layers stays eager. This is not a wider
GEMM and not a workspace ring.

``evaluate`` runs under ``torch.inference_mode``. FLA's autograd
``rms_norm`` calls ``save_for_backward`` and would raise, so the forward
calls ``layer_norm_fwd`` directly. The fp32 scale is ``1 + weight``,
matching the zero-centered affine, and it is built once at install.
"""

from __future__ import annotations

from typing import Any

RMS_NORM_EAGER = "eager"
RMS_NORM_FLA = "fla"
# FLA 0.5.2 ``layer_norm_fwd`` uses the length-independent kernel only
# above this feature count. At or below it, the autotune key includes
# ``cdiv(T, 2048)`` and Public231 lengths would rebenchmark.
FLA_RMS_LENGTH_STABLE_FEATURES = 512
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


def _fused_forward(self: Any, x: Any) -> Any:
    if _LAYER_NORM_FWD is None:
        raise RuntimeError("fused residual RMSNorm is not installed")
    return fused_residual_rms_norm(self, x, _LAYER_NORM_FWD)


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
        norm._pplx_rms_scale = scale
        norm.forward = _fused_forward.__get__(norm, type(norm))
    return len(norms)
