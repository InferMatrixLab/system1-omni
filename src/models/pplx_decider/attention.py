"""Unpadded full-attention path for the checkpoint's SDPA hook.

Transformers 5.17 ``sdpa_attention_forward`` keeps the grouped-query kernel
only when the mask is ``None`` and the module is not causal
(``use_gqa_in_sdpa`` also requires ``head_dim <= 256``; this checkpoint's
full-attention ``head_dim`` is 256). The checkpoint hook always writes a
bool ``[batch, 1, 1, time]`` mask, so those layers stay on the masked math
kernel even when every bit is true.

``install_unpadded_full_attention`` runs after ``DecisionModel`` loads. It
does not edit hashed checkpoint files. Full-attention modules are marked
noncausal first, so a later hook failure still cannot make an all-ones mask
causal: the original hook keeps supplying that mask. When every padding bit
is true the replacement hook omits the explicit full-attention mask. Any
false bit keeps ``padding[:, None, None, :].bool()``. Linear attention still
receives ``create_recurrent_attention_mask``.

Flash and math SDPA are not bit-exact. A latency sample for this path is
unfinished until decision disagreements against head ``1ba07d1`` are counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_CHECKPOINT_HOOK = "mask_inputs"
SDPA_DEFAULT = "default"


def omit_explicit_full_attention_mask(valid_bits: Sequence[bool]) -> bool:
    """Return whether the bool full-attention mask would be all true.

    The runtime hook passes the one-bit result of ``padding.bool().all()``.
    An empty mask is the checkpoint's rejected shape, not an unpadded row.
    """

    if not valid_bits:
        raise ValueError("Padding mask must match the complete nonempty input")
    return all(valid_bits)


def noncausal_attention_mask(fully_valid: bool, bool_mask: Any, linear_mask: Any) -> dict[str, Any]:
    """Build the pre-hook ``attention_mask`` dict.

    ``linear_mask`` is returned as the same object. ``bool_mask`` is omitted
    when ``fully_valid`` is true.
    """

    if type(fully_valid) is not bool:
        raise TypeError("fully_valid must be bool")
    return {
        "full_attention": None if fully_valid else bool_mask,
        "linear_attention": linear_mask,
    }


def _sdpa(language_model: Any) -> None:
    config = getattr(language_model, "config", None)
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("unpadded full attention requires SDPA")


def _mask_hook_id(language_model: Any) -> int:
    hooks = getattr(language_model, "_forward_pre_hooks", None)
    if not hooks:
        raise RuntimeError("checkpoint noncausal pre-hook is missing")
    found = []
    for key, current in hooks.items():
        fn = getattr(current, "__func__", current)
        if getattr(fn, "__name__", "") == _CHECKPOINT_HOOK:
            found.append(key)
    if len(found) != 1:
        raise RuntimeError("checkpoint noncausal pre-hook is missing")
    hook_id = found[0]
    with_kwargs = getattr(language_model, "_forward_pre_hooks_with_kwargs", None)
    if with_kwargs is not None and hook_id not in with_kwargs:
        raise RuntimeError("checkpoint noncausal pre-hook is missing keyword inputs")
    return hook_id


def mark_full_attention_noncausal(language_model: Any) -> int:
    """Set ``is_causal = False`` on full-attention modules only.

    Validates every full-attention module before writing any of them.
    """

    modules = []
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("language model has no layers")
    for layer in layers:
        if getattr(layer, "block_type", None) != "full_attention":
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "is_causal", None) is not True:
            raise RuntimeError("full-attention module was not causal")
        modules.append(attn)
    if not modules:
        raise RuntimeError("no full-attention layers")
    for attn in modules:
        attn.is_causal = False
    return len(modules)


def mask_inputs_unpadded(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Checkpoint pre-hook, with the all-true full-attention mask omitted."""

    import torch
    from transformers.masking_utils import create_recurrent_attention_mask

    if args:
        raise ValueError("The attention ablation requires keyword inputs")
    if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
        raise ValueError("Noncausal classification does not support a KV cache")
    embeddings = kwargs.get("inputs_embeds")
    if embeddings is None:
        embeddings = module.embed_tokens(kwargs["input_ids"])
    padding = kwargs.get("attention_mask")
    if padding is None:
        padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
    if not isinstance(padding, torch.Tensor) or padding.ndim != 2:
        raise ValueError("Expected the processor's 2D padding mask")
    if padding.shape != embeddings.shape[:2]:
        raise ValueError("Padding mask must match the complete nonempty input")
    valid = padding.bool()
    # One device sync. The checkpoint hook already syncs on the nonempty check.
    packed = torch.stack((valid.any(dim=-1).all(), valid.all()))
    row_nonempty, fully_valid_bit = (bool(value) for value in packed.tolist())
    if not row_nonempty:
        raise ValueError("Padding mask must match the complete nonempty input")
    fully_valid = omit_explicit_full_attention_mask([fully_valid_bit])
    bool_mask = None if fully_valid else valid[:, None, None, :]
    kwargs["attention_mask"] = noncausal_attention_mask(
        fully_valid,
        bool_mask,
        create_recurrent_attention_mask(
            config=module.config,
            inputs_embeds=embeddings,
            attention_mask=padding,
        ),
    )
    return args, kwargs


def sdpa_preference(device: str, selected: str | None = None) -> str:
    """Keep the checkpoint SDPA order on every device.

    The checkpoint turns cuDNN SDPA off. ``PPLX_SDPA=cudnn-first`` missed
    the gate and is not selectable. Unset and ``default`` are the primary.
    """

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if selected in {None, SDPA_DEFAULT}:
        return SDPA_DEFAULT
    if selected == "cudnn-first":
        raise RuntimeError("PPLX_SDPA=cudnn-first missed the gate and is not selectable")
    raise RuntimeError("PPLX_SDPA must be unset or default")


def install_unpadded_full_attention(language_model: Any) -> int:
    """Replace the checkpoint mask hook. Returns the number of retargeted layers."""

    _sdpa(language_model)
    hook_id = _mask_hook_id(language_model)
    changed = mark_full_attention_noncausal(language_model)
    language_model._forward_pre_hooks[hook_id] = mask_inputs_unpadded
    return changed
