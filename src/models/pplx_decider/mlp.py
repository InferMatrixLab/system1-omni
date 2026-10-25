"""Fuse the text MLP gate and up projections.

Qwen3.5 runs ``gate_proj`` and ``up_proj`` as two GEMMs, then SiLU and the
down projection. Those two GEMMs read the same input. One GEMM reads it
once. The fused weight is allocated per layer and the separate parameters
are dropped, so the resident set returns to the same projections.

cuBLAS may accumulate the wide GEMM differently from the two narrow ones.
This is not bit-exact. ``PPLX_MLP=default`` keeps the checkpoint modules.
"""

from __future__ import annotations

from typing import Any

MLP_FUSED = "fused-gate-up"
MLP_DEFAULT = "default"


def mlp_preference(device: str, selected: str | None = None) -> str:
    """CUDA startup fuses gate and up. Any other device keeps the checkpoint MLP.

    ``selected`` is ``PPLX_MLP``. ``default`` reproduces the primary MLP.
    """

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if selected is not None and selected not in {MLP_FUSED, MLP_DEFAULT}:
        raise RuntimeError("PPLX_MLP must be fused-gate-up or default")
    if device != "cuda":
        return MLP_DEFAULT
    if selected is None:
        return MLP_FUSED
    return selected


def prepare_fused_gate_up(mlp: Any) -> int:
    """Check one text MLP and return its intermediate size.

    Raises before any weight is moved when the module is not the unbiased
    gate/up/down stack this fusion knows.
    """

    if getattr(mlp, "fused_gate_up", None) is not None:
        raise RuntimeError("gate-up is already fused")
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if gate is None or up is None:
        raise RuntimeError("text MLP is missing gate_proj or up_proj")
    if getattr(gate, "bias", None) is not None or getattr(up, "bias", None) is not None:
        raise RuntimeError("fused gate-up expects unbiased projections")
    if gate.in_features != up.in_features or gate.out_features != up.out_features:
        raise RuntimeError("gate and up projections differ")
    intermediate = getattr(mlp, "intermediate_size", None)
    if intermediate != gate.out_features:
        raise RuntimeError("intermediate size does not match gate_proj")
    down = getattr(mlp, "down_proj", None)
    if down is None or getattr(down, "in_features", None) != intermediate:
        raise RuntimeError("down_proj does not match the fused MLP")
    if getattr(mlp, "act_fn", None) is None:
        raise RuntimeError("text MLP is missing act_fn")
    return intermediate


def fused_gate_up_forward(self: Any, hidden: Any) -> Any:
    """One gate-up GEMM, then the checkpoint SiLU, multiply, and down projection."""

    import torch.nn.functional as F

    gate_up = F.linear(hidden, self.fused_gate_up)
    gate, up = gate_up.split(self.intermediate_size, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


def install_fused_gate_up(language_model: Any) -> int:
    """Fuse every text-layer MLP. Returns the number of layers changed."""

    import torch

    layers = getattr(language_model, "layers", None)
    if not layers:
        raise RuntimeError("language model has no layers")
    planned = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise RuntimeError("text layer has no MLP")
        planned.append((mlp, prepare_fused_gate_up(mlp)))
    for mlp, intermediate in planned:
        gate = mlp.gate_proj
        fused = torch.empty(
            intermediate * 2,
            gate.in_features,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        with torch.no_grad():
            fused[:intermediate].copy_(gate.weight.detach())
            fused[intermediate:].copy_(mlp.up_proj.weight.detach())
        mlp.fused_gate_up = torch.nn.Parameter(fused, requires_grad=False)
        del mlp.gate_proj
        del mlp.up_proj
        mlp.forward = fused_gate_up_forward.__get__(mlp, type(mlp))
    return len(planned)
