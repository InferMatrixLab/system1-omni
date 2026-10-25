"""Text MLP startup choice.

The fused gate-up GEMM was measured on ``8b4ee326`` and missed the gate.
It is not selectable. Startup keeps the checkpoint's separate gate and up
projections. The SwiGLU elementwise lever does not set ``PPLX_MLP`` and
does not stack those GEMMs.
"""

from __future__ import annotations

MLP_DEFAULT = "default"


def mlp_preference(device: str, selected: str | None = None) -> str:
    """Keep the checkpoint MLP on every device.

    ``PPLX_MLP=fused-gate-up`` missed the gate and is not selectable.
    """

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if selected in {None, MLP_DEFAULT}:
        return MLP_DEFAULT
    if selected == "fused-gate-up":
        raise RuntimeError("PPLX_MLP=fused-gate-up missed the gate and is not selectable")
    raise RuntimeError("PPLX_MLP must be unset or default")
