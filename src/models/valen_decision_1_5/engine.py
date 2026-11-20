"""Valen-Decision-1.5 executor for one pinned released checkpoint.

The executor owns the loaded model (backbone, block-structured attention,
decision head), the forward pass, device state, and warmup. It consumes a
compiled shared-state sequence and returns logits and token accounting only;
API probabilities and response reconstruction remain in postprocess.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .pinned import Checkpoint
from .preprocess import CompiledInput

DTYPES = ("bf16", "fp32")


@dataclass(frozen=True)
class ExecutorOutput:
    logits: tuple[tuple[float, ...], ...]
    logical_tokens: int
    compute_tokens: int


class ValenDecisionExecutor:
    """Load and execute one immutable Valen-Decision-1.5 model instance."""

    def __init__(self, checkpoint: Checkpoint, device: str = "cuda", dtype: str = "bf16") -> None:
        if dtype not in DTYPES:
            raise ValueError("dtype must be bf16 or fp32")
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "install the reference dependencies before starting"
            ) from exc

        # The settings of the released Transformers quickstart: exact FP32
        # matmul in the FP32 heads and no TF32 convolutions.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        # trust_remote_code runs the checkpoint's own runtime files, which
        # pinned.verify_checkpoint has hashed. The decision head stays FP32
        # under either backbone dtype.
        self.model = (
            AutoModel.from_pretrained(
                str(checkpoint.path),
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch.bfloat16 if dtype == "bf16" else torch.float32,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        self.model_name = checkpoint.release.model_name
        self._lock = threading.RLock()
        self._closed = False

    def execute(self, compiled: CompiledInput) -> ExecutorOutput:
        """Run one shared-state forward and return CPU FP32 logits per question."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Valen-Decision-1.5 executor is closed")
            import torch

            state = compiled.state
            if state.inputs is None:
                raise RuntimeError("Valen-Decision-1.5 requires a shared-state sequence")
            with torch.inference_mode():
                rows = []
                for logits in self.model.forward_state(state):
                    if logits.ndim != 1:
                        raise RuntimeError("Valen-Decision-1.5 returned non-vector logits")
                    rows.append(tuple(logits.float().cpu().tolist()))
            return ExecutorOutput(
                tuple(rows),
                int(state.logical_tokens),
                int(state.compute_tokens),
            )

    def warmup(self, compiled: CompiledInput) -> None:
        """Run a real representative compile and forward before readiness."""

        self.execute(compiled)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.model = None
