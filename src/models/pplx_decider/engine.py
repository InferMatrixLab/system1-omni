"""Load the pinned pplx-decider checkpoint and run its own decision code.

The executor hashes the files it executes, or that define the request
contract, before importing them. Weight shards are not read here: hashing
about 49 GiB is a separate operator step in the recipe, and it is not part
of the latency clock. After load, full-attention modules are marked
noncausal and an all-true padding mask no longer becomes an explicit SDPA
mask. That retarget does not change the hashed files.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .attention import install_unpadded_full_attention, sdpa_preference
from .compile_forward import (
    compile_median,
    compile_min_tokens,
    compile_preference,
    install_compiled_forward,
    warmup_compiled_forward,
)
from .mlp import mlp_preference
from .rmsnorm import RMS_NORM_EAGER
from .contract import (
    ATTENTION_MODE,
    BASE_MODEL,
    BASE_REVISION,
    CODES,
    MAX_LENGTH,
    MODEL_NAME,
    POOLING,
    REPO_ID,
    REVISION,
    TEMPERATURE,
    TOKEN_IDS,
    Request,
)
from .conv import assert_qwen_conv_uses_package, prepare_causal_conv1d
from .gdn import (
    GDN_LAYOUT_GVA,
    GDN_LAYOUT_REPEAT,
    GDN_USE_CP,
    LINEAR_ATTENTION_LAYERS,
    MODELING,
    assert_qwen_chunk_uses_fla,
    gdn_prefill_mode,
    install_flashinfer_gdn_prefill,
    install_gva_prefill,
    install_short_flashinfer_prefill,
    prepare_fla,
    prepare_flashinfer,
    require_bound_linear_attention,
    uses_flashinfer_prefill,
)

# SHA-256 of the files at REVISION, computed in this integration from the
# public blobs. They match release-manifest.json except the manifest's own
# README entry, which still records an older README byte length.
PINNED_SHA256 = {
    "source/src/autojev/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "source/src/autojev/model.py":
        "02940963e9fba5736b8f625f860812f31de2e6658a49606fbed49fd7215c20f9",
    "source/src/autojev/types.py":
        "250ba2eea1cd9bc5501bc76ba76b2f4a1d6c692a7fb1bba4b01affff38cb81ca",
    "source/src/autojev/server.py":
        "840088936dbcd19c9c1a0503f348fdc3e5aa07b7b58c690eaccc9dda314691c1",
    "decision_config.json":
        "68d4a4c704a0c42837f749b0287bd80f70d186ed3be4dc6a90802e78223925e8",
    "chat_template.jinja":
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "config.json":
        "687f20bf493b8ee787dadaf338cf6df644dad44e32bcd2c3677a47c6c213d8b3",
    "requirements.txt":
        "60ce3c0ba6ec117b114e2ec1641baef7f50cb68020fc9bf46653a86e885e32e8",
    "tokenizer_config.json":
        "09564f0b57b433b55d536630be3044b4183dbe04d0cbc7b1e9b1154db3671b0b",
    "processor_config.json":
        "f76f21cad63dc9da928676e84e4f4889f154b9122e8c3f6b38093d39729fc266",
    "tokenizer.json":
        "6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706",
    "readout.safetensors":
        "774eadedb750f68d38cd7d0d7311a646dc9aed00636910e298951b5d78f16e49",
    "model.safetensors.index.json":
        "dc39d771cb240f775399e1845b11d081d8e007f14eecae3b780bc6272b128a46",
}

DECISION_FIELDS = (
    "format_version",
    "base_model",
    "revision",
    "codes",
    "token_ids",
    "temperature",
    "attention_mode",
    "pooling",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(root: Path) -> dict[str, Any]:
    """Refuse a directory whose pinned files are missing or different.

    The error names repository-relative files only.
    """

    if not root.is_dir():
        raise ValueError("checkpoint directory does not exist")
    problems: list[str] = []
    for relative, expected in PINNED_SHA256.items():
        path = root.joinpath(*relative.split("/"))
        if not path.is_file() or _sha256(path) != expected:
            problems.append(relative)
    if problems:
        raise ValueError("pinned checkpoint files differ: " + ", ".join(problems))
    saved = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
    expected = {
        "format_version": 1,
        "base_model": BASE_MODEL,
        "revision": BASE_REVISION,
        "codes": list(CODES),
        "token_ids": list(TOKEN_IDS),
        "temperature": TEMPERATURE,
        "attention_mode": ATTENTION_MODE,
        "pooling": POOLING,
    }
    for field in DECISION_FIELDS:
        if saved.get(field) != expected[field]:
            raise ValueError(f"decision_config.json field {field} differs from the pin")
    return saved


def _import_decision_model(root: Path):
    source = root / "source" / "src"
    resolved = str(source.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != resolved]
    sys.path.insert(0, resolved)
    from autojev.model import DecisionModel, answer

    return DecisionModel, answer


class PplxExecutor:
    """One loaded checkpoint. ``evaluate`` matches ``server.predict``."""

    model_name = MODEL_NAME
    max_length = MAX_LENGTH
    modalities = ("text", "image")
    attention_mode = ATTENTION_MODE
    repo_id = REPO_ID
    revision = REVISION

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        batch_size: int = 8,
        max_image_pixels: int = 2048 * 2048,
    ) -> None:
        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be cuda or cpu")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        root = Path(checkpoint)
        verify_checkpoint(root)
        # Import conv and FLA before Qwen3.5 modeling freezes the torch refs.
        # Compile tips raised p50 even after the median route proved eager,
        # so startup leaves the forward eager. Capability 9.0 uses
        # all-sequence FlashInfer. The kept primary rotates 48 cu_seqlens
        # buffers, passes use_cp false, and repeats q and k. use_cp auto
        # missed both percentiles, so it stays false. This candidate keeps
        # the ring and skips that repeat. FlashInfer allocates its own
        # state workspace. Residual RMSNorm stays eager.
        # PPLX_GDN_PREFILL=fla reproduces the prior f968e576 chunk.
        # PPLX_GDN_PREFILL=short is the rejected length dispatch.
        # cuDNN-first SDPA and the fused gate-up GEMM are not selectable.
        # A lab GPU named NVIDIA L20X is accepted at capability 8.9 or 9.0.
        cuda = device == "cuda"
        self.causal_conv1d_version = prepare_causal_conv1d(required=cuda)
        self.fla_version = prepare_fla(required=cuda)
        self.flashinfer_version = None
        self.flashinfer_status = None
        self.compute_capability = None
        self.gpu_name = None
        self.linear_attention = None
        self.sdpa = sdpa_preference(device, os.environ.get("PPLX_SDPA"))
        self.mlp = mlp_preference(device, os.environ.get("PPLX_MLP"))
        self.fused_gate_up_layers = 0
        self.rms_norm = RMS_NORM_EAGER
        self.fused_rms_norm_layers = 0
        self.fused_rms_norm_min_tokens = 0
        self.gdn_use_cp = "off"
        self.gdn_cp_heuristic = None
        self.gdn_layout = GDN_LAYOUT_REPEAT
        self.gva_layers = 0
        self.forward_compile = compile_preference(device, os.environ.get("PPLX_COMPILE"))
        self.compile_min_tokens = compile_min_tokens(self.forward_compile)
        self.compile_median = compile_median(self.forward_compile)
        self.gdn_prefill = "fla"
        self.cu_seqlens_slots = 0
        self.language_model = None
        if cuda:
            import torch

            requested = torch.device(device)
            index = requested.index
            if index is None:
                index = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(index)
            self.compute_capability = f"{capability[0]}.{capability[1]}"
            self.gpu_name = torch.cuda.get_device_name(index)
            if uses_flashinfer_prefill(capability):
                self.flashinfer_version, self.flashinfer_status = prepare_flashinfer()
                gdn_mode = gdn_prefill_mode(os.environ.get("PPLX_GDN_PREFILL"))
                if self.flashinfer_status != "bound" or gdn_mode == "fla":
                    self.gdn_prefill = "fla"
                    if self.flashinfer_status == "bound":
                        self.flashinfer_status = "held"
                else:
                    self.gdn_prefill = gdn_mode
            else:
                self.flashinfer_status = "not-selected"
            if self.flashinfer_status == "bound":
                self.cu_seqlens_slots = LINEAR_ATTENTION_LAYERS
            if self.flashinfer_status == "bound" and self.gdn_prefill != "fla":
                self.gdn_use_cp = GDN_USE_CP
        decision_model, reference_answer = _import_decision_model(root)
        if self.causal_conv1d_version is not None:
            assert_qwen_conv_uses_package()
        if self.fla_version is not None:
            assert_qwen_chunk_uses_fla()
        if cuda:
            modeling = importlib.import_module(MODELING)
            if self.flashinfer_status == "bound" and self.gdn_prefill == "short":
                install_short_flashinfer_prefill(modeling)
            elif self.flashinfer_status == "bound":
                install_flashinfer_gdn_prefill(modeling)
            self.linear_attention = require_bound_linear_attention(modeling)
        self.reference_answer = reference_answer
        self.batch_size = batch_size
        self.forward_batch_size = batch_size
        self.device = device
        self.model = decision_model(
            root,
            device=device,
            max_image_pixels=max_image_pixels,
        )
        # Hashed files stay byte-identical. This only retargets the loaded
        # full-attention modules and the noncausal pre-hook.
        language_model = self.model.backbone.language_model
        self.language_model = language_model
        self.unpadded_full_attention_layers = install_unpadded_full_attention(language_model)
        if self.flashinfer_status == "bound" and self.gdn_prefill == "flashinfer":
            self.gva_layers = install_gva_prefill(language_model)
            self.gdn_layout = GDN_LAYOUT_GVA
        if self.forward_compile != "default":
            install_compiled_forward(language_model, self.forward_compile)
            warmup_compiled_forward(self.model.backbone, device, self.forward_compile)

    def _language_stat(self, name: str, default: Any) -> Any:
        language_model = getattr(self, "language_model", None)
        if language_model is None:
            return default
        return getattr(language_model, name, default)

    @property
    def compile_eager_forwards(self) -> int:
        return int(self._language_stat("pplx_compile_eager_forwards", 0) or 0)

    @property
    def compile_compiled_forwards(self) -> int:
        return int(self._language_stat("pplx_compile_compiled_forwards", 0) or 0)

    @property
    def compile_last_length(self) -> int | None:
        return self._language_stat("pplx_last_length", None)

    @property
    def compile_median_frame_delta(self) -> int | None:
        return self._language_stat("pplx_median_frame_delta", None)

    def evaluate(self, request: Request) -> dict[str, Any]:
        import torch

        questions = dict(request.questions)
        identifiers = list(questions)
        rows = [
            {"state": request.state, "question": question, "images": list(request.images)}
            for question in questions.values()
        ]
        answers: dict[str, Any] = {}
        input_tokens = 0
        with torch.inference_mode():
            step = self.forward_batch_size
            for start in range(0, len(rows), step):
                batch = self.model.prepare(rows[start : start + step])
                distributions = (
                    (self.model(batch) / self.model.temperature).softmax(-1).cpu().tolist()
                )
                stop = start + step
                for identifier, values, count in zip(
                    identifiers[start:stop],
                    distributions,
                    batch.counts,
                    strict=True,
                ):
                    answers[identifier] = self.reference_answer(
                        questions[identifier], values[:count]
                    )
                input_tokens += batch.input_tokens
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }

    def close(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        if model is not None:
            del model
