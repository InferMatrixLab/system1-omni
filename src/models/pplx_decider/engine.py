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
import json
import sys
from pathlib import Path
from typing import Any

from .attention import install_unpadded_full_attention
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
    assert_qwen_chunk_uses_fla,
    bind_flashinfer_prefill,
    flashinfer_prefill_available,
    linear_head_dims,
    prepare_fla,
    select_gdn_prefill,
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
        text_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        key_dim, value_dim = linear_head_dims(text_config)
        # Import conv and FLA before Qwen3.5 modeling freezes the torch refs.
        cuda = device == "cuda"
        self.causal_conv1d_version = prepare_causal_conv1d(required=cuda)
        self.fla_version = prepare_fla(required=cuda)
        self.gdn_backend = None
        self.flashinfer_version = None
        decision_model, reference_answer = _import_decision_model(root)
        if self.causal_conv1d_version is not None:
            assert_qwen_conv_uses_package()
        if self.fla_version is not None:
            assert_qwen_chunk_uses_fla()
            self.gdn_backend = "fla-chunk"
            if cuda:
                import torch

                choice = select_gdn_prefill(
                    torch.cuda.get_device_capability(),
                    key_dim,
                    value_dim,
                    flashinfer_available=flashinfer_prefill_available(),
                )
                if choice == "flashinfer":
                    self.flashinfer_version = bind_flashinfer_prefill()
                    self.gdn_backend = choice
        self.reference_answer = reference_answer
        self.batch_size = batch_size
        self.device = device
        self.model = decision_model(
            root,
            device=device,
            max_image_pixels=max_image_pixels,
        )
        # Hashed files stay byte-identical. This only retargets the loaded
        # full-attention modules and the noncausal pre-hook.
        self.unpadded_full_attention_layers = install_unpadded_full_attention(
            self.model.backbone.language_model
        )

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
            for start in range(0, len(rows), self.batch_size):
                batch = self.model.prepare(rows[start : start + self.batch_size])
                distributions = (
                    (self.model(batch) / self.model.temperature).softmax(-1).cpu().tolist()
                )
                stop = start + self.batch_size
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
