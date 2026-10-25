"""Clef executor for one pinned Cloudflare snapshot.

The executor owns checkpoint identity, the hash gate, and the loaded joint
schema model. It does not admit HTTP requests. ``load_release_model`` and
``systemone`` are the checkpoint's own functions, imported only after every
executed and tokenized file matches the pin.

Clef-flash (9B, Qwen3.5-9B) and Clef (27B, Qwen3.8-27B) share one loader and
one tokenizer. They do not share ``config.json`` or ``joint_head_config.json``.
Neither snapshot is an Open-Jev export: the decision head is
``joint_head.safetensors``, and the backbone index has no separate LoRA keys.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.clef.preprocess import MAX_INPUT_TOKENS, enforce_limit
from models.clef.prompt import render

DEFAULT_VARIANT = "clef-flash"
LOADER_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
TOKENIZER_CONFIG_SHA256 = "91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72"
PROCESSOR_CONFIG_SHA256 = "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1"
SHARED_FILES = {
    "joint_schema_model.py": LOADER_SHA256,
    "tokenizer.json": TOKENIZER_SHA256,
    "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
    "processor_config.json": PROCESSOR_CONFIG_SHA256,
}
EXECUTABLE = "joint_schema_model.py"
MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    """One public Clef release. Hashes are of the file bytes at ``revision``."""

    name: str
    repo_id: str
    revision: str
    base_model: str
    hidden_size: int
    num_hidden_layers: int
    config_sha256: str
    joint_head_config_sha256: str


VARIANTS = {
    "clef-flash": Variant(
        name="clef-flash",
        repo_id="Cloudflare/clef-flash",
        revision="f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
        base_model="Qwen/Qwen3.5-9B",
        hidden_size=4096,
        num_hidden_layers=32,
        config_sha256="66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238",
        joint_head_config_sha256="77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0",
    ),
    "clef": Variant(
        name="clef",
        repo_id="Cloudflare/clef",
        revision="a30162b99d68a481d8bd75d38df069a0cf83df40",
        base_model="Qwen/Qwen3.8-27B",
        hidden_size=5120,
        num_hidden_layers=64,
        config_sha256="c42e88892bd3fd84e8276b2ad90df58c1c3b797676ea161035006a72ad468c58",
        joint_head_config_sha256="890be585d75b981201eb96a35f98a9967afe37bc8d220cfdea2e72d56507534f",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(root: Path, name: str, expected: str) -> None:
    path = root / name
    if not path.is_file():
        raise ValueError(f"missing {name}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 {actual} is not the pinned digest")


def verify_snapshot(
    root: Path, variant: Variant, shared_files: dict[str, str] | None = None
) -> None:
    """Hash the loader, tokenizer and variant configs before importing them.

    Weight shards are not hashed here. Download the snapshot at
    ``variant.revision`` so those shards belong to the same commit as the
    hashed files. An extra ``.py`` or a ``__pycache__`` in the snapshot root
    is refused so the import cannot pick up a different module.
    """

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"the Clef snapshot directory does not exist: {root}")
    for name, digest in (SHARED_FILES if shared_files is None else shared_files).items():
        _require_hash(root, name, digest)
    _require_hash(root, "config.json", variant.config_sha256)
    _require_hash(root, "joint_head_config.json", variant.joint_head_config_sha256)
    for entry in root.iterdir():
        if entry.name == "__pycache__":
            raise ValueError("unexpected bytecode cache __pycache__/ in the snapshot root")
        if entry.is_file() and entry.suffix == ".py" and entry.name != EXECUTABLE:
            raise ValueError(f"unexpected Python file {entry.name} in the snapshot root")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") or {}
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("config.json is not a Qwen3_5ForConditionalGeneration backbone")
    if text.get("hidden_size") != variant.hidden_size:
        raise ValueError(
            f"config hidden_size {text.get('hidden_size')} is not {variant.hidden_size}"
        )
    if text.get("num_hidden_layers") != variant.num_hidden_layers:
        raise ValueError(
            f"config num_hidden_layers {text.get('num_hidden_layers')} "
            f"is not {variant.num_hidden_layers}"
        )
    head = json.loads((root / "joint_head_config.json").read_text(encoding="utf-8"))
    if head.get("hidden_size") != variant.hidden_size:
        raise ValueError(
            f"joint head hidden_size {head.get('hidden_size')} is not {variant.hidden_size}"
        )


def untruncated_budget(total: int, state_tokens: int, max_length: int) -> None:
    """Reject a prompt whose full encoding does not fit, without slicing state.

    ``state_tokens`` is the rendered state counted by the same tokenizer the
    encoder uses for that piece. A negative fixed length means the two counts
    disagree; the request is not served.
    """

    if state_tokens < 0 or state_tokens > total:
        raise RuntimeError(
            f"token accounting mismatch: encoded {total}, state piece {state_tokens}"
        )
    enforce_limit(total - state_tokens, total, max_length)


def require_token_agreement(admitted: int, response: dict[str, Any]) -> dict[str, Any]:
    """Refuse a response whose prompt length is not the untruncated admission count.

    The pinned ``systemone`` slices state to fit. Admission already rejected
    that case; a shorter encoder count means the slice still happened.
    """

    usage = response.get("usage") if isinstance(response, dict) else None
    got = usage.get("input_tokens") if isinstance(usage, dict) else None
    if got != admitted:
        raise RuntimeError(
            f"token accounting mismatch: admitted {admitted} untruncated tokens, "
            f"encoder returned {got}"
        )
    return response


def bound_linear_attention(modeling: Any) -> str:
    """Name the linear-attention implementation Transformers actually bound.

    ``modeling_qwen3_5`` keeps the fast callables as module globals when the
    optional packages imported, and ``None`` when it kept the PyTorch
    fallback. The report is that binding, not an install attempt.
    """

    if modeling is None:
        return "unloaded"

    fla = getattr(modeling, "chunk_gated_delta_rule", None) is not None
    conv = getattr(modeling, "causal_conv1d_fn", None) is not None
    if fla and conv:
        return "fla+causal_conv1d"
    if fla:
        return "fla"
    if conv:
        return "causal_conv1d"
    return "torch_chunk"


def activate_linear_attention() -> None:
    """Import the optional kernels before Transformers binds Qwen3.5.

    The modeling module chooses ``chunk_gated_delta_rule`` and
    ``causal_conv1d_fn`` once, at import. A later install does not move a
    process that already imported the module. A failed import leaves the
    PyTorch fallback in place.
    """

    if MODELING in sys.modules:
        return
    for name in ("causal_conv1d", "fla.ops.gated_delta_rule"):
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as exc:
            LOG.info("clef linear-attention package %s is not active: %s", name, type(exc).__name__)


def load_joint_schema_module(path: Path) -> Any:
    """Import one verified ``joint_schema_model.py`` without putting its directory on ``sys.path``."""

    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ClefExecutor:
    """Load and run one immutable pinned Clef snapshot."""

    def __init__(self, snapshot: str | Path, variant: str = DEFAULT_VARIANT, device: str = "cuda") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        self.variant = VARIANTS[variant]
        self.root = Path(snapshot)
        self.device = device
        self.max_length = MAX_INPUT_TOKENS
        verify_snapshot(self.root, self.variant)
        activate_linear_attention()
        self.module = load_joint_schema_module(self.root / EXECUTABLE)
        self.model, self.processor = self.module.load_release_model(self.root, device=device)
        self.linear_attention = bound_linear_attention(sys.modules.get(MODELING))
        if self.processor.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer pad_token_id is required")

    def count_tokens(self, text: str) -> int:
        return len(self.processor.tokenizer(text, add_special_tokens=False).input_ids)

    def serve(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Encode once, refuse an over-long prompt, then score that encoding.

        The pinned ``systemone`` helper tokenizes and then slices state to
        ``max_length``. Calling it after a separate admission pass would
        tokenize the schema twice and could still slice if the counts drifted.
        This path keeps the full ``encode_record`` result and runs the joint
        head on it. The duplicate BPE pass is the only serve-path change; the
        head and backbone are the pinned modules. No latency delta is claimed.
        """

        encoded = self.module.encode_record(
            self.processor.tokenizer,
            payload,
            max_length=10**9,
            processor=self.processor,
        )
        total = len(encoded.input_ids)
        state_tokens = self.count_tokens(render(payload["state"]))
        untruncated_budget(total, state_tokens, self.max_length)
        response = self._forward(payload, encoded)
        return require_token_agreement(total, response)

    def _forward(self, payload: dict[str, Any], encoded: Any) -> dict[str, Any]:
        import torch

        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            raise RuntimeError("tokenizer pad_token_id is required")
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            logits = self.model(
                self.module.collate_records([encoded], pad, device)
            )[0]
        questions = payload["questions"]
        answers = {
            question.question_id: self.module.systemone_answer(
                questions[question.question_id],
                dict(
                    zip(
                        question.option_ids,
                        question_logits.float().softmax(-1).tolist(),
                    )
                ),
            )
            for question, question_logits in zip(encoded.questions, logits)
        }
        return {
            "model": payload["model"],
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": self.variant.repo_id,
            "variant": self.variant.name,
            "revision": self.variant.revision,
            "base_model": self.variant.base_model,
            "max_input_tokens": self.max_length,
            "modalities": ["text"],
            "prompt_mode": "non-thinking",
            "linear_attention": self.linear_attention,
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.module = None
