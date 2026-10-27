"""Pinned executor for Vela 2.0 0.8B (``vela2-decoder``), the first hybrid member.

Mirrors :mod:`models.vela.engine` for the compact hybrid member. The
``vela2-decoder`` package ships the same flat runtime modules as the
0.3B (with decoder-specific contents), so the gate reuses the same
helpers and the same one-package-per-process discipline; what differs
is the pin table, the structural identity of ``config.json``, and the
weights layout — one 2 GB shard checked by exact size plus the
side-loaded second span head.

The runtime loads parameters in FP32 either way; on ``cuda`` it runs
the backbone under bf16 autocast with FP32 heads (the evaluated
setting), on ``cpu`` it runs FP32 throughout. The served model name
comes from ``config.json`` ``model_name`` (upstream
``DEFAULT_MODEL_NAME`` is the family-generic ``vllm-sr/Vela-2.0`` and
must not be used as a default here).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from .engine import (
    put_first_on_sys_path,
    verify_files,
    verify_no_foreign_runtime,
)

HYBRID_REPO = "vllm-sr/Vela-2.0-0.8B"
HYBRID_REVISION = "326b01d81f61fff7fd4ae570e54b87a49119e490"
# Structural identity of the pinned config.json. model_name is unique
# per member, so a 4B or 9B package fails with its own name.
EXPECTED_MODEL_TYPE = "vela2-decoder"
EXPECTED_ARCHITECTURE = "Vela2DecoderModel"
EXPECTED_MAX_LENGTH = 16384
EXPECTED_MODEL_NAME = "vllm-sr/Vela-2.0-0.8B"
WEIGHTS_NAME = "model-00001-of-00001.safetensors"
EXPECTED_WEIGHTS_BYTES = 2022189488
# The files this gate hashes, transcribed from the pinned revision and
# cross-checked against the package's own SHA256SUMS; repinning means
# updating this table together with HYBRID_REVISION. Every file the
# runtime reads or executes is here: the four runtime modules,
# ``vela2_serve.py`` (never imported, part of the pinned surface),
# ``config.json``, ``calibration.json``, ``tokenizer.json`` (the loaded
# tokenizer), ``tokenizer_config.json`` (listed by the runtime's own
# download manifest), ``model.safetensors.index.json`` (the shard map
# the loader reads) and ``broad_head.safetensors`` (the side-loaded
# second span head; 4.5 MB hashes in milliseconds). The 2 GB shard is
# checked by exact size only, as on 0.3B.
PINNED_FILES_SHA256 = {
    "vela2_serve.py": "68e1ef278561f452aa0ca6afd6aa288636cecdb0b37fd7f7146a3c9bf17aa787",
    "vela2_inference.py": "11de1322788942153a22db164a3fb26abcaa4c5bb4b057cd572e277457c17dc2",
    "modeling_vela2.py": "b7ecd8dcacfb24dbe8249c7b96829a737f4e2b493fc2157497f1b62d9e02727d",
    "configuration_vela2.py": "c9a41e434a9d2cfeb3ab6581c00679d241b3468c04d160a811d6fe2c85cd071f",
    "config.json": "0637ca193b9159b6ece25837913956477164d3392d9b5dd1e6b74b95a484004a",
    "calibration.json": "d96e447962d0dc1d58fa8422dddbae8b99960e21fd75a10ef9b096f700a211f5",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
    "model.safetensors.index.json": "b53d1c79219677289bdb1541a1373dfe53119884f248b23c94a27aef7338919d",
    "broad_head.safetensors": "687d7a435d6de846796274e417b6333f83df3085a6e7c3bca4c01cb4d75a30f0",
}
# A warmup that touches every question kind's readout path on both span
# heads: the CandidateHead option rows (choice/set share it, noul and
# score ride the same candidate readout), the span decoder and the
# threshold echo, one router-head span (the unsupported label is a
# trained alias) and one broad-head span (an open label with no trained
# match). Model text only; the result is discarded.
WARMUP_REQUEST = {
    "state": {
        "request": "Hello, I would like to check the status of my order.",
        "answer": "Your order was delivered yesterday.",
    },
    "model": None,  # filled with the served name at warmup time
    "questions": {
        "kind": {
            "type": "choice",
            "instructions": "What is this message about?",
            "criteria": {"order": "an order's status", "other": "anything else"},
        },
        "claims": {
            "type": "set",
            "instructions": "Which topics does the message mention?",
            "criteria": {"delivery": "delivery of an order", "billing": "payments or charges"},
        },
        "tone": {
            "type": "score",
            "instructions": "How formal is the message?",
            "criteria": ["casual", "neutral", "formal"],
        },
        "polite": {
            "type": "noul",
            "instructions": "Is the message polite?",
        },
        "unsupported": {
            "type": "span",
            "instructions": "Which spans of the answer are not supported by the request?",
            "criteria": {"unsupported": "a claim not supported by the request"},
        },
        "products": {
            "type": "span",
            "instructions": "Which products or services are mentioned?",
            "criteria": {"product": "a product or a service"},
            "over": "request",
        },
    },
}


def verify_config(root: Path) -> dict[str, Any]:
    """Read config.json and check the pinned member's structural identity."""

    path = Path(root) / "config.json"
    if not path.is_file():
        raise ValueError(f"the package is missing config.json; this worker only serves {HYBRID_REPO}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"config.json declares model_type {config.get('model_type')!r}; this worker only "
            f"serves the {EXPECTED_MODEL_TYPE} member {HYBRID_REPO} at {HYBRID_REVISION}"
        )
    if config.get("architectures") != [EXPECTED_ARCHITECTURE]:
        raise ValueError(f"config.json architectures must be [{EXPECTED_ARCHITECTURE!r}]")
    if config.get("max_length") != EXPECTED_MAX_LENGTH:
        raise ValueError(f"config.json max_length must be {EXPECTED_MAX_LENGTH}")
    if config.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError(
            f"config.json model_name must be {EXPECTED_MODEL_NAME!r}; another Vela 2.0 member "
            "needs its own pinned executor"
        )
    return config


def verify_weights(root: Path) -> None:
    """Check the pinned shard is present at its exact pinned size."""

    target = Path(root) / WEIGHTS_NAME
    if not target.is_file():
        raise ValueError(f"the package is missing {WEIGHTS_NAME}")
    size = target.stat().st_size
    if size != EXPECTED_WEIGHTS_BYTES:
        raise ValueError(
            f"{WEIGHTS_NAME} is {size} bytes; the pinned revision's file is {EXPECTED_WEIGHTS_BYTES}"
        )


def verify_package(root: Path, table: dict[str, str] | None = None) -> None:
    """The full pre-import gate for one hybrid package directory."""

    verify_config(root)
    verify_files(root, table or PINNED_FILES_SHA256, HYBRID_REVISION)
    verify_weights(root)


class Vela2HybridExecutor:
    """Load and drive one immutable pinned Vela 2.0 0.8B runtime instance."""

    is_hybrid = True

    def __init__(
        self,
        package: str | Path,
        *,
        device: str = "cpu",
        model_name: str | None = None,
        batch_size: int | None = None,
        token_budget: int | None = None,
    ) -> None:
        root = Path(package)
        if not root.is_dir():
            raise ValueError(f"the Vela 2.0 0.8B package directory does not exist: {root}")
        verify_package(root)

        # Same discipline as the 0.3B executor: the just-verified package
        # root goes first on sys.path so the flat runtime modules resolve
        # to the checkpoint's own copies, and a runtime module already
        # imported from another directory is refused — one worker process
        # serves one package.
        verify_no_foreign_runtime(root)
        put_first_on_sys_path(root)
        # Import with bytecode writing off: the gate refuses existing
        # bytecode caches, so a __pycache__ this import wrote would fail
        # every later gate run of the same package.
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            import vela2_inference
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode

        self.SystemOneError = vela2_inference.SystemOneError
        self._system_one_request = vela2_inference.system_one_request
        self.engine = vela2_inference.Vela2.from_pretrained(
            root, backend="torch", device=device,
            **({"batch_size": batch_size} if batch_size is not None else {}),
            **({"token_budget": token_budget} if token_budget is not None else {}),
        )
        # The decoder members declare their served name in config.json;
        # the runtime's DEFAULT_MODEL_NAME is family-generic.
        self.model_name = model_name or self.engine.model_name
        self.max_length = int(self.engine.max_len)
        self._lock = threading.RLock()
        self._closed = False

    @property
    def served(self) -> set[str]:
        """The model names this executor answers to (upstream: one)."""

        return {self.model_name}

    def system_one(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run one admitted request body; the response is upstream's, verbatim.

        Hybrid responses carry one extra top-level field, ``span_heads``,
        alongside ``spans``; response finishing checks it in
        :func:`models.vela.postprocess.finish_response`.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("Vela 2.0 hybrid executor is closed")
            return self._system_one_request(self.engine, body, served=self.served)

    def warmup(self) -> None:
        """Run one real request of every question kind on both span heads."""

        request = dict(WARMUP_REQUEST)
        request["model"] = self.model_name
        response = self.system_one(request)
        if not isinstance(response, dict) or "answers" not in response:
            raise RuntimeError("warmup did not produce a SystemOne response")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.engine = None
