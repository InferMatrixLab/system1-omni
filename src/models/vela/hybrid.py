"""Pinned executors for the Vela 2.0 hybrid members (``vela2-decoder``).

Mirrors :mod:`models.vela.engine` for the decoder members — 0.8B, 4B
and 9B. The three packages ship byte-identical runtime modules (the
four flat files plus ``tokenizer.json`` hash the same at every pinned
revision), so one executor drives all of them; what differs per member
is the pin table, the structural identity of ``config.json``
(``model_name`` is unique per member) and the weights layout — the
shard list checked by exact size plus the side-loaded second span head.
The 4B/9B packages add non-runtime files (``MODEL_MANIFEST.json``,
``pii_calibration_fit.json``, docs, license texts); the runtime does
not read them, so they pass the gate the way the 0.3B's ``assets/``
does, by not being importable and not being pinned.

The runtime loads parameters in FP32 either way; on ``cuda`` it runs
the backbone under bf16 autocast with FP32 heads (the evaluated
setting; fp16 is unsupported), on ``cpu`` it runs FP32 throughout.
Memory scales with the member: 0.8B ~2 GB, 4B ~9.7 GB on disk (3
shards), 9B ~17.9 GB (5 shards); FP32 in memory is roughly double the
backbone share of those. The served model name comes from
``config.json`` ``model_name`` (upstream ``DEFAULT_MODEL_NAME`` is the
family-generic ``vllm-sr/Vela-2.0`` and must not be used as a default
here).
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import (
    put_first_on_sys_path,
    verify_files,
    verify_no_foreign_runtime,
)

EXPECTED_MODEL_TYPE = "vela2-decoder"
EXPECTED_ARCHITECTURE = "Vela2DecoderModel"
# The runtime modules are one program shared by every decoder member:
# these five pins are byte-identical across the 0.8B/4B/9B revisions.
SHARED_RUNTIME_FILES_SHA256 = {
    "vela2_serve.py": "68e1ef278561f452aa0ca6afd6aa288636cecdb0b37fd7f7146a3c9bf17aa787",
    "vela2_inference.py": "11de1322788942153a22db164a3fb26abcaa4c5bb4b057cd572e277457c17dc2",
    "modeling_vela2.py": "b7ecd8dcacfb24dbe8249c7b96829a737f4e2b493fc2157497f1b62d9e02727d",
    "configuration_vela2.py": "c9a41e434a9d2cfeb3ab6581c00679d241b3468c04d160a811d6fe2c85cd071f",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}


@dataclass(frozen=True)
class HybridMember:
    """One pinned ``vela2-decoder`` package and its gate identity."""

    repo: str
    revision: str
    model_name: str
    max_length: int
    # Shard name -> exact byte size at the pinned revision, read from
    # the shard map the loader follows (model.safetensors.index.json).
    weights_shards: dict[str, int]
    # The files this member's gate hashes: the shared runtime modules
    # plus this member's config.json, calibration.json,
    # tokenizer_config.json, model.safetensors.index.json and the
    # side-loaded broad_head.safetensors. Transcribed from the pinned
    # revision and cross-checked against the package's own SHA256SUMS;
    # repinning means updating this table together with `revision`.
    pinned_files: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        table = dict(SHARED_RUNTIME_FILES_SHA256)
        table.update(self.pinned_files)
        object.__setattr__(self, "pinned_files", table)

    @property
    def label(self) -> str:
        return self.model_name.rsplit("/", 1)[-1]


def _member(
    repo: str,
    revision: str,
    model_name: str,
    weights_shards: dict[str, int],
    pinned: dict[str, str],
) -> HybridMember:
    return HybridMember(
        repo=repo,
        revision=revision,
        model_name=model_name,
        max_length=16384,
        weights_shards=weights_shards,
        pinned_files=pinned,
    )


# Keyed by config.json `model_name`, the only per-member field the
# packages themselves declare.
MEMBERS: dict[str, HybridMember] = {
    member.model_name: member
    for member in (
        _member(
            repo="vllm-sr/Vela-2.0-0.8B",
            revision="326b01d81f61fff7fd4ae570e54b87a49119e490",
            model_name="vllm-sr/Vela-2.0-0.8B",
            weights_shards={
                "model-00001-of-00001.safetensors": 2022189488,
            },
            pinned={
                "config.json": "0637ca193b9159b6ece25837913956477164d3392d9b5dd1e6b74b95a484004a",
                "calibration.json": "d96e447962d0dc1d58fa8422dddbae8b99960e21fd75a10ef9b096f700a211f5",
                "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
                "model.safetensors.index.json": "b53d1c79219677289bdb1541a1373dfe53119884f248b23c94a27aef7338919d",
                "broad_head.safetensors": "687d7a435d6de846796274e417b6333f83df3085a6e7c3bca4c01cb4d75a30f0",
            },
        ),
        _member(
            repo="vllm-sr/Vela-2.0-4B",
            revision="d692394127f6f87fc581d8634c9558fe4776efce",
            model_name="vllm-sr/Vela-2.0-4B",
            weights_shards={
                "model-00001-of-00003.safetensors": 4280713812,
                "model-00002-of-00003.safetensors": 4247826344,
                "model-00003-of-00003.safetensors": 1176485676,
            },
            pinned={
                "config.json": "c0318849c586fe42f98db289c9d959642a8e2020726f2b3cb4685ccfc9912c40",
                "calibration.json": "2a3f3cfa2d1cf6fa133d887960261b71c09420899f817f4156677962ac02460f",
                "tokenizer_config.json": "353c083805d863719e54973335a38e252a92e1ac64d32b83bb0b333cc1644ef2",
                "model.safetensors.index.json": "24e5eea6b54282cdf32a7f8d746f9c7b3f8c2ef66bcd57b333fd0022b7c74ca0",
                "broad_head.safetensors": "45dd5eafa620a5b4d3e060eaf8baadbfeaab4d3086dc3a63a6734068e441e20c",
            },
        ),
        _member(
            repo="vllm-sr/Vela-2.0-9B",
            revision="068425bb145c05884a98b870c219047da22d9aea",
            model_name="vllm-sr/Vela-2.0-9B",
            weights_shards={
                "model-00001-of-00005.safetensors": 4203284492,
                "model-00002-of-00005.safetensors": 4232405584,
                "model-00003-of-00005.safetensors": 4282673120,
                "model-00004-of-00005.safetensors": 4232830648,
                "model-00005-of-00005.safetensors": 991768516,
            },
            pinned={
                "config.json": "4770113c223c399500382f07746cfe384465409e16f07b2cea49ced9f2547aa3",
                "calibration.json": "9fa0f31ce1c7b784220fd73320ee64da9b3b789c403755af91bcbefb5740d766",
                "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
                "model.safetensors.index.json": "b8c6ef31bad3b3f55952261553827df322e666e2393f85856e435af70e393c4e",
                "broad_head.safetensors": "c376f1668247ff902947d1ffd69c92195c7b74116dedd958aebd31ff5f1b580c",
            },
        ),
    )
}

# The 0.8B entry keeps its historical module-level names: the tests and
# the oracle fixtures are written against the first hybrid member.
HYBRID_MEMBER = MEMBERS["vllm-sr/Vela-2.0-0.8B"]
HYBRID_REPO = HYBRID_MEMBER.repo
HYBRID_REVISION = HYBRID_MEMBER.revision
EXPECTED_MODEL_NAME = HYBRID_MEMBER.model_name
EXPECTED_MAX_LENGTH = HYBRID_MEMBER.max_length
WEIGHTS_NAME = next(iter(HYBRID_MEMBER.weights_shards))
EXPECTED_WEIGHTS_BYTES = HYBRID_MEMBER.weights_shards[WEIGHTS_NAME]
PINNED_FILES_SHA256 = HYBRID_MEMBER.pinned_files


def members_summary() -> str:
    """The served decoder members, for gate error messages."""

    return ", ".join(f"{member.label} ({member.repo} at {member.revision})" for member in MEMBERS.values())


def resolve_member(config: dict[str, Any]) -> HybridMember:
    """Check a config's structural identity and return its member."""

    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"config.json declares model_type {config.get('model_type')!r}; this worker only "
            f"serves the {EXPECTED_MODEL_TYPE} members: {members_summary()}"
        )
    if config.get("architectures") != [EXPECTED_ARCHITECTURE]:
        raise ValueError(f"config.json architectures must be [{EXPECTED_ARCHITECTURE!r}]")
    name = config.get("model_name")
    member = MEMBERS.get(name)
    if member is None:
        raise ValueError(
            f"config.json model_name {name!r} is not a pinned member; this worker serves "
            f"{members_summary()}"
        )
    if config.get("max_length") != member.max_length:
        raise ValueError(f"config.json max_length must be {member.max_length}")
    return member


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
    """Read config.json, check the structural identity, return it."""

    path = Path(root) / "config.json"
    if not path.is_file():
        raise ValueError(
            f"the package is missing config.json; this worker only serves the "
            f"{EXPECTED_MODEL_TYPE} members: {members_summary()}"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    resolve_member(config)
    return config


def verify_weights(root: Path, member: HybridMember | None = None) -> None:
    """Check every pinned shard is present at its exact pinned size."""

    if member is None:
        member = resolve_member(verify_config(root))
    for name, expected in member.weights_shards.items():
        target = Path(root) / name
        if not target.is_file():
            raise ValueError(f"the package is missing {name}")
        size = target.stat().st_size
        if size != expected:
            raise ValueError(
                f"{name} is {size} bytes; the pinned revision's file is {expected}"
            )


def verify_package(root: Path, table: dict[str, str] | None = None) -> HybridMember:
    """The full pre-import gate for one hybrid package directory.

    Returns the member the package resolved to.
    """

    member = resolve_member(verify_config(root))
    verify_files(root, table or member.pinned_files, member.revision)
    verify_weights(root, member)
    return member


class Vela2HybridExecutor:
    """Load and drive one immutable pinned Vela 2.0 hybrid runtime instance."""

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
            raise ValueError(f"the Vela 2.0 hybrid package directory does not exist: {root}")
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
