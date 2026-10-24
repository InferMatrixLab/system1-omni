"""Pinned executor for Vela 2.0 0.3B (``vela2-unified``).

The executor owns the pinned runtime, the file-identity gate that
precedes its import, device state, and warmup. It drives the checkpoint's
own ``vela2_inference`` module through its ``system_one_request`` surface
and stays free of HTTP concerns; wire admission lives in protocol.py and
preprocess.py, response finishing in postprocess.py.

The checkpoint ships its runtime as flat Python files next to the
weights, and unlike d3 there is no package manifest to lean on. The gate
therefore hashes every small file the runtime reads or executes — the
four runtime modules, ``vela2_serve.py`` (never imported, but part of the
pinned surface), ``config.json``, ``calibration.json`` and
``tokenizer.json`` — against the table checked in for
``REFERENCE_REVISION``, before anything is imported, and refuses
importable entries the table does not name (the package root goes first
on ``sys.path``). ``model.safetensors`` is checked by exact size only:
hashing 1.2 GB at startup buys nothing over the runtime's own strict
safetensors load, which fails on truncated or mismatched weights.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from importlib.machinery import EXTENSION_SUFFIXES, SOURCE_SUFFIXES
from pathlib import Path
from typing import Any

REFERENCE_REPO = "vllm-sr/Vela-2.0-0.3B"
REFERENCE_REVISION = "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409"
# Structural identity of the pinned config.json, checked before hashing
# so a foreign config fails with its own name.
EXPECTED_MODEL_TYPE = "vela2-unified"
EXPECTED_ARCHITECTURE = "Vela2Model"
EXPECTED_MAX_LENGTH = 8192
EXPECTED_MARKER_COUNT = 8
WEIGHTS_NAME = "model.safetensors"
EXPECTED_WEIGHTS_BYTES = 1236473252
# The files this gate hashes, transcribed from the pinned revision;
# repinning means updating this table together with REFERENCE_REVISION.
PINNED_FILES_SHA256 = {
    "vela2_serve.py": "6c7090b56d66fba527f2eba4ea38aa3ca5c7c7cf167a38a3faa75a558e8b0964",
    "vela2_inference.py": "d3e140c7bc9d291e0862b85894f03ecb1d4c95f0c58bbf705da775b1f63cbf22",
    "modeling_vela2.py": "a536b7ffa832693b2aaaa6f499da6c594bebc1be2bd8d85d2ef5e794124cda94",
    "configuration_vela2.py": "94b22bf266a45eaa4877123ca59e62e0908d4054a985712f12de8d1e230f0e3c",
    "config.json": "81e08c8f8f2a4a6205f05b67d5682e42fc5249cc11e5231c456929ae0534d675",
    "calibration.json": "f8557b3922e9bef33e688ece12da20436b5ae868ed7ff1375b346bd26403506a",
    "tokenizer.json": "e4b670c5ab72158f35de150792133e8c92adb4ba677d26b99d22c77f34c63faf",
}
# The importable runtime modules of the pinned table; a stale in-process
# import of any of them from another package directory would silently
# mix two runtimes once this package's root goes first on ``sys.path``.
RUNTIME_MODULES = tuple(name[:-3] for name in PINNED_FILES_SHA256 if name.endswith(".py"))
# Every module-file form importable from a plain sys.path entry: source,
# sourceless and CPython extension suffixes.
IMPORTABLE_FILE_SUFFIXES = frozenset(SOURCE_SUFFIXES) | {".pyc"} | frozenset(EXTENSION_SUFFIXES)
# A warmup that touches every question kind's readout path (cosine + MLP
# option heads, the span head and the span decoder, the set sigmoid and
# threshold echo). Model text only; the result is discarded.
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
    },
}


def _sha256_safe(path: Path) -> str:
    """Might encounter a large file, read in chunks"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_config(root: Path) -> dict[str, Any]:
    """Read config.json and check the pinned member's structural identity."""

    path = Path(root) / "config.json"
    if not path.is_file():
        raise ValueError(f"the package is missing config.json; this worker only serves {REFERENCE_REPO}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"config.json declares model_type {config.get('model_type')!r}; this worker only "
            f"serves the {EXPECTED_MODEL_TYPE} member {REFERENCE_REPO} at {REFERENCE_REVISION}"
        )
    if config.get("architectures") != [EXPECTED_ARCHITECTURE]:
        raise ValueError(f"config.json architectures must be [{EXPECTED_ARCHITECTURE!r}]")
    if config.get("max_length") != EXPECTED_MAX_LENGTH:
        raise ValueError(f"config.json max_length must be {EXPECTED_MAX_LENGTH}")
    if len(config.get("marker_ids") or {}) != EXPECTED_MARKER_COUNT:
        raise ValueError(f"config.json must carry the {EXPECTED_MARKER_COUNT} marker token ids")
    return config


def verify_files(root: Path, table: dict[str, str] | None = None) -> list[str]:
    """Hash every pinned file in full and return their names.

    Must run before any package code is imported: the table names the
    files this process is about to execute or read as calibration, so
    there is no size shortcut. It also refuses importable entries the
    table does not name — the package root is about to go first on
    ``sys.path``, where an extra module file (source, sourceless or
    extension) or a directory carrying any importable entry — an
    ``__init__`` in every module form, or a namespace package's nested
    modules — would shadow one of the pinned runtime's dependencies —
    and any ``__pycache__``, whose existing bytecode is not verifiable
    against the pin.
    """

    table = dict(table or PINNED_FILES_SHA256)
    problems: list[str] = []
    for name, digest in sorted(table.items()):
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"unsafe pinned file name {name!r}")
        target = Path(root) / name
        if not target.is_file():
            problems.append(f"missing pinned file {name}")
            continue
        if _sha256_safe(target) != digest:
            problems.append(f"{name}: SHA-256 differs from the pin for {REFERENCE_REVISION}")
    for entry in sorted(Path(root).iterdir()):
        if entry.name in table:
            continue
        if entry.name == "__pycache__":
            problems.append(
                f"existing bytecode cache {entry.name}/ in the package root is not verifiable against the pin"
            )
        elif entry.is_file() and entry.suffix in IMPORTABLE_FILE_SUFFIXES:
            problems.append(f"unexpected importable file {entry.name} in the package root")
        elif entry.is_dir() and any(
            child.is_file() and child.suffix in IMPORTABLE_FILE_SUFFIXES
            for child in entry.rglob("*")
        ):
            # Any importable file at any depth makes the directory an
            # importable package: through an __init__ in source, sourceless
            # or extension form, or as a namespace package whose nested
            # modules import directly.
            problems.append(f"unexpected importable package {entry.name}/ in the package root")
    if problems:
        raise ValueError("package verification failed: " + "; ".join(problems[:8]))
    return sorted(table)


def verify_weights(root: Path) -> None:
    """Check the pinned weights are present at their exact pinned size."""

    target = Path(root) / WEIGHTS_NAME
    if not target.is_file():
        raise ValueError(f"the package is missing {WEIGHTS_NAME}")
    size = target.stat().st_size
    if size != EXPECTED_WEIGHTS_BYTES:
        raise ValueError(
            f"{WEIGHTS_NAME} is {size} bytes; the pinned revision's file is {EXPECTED_WEIGHTS_BYTES}"
        )


def verify_no_foreign_runtime(root: Path) -> None:
    """Refuse runtime modules already imported from another directory.

    Must run before the package root goes first on ``sys.path``: a
    runtime module already in ``sys.modules`` keeps resolving to its
    original directory, so serving a second package in one process
    would silently mix two runtimes.
    """

    for name in RUNTIME_MODULES:
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve().parent != Path(root).resolve():
            raise ValueError(
                f"a {name} module from another package directory is already "
                "imported in this process; serve one package per process"
            )


def put_first_on_sys_path(root: Path) -> str:
    """Put the resolved package root at the front of ``sys.path``.

    Equivalent entries are dropped, not reused: a ``PYTHONPATH`` copy of
    the root listed later would leave foreign directories ahead of it,
    and ``verify_no_foreign_runtime`` only guards ``sys.modules``, which
    is empty before the first import.
    """

    resolved = Path(root).resolve()
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != resolved]
    sys.path.insert(0, str(resolved))
    return str(resolved)


def verify_package(root: Path, table: dict[str, str] | None = None) -> None:
    """The full pre-import gate for one package directory."""

    verify_config(root)
    verify_files(root, table)
    verify_weights(root)


class Vela2Executor:
    """Load and drive one immutable pinned Vela 2.0 0.3B runtime instance."""

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
            raise ValueError(f"the Vela 2.0 0.3B package directory does not exist: {root}")
        verify_package(root)

        # Import the just-verified runtime the way upstream vela2_serve.py
        # does: the package directory goes first on sys.path so the flat
        # runtime modules resolve to the checkpoint's own copies. One
        # worker process serves one package; a runtime module already
        # imported from another directory would silently be the wrong
        # runtime.
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
        self.model_name = model_name or vela2_inference.DEFAULT_MODEL_NAME
        self.max_length = int(self.engine.max_len)
        self._lock = threading.RLock()
        self._closed = False

    @property
    def served(self) -> set[str]:
        """The model names this executor answers to (upstream: one)."""

        return {self.model_name}

    def system_one(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run one admitted request body; the response is upstream's, verbatim.

        Raises the pinned runtime's ``SystemOneError`` (exposed as
        ``self.SystemOneError``) for whole-request 422/413 rejections;
        the worker maps it through ``preprocess.to_request_error``.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("Vela 2.0 executor is closed")
            return self._system_one_request(self.engine, body, served=self.served)

    def warmup(self) -> None:
        """Run one real request of every question kind before readiness.

        The upstream server has no warmup; this follows the repository's
        worker convention that readiness follows a real first inference,
        so the first client request never pays kernel-selection or
        tokenisation first-touch costs.
        """

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
