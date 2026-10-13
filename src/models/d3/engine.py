"""D3 executor for the pinned code-readout checkpoint.

The executor owns the pinned runtime, the manifest verification that gates its
import, device state, and warmup. It consumes raw state/questions through the
upstream ``prepare``/``run``/``respond`` surface and stays free of HTTP
concerns; wire admission lives in protocol.py and preprocess.py.

The checkpoint ships its own runtime as flat Python files next to the
weights. Those files are the code this process executes, so — stronger than
the upstream default, whose ``verify_package`` walks only the 39-entry
``files_sha256`` table and skips hashing files over 64 MiB in fast mode — the
executor hashes every file of the manifest's ``runtime.files_sha256`` table
in full before importing anything from the package directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

REFERENCE_REPO = "vllm-sr/d3"
REFERENCE_REVISION = "cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75"
MANIFEST_NAME = "MODEL_MANIFEST.json"
MANIFEST_SCHEMA = "d3-package-manifest/1"
VERIFY_MODES = ("fast", "full", "none")


def _sha256_safe(path: Path) -> str:
    """Might encounter a large file, read in chunks"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> dict[str, Any]:
    """Read and schema-check the package manifest; refuse packages without one."""

    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(
            f"the package is missing {MANIFEST_NAME}; this worker only serves verified packages"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{MANIFEST_NAME} is not a {MANIFEST_SCHEMA} manifest")
    return manifest


def verify_runtime_files(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Hash every runtime file of the manifest in full and return their names.

    Must run before any package code is imported: the runtime table names the
    files this process is about to execute, so unlike the upstream shard
    verification there is no size shortcut.
    """

    table = (manifest.get("runtime") or {}).get("files_sha256")
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{MANIFEST_NAME} has no runtime.files_sha256 table")
    problems: list[str] = []
    for name, digest in sorted(table.items()):
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"unsafe runtime file name {name!r}")
        target = Path(root) / name
        if not target.is_file():
            problems.append(f"missing runtime file {name}")
            continue
        if _sha256_safe(target) != digest:
            problems.append(f"{name}: runtime SHA-256 differs from {MANIFEST_NAME}")
    if problems:
        raise ValueError("runtime verification failed: " + "; ".join(problems[:8]))
    return sorted(table)


class D3Executor:
    """Load and drive one immutable pinned d3 runtime instance."""

    model_name = "d3"

    def __init__(
        self,
        package: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 8,
        verify: str = "fast",
        model_name: str | None = None,
        max_length: int | None = None,
        readout_dtype: str | None = None,
        permutation_average: bool = False,
    ) -> None:
        if verify not in VERIFY_MODES:
            raise ValueError(f"verify must be one of {VERIFY_MODES}")
        root = Path(package)
        if not root.is_dir():
            raise ValueError(f"the d3 package directory does not exist: {root}")
        manifest = load_manifest(root)
        verify_runtime_files(root, manifest)

        # Import the just-verified runtime the way upstream d3_server.py does:
        # the package directory goes first on sys.path so the flat runtime
        # modules (d3_runtime, d3_format, ...) resolve to the checkpoint's own
        # copies. One worker process serves one package.
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        import d3_runtime

        self.model = d3_runtime.D3.from_pretrained(
            root,
            device=device,
            batch_size=batch_size,
            verify=verify,
            model_name=model_name,
            max_length=max_length,
            readout_dtype=readout_dtype,
            permutation_average=permutation_average,
        )
        self.model_name = self.model.model_name
        self.max_length = self.model.max_length
        self._image_unavailable = self.model.image_unavailable
        self._video_unavailable = self.model.video_unavailable
        self._lock = threading.RLock()
        self._closed = False

    @property
    def modalities(self) -> list[str]:
        """The loaded model's input modalities, in upstream /health order."""

        names = ["text", "image"] if self._image_unavailable is None else ["text"]
        return names + ["video"] if self._video_unavailable is None else names

    def prepare(self, state: Any, questions: dict[str, Any]) -> Any:
        """Validate and tokenize one request; per-question failures land in
        ``prepared.errors`` instead of raising (envelope problems raise)."""

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            return self.model.prepare(state, questions)

    def run(self, prepared: Any) -> tuple[dict[str, list[float]], int]:
        """Probabilities per runnable question and the input token count."""

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            return self.model.run(prepared)

    def respond(self, prepared: Any, probabilities: dict[str, list[float]], tokens: int) -> dict:
        """Build the System One response, embedding per-question model failures."""

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            return self.model.respond(prepared, probabilities, tokens)

    def warmup(self) -> None:
        """Compile and autotune the kernels before readiness."""

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            self.model.warmup()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.model = None
