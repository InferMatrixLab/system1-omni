"""Launch the Rust Clef server.

HTTP used to be ``ThreadingHTTPServer`` in this module: one new thread per
request, and the first CUDA call on that thread paid the context-init tax.
The public server is now the ``omni-clef`` binary. It admits through
``omni-runtime`` and scores on one persistent executor process.

Running this module replaces the process with that binary. It does not
bind a port and it does not fall back to a Python HTTP server. ``decide``
remains importable for the executor and for contract tests.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from models.clef.engine import DEFAULT_VARIANT, FLA_PREFILL, GDN_PREFILL, VARIANTS
from models.clef.piecewise import BACKBONE_EAGER, BACKBONE_PIECEWISE
from models.clef.worker import decide

__all__ = ["decide", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MISSING = (
    "omni-clef is not built. From the repository root run "
    "`cargo build --release --locked -p omni-clef` and retry. "
    "This launcher does not start a Python HTTP server."
)


def candidate_binaries() -> list[Path]:
    """In-tree build outputs, release before debug."""

    return [
        _REPO_ROOT / "target" / "release" / "omni-clef",
        _REPO_ROOT / "target" / "debug" / "omni-clef",
    ]


def find_binary() -> Path:
    override = os.environ.get("OMNI_CLEF_BIN")
    if override:
        path = Path(override)
        if not path.is_file():
            raise SystemExit(f"OMNI_CLEF_BIN is not a file: {path}")
        return path
    for candidate in candidate_binaries():
        if candidate.is_file():
            return candidate
    found = shutil.which("omni-clef")
    if found:
        return Path(found)
    raise SystemExit(_MISSING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exec the omni-clef Rust server for the pinned Clef snapshot."
    )
    parser.add_argument("--model", required=True, help="snapshot directory of the pinned revision")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gdn-prefill",
        default=FLA_PREFILL,
        choices=(FLA_PREFILL, GDN_PREFILL),
        help=(
            "Linear-attention prefill. fla is the measured default. "
            "flashinfer is opt-in and has no answer parity against that path."
        ),
    )
    parser.add_argument(
        "--backbone",
        default=BACKBONE_EAGER,
        choices=(BACKBONE_EAGER, BACKBONE_PIECEWISE),
        help=(
            "Qwen3.5 backbone. eager is the default. piecewise is the opt-in "
            "inductor CUDA-graph split and is not the default."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    binary = find_binary()
    forwarded = [
        str(binary),
        "--model",
        args.model,
        "--variant",
        args.variant,
        "--device",
        args.device,
        "--gdn-prefill",
        args.gdn_prefill,
        "--backbone",
        args.backbone,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    try:
        os.execv(str(binary), forwarded)
    except OSError as exc:
        raise SystemExit(f"failed to start {binary}: {exc}") from exc


if __name__ == "__main__":
    main()
