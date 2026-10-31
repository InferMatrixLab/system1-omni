#!/usr/bin/env python3
"""Export the pinned Vela 2.0 0.8B package into a native bundle.

The bundle re-keys the hybrid
backbone for the shared qwen3_5 native loader — ``backbone.X`` tensors
become ``model.X`` (the loader's prefix probe accepts ``model.``), with
**per-tensor dtypes preserved verbatim** (Linear/conv1d in BF16,
embeddings/norms/A_log/dt_bias in FP32 — the reference's own mixed
layout, no re-save rounding) — and copies the Vela-owned F32 heads and
the side-loaded broad span head as separate files, plus calibration and
tokenizer. ``config.json`` is the package's ``backbone_config`` written
at the root, which is exactly the shape the family ``Config::load``
reads. The manifest pins every source file by SHA-256 (re-using the
worker's own gate table) and records the exported tensor inventory.

    PYTHONPATH=src .venv/bin/python recipe/vela/export_native.py \
      --package weights/vela-2.0-0.8b --out weights/vela-2.0-0.8b-native
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

BACKBONE_PREFIX = "backbone."
HEAD_PREFIXES = ("head.", "span2.", "set_bias")
FORMAT = "vela2-native-bundle/1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, help="pinned Vela 2.0 0.8B package directory")
    parser.add_argument("--out", required=True, help="output bundle directory (created)")
    parser.add_argument(
        "--bf16-out",
        default=None,
        help="also write a uniform-BF16 backbone directory for the shared qwen3_5 "
        "loader (model.safetensors + config.json)",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_into_place(target: Path, write_files) -> None:
    """Stage the artifact set in a sibling directory and move it into place
    with one rename, so the target only ever holds a complete export: a run
    killed mid-write leaves at most the staging directory, which the next
    run with the same target removes."""
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"output directory {target} already exists and is not empty")
    staging = target.with_name(target.name + ".export-partial")
    if staging.is_dir():
        # Remains of a run killed mid-write with the same target directory.
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        write_files(staging)
        if target.exists():
            # Empty, checked above; the rename replaces the directory whole.
            target.rmdir()
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import torch
    from safetensors.torch import load_file, save_file

    from models.vela.hybrid import (
        HYBRID_REPO,
        HYBRID_REVISION,
        PINNED_FILES_SHA256,
        verify_package,
    )

    package = Path(args.package)
    out = Path(args.out)
    verify_package(package)  # the worker's own gate: provenance before export
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    backbone_config = config["backbone_config"]

    # The shard map names every backbone and head tensor; broad_head is a
    # separate side-loaded file.
    index = json.loads((package / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    tensors = {}
    for name in shard_files:
        tensors.update(load_file(str(package / name)))
    broad = load_file(str(package / "broad_head.safetensors"))

    backbone, heads = {}, {}
    dtypes: dict[str, dict[str, str]] = {"backbone": {}, "heads": {}}
    for name, tensor in sorted(weight_map.items()):
        value = tensors[name]
        dtype = str(value.dtype).removeprefix("torch.")
        if name.startswith(BACKBONE_PREFIX):
            rekeyed = "model." + name[len(BACKBONE_PREFIX):]
            backbone[rekeyed] = value
            dtypes["backbone"][rekeyed] = f"{dtype}:{list(value.shape)}"
        elif name.startswith(HEAD_PREFIXES):
            heads[name] = value
            dtypes["heads"][name] = f"{dtype}:{list(value.shape)}"
        else:
            raise SystemExit(f"unexpected tensor key {name!r}; update the export for the new prefix")
    for name, value in sorted(broad.items()):
        heads[name] = value
        dtypes["heads"][name] = f"{str(value.dtype).removeprefix('torch.')}:{list(value.shape)}"

    if args.bf16_out:
        # Uniform-BF16 re-save for the family crate's loader: every
        # backbone tensor rounds to bfloat16 once, at export (the
        # reference's own GPU setting runs the backbone under bf16
        # autocast too).
        bf16_dir = Path(args.bf16_out)
        uniform = {name: value.to(torch.bfloat16) for name, value in backbone.items()}

        def write_bf16(directory: Path) -> None:
            save_file(uniform, str(directory / "model.safetensors"))
            (directory / "config.json").write_text(json.dumps(backbone_config, indent=1) + "\n", encoding="utf-8")

        stage_into_place(bf16_dir, write_bf16)
        print(f"bf16 backbone: {len(uniform)} tensors -> {bf16_dir}")

    def write_bundle(out: Path) -> None:
        save_file(backbone, str(out / "backbone.safetensors"))
        save_file(heads, str(out / "heads.safetensors"))
        # config.json for the family loader: backbone_config at the root.
        (out / "config.json").write_text(json.dumps(backbone_config, indent=1) + "\n", encoding="utf-8")
        for name in ("calibration.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(package / name, out / name)
        for name in [
            "backbone.safetensors", "heads.safetensors", "config.json",
            "calibration.json", "tokenizer.json", "tokenizer_config.json",
        ]:
            manifest["files_sha256"][name] = sha256(out / name)
        if args.bf16_out:
            for name in ("model.safetensors", "config.json"):
                manifest["files_sha256"][f"bf16/{name}"] = sha256(bf16_dir / name)
        (out / "vela_native_manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )

    stage_into_place(out, write_bundle)

    print(f"export ok: {out}")
    print(f"  backbone: {len(backbone)} tensors ({bf16} bf16, {fp32} fp32), re-keyed {BACKBONE_PREFIX} -> model.")
    print(f"  heads:    {len(heads)} tensors (F32, incl. span_broad)")
    print(f"  config:   backbone_config at root, model_type={backbone_config.get('model_type')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
