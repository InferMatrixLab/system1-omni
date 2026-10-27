#!/usr/bin/env python3
"""Export a pinned Vela 2.0 decoder package (0.8B/4B/9B) into a native bundle.

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

The member is whatever the package's own gate resolves to
(config.json `model_name`), so the same script exports 0.8B, 4B and 9B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BACKBONE_PREFIX = "backbone."
HEAD_PREFIXES = ("head.", "span2.", "set_bias")
FORMAT = "vela2-native-bundle/1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--package", required=True,
        help="pinned Vela 2.0 package directory (a decoder member: 0.8B/4B/9B)",
    )
    parser.add_argument("--out", required=True, help="output bundle directory (created)")
    parser.add_argument(
        "--bf16-out",
        default=None,
        help="also write a uniform-BF16 backbone directory for the shared qwen3_5 "
        "loader (model.safetensors + config.json)",
    )
    parser.add_argument(
        "--executor-only",
        action="store_true",
        help="skip the faithful mixed-dtype backbone.safetensors copy and keep only "
        "what the executor path reads (heads, the bf16 backbone, config, calibration, "
        "tokenizer); for hosts whose disk cannot hold both backbone copies. The "
        "manifest still records the full tensor inventory from the shard map.",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import torch
    from safetensors.torch import load_file, save_file

    from models.vela.hybrid import verify_package

    package = Path(args.package)
    out = Path(args.out)
    # The worker's own gate: provenance before export; it also resolves
    # the member (0.8B/4B/9B) the package declares.
    member = verify_package(package)
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

    out.mkdir(parents=True, exist_ok=True)
    if not args.executor_only:
        save_file(backbone, str(out / "backbone.safetensors"))
    save_file(heads, str(out / "heads.safetensors"))
    # config.json for the family loader: backbone_config at the root.
    (out / "config.json").write_text(json.dumps(backbone_config, indent=1) + "\n", encoding="utf-8")
    for name in ("calibration.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(package / name, out / name)

    bf16 = sum(1 for d in dtypes["backbone"].values() if d.startswith("bfloat16"))
    fp32 = len(dtypes["backbone"]) - bf16
    manifest = {
        "format": FORMAT,
        "model_id": member.repo,
        "model_name": config["model_name"],
        "max_length": config["max_length"],
        "description": config.get("description", ""),
        "source": {
            "repo": member.repo,
            "revision": member.revision,
            # The worker gate's own table, re-verified above.
            "pinned_files_sha256": dict(member.pinned_files),
            "shard_files": shard_files,
        },
        "backbone": {
            "tensors": len(backbone),
            "bf16": bf16,
            "fp32": fp32,
            "note": "per-tensor dtypes preserved from the package; backbone.X re-keyed to model.X",
            "inventory": dtypes["backbone"],
        },
        "heads": {
            "tensors": len(heads),
            "inventory": dtypes["heads"],
            "note": "Vela-owned F32 heads (CandidateHead, set_bias, span2 router, span_broad) in one file",
        },
        "files_sha256": {},
    }
    if args.bf16_out:
        # Uniform-BF16 re-save for the family crate's loader: every
        # backbone tensor rounds to bfloat16 once, at export (the
        # reference's own GPU setting runs the backbone under bf16
        # autocast too).
        bf16_dir = Path(args.bf16_out)
        bf16_dir.mkdir(parents=True, exist_ok=True)
        uniform = {name: value.to(torch.bfloat16) for name, value in backbone.items()}
        save_file(uniform, str(bf16_dir / "model.safetensors"))
        (bf16_dir / "config.json").write_text(json.dumps(backbone_config, indent=1) + "\n", encoding="utf-8")
        print(f"bf16 backbone: {len(uniform)} tensors -> {bf16_dir}")

    bundle_files = ["heads.safetensors", "config.json",
                   "calibration.json", "tokenizer.json", "tokenizer_config.json"]
    if not args.executor_only:
        bundle_files.insert(0, "backbone.safetensors")
    for name in bundle_files:
        manifest["files_sha256"][name] = sha256(out / name)
    if args.bf16_out:
        for name in ("model.safetensors", "config.json"):
            manifest["files_sha256"][f"bf16/{name}"] = sha256(bf16_dir / name)
    (out / "vela_native_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"export ok: {out}")
    print(f"  backbone: {len(backbone)} tensors ({bf16} bf16, {fp32} fp32), re-keyed {BACKBONE_PREFIX} -> model.")
    print(f"  heads:    {len(heads)} tensors (F32, incl. span_broad)")
    print(f"  config:   backbone_config at root, model_type={backbone_config.get('model_type')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
