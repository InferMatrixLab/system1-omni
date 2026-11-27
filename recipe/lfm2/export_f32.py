"""Widen the pinned LFM2.5-350M BF16 checkpoint into a local F32 artifact."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "src/models/lfm2/native/profile.json"
SOURCE = "model.safetensors"
WEIGHTS = "weights.f32.safetensors"
FORMAT = "omni-lfm2-f32-v1"
CHUNK = 1 << 20


def identity(path):
    with path.open("rb") as handle:
        return {"bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_source(checkpoint, profile):
    files = {name: identity(checkpoint / name) for name in profile["checkpoint"]}
    for name, info in files.items():
        require(info["sha256"] == profile["checkpoint"][name], f"wrong source hash: {name}")
    return files


def widen(source, output):
    """Write a deterministic safetensors header and widen unsigned BF16 bits."""
    with source.open("rb") as handle:
        header_bytes, = struct.unpack("<Q", handle.read(8))
        require(0 < header_bytes <= 32768, "unexpected BF16 header length")
        header = json.loads(handle.read(header_bytes))
    header.pop("__metadata__", None)
    names = sorted(header)
    require(len(names) == 148 and "lm_head.weight" not in header
            and "model.embed_tokens.weight" in header, "wrong inventory or broken tied LM head")

    out_header = {"__metadata__": {"format": "pt"}}
    cursor = 0
    with safe_open(source, framework="pt", device="cpu") as tensors:
        require(tensors.keys() == names, "BF16 safetensors index mismatch")
        for name in names:
            entry = header[name]
            start, end = entry["data_offsets"]
            shape = tensors.get_slice(name).get_shape()
            require(entry["dtype"] == "BF16" and shape == entry["shape"],
                    f"unexpected source dtype or shape: {name}")
            byte_count = (end - start) * 2
            out_header[name] = {"dtype": "F32", "shape": shape,
                                "data_offsets": [cursor, cursor + byte_count]}
            cursor += byte_count
    encoded = json.dumps(out_header, separators=(",", ":")).encode("ascii")
    encoded += b" " * (-len(encoded) % 8)
    with output.open("xb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name in names:
            start, end = header[name]["data_offsets"]
            values = np.memmap(source, dtype="<u2", mode="r",
                               offset=8 + header_bytes + start, shape=((end - start) // 2,))
            for offset in range(0, values.size, CHUNK):
                widened = np.left_shift(values[offset:offset + CHUNK].astype("<u4"), 16)
                handle.write(widened.astype("<u4", copy=False).tobytes())
            del values
        handle.flush()
        os.fsync(handle.fileno())


def verify_values(checkpoint, artifact):
    """Independently compare all F32 bits to PyTorch CPU widening."""
    import torch

    with safe_open(checkpoint / SOURCE, framework="pt", device="cpu") as src, \
            safe_open(artifact / WEIGHTS, framework="pt", device="cpu") as dst:
        require(src.keys() == dst.keys(), "tensor names changed during export")
        parameters = 0
        for name in src.keys():
            original = src.get_tensor(name)
            result = dst.get_tensor(name)
            require(original.dtype == torch.bfloat16 and result.dtype == torch.float32
                    and original.shape == result.shape, f"dtype or shape mismatch: {name}")
            original = original.flatten()
            result = result.flatten()
            for start in range(0, original.numel(), CHUNK):
                expected = original[start:start + CHUNK].float()
                actual = result[start:start + CHUNK]
                require(torch.equal(actual.view(torch.int32), expected.view(torch.int32)),
                        f"F32 widening mismatch: {name} from element {start}")
                require(torch.isfinite(actual).all().item(), f"non-finite weight: {name}")
            parameters += original.numel()
        return {"tensors": len(src.keys()), "parameters_bitwise_checked": parameters}


def verify(checkpoint, artifact, profile, files):
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["format"] == FORMAT and manifest["profile"] == profile,
            "artifact profile or schema mismatch")
    expected = {WEIGHTS: identity(artifact / WEIGHTS)}
    expected.update({name: info for name, info in files.items() if name != SOURCE})
    require(manifest["files"] == expected, "artifact inventory mismatch")
    for name, info in expected.items():
        if name != WEIGHTS:
            require(identity(artifact / name) == info, f"changed checkpoint file: {name}")
    report = verify_values(checkpoint, artifact)
    report.update({"manifest": identity(artifact / "manifest.json"), "files": expected})
    return report


def export(checkpoint, output, profile, files):
    require(not output.exists(), f"output already exists: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest_files = {name: info for name, info in files.items() if name != SOURCE}
        needed = files[SOURCE]["bytes"] * 2 + sum(info["bytes"] for info in manifest_files.values())
        require(shutil.disk_usage(staging).free >= needed, "insufficient storage for F32 export")
        for name in manifest_files:
            shutil.copyfile(checkpoint / name, staging / name)
        widen(checkpoint / SOURCE, staging / WEIGHTS)
        manifest_files[WEIGHTS] = identity(staging / WEIGHTS)
        manifest = {"format": FORMAT, "profile": profile, "files": manifest_files}
        with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        report = verify(checkpoint, staging, profile, files)
        require(not output.exists(), f"output already exists: {output}")
        staging.rename(output)
        return report
    except BaseException as error:
        error.add_note(f"Partial export retained at {staging}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="local pinned LFM2.5-350M snapshot")
    parser.add_argument("artifact", type=Path, help="output directory (or existing directory with --verify)")
    parser.add_argument("--verify", action="store_true", help="verify an existing export without writing it")
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    files = check_source(args.checkpoint, profile)
    if args.verify:
        report = verify(args.checkpoint, args.artifact, profile, files)
    else:
        report = export(args.checkpoint, args.artifact, profile, files)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
