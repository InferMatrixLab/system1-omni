"""Export pinned Kev-4B language weights and pointer head on CPU (no serving)."""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODEL_ID = "jaredpalmer/kev-4b"
REVISION = "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
BASE_ID = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
REFERENCE_REVISION = "2074cc44256200bb37540103fa4572901faa10f3"
SPECIAL = {
    "<|fim_prefix|>": 248060,
    "<|fim_middle|>": 248061,
    "<|box_start|>": 248049,
    "<|box_end|>": 248050,
    "<|fim_suffix|>": 248062,
}
BASE_FILES = {
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c",
    "model.safetensors.index.json": "eae340074abb0a5f31a6621f7ae8e8248a7c1790df04a722c4e4b70c2a6d1dbb",
    "model.safetensors-00001-of-00002.safetensors": "df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712",
    "model.safetensors-00002-of-00002.safetensors": "590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef",
}
CHECKPOINT_FILES = {
    "adapter_config.json": "8a05dfd6c5e7f61a62e6db5d8de8093a8dbbd105aa7ee7b5e7b73f6249f45617",
    "adapter_model.safetensors": "90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2",
    "head.pt": "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c",
}


def expected_targets():
    """The pinned 32-layer model: 96 MLP, 32 attention and 120 GDN targets."""
    targets = set()
    for layer in range(32):
        modules = [f"mlp.{p}_proj" for p in ("gate", "up", "down")]
        if layer % 4 == 3:
            modules += [f"self_attn.{p}_proj" for p in ("q", "k", "v", "o")]
        else:
            modules += [f"linear_attn.{p}" for p in (
                "in_proj_qkv", "in_proj_a", "in_proj_z", "in_proj_b", "out_proj"
            )]
        targets.update(f"model.language_model.layers.{layer}.{m}.weight" for m in modules)
    return targets


def validate_targets(targets):
    expected = expected_targets()
    if set(targets) != expected:
        raise ValueError(
            f"LoRA target mismatch: missing={sorted(expected - set(targets))}, "
            f"unexpected={sorted(set(targets) - expected)}"
        )


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(directory, expected):
    for name, digest in expected.items():
        if sha256(directory / name) != digest:
            raise ValueError(f"{name} does not match the pinned release")


def validate_tokenizer(path):
    tokens = {
        t["content"]: t["id"] for t in json.loads(path.read_text())["added_tokens"]
    }
    if any(tokens.get(token) != value for token, value in SPECIAL.items()):
        raise ValueError("base tokenizer has unexpected delimiter IDs")


def adapter_pairs(tensors):
    pairs = {}
    for name, value in tensors.items():
        prefix, marker, side = name.partition(".lora_")
        if (
            not prefix.startswith("base_model.model.layers.")
            or not marker
            or side not in ("A.weight", "B.weight")
        ):
            raise ValueError(f"unsupported adapter tensor: {name}")
        if (
            value.dtype != torch.float32
            or value.ndim != 2
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"invalid FP32 adapter tensor: {name}")
        target = (
            "model.language_model."
            + prefix.removeprefix("base_model.model.")
            + ".weight"
        )
        pairs.setdefault(target, {})[side[0]] = value
    if any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("incomplete LoRA pair")
    return pairs


def merge_weight(weight, a, b):
    """Kev's alpha/r = 2: FP32 delta and addition, one final BF16 rounding."""
    if (
        weight.ndim != 2
        or a.shape != (16, weight.shape[1])
        or b.shape != (weight.shape[0], 16)
    ):
        raise ValueError("LoRA shape mismatch")
    merged = (weight.float() + (b @ a) * 2.0).to(torch.bfloat16)
    if not torch.isfinite(merged).all():
        raise ValueError("non-finite merged weight")
    return merged.contiguous()


def pointer_head(meta):
    expected = {
        "base": BASE_ID,
        "base_revision": BASE_REVISION,
        "lora": 16,
        "head_dim": 256,
        "option_isolation": False,
        "special_embeddings": False,
        "weights_dtype": "fp32",
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported Kev head metadata")
    temperature = meta.get("temperature")
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid calibration temperature")
    shapes = {
        "q.weight": (256, 2560),
        "k.weight": (256, 2560),
        "q.bias": (256,),
        "k.bias": (256,),
    }
    head = meta["head"]
    if set(head) != set(shapes):
        raise ValueError("unexpected pointer-head keys")
    for name, shape in shapes.items():
        value = head[name]
        if (
            value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"invalid pointer-head tensor: {name}")
    return {name: value.contiguous() for name, value in head.items()}, temperature


def export(base, checkpoint, out):
    if out.exists():
        raise ValueError("output already exists; choose a new directory")
    verify_files(base, BASE_FILES)
    verify_files(checkpoint, CHECKPOINT_FILES)
    validate_tokenizer(base / "tokenizer.json")
    # Hash the pickle before parsing and use the restricted weights-only loader.
    head, temperature = pointer_head(
        torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    )
    pairs = adapter_pairs(
        load_file(checkpoint / "adapter_model.safetensors", device="cpu")
    )
    validate_targets(pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    used, weight_map = set(), {}
    try:
        for name in BASE_FILES:
            if not name.endswith(".safetensors"):
                continue
            weights = {}
            with safe_open(base / name, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if not key.startswith("model.language_model."):
                        continue  # Kev is text-only; exclude vision and MTP.
                    weight = source.get_tensor(key)
                    if key in pairs:
                        pair = pairs[key]
                        weight = merge_weight(weight, pair["A"], pair["B"])
                        used.add(key)
                    else:
                        weight = weight.to(torch.bfloat16).contiguous()
                        if not torch.isfinite(weight).all():
                            raise ValueError(f"non-finite base tensor: {key}")
                    weights[key] = weight
                    if key in weight_map:
                        raise ValueError(f"duplicate base tensor: {key}")
                    weight_map[key] = name
            if weights:
                save_file(weights, staging / name)
            del weights
        if used != set(pairs):
            raise ValueError(f"unmerged adapter targets: {sorted(set(pairs) - used)}")
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}, indent=2) + "\n"
        )
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(base / name, staging / name)
        save_file(head, staging / "head.safetensors")
        manifest = {
            "format": "kev-4b-merged/1",
            "model_id": MODEL_ID,
            "revision": REVISION,
            "base_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "reference_revision": REFERENCE_REVISION,
            "backbone_dtype": "bfloat16",
            "head_dtype": "float32",
            "hidden_size": 2560,
            "pointer_size": 256,
            "temperature": temperature,
            "special_tokens": SPECIAL,
            "merged_targets": sorted(used),
            "inputs": {"base": BASE_FILES, "checkpoint": CHECKPOINT_FILES},
            "files": {p.name: sha256(p) for p in sorted(staging.iterdir())},
        }
        (staging / "kev_export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest = export(args.base, args.checkpoint, args.out)
    print(
        f"Exported {len(manifest['merged_targets'])} merged projections to {args.out}"
    )


if __name__ == "__main__":
    main()
