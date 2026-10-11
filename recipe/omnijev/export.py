"""Export OmniJev-4B v1.1 for the native worker on CPU.

Merges the language LoRA into the pinned Qwen3.5-4B in BF16, restores the option
tokens' embeddings, and writes the base vision tensors, the FP32 heads, the tokenizer
and the calibration next to it. Use the reference environment in README.md; no GPU
is needed. Needs about 18 GB of host RAM and writes 9 GB.
"""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "tinnel123/OmniJev"
CHECKPOINT_REVISION = "ffe5f436eaf22e20e2f041f8e74e121fd057a6cb"
BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_FILES = {
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "model.safetensors.index.json": "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
    "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
OPTION_TOKENS = {"<|opt|>": 248077, "<|/opt|>": 248078}
VOCABULARY = 248079       # the tokenizer's length; the reference resizes the tied embedding to it
MAX_PIXELS = 768 * 28 * 28  # MSO1's default, which overrides the processor's 401,408
NEEDED = ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja", "head.pt", "head_meta.json",
          "new_tok_emb.pt", "ord.pt", "processor_config.json", "tokenizer.json", "tokenizer_config.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, type=Path, help=f"{BASE_MODEL_ID} @ {BASE_REVISION}")
    parser.add_argument("--checkpoint", required=True, type=Path, help=f"{MODEL_ID} @ {CHECKPOINT_REVISION}")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output already exists; choose a new export directory")

    release = json.loads((args.checkpoint / "release_manifest.json").read_text())
    if release.get("release") != "v1.1" or release.get("base_model") != BASE_MODEL_ID:
        raise ValueError("expected the OmniJev v1.1 release manifest")
    inputs = {}
    for name in NEEDED:
        digest = sha256(args.checkpoint / name)
        if digest != release["files"][name]["sha256"]:
            raise ValueError(f"{name} does not match the release manifest")
        inputs[f"checkpoint/{name}"] = digest
    for name, expected in BASE_FILES.items():
        if sha256(args.base / name) != expected:
            raise ValueError(f"{name} does not match {BASE_MODEL_ID} @ {BASE_REVISION}")
        inputs[f"base/{name}"] = expected
    meta = json.loads((args.checkpoint / "head_meta.json").read_text())
    if not (meta["norm"] == "softmax" and meta["lm_feats"] and meta["ordinal"] and meta["branch"]):
        raise ValueError("expected softmax heads with LM features, the ordinal head and branching")
    extra = set(meta.get("biases", {})) - {"noul"}
    if extra:
        raise ValueError(f"unsupported calibration biases: {sorted(extra)}")
    temperatures, noul_bias = meta["temperatures"], meta["biases"]["noul"]
    if not (sorted(temperatures) == ["choice", "noul", "score"]
            and all(math.isfinite(t) and t > 0 for t in temperatures.values()) and math.isfinite(noul_bias)):
        raise ValueError("expected positive, finite temperatures and a finite Noul bias")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if len(tokenizer) != VOCABULARY or any(tokenizer.convert_tokens_to_ids(t) != i for t, i in OPTION_TOKENS.items()):
        raise ValueError("unexpected tokenizer")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": "cpu"}, local_files_only=True)
    # mso.records.add_option_tokens: resize the tied embedding to the tokenizer and
    # restore the saved option-token rows.
    model.resize_token_embeddings(VOCABULARY)
    saved = torch.load(args.checkpoint / "new_tok_emb.pt", map_location="cpu", weights_only=True)
    if saved.shape != (2, model.config.text_config.hidden_size):
        raise ValueError("unexpected option-token embeddings")
    with torch.no_grad():
        embedding = model.get_input_embeddings().weight
        for row, token in zip(saved, OPTION_TOKENS.values()):
            embedding[token] = row.to(embedding.dtype)
    if model.get_output_embeddings().weight.data_ptr() != embedding.data_ptr():
        raise ValueError("expected tied input and output embeddings")

    model = PeftModel.from_pretrained(model, args.checkpoint).merge_and_unload(safe_merge=True)
    # Write next to the output and rename at the end, so a failed or killed export
    # leaves no output behind; the next run removes what it left.
    staging = args.out.with_name(f".{args.out.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    model.model.language_model.save_pretrained(staging, max_shard_size="5GB")
    model.config.to_json_file(staging / "config.json")
    vision = {f"model.visual.{k}": v.contiguous() for k, v in model.model.visual.state_dict().items()}
    save_file(vision, staging / "vision.safetensors")

    heads = {}
    for prefix, name in (("head", "head.pt"), ("ord", "ord.pt")):
        state = torch.load(args.checkpoint / name, map_location="cpu", weights_only=True)
        for key, value in state.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite {prefix}.{key}")
            heads[f"{prefix}.{key}"] = value.float().contiguous()
    save_file(heads, staging / "heads.safetensors")
    tokenizer.save_pretrained(staging)

    outputs = {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()}
    # Written last: the worker refuses incomplete exports.
    (staging / "omnijev_export.json").write_text(json.dumps({
        "format": "omnijev-merged/1",
        "model_id": MODEL_ID, "checkpoint_revision": CHECKPOINT_REVISION,
        "base_model_id": BASE_MODEL_ID, "base_revision": BASE_REVISION,
        "lora": {"rank": 32, "alpha": 64, "merged": "bfloat16"},
        "vocabulary": VOCABULARY, "option_tokens": OPTION_TOKENS, "max_pixels": MAX_PIXELS,
        "heads": {"norm": meta["norm"], "lm_features": True, "ordinal": True},
        "calibration": {"temperatures": temperatures, "noul_bias": noul_bias},
        "inputs": inputs, "outputs": outputs,
    }, indent=1, allow_nan=False) + "\n")
    staging.rename(args.out)


if __name__ == "__main__":
    main()
