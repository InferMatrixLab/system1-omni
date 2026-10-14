"""Export-tool tests for recipe/valen/export_checkpoint.py (spec 02, E1).

CPU-only checks of the export's observable contract: the LoRA merge and
FP32->BF16 downcast semantics on synthetic tensors (against the installed
PEFT as the independent reference), checkpoint-key classification, and the
manifest/artifact consistency on a tiny synthetic fixture. The full
determinism run against the real pinned artifacts is env-gated on
VALEN_CKPT / VALEN_BASE, following the CUA_S1_BASE precedent.

PYTHONPATH=src python -m pytest tests/valen -q
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE_VALEN = HERE.parents[1] / "recipe" / "valen"
sys.path.insert(0, str(RECIPE_VALEN))

import export_checkpoint as tool  # noqa: E402


def test_lora_scaling_is_alpha_over_rank():
    assert tool.lora_scaling(rank=32, alpha=64) == 2.0
    assert tool.lora_scaling(rank=4, alpha=8) == 2.0
    assert tool.lora_scaling(rank=8, alpha=4) == 0.5


def test_merge_matches_peft_merge_bit_exactly():
    """The pinned peft 0.18.1 Linear.merge is the independent source of truth."""
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    linear = nn.Linear(8, 6, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(6, 8).to(torch.bfloat16))

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = linear

        def forward(self, x):
            return self.lin(x)

    model = get_peft_model(
        Wrapped(),
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["lin"], bias="none"),
    )
    lora_linear = model.base_model.model.lin
    with torch.no_grad():
        lora_linear.lora_A["default"].weight.copy_(torch.randn(4, 8))
        lora_linear.lora_B["default"].weight.copy_(torch.randn(6, 4) * 0.1)
    base_weight = linear.weight.detach().clone()
    lora_a = lora_linear.lora_A["default"].weight.detach().clone()
    lora_b = lora_linear.lora_B["default"].weight.detach().clone()
    scaling = lora_linear.scaling["default"]

    lora_linear.merge(safe_merge=False)
    peft_merged = linear.weight.detach().clone()
    mine = tool.merge_lora_weight(base_weight, lora_a, lora_b, scaling)
    assert mine.dtype == torch.bfloat16
    assert torch.equal(mine, peft_merged)


def test_merge_hand_computed_rounding():
    """bf16(fp32(W) + 2*(B@A)) with one rounding of the sum, hand-computed."""
    import torch

    # bf16 neighbors around 1.0: 1.0 and 1.0078125 (ulp 2^-7); the rounding
    # boundary sits at the midpoint 1.00390625.
    base = torch.full((1, 1), 1.0, dtype=torch.bfloat16)
    # delta = 2 * B@A = 2 * 0.0015 -> sum 1.003 < midpoint -> rounds to 1.0
    small = tool.merge_lora_weight(base, torch.tensor([[0.0015]]), torch.tensor([[1.0]]), 2.0)
    assert small.item() == 1.0
    # delta = 2 * B@A = 2 * 0.002 -> sum 1.004 > midpoint -> rounds to 1.0078125
    large = tool.merge_lora_weight(base, torch.tensor([[0.002]]), torch.tensor([[1.0]]), 2.0)
    assert large.item() == 1.0078125


def test_merge_with_zero_lora_b_is_bit_exact_base():
    """Untrained LoRA (zero B, PEFT init) must merge back to the base tensor."""
    import torch

    torch.manual_seed(1)
    base = torch.randn(6, 8).to(torch.bfloat16)
    lora_a = torch.randn(4, 8)
    lora_b = torch.zeros(6, 4)
    merged = tool.merge_lora_weight(base, lora_a, lora_b, 2.0)
    assert torch.equal(merged, base)


def test_downcast_rounds_to_nearest_bf16():
    """FP32 -> BF16 uses round-to-nearest-even, matching the reference's own
    from_pretrained(dtype=bf16) cast; already-BF16 input passes through."""
    import torch

    values = torch.tensor(
        [1.0, 1.003, 1.004, 0.5, -2.75, 3.14159265], dtype=torch.float32
    )
    downcast = tool.downcast_bf16(values)
    assert downcast.dtype == torch.bfloat16
    assert torch.equal(downcast, values.to(torch.bfloat16))
    # A BF16-representable value survives the downcast bit-exactly.
    bf16_source = torch.tensor([1.0, -0.5], dtype=torch.bfloat16)
    assert torch.equal(tool.downcast_bf16(bf16_source), bf16_source)


# --- checkpoint-key classification (synthetic keys in the pinned layout) ---

def synthetic_checkpoint_keys():
    """A miniature of the 428-key layout: one LoRA pair per family, the
    vision-top blocks 20-23, the merger and the head."""

    lora_modules = [
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_z",
        "layers.0.linear_attn.in_proj_b",
        "layers.0.linear_attn.in_proj_a",
        "layers.0.linear_attn.out_proj",
        "layers.3.self_attn.q_proj",
        "layers.3.self_attn.k_proj",
        "layers.3.self_attn.v_proj",
        "layers.3.self_attn.o_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
    ]
    keys = []
    for module in lora_modules:
        for side, suffix in (("A", tool.PEFT_LORA_SUFFIX_A), ("B", tool.PEFT_LORA_SUFFIX_B)):
            keys.append(tool.PEFT_LANGUAGE_PREFIX + module + suffix)
    vision_blocks = []
    for block in (20, 21, 22, 23):
        for tail in (
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
            "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias",
            "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
        ):
            vision_blocks.append(f"backbone.visual.blocks.{block}.{tail}")
    keys.extend(vision_blocks)
    keys.extend(
        [
            "backbone.visual.merger.norm.weight",
            "backbone.visual.merger.norm.bias",
            "backbone.visual.merger.linear_fc1.weight",
            "backbone.visual.merger.linear_fc1.bias",
            "backbone.visual.merger.linear_fc2.weight",
            "backbone.visual.merger.linear_fc2.bias",
            "head.decision.weight",
            "head.candidate.weight",
        ]
    )
    return keys


def test_classify_checkpoint_weights_layout():
    classification = tool.classify_checkpoint_weights(synthetic_checkpoint_keys())
    # Every LoRA pair maps to the base's own tensor name.
    assert classification.lora["model.language_model.layers.0.mlp.down_proj.weight"] == (
        tool.PEFT_LANGUAGE_PREFIX + "layers.0.mlp.down_proj" + tool.PEFT_LORA_SUFFIX_A,
        tool.PEFT_LANGUAGE_PREFIX + "layers.0.mlp.down_proj" + tool.PEFT_LORA_SUFFIX_B,
    )
    assert len(classification.lora) == 12
    # Vision-top blocks land in the base's model.visual.* layout.
    assert (
        "model.visual.blocks.20.attn.qkv.weight",
        "backbone.visual.blocks.20.attn.qkv.weight",
    ) in classification.vision_top
    assert len(classification.vision_top) == 48
    assert (
        "model.visual.merger.linear_fc2.weight",
        "backbone.visual.merger.linear_fc2.weight",
    ) in classification.merger
    assert len(classification.merger) == 6
    assert classification.head == {
        "head.decision.weight": "head.decision.weight",
        "head.candidate.weight": "head.candidate.weight",
    }


def test_classify_checkpoint_weights_rejects_unknown_keys():
    import pytest

    for key in (
        "backbone.visual.patch_embed.proj.weight",  # frozen, never in checkpoint
        "backbone.language_model.layers.0.mlp.down_proj.weight",  # base weight, not PEFT
        "head.decision.bias",  # the head is bias-free
        "optimizer.steps",
    ):
        with pytest.raises(ValueError, match="unknown checkpoint weight"):
            tool.classify_checkpoint_weights(synthetic_checkpoint_keys() + [key])


def test_classify_checkpoint_weights_rejects_unpaired_lora():
    import pytest

    keys = synthetic_checkpoint_keys()
    keys.remove(tool.PEFT_LANGUAGE_PREFIX + "layers.0.mlp.down_proj" + tool.PEFT_LORA_SUFFIX_B)
    with pytest.raises(ValueError, match="unpaired LoRA"):
        tool.classify_checkpoint_weights(keys)


def test_classify_checkpoint_weights_rejects_non_visual_block_tower():
    import pytest

    # A trainable vision block outside the trained top-4 set contradicts the
    # pinned vision_top stage.
    with pytest.raises(ValueError, match="vision_top"):
        tool.classify_checkpoint_weights(
            synthetic_checkpoint_keys() + ["backbone.visual.blocks.19.norm1.weight"]
        )


def test_validate_inventory_counts():
    classification = tool.classify_checkpoint_weights(synthetic_checkpoint_keys())
    expected = {
        "checkpoint_tensors": 80,
        "lora_keys": 24,
        "vision_top_keys": 48,
        "merger_keys": 6,
        "head_keys": 2,
    }
    tool.validate_inventory(classification, expected)
    import pytest

    with pytest.raises(ValueError, match="lora_keys"):
        tool.validate_inventory(classification, dict(expected, lora_keys=23))


# --- run_export on a tiny synthetic fixture (no real artifacts) ---

TINY_INVENTORY = {
    "checkpoint_tensors": 11,
    "lora_keys": 6,
    "vision_top_keys": 2,
    "merger_keys": 1,
    "head_keys": 2,
    "base_tensors": 12,
    "base_fp32_tensors": 2,
}
TINY_MERGE = {"rank": 4, "alpha": 8}
TINY_HEAD = {"projection_dim": 3, "hidden_size": 6, "bias": False}
TINY_SOURCES = {
    "checkpoint": {"revision": "c0ffee", "sha256": "ab" * 32},
    "base": {"revision": "badbee", "sha256": "cd" * 32},
}


def tiny_base_tensors():
    import torch

    return {
        "model.language_model.embed_tokens.weight": torch.randn(12, 6).to(torch.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(6).to(torch.bfloat16),
        "model.language_model.layers.0.linear_attn.A_log": torch.randn(4, dtype=torch.float32),
        "model.language_model.layers.0.linear_attn.norm.weight": torch.randn(4, dtype=torch.float32),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.randn(8, 6).to(torch.bfloat16),
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": torch.randn(4, 6).to(torch.bfloat16),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(8, 6).to(torch.bfloat16),
        "model.visual.blocks.19.norm1.weight": torch.randn(4).to(torch.bfloat16),
        "model.visual.blocks.20.norm1.weight": torch.randn(4).to(torch.bfloat16),
        "model.visual.blocks.20.attn.qkv.weight": torch.randn(12, 4).to(torch.bfloat16),
        "model.visual.merger.linear_fc1.weight": torch.randn(8, 8).to(torch.bfloat16),
        "mtp.fc.weight": torch.randn(6, 6).to(torch.bfloat16),
    }


def tiny_checkpoint_weights():
    import torch

    prefix = tool.PEFT_LANGUAGE_PREFIX
    weights = {}
    for module, (out, inp) in (
        ("layers.0.linear_attn.in_proj_qkv", (8, 6)),
        ("layers.0.linear_attn.in_proj_z", (4, 6)),
        ("layers.0.mlp.gate_proj", (8, 6)),
    ):
        weights[prefix + module + tool.PEFT_LORA_SUFFIX_A] = torch.randn(4, inp)
        weights[prefix + module + tool.PEFT_LORA_SUFFIX_B] = torch.randn(out, 4) * 0.1
    weights["backbone.visual.blocks.20.norm1.weight"] = torch.randn(4).to(torch.bfloat16)
    weights["backbone.visual.blocks.20.attn.qkv.weight"] = torch.randn(12, 4).to(torch.bfloat16)
    weights["backbone.visual.merger.linear_fc1.weight"] = torch.randn(8, 8).to(torch.bfloat16)
    weights["head.decision.weight"] = torch.randn(3, 6)
    weights["head.candidate.weight"] = torch.randn(3, 6)
    return weights


def materialize_input_dir(directory, tensors=None, config_text='{"text_config": {"hidden_size": 6}}\n'):
    """Write the tiny base safetensors + config into an input directory."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True)
    if tensors is None:
        tensors = tiny_base_tensors()
    save_file({k: v.contiguous() for k, v in tensors.items()}, directory / "base.safetensors")
    (directory / "config.json").write_text(config_text, encoding="utf-8")
    return directory / "base.safetensors", directory / "config.json"


def seeded_tiny_inputs():
    """The tiny fixture with fixed values (seed 7): the export must be a pure
    function of its inputs, so every run sees byte-identical content."""

    import torch

    torch.manual_seed(7)
    return tiny_base_tensors(), tiny_checkpoint_weights()


def run_tiny_export(out_dir, input_dir):
    base, checkpoint = seeded_tiny_inputs()
    base_file, config_file = materialize_input_dir(input_dir, base)
    return tool.run_export(
        checkpoint,
        base_file,
        Path(out_dir),
        sources=TINY_SOURCES,
        merge_config=TINY_MERGE,
        head_config=TINY_HEAD,
        expected=TINY_INVENTORY,
        base_config_path=config_file,
    )


def read_manifest(out_dir):
    return json.loads((Path(out_dir) / tool.MANIFEST_FILE).read_text(encoding="utf-8"))


def tensor_map(manifest):
    return {(t["file"], t["name"]): t for t in manifest["tensors"]}


def test_run_export_tiny_fixture_contract(tmp_path):
    import torch
    from safetensors import safe_open

    out = tmp_path / "export"
    manifest = run_tiny_export(out, tmp_path / "in-a")
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json", "head.safetensors", "model.safetensors", "valen_export.json",
    ]

    # Manifest header fields (the auditable contract).
    assert manifest["format"] == tool.FORMAT
    assert manifest["sources"] == TINY_SOURCES
    assert manifest["merge"]["decision"] == "merged"
    assert manifest["merge"]["scaling"] == 2.0
    assert manifest["merge"]["lora_rank"] == 4 and manifest["merge"]["lora_alpha"] == 8
    assert manifest["merge"]["merged_tensors"] == 3 and manifest["merge"]["lora_keys"] == 6
    assert "fallback" in manifest["merge"]
    assert manifest["downcasts"] == [
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.norm.weight",
    ]
    assert manifest["head"]["projection_dim"] == 3
    assert manifest["head"]["hidden_size"] == 6
    assert manifest["head"]["bias"] is False
    assert manifest["head"]["dtype"] == "float32"
    assert manifest["head"]["tensors"] == ["head.candidate.weight", "head.decision.weight"]
    assert manifest["parity"] is None
    assert "parity_slot_shape" in manifest

    # config.json copied verbatim.
    assert (out / "config.json").read_text(encoding="utf-8") == '{"text_config": {"hidden_size": 6}}\n'

    # Every tensor on disk appears in the manifest with matching metadata.
    tensors = tensor_map(manifest)
    for file_name in (tool.BACKBONE_FILE, tool.HEAD_FILE):
        with safe_open(out / file_name, framework="pt") as f:
            keys = sorted(f.keys())
            for key in keys:
                slice_ = f.get_slice(key)
                entry = tensors[(file_name, key)]
                assert entry["shape"] == list(slice_.get_shape())
                assert entry["dtype"] == {
                    "BF16": "bfloat16", "F32": "float32"
                }[slice_.get_dtype()]
    assert len(manifest["tensors"]) == 12 + 2
    assert [t["name"] for t in manifest["tensors"]] == sorted(
        t["name"] for t in manifest["tensors"]
    )

    # Component attribution covers all five components.
    components = {}
    for entry in manifest["tensors"]:
        components.setdefault(entry["component"], []).append(entry["name"])
    assert set(components) == set(tool.COMPONENTS)
    assert len(components["lora-merged"]) == 3
    assert components["vision-top"] == [
        "model.visual.blocks.20.attn.qkv.weight", "model.visual.blocks.20.norm1.weight",
    ]
    assert components["merger"] == ["model.visual.merger.linear_fc1.weight"]
    assert components["decision-head"] == ["head.candidate.weight", "head.decision.weight"]
    assert "model.visual.blocks.19.norm1.weight" in components["backbone"]

    # File digests recorded match the bytes on disk.
    for file_name, record in manifest["files"].items():
        data = (out / file_name).read_bytes()
        assert record["size"] == len(data)
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert set(manifest["files"]) == {"model.safetensors", "head.safetensors", "config.json"}

    # The transformations are the declared ones.
    base, checkpoint = seeded_tiny_inputs()
    with safe_open(out / tool.BACKBONE_FILE, framework="pt") as f, safe_open(
        tmp_path / "in-a" / "base.safetensors", framework="pt"
    ) as original:
        merged = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        expected = tool.merge_lora_weight(
            original.get_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
            checkpoint[tool.PEFT_LANGUAGE_PREFIX + "layers.0.linear_attn.in_proj_qkv" + tool.PEFT_LORA_SUFFIX_A],
            checkpoint[tool.PEFT_LANGUAGE_PREFIX + "layers.0.linear_attn.in_proj_qkv" + tool.PEFT_LORA_SUFFIX_B],
            2.0,
        )
        assert torch.equal(merged, expected)
        assert not torch.equal(merged, base["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"])
        # Untouched backbone tensors pass through bit-exactly.
        assert torch.equal(
            f.get_tensor("mtp.fc.weight"), original.get_tensor("mtp.fc.weight")
        )
        assert torch.equal(
            f.get_tensor("model.visual.blocks.19.norm1.weight"),
            original.get_tensor("model.visual.blocks.19.norm1.weight"),
        )
        # Downcast semantics.
        assert torch.equal(
            f.get_tensor("model.language_model.layers.0.linear_attn.A_log").float(),
            base["model.language_model.layers.0.linear_attn.A_log"].to(torch.bfloat16).float(),
        )
        # Vision-top and merger are the checkpoint's trained replacements.
        assert torch.equal(
            f.get_tensor("model.visual.blocks.20.attn.qkv.weight"),
            checkpoint["backbone.visual.blocks.20.attn.qkv.weight"],
        )
        assert torch.equal(
            f.get_tensor("model.visual.merger.linear_fc1.weight"),
            checkpoint["backbone.visual.merger.linear_fc1.weight"],
        )
    with safe_open(out / tool.HEAD_FILE, framework="pt") as f:
        assert torch.equal(f.get_tensor("head.decision.weight"), checkpoint["head.decision.weight"])
        assert f.get_tensor("head.decision.weight").dtype == torch.float32

    # The downcast entries carry their origin dtype.
    for name in manifest["downcasts"]:
        assert tensors[(tool.BACKBONE_FILE, name)]["downcast_from"] == "float32"


def test_run_export_is_deterministic_and_path_independent(tmp_path):
    first_out, second_out = tmp_path / "export-1", tmp_path / "export-2"
    run_tiny_export(first_out, tmp_path / "in-a")
    run_tiny_export(second_out, tmp_path / "in-b-different-name")
    for name in ("model.safetensors", "head.safetensors", "config.json", "valen_export.json"):
        assert (first_out / name).read_bytes() == (second_out / name).read_bytes(), name
    # No absolute paths leak into the manifest.
    manifest_text = (first_out / tool.MANIFEST_FILE).read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert '"/' not in manifest_text


def test_run_export_rejects_existing_output(tmp_path):
    import pytest

    out = tmp_path / "export"
    run_tiny_export(out, tmp_path / "in-a")
    with pytest.raises(ValueError, match="already exists"):
        run_tiny_export(out, tmp_path / "in-b")


def test_run_export_rejects_unexpected_fp32_base_tensor(tmp_path):
    import pytest
    import torch
    from safetensors.torch import save_file

    base = tiny_base_tensors()
    # An FP32 tensor outside the A_log / linear_attn.norm.weight families is
    # a silent dtype surprise, not a declared downcast.
    base["model.language_model.layers.0.mlp.gate_proj.weight"] = torch.randn(8, 6)
    directory = tmp_path / "in"
    directory.mkdir()
    save_file({k: v.contiguous() for k, v in base.items()}, directory / "base.safetensors")
    (directory / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fp32"):
        tool.run_export(
            tiny_checkpoint_weights(),
            directory / "base.safetensors",
            tmp_path / "out",
            sources=TINY_SOURCES,
            merge_config=TINY_MERGE,
            head_config=TINY_HEAD,
            expected=dict(TINY_INVENTORY, base_fp32_tensors=3),
            base_config_path=directory / "config.json",
        )


def test_run_export_rejects_lora_shape_mismatch(tmp_path):
    import pytest
    import torch

    weights = tiny_checkpoint_weights()
    weights[tool.PEFT_LANGUAGE_PREFIX + "layers.0.mlp.gate_proj" + tool.PEFT_LORA_SUFFIX_B] = (
        torch.randn(7, 4)
    )
    base_file, config_file = materialize_input_dir(tmp_path / "in")
    with pytest.raises(ValueError, match="shape"):
        tool.run_export(
            weights,
            base_file,
            tmp_path / "out",
            sources=TINY_SOURCES,
            merge_config=TINY_MERGE,
            head_config=TINY_HEAD,
            expected=TINY_INVENTORY,
            base_config_path=config_file,
        )


def test_run_export_rejects_head_config_mismatch(tmp_path):
    import pytest

    base_file, config_file = materialize_input_dir(tmp_path / "in")
    with pytest.raises(ValueError, match="head"):
        tool.run_export(
            tiny_checkpoint_weights(),
            base_file,
            tmp_path / "out",
            sources=TINY_SOURCES,
            merge_config=TINY_MERGE,
            head_config=dict(TINY_HEAD, projection_dim=5),
            expected=TINY_INVENTORY,
            base_config_path=config_file,
        )


# --- validate_export: manifest/artifact consistency and tamper detection ---

def rewrite_manifest(out_dir, mutate):
    manifest = read_manifest(out_dir)
    mutate(manifest)
    (Path(out_dir) / tool.MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_validate_export_accepts_a_fresh_export(tmp_path):
    out = tmp_path / "export"
    run_tiny_export(out, tmp_path / "in-a")
    manifest = tool.validate_export(out)
    assert manifest["format"] == tool.FORMAT


def test_validate_export_streams_file_digests(tmp_path, monkeypatch):
    """File digests must be computed streaming, never via a whole-file read:
    the real model.safetensors is ~4.5 GiB and the contract is "headers and
    digests only" (validate_export docstring, export.md)."""

    out = tmp_path / "export"
    run_tiny_export(out, tmp_path / "in-a")

    def refuse(self):
        raise AssertionError(f"validate_export read {self} whole into memory")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    manifest = tool.validate_export(out)
    assert manifest["format"] == tool.FORMAT


def test_validate_export_detects_tampered_bytes(tmp_path):
    import pytest

    out = tmp_path / "export"
    run_tiny_export(out, tmp_path / "in-a")
    target = out / tool.BACKBONE_FILE
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        tool.validate_export(out)


def test_validate_export_detects_manifest_tensor_removal(tmp_path):
    import pytest

    out = tmp_path / "export"
    run_tiny_export(out, tmp_path / "in-a")

    def drop(manifest):
        # tensors are sorted by (file, name); mtp.fc.weight is the last entry.
        manifest["tensors"] = manifest["tensors"][:-1]

    rewrite_manifest(out, drop)
    with pytest.raises(ValueError, match="mtp.fc.weight"):
        tool.validate_export(out)


def test_validate_export_detects_downcast_list_drift(tmp_path):
    import pytest

    out = tmp_path / "export"
    run_tiny_export(out, tmp_path / "in-a")

    def drop_downcast(manifest):
        manifest["downcasts"] = manifest["downcasts"][:1]

    rewrite_manifest(out, drop_downcast)
    with pytest.raises(ValueError, match="downcast"):
        tool.validate_export(out)


def test_validate_export_detects_head_config_drift(tmp_path):
    import pytest

    out = tmp_path / "export"
    run_tiny_export(out, tmp_path / "in-a")

    def wrong_dim(manifest):
        manifest["head"]["projection_dim"] = 5

    rewrite_manifest(out, wrong_dim)
    with pytest.raises(ValueError, match="head"):
        tool.validate_export(out)


def test_validate_export_detects_extra_files_and_forgotten_entries(tmp_path):
    import pytest

    out = tmp_path / "export"
    run_tiny_export(out, tmp_path / "in-a")
    (out / "surprise.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected file"):
        tool.validate_export(out)
    (out / "surprise.bin").unlink()

    def drop_file_record(manifest):
        del manifest["files"][tool.HEAD_FILE]

    rewrite_manifest(out, drop_file_record)
    with pytest.raises(ValueError, match="head.safetensors"):
        tool.validate_export(out)


# --- real-artifact pin helpers (CPU-only, no multi-GB reads) ---

def test_base_weights_file_requires_single_shard(tmp_path):
    import pytest

    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors"}}), encoding="utf-8"
    )
    (base / "model-00001.safetensors").write_bytes(b"")
    assert tool.base_weights_file(base) == base / "model-00001.safetensors"

    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="single-shard"):
        tool.base_weights_file(base)


def test_checkpoint_revision_from_tree_pin(tmp_path):
    checkpoint = tmp_path / "ckpt"
    trees = checkpoint / ".cache" / "huggingface" / "trees"
    trees.mkdir(parents=True)
    assert tool.checkpoint_revision(checkpoint) is None
    (trees / "81b9c6396ded3a45a92ae8079af85f988aa40501.json").write_text("{}", encoding="utf-8")
    assert tool.checkpoint_revision(checkpoint) == "81b9c6396ded3a45a92ae8079af85f988aa40501"
    (trees / "second.json").write_text("{}", encoding="utf-8")
    assert tool.checkpoint_revision(checkpoint) is None


# --- the full determinism check against the real pinned artifacts ---
# Set VALEN_CKPT and VALEN_BASE (the recipe reference.md artifact exports) to
# run; it exports twice and compares every byte, mirroring the CUA_S1_BASE
# artifact-gating precedent.

def test_real_artifact_export_is_deterministic(tmp_path):
    import pytest

    if not (os.environ.get("VALEN_CKPT") and os.environ.get("VALEN_BASE")):
        pytest.skip("set VALEN_CKPT and VALEN_BASE to run")

    runs = []
    for index in (1, 2):
        out = tmp_path / f"run-{index}"
        manifest = tool.export(
            Path(os.environ["VALEN_CKPT"]), Path(os.environ["VALEN_BASE"]), out
        )
        runs.append((out, manifest))

    (first, manifest_one), (second, manifest_two) = runs
    for name in (tool.BACKBONE_FILE, tool.HEAD_FILE, "config.json", tool.MANIFEST_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    # The pinned inventory, end to end.
    counts = {}
    for entry in manifest_one["tensors"]:
        counts[entry["component"]] = counts.get(entry["component"], 0) + 1
    # 632 base tensors: 186 merged, 48 vision-top and 6 merger replaced,
    # 392 passing through (36 of them downcast from fp32).
    assert counts == {
        "backbone": 392,
        "lora-merged": 186,
        "vision-top": 48,
        "merger": 6,
        "decision-head": 2,
    }
    assert len(manifest_one["downcasts"]) == 36
    assert manifest_one["merge"]["scaling"] == 2.0
    assert manifest_one["head"]["projection_dim"] == 256
    assert manifest_one["head"]["hidden_size"] == 2048
    assert manifest_one["sources"]["checkpoint"]["sha256"] == (
        "836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012"
    )
    assert manifest_one["sources"]["base"]["revision"] == (
        "15852e8c16360a2fea060d615a32b45270f8a8fc"
    )
    assert manifest_one["parity"] is None
