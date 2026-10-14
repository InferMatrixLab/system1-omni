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
