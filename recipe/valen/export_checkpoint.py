#!/usr/bin/env python3
"""Export the pinned Valen checkpoint plus its base to a safetensors set.

One deterministic command turns the training-framework pickle
(``checkpoint.pt`` + the pinned Qwen3.5-2B base) into the artifact set a
native worker can load, plus a JSON manifest recording every transformation
performed (spec 02):

- ``model.safetensors`` — the backbone in the base's own key layout
  (``model.language_model.*`` / ``model.visual.*``, BF16 only): the language
  LoRA merged at BF16 (``bf16(fp32(W) + 2*(B@A))``, bit-equal to peft
  0.18.1's ``Linear.merge``), the trained vision-top blocks 20-23 and the
  merger substituted as plain BF16 replacements, and the base's 36 FP32
  tensors (``linear_attn.A_log`` / ``linear_attn.norm.weight``) downcast to
  BF16 — the same cast transformers' own BF16 runtime performs.
- ``head.safetensors`` — the decision head's two FP32 projection matrices
  (``head.decision.weight``, ``head.candidate.weight``), no bias.
- ``config.json`` — the base config copied verbatim so the reference builder
  (``Qwen3_5ForConditionalGeneration.from_pretrained(out, dtype=bf16,
  attn_implementation="eager").model``) loads the export unchanged.
- ``valen_export.json`` — the manifest: per-tensor name/shape/dtype and
  component attribution (backbone / lora-merged / vision-top / merger /
  decision-head), source revisions and SHA-256s, the merge decision and
  scaling, the downcast list, the head configuration, and a ``parity`` slot
  left ``null`` for the E2 parity gate to fill.

Determinism: no timestamps, no absolute paths, tensors and JSON written in
sorted key order, single-threaded CPU math — two runs on the same inputs
produce byte-identical artifacts and identical manifests. The export runs
CPU-only and streams tensor-by-tensor (mmap reads, an incremental
safetensors writer), so host memory stays well below the artifact sizes.

Usage:

    PYTHONPATH=src .venv/bin/python recipe/valen/export_checkpoint.py \
      --checkpoint weights/Valen-Preview-0923 --base weights/Qwen3.5-2B \
      --out weights/valen-export
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import NamedTuple

import torch

FORMAT = "valen-checkpoint-export/1"
BACKBONE_FILE = "model.safetensors"
HEAD_FILE = "head.safetensors"
MANIFEST_FILE = "valen_export.json"

# Checkpoint key layout, verified in wayfinder/research/001: PEFT wraps
# backbone.language_model, vision-top/merger tensors ride under
# backbone.visual.*, and the decision head is head.*.
PEFT_LANGUAGE_PREFIX = "backbone.language_model.base_model.model."
PEFT_LORA_SUFFIX_A = ".lora_A.default.weight"
PEFT_LORA_SUFFIX_B = ".lora_B.default.weight"
BACKBONE_PREFIX = "model."
VISUAL_PREFIX = "backbone.visual."
HEAD_PREFIX = "head."

COMPONENT_BACKBONE = "backbone"
COMPONENT_LORA_MERGED = "lora-merged"
COMPONENT_VISION_TOP = "vision-top"
COMPONENT_MERGER = "merger"
COMPONENT_HEAD = "decision-head"
COMPONENTS = (
    COMPONENT_BACKBONE,
    COMPONENT_LORA_MERGED,
    COMPONENT_VISION_TOP,
    COMPONENT_MERGER,
    COMPONENT_HEAD,
)

# The pinned Preview inventory (research 001, Part 1a): 428 trainable
# tensors decomposing as 372 LoRA keys over the 186 targeted language
# linears, 48 vision-top block tensors, 6 merger tensors and 2 head
# projections; the base file holds 632 tensors of which 36 are FP32.
PINNED_INVENTORY = {
    "checkpoint_tensors": 428,
    "lora_keys": 372,
    "vision_top_keys": 48,
    "merger_keys": 6,
    "head_keys": 2,
    "base_tensors": 632,
    "base_fp32_tensors": 36,
}

# Documented fallback if the E2 parity gate fails (spec 02, user story 9).
MERGE_FALLBACK = (
    "export the FP32 LoRA separately and serve unmerged "
    "(spec 02 user story 9; triggered only by a failed parity gate)"
)

# The shape of the parity slot E2 fills into the manifest (left null here).
PARITY_SLOT_SHAPE = {
    "status": "passed | failed",
    "device": "torch device string",
    "cases": "text and image choice fixtures compared",
    "exact_answer_agreement": True,
    "max_probability_delta": "<= 1e-6",
    "merge_drift": {
        "per_layer_max_output_delta": "max |merged - unmerged| per adapted layer",
        "aggregate": "worst-case relative drift across layers",
    },
    "evidence": "pointer to the recorded parity run",
}


def lora_scaling(rank, alpha):
    """PEFT's scaling constant alpha/r (64/32 = 2.0 for the pinned config)."""

    return alpha / rank


def merge_lora_weight(base_weight, lora_a, lora_b, scaling):
    """Merge one LoRA pair into a base weight at the reference's cast
    boundaries.

    The delta ``scaling * (B @ A)`` is computed in FP32, added to the
    FP32-upcast base weight, and the sum is rounded to the base dtype
    (BF16) exactly once — bit-equal to peft 0.18.1's ``Linear.merge``
    (``get_delta_weight`` computes the transpose-free delta in FP32 and the
    in-place add rounds once per weight element). Zero-B adapters merge
    back to the base tensor bit-exactly.
    """

    base_fp32 = base_weight.to(torch.float32)
    delta = lora_b.to(torch.float32) @ lora_a.to(torch.float32)
    return (base_fp32 + scaling * delta).to(base_weight.dtype)


def downcast_bf16(tensor):
    """Downcast a stored-FP32 base tensor to the BF16 the reference runtime
    itself keeps (transformers has no keep_in_fp32_modules for qwen3_5 and
    upcasts with ``.float()`` at use)."""

    return tensor.to(torch.bfloat16)


class Classification(NamedTuple):
    """The checkpoint's trainable tensors decomposed by provenance component.

    ``lora`` maps the base tensor name each adapter pair targets to its
    ``(lora_A, lora_B)`` checkpoint keys; ``vision_top``/``merger`` pair each
    export key (the base's own ``model.visual.*`` layout) with its checkpoint
    source; ``head`` maps the export names identically. Everything is kept in
    sorted key order so downstream output is deterministic.
    """

    lora: dict  # base tensor name -> (a_key, b_key)
    vision_top: list  # sorted (export_key, checkpoint_key)
    merger: list  # sorted (export_key, checkpoint_key)
    head: dict  # export key -> checkpoint key


_VISION_BLOCK = re.compile(r"^backbone\.visual\.blocks\.(\d+)\.(.+)$")
_HEAD_TENSORS = ("head.decision.weight", "head.candidate.weight")


def classify_checkpoint_weights(keys):
    """Map the checkpoint's weight keys onto export names by component.

    Rejects anything outside the pinned ``vision_top`` layout: frozen
    backbone tensors, non-PEFT language keys, head biases and unpaired or
    non-top-4 vision adapters are all errors, because the export recipe is
    only valid for the component inventory research 001 verified.
    """

    lora_a = {}
    lora_b = {}
    vision_top = []
    merger = []
    head = {}
    for key in sorted(keys):
        if key.startswith(PEFT_LANGUAGE_PREFIX) and key.endswith(PEFT_LORA_SUFFIX_A):
            base_name = (
                "model.language_model."
                + key[len(PEFT_LANGUAGE_PREFIX): -len(PEFT_LORA_SUFFIX_A)]
                + ".weight"
            )
            lora_a[base_name] = key
        elif key.startswith(PEFT_LANGUAGE_PREFIX) and key.endswith(PEFT_LORA_SUFFIX_B):
            base_name = (
                "model.language_model."
                + key[len(PEFT_LANGUAGE_PREFIX): -len(PEFT_LORA_SUFFIX_B)]
                + ".weight"
            )
            lora_b[base_name] = key
        elif key.startswith("backbone.visual.merger."):
            merger.append(("model.visual.merger." + key[len("backbone.visual.merger."):], key))
        elif _VISION_BLOCK.match(key):
            block, tail = _VISION_BLOCK.match(key).groups()
            if int(block) < 20:
                raise ValueError(
                    f"unknown checkpoint weight {key!r}: the vision_top stage "
                    "trains only the last four blocks (20-23)"
                )
            vision_top.append((f"model.visual.blocks.{block}.{tail}", key))
        elif key in _HEAD_TENSORS:
            head[key] = key
        else:
            raise ValueError(f"unknown checkpoint weight {key!r}")
    if set(lora_a) != set(lora_b):
        missing = sorted(set(lora_a) ^ set(lora_b))
        raise ValueError(f"unpaired LoRA adapters for {missing}")
    return Classification(
        lora={name: (lora_a[name], lora_b[name]) for name in sorted(lora_a)},
        vision_top=sorted(vision_top),
        merger=sorted(merger),
        head={name: head[name] for name in sorted(head)},
    )


def validate_inventory(classification, expected):
    """Check the classified component counts against the expected inventory."""

    counts = {
        "checkpoint_tensors": 2 * len(classification.lora)
        + len(classification.vision_top)
        + len(classification.merger)
        + len(classification.head),
        "lora_keys": 2 * len(classification.lora),
        "vision_top_keys": len(classification.vision_top),
        "merger_keys": len(classification.merger),
        "head_keys": len(classification.head),
    }
    for name, actual in counts.items():
        wanted = expected[name]
        if actual != wanted:
            raise ValueError(
                f"checkpoint inventory mismatch: {name} is {actual}, expected {wanted}"
            )
