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


# --- streaming safetensors writer ---
# The pinned base is a single ~4.5 GiB file and host memory is a fraction of
# that, so the export never holds the artifact set in memory: every tensor is
# materialized once, in sorted name order, straight into the output file.
# The byte format is the safetensors v1 spec (8-byte LE header length, JSON
# header padded to an 8-byte boundary, then contiguous tensor data), written
# with fixed ordering so two runs are byte-identical.

_DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float32: "F32"}
_DTYPE_SIZES = {torch.bfloat16: 2, torch.float32: 4}
_READER_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}
_MANIFEST_DTYPES = {"BF16": "bfloat16", "F32": "float32"}
_FP32_TENSOR = re.compile(
    r"^model\.language_model\.layers\.\d+\.linear_attn\.(A_log|norm\.weight)$"
)


def _tensor_bytes(tensor):
    """Raw little-endian bytes of one contiguous tensor (BF16-safe).

    Flattened to 1-D so the byte length is the full payload: ``len()`` of a
    multi-dimensional memoryview only reports its first dimension.
    """

    view = tensor.contiguous().view(torch.uint8).reshape(-1)
    return memoryview(view.numpy())


def write_safetensors(path, entries):
    """Stream ``(name, dtype, shape, materialize)`` entries to ``path``.

    ``materialize`` is called once per entry, in order, so callers can pull
    each tensor from an mmap'd source and release it before the next.
    Returns ``{"size", "sha256"}`` of the bytes written.
    """

    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, dtype, shape, _ in entries:
        nbytes = _DTYPE_SIZES[dtype]
        for dim in shape:
            nbytes *= dim
        header[name] = {
            "dtype": _DTYPE_NAMES[dtype],
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-(8 + len(header_bytes)) % 8)
    digest = hashlib.sha256()
    with open(path, "wb") as handle:

        def write(chunk):
            digest.update(chunk)
            handle.write(chunk)

        write(struct.pack("<Q", len(header_bytes)))
        write(header_bytes)
        for name, dtype, shape, materialize in entries:
            data = _tensor_bytes(materialize())
            expected = _DTYPE_SIZES[dtype]
            for dim in shape:
                expected *= dim
            if len(data) != expected:
                raise ValueError(
                    f"tensor {name!r} serialized as {len(data)} bytes, expected {expected}"
                )
            write(data)
    return {"size": 8 + len(header_bytes) + offset, "sha256": digest.hexdigest()}


def run_export(
    checkpoint_weights,
    base_file,
    out_dir,
    *,
    sources,
    merge_config,
    head_config,
    expected,
    base_config_path=None,
):
    """Write the exported artifact set and its manifest, streaming.

    ``checkpoint_weights`` is the unpickled ``payload["weights"]`` dict (mmap
    views are fine); ``base_file`` is the base safetensors path, read lazily
    through ``safe_open``. ``sources`` is recorded verbatim in the manifest;
    ``merge_config``/``head_config``/``expected`` are validated against the
    tensors before anything is written. Returns the manifest dict.
    """

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"output directory {out_dir} already exists and is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Single-threaded CPU math keeps the FP32 merge GEMM bit-reproducible
    # regardless of the ambient thread/env configuration.
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    classification = classify_checkpoint_weights(checkpoint_weights)
    validate_inventory(classification, expected)
    scaling = lora_scaling(merge_config["rank"], merge_config["alpha"])

    from safetensors import safe_open

    with safe_open(base_file, framework="pt", device="cpu") as base:
        base_names = sorted(base.keys())
        if len(base_names) != expected["base_tensors"]:
            raise ValueError(
                f"base holds {len(base_names)} tensors, expected {expected['base_tensors']}"
            )
        base_dtypes = {}
        base_shapes = {}
        for name in base_names:
            slice_ = base.get_slice(name)
            base_dtypes[name] = _READER_DTYPES[slice_.get_dtype()]
            base_shapes[name] = list(slice_.get_shape())

        # Every FP32 base tensor must be a declared downcast target; anything
        # else is a silent dtype surprise (spec 02, user story 7).
        downcasts = [name for name in base_names if base_dtypes[name] == torch.float32]
        for name in downcasts:
            if not _FP32_TENSOR.match(name):
                raise ValueError(
                    f"unexpected fp32 base tensor {name!r}: only linear_attn "
                    "A_log / norm.weight may be stored in fp32"
                )
        if len(downcasts) != expected["base_fp32_tensors"]:
            raise ValueError(
                f"base holds {len(downcasts)} fp32 tensors, "
                f"expected {expected['base_fp32_tensors']}"
            )

        # Validate every checkpoint-driven replacement against the base.
        vision_by_export = dict(classification.vision_top)
        merger_by_export = dict(classification.merger)
        for export_name, checkpoint_name in classification.vision_top + classification.merger:
            if export_name not in base_dtypes:
                raise ValueError(
                    f"checkpoint tensor {checkpoint_name!r} has no base counterpart"
                )
            tensor = checkpoint_weights[checkpoint_name]
            if tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"checkpoint tensor {checkpoint_name!r} is {tensor.dtype}, expected bf16"
                )
            if list(tensor.shape) != base_shapes[export_name]:
                raise ValueError(
                    f"checkpoint tensor {checkpoint_name!r} shape {list(tensor.shape)} "
                    f"!= base shape {base_shapes[export_name]}"
                )
        for base_name, (a_key, b_key) in classification.lora.items():
            if base_name not in base_dtypes:
                raise ValueError(f"LoRA target {base_name!r} has no base tensor")
            lora_a = checkpoint_weights[a_key]
            lora_b = checkpoint_weights[b_key]
            base_shape = base_shapes[base_name]
            if lora_a.dtype != torch.float32 or lora_b.dtype != torch.float32:
                raise ValueError(
                    f"LoRA pair for {base_name!r} must be fp32, "
                    f"got {lora_a.dtype}/{lora_b.dtype}"
                )
            if lora_a.shape[0] != merge_config["rank"]:
                raise ValueError(
                    f"LoRA rank for {base_name!r} is {lora_a.shape[0]}, "
                    f"expected {merge_config['rank']}"
                )
            if list(lora_a.shape) != [merge_config["rank"], base_shape[1]] or list(
                lora_b.shape
            ) != [base_shape[0], merge_config["rank"]]:
                raise ValueError(
                    f"LoRA pair shape mismatch for {base_name!r}: "
                    f"A {list(lora_a.shape)}, B {list(lora_b.shape)}, base {base_shape}"
                )

        head_entries = []
        for export_name, checkpoint_name in sorted(classification.head.items()):
            tensor = checkpoint_weights[checkpoint_name]
            if tensor.dtype != torch.float32:
                raise ValueError(f"head tensor {checkpoint_name!r} must be fp32")
            if list(tensor.shape) != [
                head_config["projection_dim"],
                head_config["hidden_size"],
            ]:
                raise ValueError(
                    f"head tensor {checkpoint_name!r} shape {list(tensor.shape)} "
                    f"contradicts the head config (projection_dim="
                    f"{head_config['projection_dim']}, hidden_size="
                    f"{head_config['hidden_size']})"
                )
            head_entries.append(
                (export_name, torch.float32, list(tensor.shape), (lambda t=tensor: t))
            )

        # Plan and stream the backbone file entry by entry.
        tensors = []

        def plan(name):
            if name in classification.lora:
                a_key, b_key = classification.lora[name]
                if base_dtypes[name] != torch.bfloat16:
                    raise ValueError(f"LoRA target {name!r} must be a bf16 base tensor")
                tensors.append(
                    {
                        "name": name,
                        "shape": base_shapes[name],
                        "dtype": "bfloat16",
                        "component": COMPONENT_LORA_MERGED,
                        "source": "base+checkpoint",
                    }
                )
                return (
                    name,
                    torch.bfloat16,
                    base_shapes[name],
                    (
                        lambda n=name, a=a_key, b=b_key: merge_lora_weight(
                            base.get_tensor(n),
                            checkpoint_weights[a],
                            checkpoint_weights[b],
                            scaling,
                        )
                    ),
                )
            checkpoint_name = vision_by_export.get(name) or merger_by_export.get(name)
            if checkpoint_name is not None:
                tensors.append(
                    {
                        "name": name,
                        "shape": base_shapes[name],
                        "dtype": "bfloat16",
                        "component": (
                            COMPONENT_VISION_TOP
                            if name in vision_by_export
                            else COMPONENT_MERGER
                        ),
                        "source": "checkpoint",
                    }
                )
                return (
                    name,
                    torch.bfloat16,
                    base_shapes[name],
                    (lambda c=checkpoint_name: checkpoint_weights[c]),
                )
            if base_dtypes[name] == torch.float32:
                tensors.append(
                    {
                        "name": name,
                        "shape": base_shapes[name],
                        "dtype": "bfloat16",
                        "component": COMPONENT_BACKBONE,
                        "source": "base",
                        "downcast_from": "float32",
                    }
                )
                return (
                    name,
                    torch.bfloat16,
                    base_shapes[name],
                    (lambda n=name: downcast_bf16(base.get_tensor(n))),
                )
            if base_dtypes[name] != torch.bfloat16:
                raise ValueError(
                    f"base tensor {name!r} is {base_dtypes[name]}, expected bf16"
                )
            tensors.append(
                {
                    "name": name,
                    "shape": base_shapes[name],
                    "dtype": "bfloat16",
                    "component": COMPONENT_BACKBONE,
                    "source": "base",
                }
            )
            return (
                name,
                torch.bfloat16,
                base_shapes[name],
                (lambda n=name: base.get_tensor(n)),
            )

        backbone_entries = [plan(name) for name in base_names]
        for record in tensors:
            record["file"] = (
                HEAD_FILE if record["component"] == COMPONENT_HEAD else BACKBONE_FILE
            )
        for entry in head_entries:
            tensors.append(
                {
                    "name": entry[0],
                    "shape": entry[2],
                    "dtype": "float32",
                    "component": COMPONENT_HEAD,
                    "source": "checkpoint",
                    "file": HEAD_FILE,
                }
            )

        # Stream both files while the base handle (the lambdas' data source)
        # is still open.
        files = {BACKBONE_FILE: write_safetensors(out_dir / BACKBONE_FILE, backbone_entries)}
        files[HEAD_FILE] = write_safetensors(out_dir / HEAD_FILE, head_entries)

    if base_config_path is not None:
        config_bytes = Path(base_config_path).read_bytes()
        (out_dir / "config.json").write_bytes(config_bytes)
        files["config.json"] = {
            "size": len(config_bytes),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        }

    tensors.sort(key=lambda record: (record["file"], record["name"]))
    manifest = {
        "format": FORMAT,
        "sources": sources,
        "merge": {
            "decision": "merged",
            "formula": f"bf16(fp32(W) + {scaling:g} * (B @ A))",
            "scaling": scaling,
            "lora_rank": merge_config["rank"],
            "lora_alpha": merge_config["alpha"],
            "merged_tensors": len(classification.lora),
            "lora_keys": 2 * len(classification.lora),
            "fallback": MERGE_FALLBACK,
        },
        "downcasts": downcasts,
        "head": {
            "file": HEAD_FILE,
            "dtype": "float32",
            "bias": bool(head_config.get("bias", False)),
            "projection_dim": head_config["projection_dim"],
            "hidden_size": head_config["hidden_size"],
            "scale_formula": "sqrt(projection_dim)",
            "logit_formula": "(W_cand @ h_cand) . (W_dec @ h_dec) / sqrt(projection_dim), fp32",
            "tensors": [entry[0] for entry in head_entries],
        },
        "tensors": tensors,
        "files": dict(sorted(files.items())),
        "consumption": {
            "backbone": (
                "Qwen3_5ForConditionalGeneration.from_pretrained(out, dtype=bf16, "
                "attn_implementation='eager', local_files_only=True).model"
            ),
            "head": (
                f"{HEAD_FILE}: DecisionHead(hidden_size={head_config['hidden_size']}, "
                f"projection_dim={head_config['projection_dim']}), fp32, no bias"
            ),
        },
        "parity": None,
        "parity_slot_shape": PARITY_SLOT_SHAPE,
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_export(out_dir):
    """Cross-check a finished export against its manifest.

    Reads only headers and digests, not tensor payloads, so it also runs
    cheaply against the multi-GiB real artifact set. Returns the manifest
    when every recorded claim matches the bytes on disk.
    """

    from safetensors import safe_open

    out_dir = Path(out_dir)
    present = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    allowed = sorted([BACKBONE_FILE, HEAD_FILE, MANIFEST_FILE, "config.json"])
    if present != allowed:
        extra = sorted(set(present) - set(allowed))
        missing = sorted(set(allowed) - set(present))
        raise ValueError(
            f"export directory must hold exactly {allowed}: "
            f"unexpected files {extra}, missing {missing}"
        )
    manifest = json.loads((out_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"manifest format {manifest.get('format')!r} != {FORMAT!r}")
    artifact_files = {BACKBONE_FILE, HEAD_FILE, "config.json"}
    if set(manifest["files"]) != artifact_files:
        raise ValueError(
            f"manifest files record {sorted(manifest['files'])} != "
            f"{sorted(artifact_files)}"
        )

    # Recorded file digests must match the bytes on disk.
    for name, record in sorted(manifest["files"].items()):
        data = (out_dir / name).read_bytes()
        if record["size"] != len(data):
            raise ValueError(f"{name} is {len(data)} bytes, manifest says {record['size']}")
        digest = hashlib.sha256(data).hexdigest()
        if record["sha256"] != digest:
            raise ValueError(f"{name} sha256 {digest} != manifest {record['sha256']}")

    # Every safetensors tensor is listed with matching metadata, and vice versa.
    listed = {}
    for entry in manifest["tensors"]:
        for field in ("name", "shape", "dtype", "component", "source", "file"):
            if field not in entry:
                raise ValueError(f"manifest tensor entry missing {field!r}: {entry}")
        if entry["component"] not in COMPONENTS:
            raise ValueError(f"unknown component {entry['component']!r} for {entry['name']}")
        if entry["file"] not in (BACKBONE_FILE, HEAD_FILE):
            raise ValueError(
                f"tensor {entry['name']!r} points at unknown file {entry['file']!r}"
            )
        if (entry["file"], entry["name"]) in listed:
            raise ValueError(f"manifest lists {entry['name']!r} twice")
        listed[(entry["file"], entry["name"])] = entry
    on_disk = set()
    for file_name in (BACKBONE_FILE, HEAD_FILE):
        with safe_open(out_dir / file_name, framework="pt") as handle:
            on_disk.update((file_name, key) for key in handle.keys())
    orphans = sorted(set(listed) - on_disk)
    if orphans:
        raise ValueError(f"manifest lists tensors absent from the files: {orphans[:4]}")
    unlisted = sorted(on_disk - set(listed))
    if unlisted:
        raise ValueError(
            f"{unlisted[0][0]} holds unlisted tensor {unlisted[0][1]!r}"
            f" ({len(unlisted)} total)"
        )
    for file_name, key in sorted(on_disk):
        with safe_open(out_dir / file_name, framework="pt") as handle:
            slice_ = handle.get_slice(key)
            entry = listed[(file_name, key)]
            if entry["shape"] != list(slice_.get_shape()):
                raise ValueError(
                    f"{key!r} shape {entry['shape']} != file {list(slice_.get_shape())}"
                )
            if entry["dtype"] != _MANIFEST_DTYPES[slice_.get_dtype()]:
                raise ValueError(
                    f"{key!r} dtype {entry['dtype']} != file "
                    f"{_MANIFEST_DTYPES[slice_.get_dtype()]}"
                )

    # Downcast bookkeeping: the list and the per-tensor flags must agree.
    downcasts = set(manifest["downcasts"])
    flagged = {
        entry["name"]
        for entry in manifest["tensors"]
        if entry.get("downcast_from") is not None
    }
    if downcasts != flagged:
        raise ValueError(
            "downcast list and tensor flags disagree: "
            f"list-only {sorted(downcasts - flagged)}, flag-only {sorted(flagged - downcasts)}"
        )
    for name in sorted(downcasts):
        entry = listed.get((BACKBONE_FILE, name))
        if (
            entry is None
            or entry.get("downcast_from") != "float32"
            or entry["dtype"] != "bfloat16"
        ):
            raise ValueError(f"downcast entry {name!r} is not a bf16 tensor from fp32")

    # Head config must match the exported head tensors.
    head = manifest["head"]
    projection = (head["projection_dim"], head["hidden_size"])
    for name in head["tensors"]:
        entry = listed.get((HEAD_FILE, name))
        if entry is None:
            raise ValueError(f"head config lists unknown tensor {name!r}")
        if tuple(entry["shape"]) != projection:
            raise ValueError(
                f"head tensor {name!r} shape {entry['shape']} contradicts the head config "
                f"(projection_dim, hidden_size)={projection}"
            )
        if entry["dtype"] != "float32" or entry["component"] != COMPONENT_HEAD:
            raise ValueError(f"head tensor {name!r} must be an fp32 decision-head tensor")

    # The merge decision must match the merged-tensor inventory.
    merged = [e for e in manifest["tensors"] if e["component"] == COMPONENT_LORA_MERGED]
    if manifest["merge"]["merged_tensors"] != len(merged):
        raise ValueError(
            f"merge block records {manifest['merge']['merged_tensors']} merged tensors, "
            f"manifest lists {len(merged)}"
        )
    if manifest["parity"] is not None and not isinstance(manifest["parity"], dict):
        raise ValueError("parity slot must be null or the parity-gate record")
    return manifest


# --- the real-artifact path: pinned checkpoint + pinned base -> export ---

CHECKPOINT_FILE = "checkpoint.pt"
TRAINING_METADATA_FILE = "training_metadata.json"
BASE_MANIFEST_FILE = "valen_manifest.json"
BASE_INDEX_FILE = "model.safetensors.index.json"
PINNED_STAGE = "vision_top"


def sha256_file(path):
    """Stream a file through SHA-256 without loading it."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_revision(checkpoint_dir):
    """The HF revision the Preview directory was pinned to, if recorded.

    ``hf download`` leaves a per-revision tree file under .cache; its file
    name is the revision. Returns None when the directory carries no pin
    (the sha256 remains the authoritative identity either way).
    """

    trees = Path(checkpoint_dir) / ".cache" / "huggingface" / "trees"
    if not trees.is_dir():
        return None
    revisions = sorted(p.stem for p in trees.glob("*.json"))
    if len(revisions) != 1:
        return None
    return revisions[0]


def base_weights_file(base_dir):
    """The single safetensors shard the pinned base stores its weights in."""

    index_path = Path(base_dir) / BASE_INDEX_FILE
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 1:
        raise ValueError(
            f"this recipe exports single-shard bases; {index_path} lists {shards}"
        )
    path = Path(base_dir) / shards[0]
    if not path.is_file():
        raise ValueError(f"base weights file {path} does not exist")
    return path


def export(checkpoint_dir, base_dir, out_dir):
    """Export the pinned checkpoint + base into ``out_dir`` (must not exist).

    Enforces the artifact-internal pins before writing anything: the
    checkpoint digest against training_metadata.json, the base weights
    digest against both valen_manifest.json and the checkpoint's own
    base_manifest, and the revision agreement between them. Returns the
    validated manifest.
    """

    checkpoint_dir = Path(checkpoint_dir)
    base_dir = Path(base_dir)

    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE
    checkpoint_sha = sha256_file(checkpoint_path)
    metadata_path = checkpoint_dir / TRAINING_METADATA_FILE
    if metadata_path.is_file():
        declared = json.loads(metadata_path.read_text(encoding="utf-8"))["checkpoint_sha256"]
        if declared != checkpoint_sha:
            raise ValueError(
                f"checkpoint sha256 {checkpoint_sha} != training_metadata {declared}"
            )

    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = payload["config"]
    if config.get("stage") != PINNED_STAGE:
        raise ValueError(
            f"checkpoint stage {config.get('stage')!r} != {PINNED_STAGE!r}: this "
            "recipe only exports the vision_top Preview layout"
        )

    weights_file = base_weights_file(base_dir)
    shard_sums = {weights_file.name: sha256_file(weights_file)}
    valen_manifest = json.loads(
        (base_dir / BASE_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    base_manifest = payload.get("base_manifest") or {}
    for label, declared in (
        ("valen_manifest.json", valen_manifest.get("weight_sha256")),
        ("checkpoint base_manifest", base_manifest.get("weight_sha256")),
    ):
        # Both pins record a per-shard sha mapping; they must cover exactly
        # the shards on disk and match every digest.
        if declared is not None and dict(declared) != shard_sums:
            raise ValueError(
                f"base weights sha256 {shard_sums} != {label} {dict(declared)}"
            )
    base_revision = valen_manifest.get("revision")
    if base_manifest.get("revision") not in (None, base_revision):
        raise ValueError(
            f"base revision {base_revision!r} != checkpoint base_manifest "
            f"{base_manifest['revision']!r}"
        )

    base_config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    hidden_size = base_config["text_config"]["hidden_size"]
    import safetensors

    environment = {
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
    }
    try:
        import peft

        environment["peft"] = peft.__version__
    except ImportError:  # pragma: no cover - the pinned venv always has peft
        pass

    sources = {
        "checkpoint": {
            "sha256": checkpoint_sha,
            "revision": checkpoint_revision(checkpoint_dir),
            "stage": config["stage"],
        },
        "base": {
            "weights_sha256": shard_sums,
            "revision": base_revision,
            "repo": valen_manifest.get("repo"),
        },
        "environment": environment,
    }
    manifest = run_export(
        payload["weights"],
        weights_file,
        out_dir,
        sources=sources,
        merge_config={"rank": config["lora_rank"], "alpha": config["lora_alpha"]},
        head_config={
            "projection_dim": config["projection_dim"],
            "hidden_size": hidden_size,
            "bias": False,
        },
        expected=PINNED_INVENTORY,
        base_config_path=base_dir / "config.json",
    )
    return validate_export(Path(out_dir))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="Valen-Preview-0923 directory")
    parser.add_argument("--base", required=True, help="Qwen3.5-2B base directory")
    parser.add_argument("--out", required=True, type=Path, help="export output directory")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = export(args.checkpoint, args.base, args.out)
    counts = {}
    for entry in manifest["tensors"]:
        counts[entry["component"]] = counts.get(entry["component"], 0) + 1
    print(f"exported {args.out}")
    for component in sorted(counts):
        print(f"  {component}: {counts[component]} tensors")
    print(f"  downcasts: {len(manifest['downcasts'])} fp32 -> bf16")
    print(f"  manifest: {args.out / MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
