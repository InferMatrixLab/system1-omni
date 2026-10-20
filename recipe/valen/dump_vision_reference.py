#!/usr/bin/env python3
"""Record the pinned reference vision tower on the recipe's image fixture.

Builds the reference stack exactly as the worker does (``ValenExecutor`` ->
``valen.modeling.model.build_model`` + ``load_checkpoint``: pinned base,
trained vision-top blocks 20-23, merger and FP32 LoRA from ``checkpoint.pt``),
compiles the fixture image through the pinned ``Compiler`` with the worker's
own record rules, and records, bit for bit:

- the BF16 tower input the reference forward receives (the processor's
  ``pixel_values`` after the model's own dtype cast),
- the tower output (the merged ``pooler_output``, width 2048) that the
  language model consumes,
- the grid, the provenance (device, versions, revisions) and a tolerance
  slot for the native parity gate (spec 03, N1).

The output is the committed fixture consumed by the env-gated Rust test
``tests/cua_s1/valen_vision.rs``; it is a reproducible command output, never
a hand edit. ``--tolerance`` declares the native gate's absolute elementwise
tolerance (measured first, then declared); without it the fixture records
``null`` and the Rust test refuses to pass until a tolerance is declared.

Usage (from the repository root, on the GPU host):

    PYTHONPATH=src .venv/bin/python recipe/valen/dump_vision_reference.py \
      --valen-source weights/Valen --checkpoint weights/Valen-Preview-0923 \
      --base weights/Qwen3.5-2B --device cuda --dtype bf16

Exit status 0 means the fixture was written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare_reference as reference  # noqa: E402  (the recipe's fixture helpers)

FIXTURE_FORMAT = "valen-vision-reference/1"
DEFAULT_FIXTURE_NAME = "vision-tower-example-image.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valen-source", required=True, help="pinned Valen source checkout")
    parser.add_argument("--checkpoint", required=True, help="Valen-Preview-0923 directory")
    parser.add_argument("--base", required=True, help="Qwen3.5-2B base directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--fixture",
        default=str(HERE / "example-request.json"),
        help="image-case request body (default: the recipe fixture)",
    )
    parser.add_argument(
        "--out",
        default=str(HERE / "fixtures" / DEFAULT_FIXTURE_NAME),
        help=f"output fixture path (default: fixtures/{DEFAULT_FIXTURE_NAME})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="declared absolute elementwise tolerance for the native parity gate",
    )
    parser.add_argument(
        "--stages-out",
        default=None,
        help="optional diagnostic: record per-stage tower activations here",
    )
    return parser.parse_args(argv)


def bf16_bits(tensor) -> str:
    """Hex-encode a contiguous BF16 tensor's raw bits, row-major."""
    import torch

    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16, got {tensor.dtype}")
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy().tobytes().hex()


def main(argv=None):
    args = parse_args(argv)
    if args.dtype != "bf16":
        raise SystemExit("the native vision tower is BF16; --dtype bf16 only")

    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from models.valen.engine import BASE_REVISION, CHECKPOINT_SHA256, ValenExecutor
    from models.valen.preprocess import ValenProcessor

    source = Path(args.valen_source).resolve()
    from frontend.valen import _verify_source_revision

    _verify_source_revision(source)
    reference._verify_clean_checkout(source)
    sys.path.insert(0, str(source))

    import torch
    from valen.data.compiler import Compiler

    checkpoint = Path(args.checkpoint).resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    max_length = int(config["max_length"])
    media_kwargs = config.get("media_kwargs") or {}
    processor = ValenProcessor(Path(args.base), max_length, config.get("media_kwargs"))
    executor = ValenExecutor(checkpoint, Path(args.base), args.device, args.dtype)

    body = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    image_bytes, image_format = reference.decode_image_data_url(body["state"]["image"])
    questions = reference.reference_questions(body, processor.special_tokens)

    try:
        with TemporaryDirectory(prefix="valen-vision-dump-") as directory:
            media_root = Path(directory)
            record = reference.reference_image_record(
                image_bytes, image_format, questions, media_root
            )
            compiler = Compiler(
                processor.processor,
                media_root=media_root,
                max_length=max_length,
                media_kwargs=media_kwargs,
            )
            state = compiler.compile(record)
            branch = state.questions[0].branches[0]
            pixel_values = branch.inputs["pixel_values"]
            grid = branch.inputs["image_grid_thw"]
            device = torch.device(args.device)
            is_cuda = device.type == "cuda"

            backbone = executor.model.backbone
            # get_image_features casts to the visual tower's dtype itself;
            # replicate that cast so the recorded input is the exact tensor
            # the reference tower consumes.
            tower_input = pixel_values.to(device).type(backbone.visual.dtype)
            stages = {}
            handles = []
            if args.stages_out:

                def record(name):
                    def hook(_module, _inputs, result):
                        tensor = result[0] if isinstance(result, tuple) else result
                        stages.setdefault(name, bf16_bits(tensor.detach()))

                    return hook

                visual = backbone.visual
                handles.append(visual.patch_embed.register_forward_hook(record("patch_embed")))
                for index, block in enumerate(visual.blocks):
                    handles.append(
                        block.register_forward_hook(record(f"blocks.{index}"))
                    )
                handles.append(visual.merger.register_forward_hook(record("merger")))
            with torch.no_grad():
                vision_outputs = backbone.get_image_features(
                    tower_input.clone(), grid.to(device)
                )
            for handle in handles:
                handle.remove()
            output = vision_outputs.pooler_output[0]
    finally:
        executor.close()

    grid_list = grid.tolist()[0]
    patches, values = tower_input.shape
    tokens, width = output.shape
    if (
        grid_list[1] * grid_list[2] != patches
        or values != 1536
        or tokens != patches // 4
        or width != backbone.visual.config.out_hidden_size
    ):
        raise SystemExit(
            f"unexpected tower geometry: grid={grid_list}, input={tuple(tower_input.shape)}, "
            f"output={tuple(output.shape)}"
        )

    fixture = {
        "format": FIXTURE_FORMAT,
        "fixture": str(Path(args.fixture).resolve()),
        "fixture_image_sha256": reference.hashlib.sha256(image_bytes).hexdigest(),
        "image_grid_thw": grid_list,
        "tower_input_shape": [patches, values],
        "tower_output_shape": [tokens, width],
        "pixel_values_bf16_hex": bf16_bits(tower_input),
        "tower_output_bf16_hex": bf16_bits(output),
        "tolerance": args.tolerance,
        "tolerance_semantics": (
            "max absolute elementwise difference, in FP32, between this "
            "reference tower output and the native tower output on the same "
            "device and dtype (BF16); measure first, then declare"
        ),
        "reference": {
            "device": args.device,
            "device_name": torch.cuda.get_device_name(0) if is_cuda else "cpu",
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))
            if is_cuda
            else None,
            "dtype": args.dtype,
            "torch": torch.__version__,
            "valen_source_revision": reference.subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "base_revision": BASE_REVISION,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"recorded {out}")
    print(f"  grid: {grid_list}, patches: {patches}, tokens: {tokens}, width: {width}")
    print(f"  tolerance: {args.tolerance!r} (declare after measuring the native delta)")
    if args.stages_out:
        Path(args.stages_out).write_text(
            json.dumps(
                {
                    "format": "valen-vision-reference-stages/1",
                    "reference": fixture["reference"],
                    "stages": stages,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"recorded stages: {sorted(stages)} -> {args.stages_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
