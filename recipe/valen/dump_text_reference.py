#!/usr/bin/env python3
"""Record the pinned reference text path on a compiled text-choice branch.

Builds the reference stack exactly as the worker does (``ValenExecutor`` ->
``valen.modeling.model.build_model`` + ``load_checkpoint``: pinned base,
FP32 language LoRA from ``checkpoint.pt``, eager attention, BF16), compiles
a fixed text state + choice question through the pinned ``Compiler``, and
records, bit for bit:

- the branch token ids the reference language model consumes,
- the fused projection inputs (the norm outputs the GEMMs see) and the tail
  of every adapted projection output for the first GDN layer (0) and the
  first full-attention layer (3),
- the residual stream's last row after every layer (drift attribution), and
- the final-norm last-position hidden state — the native ``forward`` output.

The output is the committed fixture consumed by the env-gated Rust tests
(``tests/valen/text_reference.rs``); it is a reproducible command output,
never a hand edit. ``--tolerance`` (end-to-end final output) and
``--projection-tolerance`` (per-projection outputs) declare the native
gates' absolute elementwise tolerances (measured first, then declared);
without them the fixtures record ``null`` and the Rust tests refuse to pass
until tolerances are declared.

Usage (from the repository root, on the GPU host):

    PYTHONPATH=src .venv/bin/python recipe/valen/dump_text_reference.py \
      --valen-source weights/Valen --checkpoint weights/Valen-Preview-0923 \
      --base weights/Qwen3.5-2B --device cuda

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

FIXTURE_FORMAT = "valen-text-reference/1"
DEFAULT_FIXTURE_NAME = "text-choice-reference.json"
DEFAULT_STATE = "The QA log lists three runs: r1 passed in 12s, r2 failed on a timeout, r3 passed after a retry."
DEFAULT_INSTRUCTIONS = "Which runs passed?"
DEFAULT_CANDIDATES = {
    "r1_only": "Only run r1 passed.",
    "r1_and_r3": "Runs r1 and r3 passed.",
}
# The projection tails: the last OUTPUT_ROWS rows of every recorded output.
OUTPUT_ROWS = 8
# The two instrumented layers: the first GDN layer and the first full-attn layer.
LAYERS = (0, 3)
PROJECTIONS = {
    0: [
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    3: [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ],
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valen-source", required=True, help="pinned Valen source checkout")
    parser.add_argument("--checkpoint", required=True, help="Valen-Preview-0923 directory")
    parser.add_argument("--base", required=True, help="Qwen3.5-2B base directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--out",
        default=str(HERE / "fixtures" / DEFAULT_FIXTURE_NAME),
        help=f"output fixture path (default: fixtures/{DEFAULT_FIXTURE_NAME})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="declared absolute elementwise tolerance for the end-to-end final output",
    )
    parser.add_argument(
        "--projection-tolerance",
        type=float,
        default=None,
        help="declared absolute elementwise tolerance for the projection outputs",
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
        raise SystemExit("the native text path is BF16; --dtype bf16 only")

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

    # The native FP32 adapter GEMMs run with TF32 disabled; pin the reference
    # to strict FP32 the same way the export-parity run did.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = Path(args.checkpoint).resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    max_length = int(config["max_length"])
    media_kwargs = config.get("media_kwargs") or {}
    processor = ValenProcessor(Path(args.base), max_length, config.get("media_kwargs"))
    executor = ValenExecutor(checkpoint, Path(args.base), args.device, args.dtype)

    try:
        with TemporaryDirectory(prefix="valen-text-dump-") as directory:
            record = reference.reference_text_record(
                DEFAULT_STATE,
                {
                    "which_passed": {
                        "type": "choice",
                        "instructions": DEFAULT_INSTRUCTIONS,
                        "criteria": DEFAULT_CANDIDATES,
                    }
                },
            )
            compiler = Compiler(
                processor.processor,
                media_root=Path(directory),
                max_length=max_length,
                media_kwargs=media_kwargs,
            )
            state = compiler.compile(record)
            branch = state.questions[0].branches[0]
            inputs = {
                key: value
                for key, value in branch.inputs.items()
                if key not in ("mm_token_type_ids", "pixel_values", "image_grid_thw")
            }
            device = torch.device(args.device)
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            token_ids = inputs["input_ids"][0].tolist()
            tokens = len(token_ids)

            language = executor.model.backbone.language_model
            hooks = []
            captured: dict[str, dict] = {}

            def pre_hook(name):
                def hook(_module, args, _kwargs):
                    captured.setdefault(name, {})["input"] = args[0].detach()

                return hook

            def post_hook(name):
                def hook(_module, _args, result):
                    tensor = result[0] if isinstance(result, tuple) else result
                    captured.setdefault(name, {})["output"] = tensor.detach()

                return hook

            for layer, names in PROJECTIONS.items():
                for name in names:
                    module = getattr(
                        getattr(language.base_model.model.layers[layer], name.split(".")[0]),
                        name.split(".")[1],
                    )
                    hooks.append(module.register_forward_pre_hook(pre_hook(f"layers.{layer}.{name}"), with_kwargs=True))
                    hooks.append(module.register_forward_hook(post_hook(f"layers.{layer}.{name}")))

            def run():
                with torch.no_grad():
                    return executor.model.backbone(
                        **inputs, use_cache=False, return_dict=True,
                        output_hidden_states=True,
                    )

            first = run()
            second = run()
            for handle in hooks:
                handle.remove()
            if not torch.equal(first.last_hidden_state, second.last_hidden_state):
                raise SystemExit("the reference forward is not deterministic; refusing to record")

            last_hidden = first.last_hidden_state[0]
            if last_hidden.shape[0] != tokens:
                raise SystemExit(f"unexpected hidden length {last_hidden.shape[0]} != {tokens}")
            hidden_states = first.hidden_states
    finally:
        executor.close()

    projections = {}
    for layer, names in PROJECTIONS.items():
        for name in names:
            key = f"layers.{layer}.{name}"
            entry = captured[key]
            tensor_in, tensor_out = entry["input"][0], entry["output"][0]
            if tensor_in.dtype is not torch.bfloat16 or tensor_out.dtype is not torch.bfloat16:
                raise SystemExit(f"{key}: expected BF16 tensors, got {tensor_in.dtype}/{tensor_out.dtype}")
            if tensor_in.shape[0] != tokens:
                raise SystemExit(f"{key}: input covers {tensor_in.shape[0]} rows, expected {tokens}")
            projections[key] = {
                "input_shape": list(tensor_in.shape),
                "output_shape": list(tensor_out.shape),
                "input_bf16_hex": bf16_bits(tensor_in),
                "output_tail_bf16_hex": bf16_bits(tensor_out[-OUTPUT_ROWS:]),
                "output_tail_rows": OUTPUT_ROWS,
            }

    # The residual stream's last row after the embeddings (0) and each layer.
    hidden_tail = {
        str(index): bf16_bits(hidden[0, -1]) for index, hidden in enumerate(hidden_states)
    }

    fixture = {
        "format": FIXTURE_FORMAT,
        "state": DEFAULT_STATE,
        "instructions": DEFAULT_INSTRUCTIONS,
        "candidates": DEFAULT_CANDIDATES,
        "token_ids": token_ids,
        "tokens": tokens,
        "hidden_width": int(last_hidden.shape[-1]),
        "layers": list(LAYERS),
        "projections": projections,
        "hidden_state_last_row_bf16_hex": hidden_tail,
        "final_output_bf16_hex": bf16_bits(last_hidden[-1]),
        "tolerance": args.tolerance,
        "tolerance_semantics": (
            "max absolute elementwise difference, in FP32, between this "
            "reference final-norm last-position hidden state and the native "
            "forward output on the same device and dtype (BF16); measure "
            "first, then declare"
        ),
        "projection_tolerance": args.projection_tolerance,
        "projection_tolerance_semantics": (
            "max absolute elementwise difference, in FP32, between each "
            "recorded reference projection output tail and the native base "
            "GEMM + LoRA application on the recorded input, same device/dtype"
        ),
        "reference": {
            "device": args.device,
            "device_name": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "dtype": args.dtype,
            "torch": torch.__version__,
            "tf32": False,
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
    print(f"  tokens: {tokens}, hidden width: {last_hidden.shape[-1]}")
    print(f"  projections: {sorted(projections)}")
    print(
        f"  tolerances: end-to-end {args.tolerance!r}, projections "
        f"{args.projection_tolerance!r} (declare after measuring the native deltas)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
