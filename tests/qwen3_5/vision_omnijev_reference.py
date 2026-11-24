"""Generate an external, exact BF16 vision-stage oracle in the pinned reference env.

No weights or run artifacts are committed. See tests/qwen3_5/README.md for use.
The complete HF visual output must equal an actual official MSO1 capture before
its intermediate tensors are accepted as an oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path

import peft
import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True, help="official FP32 pixel_values, flat little-endian")
    parser.add_argument("--grid", type=int, nargs=3, required=True)
    parser.add_argument("--mso1-output", type=Path, required=True, help="actual official visual BF16 capture")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert torch.__version__.split("+")[0] == "2.14.0"
    assert transformers.__version__ == "5.17.0" and peft.__version__ == "0.19.1"
    assert args.grid[0] == 1
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(args.base, local_files_only=True).vision_config
    assert config.hidden_size == 1024 and config.depth == 24
    config._attn_implementation = "sdpa"
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = modeling.Qwen3_5VisionModel(config)
    finally:
        torch.set_default_dtype(previous)
    assert model.rotary_pos_emb.inv_freq.dtype == torch.float32
    model.load_state_dict({k.removeprefix("model.visual."): v for k, v in load_file(args.export / "vision.safetensors").items()}, strict=True)
    model = model.cuda().eval()
    stages = []

    def record(name, tensor):
        tensor = tensor.detach().cpu().contiguous()
        assert tensor.dtype == torch.bfloat16
        path = args.out / (name + ".bf16")
        path.write_bytes(tensor.view(torch.uint16).numpy().tobytes())
        stages.append({"name": name, "shape": list(tensor.shape), "sha256": digest(path)})

    model.patch_embed.register_forward_hook(lambda mod, inputs, out: record("patch_embed", out))
    model.blocks[0].register_forward_pre_hook(lambda mod, inputs: record("position", inputs[0]))
    for index, block in enumerate(model.blocks):
        block.register_forward_hook(lambda mod, inputs, out, index=index: record(f"blocks.{index}", out))
    for name in ["norm1", "attn.qkv", "attn.proj", "norm2", "mlp.linear_fc1", "mlp.act_fn", "mlp.linear_fc2"]:
        target = "mlp.activation" if name == "mlp.act_fn" else name
        model.blocks[0].get_submodule(name).register_forward_hook(lambda mod, inputs, out, target=target: record("blocks.0." + target, out))
    model.blocks[0].attn.proj.register_forward_pre_hook(lambda mod, inputs: record("blocks.0.attn.output", inputs[0]))
    model.blocks[0].norm2.register_forward_pre_hook(lambda mod, inputs: record("blocks.0.attn.residual", inputs[0]))
    model.merger.norm.register_forward_hook(lambda mod, inputs, out: record("merger.norm", out))
    model.merger.linear_fc1.register_forward_hook(lambda mod, inputs, out: record("merger.linear_fc1", out))
    original_rope = modeling.apply_rotary_pos_emb_vision
    seen_rope = False

    def rope(*inputs):
        nonlocal seen_rope
        result = original_rope(*inputs)
        if not seen_rope:
            record("blocks.0.attn.q", result[0])
            record("blocks.0.attn.k", result[1])
            seen_rope = True
        return result

    pixels_bytes = args.pixels.read_bytes()
    pixels = torch.frombuffer(bytearray(pixels_bytes), dtype=torch.float32).reshape(-1, 1536)
    assert pixels.shape[0] == args.grid[1] * args.grid[2] and torch.isfinite(pixels).all()
    (args.out / "pixels.f32").write_bytes(pixels_bytes)
    modeling.apply_rotary_pos_emb_vision = rope
    try:
        with torch.inference_mode():
            result = model(pixels.cuda().bfloat16(), grid_thw=torch.tensor([args.grid], device="cuda"), return_dict=True).pooler_output
    finally:
        modeling.apply_rotary_pos_emb_vision = original_rope
    actual = torch.frombuffer(bytearray(args.mso1_output.read_bytes()), dtype=torch.bfloat16).reshape_as(result)
    assert torch.equal(result.cpu(), actual), "reject oracle: HF visual differs from actual official MSO1"
    record("merger.output", result)
    manifest = {
        "schema": "omnijev-4b-vision-reference/1", "grid": args.grid,
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "export_manifest_sha256": digest(args.export / "omnijev_export.json"),
        "vision_weights_sha256": digest(args.export / "vision.safetensors"),
        "pixels_sha256": digest(args.pixels), "mso1_output_sha256": digest(args.mso1_output),
        "mso1_output_exact": True, "stages": stages,
    }
    (args.out / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"mso1_output_exact": True, "stages": len(stages), "output": str(args.out)}))


if __name__ == "__main__":
    main()
