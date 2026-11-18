"""Generate a small independent Transformers oracle; no pretrained weights.

Run with torch 2.12.1+cu130 and transformers 5.17.0 on the qualification GPU:
python tests/qwen3_5/prefix_bidir_reference.py /absolute/output-directory
Generated weights/arrays are external evidence, not committed fixtures.
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel


@torch.inference_mode()
def generate(output):
    if torch.__version__ != "2.12.1+cu130" or transformers.__version__ != "5.17.0":
        raise RuntimeError("use the documented pinned oracle dependencies")
    torch.manual_seed(20261101)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=512,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["full_attention", "linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5TextModel(config).to(device="cuda", dtype=torch.bfloat16).eval()
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    rows = []
    for length, state in [(3, 1), (65, 31), (129, 63), (137, 65), (257, 129)]:
        ids = (torch.arange(length, device="cuda") * 17 + 7).remainder(128).view(1, -1)
        # Dense boolean oracle, independent of the native two-pass composition.
        visible = torch.ones((length, length), device="cuda", dtype=torch.bool)
        visible[:state, state:] = False
        mask = {"full_attention": visible[None, None], "linear_attention": None}
        hidden = (
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            .last_hidden_state[0, -1]
            .float()
        )
        rows.append(
            {
                "id": f"t{length}-s{state}",
                "ids": ids[0].tolist(),
                "state_tokens": state,
                "hidden": hidden.tolist(),
            }
        )
    result = {
        "schema": "qwen-prefix-bidir-reference/1",
        "rows": rows,
        "metadata": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "seed": 20261101,
            "precision": "all BF16; dense block SDPA; causal GDN",
        },
    }
    (output / "reference.json").write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
