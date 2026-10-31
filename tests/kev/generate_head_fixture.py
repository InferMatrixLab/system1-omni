"""Run the pinned upstream PointerHead on deterministic hidden states, on CPU."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import torch

SOURCE_SHA256 = "2ff3b2e67c048e3bc5a0d1a32aafe50804ed315a5564fce06bb4b95697c5271c"
HEAD_SHA256 = "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c"


def hidden(row):
    return torch.tensor(
        [((i * 37 + row * 19) % 101 - 50) / 64 for i in range(2560)],
        dtype=torch.float32,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.reference_model.read_bytes()
    if (
        hashlib.sha256(source).hexdigest() != SOURCE_SHA256
        or hashlib.sha256(args.head.read_bytes()).hexdigest() != HEAD_SHA256
    ):
        raise ValueError("reference source or head does not match the pinned release")
    # Execute the actual pinned class without importing the unrelated backbone and
    # training dependencies. The hash above covers the entire upstream model.py.
    definition = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "PointerHead"
    )
    namespace = {"nn": torch.nn, "math": math}
    exec(
        compile(
            ast.Module(body=[definition], type_ignores=[]),
            str(args.reference_model),
            "exec",
        ),
        namespace,
    )
    meta = torch.load(args.head, map_location="cpu", weights_only=True)
    head = namespace["PointerHead"](2560, 256)
    head.load_state_dict(meta["head"], strict=True)
    head.temperature = meta["temperature"]
    torch.set_num_threads(1)
    cases = []
    with torch.no_grad():
        for n in (1, 2, 7, 31, 255):
            options = torch.stack([hidden(i + 1) for i in range(n)])
            head.train()
            logits = head(hidden(0), options)
            head.eval()
            probabilities = head(hidden(0), options).softmax(-1)
            cases.append(
                {
                    "options": n,
                    "logits": logits.tolist(),
                    "probabilities": probabilities.tolist(),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "reference_revision": "2074cc44256200bb37540103fa4572901faa10f3",
                "reference_model_sha256": SOURCE_SHA256,
                "head_pt_sha256": HEAD_SHA256,
                "torch": torch.__version__,
                "device": "cpu",
                "temperature": meta["temperature"],
                "hidden_rule": "((i * 37 + row * 19) % 101 - 50) / 64; row 0 = decide",
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
