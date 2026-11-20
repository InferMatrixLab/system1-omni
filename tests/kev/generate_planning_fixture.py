"""Regenerate the CPU geometry oracle from the pinned official implementation."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
source = args.reference / "kev/cuda_graphs.py"
assert (
    hashlib.sha256(source.read_bytes()).hexdigest()
    == "5ce84f0e4e5ad2177314ed3ecc9564d45d14ad9eb422e0fb079c9e1cbbb011b6"
)
spec = importlib.util.spec_from_file_location("graphs", source)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
fixtures = []
for split, lengths in [
    (1, [1]),
    (29, [27, 12]),
    (67, [66, 70]),
    (52, [69]),
    (7, [1679]),
    (29, [12] * 5),
    (29, [12] * 64),
    (4096, [18] * 32),
    (1025, [999, 32, 400, 33, 8]),
    (29, [16, 200, 900, 16]),
    (8000, [5000, 32, 500, 200]),
    (7, [1679, 14, 1680]),
]:
    state = g.bucket(split) if g.bucket(split) <= 1024 else split
    graphed = g.bucket(max(lengths)) <= 1024 and split <= 4096
    prefix = g.pow2(max(16, split)) if graphed else split
    groups = (
        g.length_groups(lengths, 32)
        if graphed
        else [
            list(range(len(lengths)))[i : i + max(1, 16384 // (split + max(lengths)))]
            for i in range(0, len(lengths), max(1, 16384 // (split + max(lengths))))
        ]
    )
    result = [None] * len(lengths)
    for group in groups:
        question = (
            g.bucket(max(lengths[i] for i in group))
            if graphed
            else max(lengths[i] for i in group)
        )
        cap = min(32, 32768 // (prefix + question)) if graphed else len(group)
        limit = max(
            n for n in range(1, cap + 1) if not graphed or g.count_bucket(n) <= cap
        )
        for start in range(0, len(group), limit):
            part = group[start : start + limit]
            count = g.count_bucket(len(part)) if graphed else len(part)
            for index, i in enumerate(part):
                result[i] = {
                    "split": split,
                    "state": state,
                    "prefix": prefix,
                    "question": question,
                    "count": count,
                    "index": index,
                }
    fixtures.append({"split": split, "lengths": lengths, "plans": result})
args.out.write_text(
    json.dumps(
        {"reference": "2074cc44256200bb37540103fa4572901faa10f3", "cases": fixtures},
        separators=(",", ":"),
    )
    + "\n"
)
