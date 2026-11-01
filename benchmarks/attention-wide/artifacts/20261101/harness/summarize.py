#!/usr/bin/env python3
"""Rebuild summary.json from the raw JSONL records in this directory.

Usage: python3 summarize.py [artifact_dir]
"""
import collections
import json
import os
import re
import statistics
import sys


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def kernel_arms(rows):
    by_arm = collections.defaultdict(list)
    medians = collections.defaultdict(dict)
    for r in rows:
        if r["kind"] == "sample":
            by_arm[(r["trial"], r["tokens"], r["arm"])].append(r["ms"])
        elif r["kind"] == "median":
            medians[(r["trial"], r["tokens"])] = (r["bm64"], r["bm128"])
    out = []
    keys = sorted({(r["trial"], r["tokens"]) for r in rows if r["kind"] == "median"})
    for trial, tokens in keys:
        a = by_arm[(trial, tokens, "bm64")]
        b = by_arm[(trial, tokens, "bm128")]
        ma, mb = statistics.median(a), statistics.median(b)
        out.append({
            "trial": trial,
            "tokens": tokens,
            "bm64_ms": round(ma, 6),
            "bm128_ms": round(mb, 6),
            "delta_pct": round((mb - ma) / ma * 100.0, 3),
            "bm64_p10_ms": round(pct(a, 0.1), 6),
            "bm64_p90_ms": round(pct(a, 0.9), 6),
            "bm128_p10_ms": round(pct(b, 0.1), 6),
            "bm128_p90_ms": round(pct(b, 0.9), 6),
            "samples_per_arm": len(a),
        })
    return out


def lib_arms(rows):
    by_rev = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "lib_sample":
            by_rev[(r["trial"], r["tokens"], r["revision"])].append(r["ms"])
    out = []
    keys = sorted({(r["trial"], r["tokens"]) for r in rows if r["kind"] == "lib_sample"})
    for trial, tokens in keys:
        a = by_rev[(trial, tokens, "old")]
        b = by_rev[(trial, tokens, "new")]
        if not a or not b:
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        out.append({
            "trial": trial,
            "tokens": tokens,
            "old_ms": round(ma, 6),
            "new_ms": round(mb, 6),
            "delta_pct": round((mb - ma) / ma * 100.0, 3),
        })
    return out


def aggregate(rows, key_a, key_b):
    by_tokens = collections.defaultdict(list)
    for r in rows:
        by_tokens[r["tokens"]].append((r[key_a], r[key_b]))
    out = {}
    for tokens, pairs in sorted(by_tokens.items()):
        a = statistics.median(p[0] for p in pairs)
        b = statistics.median(p[1] for p in pairs)
        deltas = [(p[1] - p[0]) / p[0] * 100.0 for p in pairs]
        out[str(tokens)] = {
            "trials": len(pairs),
            "baseline_ms": round(a, 6),
            "candidate_ms": round(b, 6),
            "delta_pct": round((b - a) / a * 100.0, 3),
            "delta_pct_range": [round(min(deltas), 3), round(max(deltas), 3)],
        }
    return out


def parity(path):
    text = open(path).read() if os.path.exists(path) else ""
    lines = [l for l in text.splitlines() if l.strip()]
    return {"lines": len(lines), "all_exact": bool(lines) and all(l.endswith("exact") for l in lines)}


def sanitizer(path):
    text = open(path).read() if os.path.exists(path) else ""
    errors = [int(m) for m in re.findall(r"ERROR SUMMARY: (\d+) errors", text)]
    exits = [int(m) for m in re.findall(r"exit=(\d+)", text)]
    return {"runs": len(errors), "errors": errors, "all_zero": bool(errors) and not any(errors),
            "all_exit_zero": bool(exits) and not any(exits)}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    summary = {"protocol": "artifacts/20261101/plan.json", "devices": {}}
    for device in ("l20", "sm120"):
        krows = load(os.path.join(d, f"results_{device}.jsonl"))
        lrows = load(os.path.join(d, f"lib_{device}.jsonl"))
        kdeets = kernel_arms(krows)
        ldeets = lib_arms(lrows)
        dev = {
            "kernel_level": {
                "per_trial": kdeets,
                "aggregate": aggregate(kdeets, "bm64_ms", "bm128_ms"),
                "parity_all_identical": all(r["identical"] for r in krows if r["kind"] == "parity"),
                "occupancy": [r for r in krows if r["kind"] == "occupancy"],
            },
            "library_level": {"per_trial": ldeets, "aggregate": aggregate(ldeets, "old_ms", "new_ms")},
            "parity_full_pass": parity(os.path.join(d, f"parity_{device}.log")),
            "parity_prefix_windows": parity(os.path.join(d, f"parity_prefix_{device}.log")),
            "sanitizer": sanitizer(os.path.join(d, f"sanitizer_{device}.log")),
        }
        summary["devices"][device] = dev
    out = os.path.join(d, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
