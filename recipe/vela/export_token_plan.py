#!/usr/bin/env python3
"""Freeze the token-level row plan of the pinned Vela 2.0 0.8B runtime.

This tool records, for a set of request
bodies, exactly what the pinned runtime feeds the backbone — the
SystemOne → predict mapping, the per-row decomposition (parts pass +
question blocks; one span question per row), and the rendered records
(``p_ids``, block token ids with option/label end positions, query
positions, span word positions, truncation and segment lengths). The
native Rust processor must reproduce these byte-for-byte; the frozen
files under ``tests/vela/fixtures/native-plan-*.json`` are that parity
target. Model outputs are not recorded here — the oracle fixtures from
``compare_reference.py`` own the output side.

    PYTHONPATH=src .venv/bin/python recipe/vela/export_token_plan.py \
      --model weights/vela-2.0-0.8b --out tests/vela/fixtures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_NAME = "vllm-sr/Vela-2.0-0.8B"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="pinned Vela 2.0 0.8B package directory")
    parser.add_argument("--out", default=None, help="fixture output directory")
    return parser.parse_args(argv)


def long_span_body():
    """A span target beyond HYBRID_RMAX (2,048 tokens) to exercise the
    windowed layout: the full-render span block is skipped and one row
    per 1,800-token window (stride 1,536, cut at word starts) carries
    the span question."""

    filler = " ".join(f"Sentence {i} reports a routine delivery event." for i in range(220))
    return {
        "model": MODEL_NAME,
        "state": {"request": filler},
        "questions": {
            "claims": {
                "type": "span",
                "instructions": "Which spans mention a delivery?",
                "criteria": {"delivery": "a delivery event"},
                "over": "request",
            }
        },
    }


def cases():
    named = [
        ("example-request", HERE / "example-request.json"),
        ("example-request-set-threshold", HERE / "example-request-set-threshold.json"),
        ("example-request-cjk", HERE / "example-request-cjk.json"),
        ("example-request-score", HERE / "example-request-score.json"),
    ]
    loaded = [
        (
            name,
            {**json.loads(path.read_text(encoding="utf-8")), "model": MODEL_NAME},
        )
        for name, path in named
    ]
    loaded.append(("long-span-windowed", long_span_body()))
    return loaded


def plain(value):
    """numpy-free JSON for the rendered records."""

    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def rows_of(engine, body):
    """Mirror ``Vela2._predict``'s row decomposition and render each row."""

    from vela2_inference import system_one_to_predict

    req, plan = system_one_to_predict(body["state"], body["questions"])
    parts, qs, _thr = engine._to_generic(req)
    spans = [q for q in qs if q["type"] == "span"]
    rest = [q for q in qs if q["type"] != "span"]
    rows = [dict(parts=parts, questions=rest + spans[:1])] + [
        dict(parts=parts, questions=[s]) for s in spans[1:]
    ]
    rendered, windows = [], []
    for row in rows:
        if not row["questions"]:
            continue
        prepped = engine._prep(row)
        rec = plain(engine._render(prepped))
        rec["head"] = prepped["head"]
        rec["head_reason"] = prepped["head_reason"]
        rendered.append(rec)
        for wrow in engine._window_rows(prepped) or []:
            wrec = plain(engine._render(wrow))
            wrec["head"] = prepped["head"]
            windows.append(wrec)
    return plain(req), plain(plan), rendered, windows


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(Path(args.model).resolve()))
    from vela2_inference import Vela2

    engine = Vela2.from_pretrained(args.model, backend="torch", device="cpu")
    out = Path(args.out) if args.out else HERE.parents[1] / "tests" / "vela" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    for name, body in cases():
        req, plan, rows, windows = rows_of(engine, body)
        fixture = {
            "case": name,
            "request": body,
            "predict_request": req,
            "plan": plan,
            "rows": rows,
            "window_rows": windows,
        }
        target = out / f"native-plan-{name}.json"
        target.write_text(json.dumps(fixture, ensure_ascii=False) + "\n", encoding="utf-8")
        tokens = sum(r["n_tokens"] for r in rows) + sum(r["n_tokens"] for r in windows)
        print(
            f"{name}: {len(rows)} row(s) + {len(windows)} window row(s), "
            f"{tokens} tokens -> {target.name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
