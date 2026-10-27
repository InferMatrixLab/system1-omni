#!/usr/bin/env python3
"""Freeze the head inputs/outputs of the pinned Vela 2.0 0.8B runtime.

Records, per frozen case, exactly what
the two F32 heads consume and produce — CandidateHead (option-end
hidden states + the block query hidden) and SpanHeadV2 (word hidden
states + label-mean hiddens), on CPU FP32 with autocast off (the
heads' own setting, bit-deterministic) — so the Rust heads and the
scoring/decode mirror are checked against these without any GPU. Also
records the reference's calibrated predict answers for the same cases
(the scoring target). Hidden matrices go to a per-case safetensors
file; structure, small logits and answers go to JSON next to it.

    PYTHONPATH=src .venv/bin/python recipe/vela/export_head_inputs.py \
      --model weights/vela-2.0-0.8b --out tests/vela/fixtures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

HERE = Path(__file__).resolve().parent
MODEL_NAME = "vllm-sr/Vela-2.0-0.8B"



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="pinned Vela 2.0 decoder package directory")
    parser.add_argument("--out", default=None, help="fixture output directory")
    parser.add_argument("--device", default="cpu", help="torch device (default: the CPU FP32 oracle)")
    return parser.parse_args(argv)


def long_span_body():
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
        (name, {**json.loads(path.read_text(encoding="utf-8")), "model": MODEL_NAME})
        for name, path in named
    ]
    loaded.append(("long-span-windowed", long_span_body()))
    return loaded


def plain(value):
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


class Recorder:
    """Class-level wrappers around the two heads; records inputs/outputs."""

    def __init__(self, module, model):
        self.calls: list[dict] = []
        self.tensors: dict[str, np.ndarray] = {}
        self._on = False
        # Tensor names must never collide across rows: the counter keeps
        # growing for the whole case.
        self.next_index = 0
        self.module = module
        self.model = model
        original_head = module.CandidateHead.forward
        original_span = module.SpanHeadV2.forward
        recorder = self

        def head_forward(self, candidates, query):
            out = original_head(self, candidates, query)
            if recorder._on:
                index = recorder.next_index
                cname, qname = f"q{index}_cand", f"q{index}_query"
                recorder.tensors[cname] = candidates.detach().cpu().numpy().astype(np.float32)
                recorder.tensors[qname] = query.detach().cpu().numpy().astype(np.float32)
                recorder.next_index = index + 1
                recorder.calls.append(
                    dict(kind="q", cand=cname, query=qname,
                         logits=[row.tolist() for row in out.detach().cpu().numpy()])
                )
            return out

        def span_forward(self, words, labels):
            out = original_span(self, words, labels)
            if recorder._on:
                index = recorder.next_index
                wname, lname = f"s{index}_words", f"s{index}_labels"
                which = "broad" if self is getattr(recorder.model, "span_broad", None) else "router"
                recorder.tensors[wname] = words.detach().cpu().numpy().astype(np.float32)
                recorder.tensors[lname] = labels.detach().cpu().numpy().astype(np.float32)
                recorder.next_index = index + 1
                recorder.calls.append(
                    dict(kind="span", head=which, words=wname, labels=lname,
                         logits=out.detach().cpu().numpy().tolist())
                )
            return out

        module.CandidateHead.forward = head_forward
        module.SpanHeadV2.forward = span_forward


def rows_of(engine, body):
    from vela2_inference import system_one_to_predict

    req, plan = system_one_to_predict(body["state"], body["questions"])
    parts, qs, _thr = engine._to_generic(req)
    spans = [q for q in qs if q["type"] == "span"]
    rest = [q for q in qs if q["type"] != "span"]
    rows = [dict(parts=parts, questions=rest + spans[:1])] + [
        dict(parts=parts, questions=[s]) for s in spans[1:]
    ]
    rendered, preps, windows = [], [], []
    for row in rows:
        if not row["questions"]:
            continue
        prepped = engine._prep(row)
        rec = engine._render(prepped)
        rec["head"] = prepped["head"]
        rec["head_reason"] = prepped["head_reason"]
        rec["label_alias"] = plain(prepped.get("label_alias"))
        preps.append(prepped)
        rendered.append(rec)
        for wrow in engine._window_rows(prepped) or []:
            wrec = engine._render(wrow)
            wrec["head"] = prepped["head"]
            windows.append(wrec)
    return req, plan, rendered, windows


def main(argv=None) -> int:
    args = parse_args(argv)
    label = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))["model_name"].rsplit("/", 1)[-1]
    sys.path.insert(0, str(Path(args.model).resolve()))
    from vela2_inference import Vela2

    engine = Vela2.from_pretrained(args.model, backend="torch", device=args.device)
    import modeling_vela2

    recorder = Recorder(modeling_vela2, engine.run.model)
    out = Path(args.out) if args.out else HERE.parents[1] / "tests" / "vela" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    for name, body in cases():
        req, plan, rows, windows = rows_of(engine, body)
        # One row per backend call so every recorded head call is
        # attributable; question blocks arrive stacked in row order.
        calls = []
        tensors: dict[str, np.ndarray] = {}
        recorder.calls = []
        recorder.tensors = {}
        recorder.next_index = 0
        recorder._on = True
        row_of_call: list[int] = []
        try:
            for index, rec in enumerate(rows + windows):
                # Windows render after the main rows (row index continues).
                engine.run([rec])
                for call in recorder.calls:
                    row_of_call.append(index)
                calls.extend(recorder.calls)
                tensors.update(recorder.tensors)
                recorder.calls = []
            # The reference's calibrated answers for the same request.
            predict = plain(engine._predict(req)[0])
        finally:
            recorder._on = False
        for call, row in zip(calls, row_of_call):
            call["row"] = row
        save_file(tensors, str(out / f"head-input-{label}-{name}.safetensors"))
        fixture = {
            "case": name,
            "request": body,
            # The rows block carries the span-head dispatch and alias
            # provenance for human inspection; the tests read the
            # rendered rows from the native-plan fixtures.
            "rows": [
                {key: rec[key] for key in ("head", "head_reason", "label_alias", "n_tokens", "seg_lens")
                 if key in rec}
                for rec in rows
            ],
            "head_calls": calls,
            "predict": predict,
        }
        (out / f"head-input-{label}-{name}.json").write_text(
            json.dumps(fixture, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        span_calls = sum(1 for call in calls if call["kind"] == "span")
        print(f"{name}: {len(rows)} row(s), {len(calls)} head call(s) ({span_calls} span), predict answers: {len(predict['answers'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
