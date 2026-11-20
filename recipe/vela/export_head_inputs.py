#!/usr/bin/env python3
"""Freeze the head inputs/outputs of a pinned Vela 2.0 decoder member.

Records, per frozen case, exactly what
the two F32 heads consume and produce — CandidateHead (option-end
hidden states + the block query hidden) and SpanHeadV2 (word hidden
states + label-mean hiddens), on CPU FP32 with autocast off (the
heads' own setting, bit-deterministic) — so the Rust heads and the
scoring/decode mirror are checked against these without any GPU. Also
records the reference's calibrated predict answers from that same
`_predict` execution (the scoring target). Hidden matrices go to a
per-case safetensors file; structure, small logits and answers go to
JSON next to it. The package must pass the worker's pin gate first;
``--device`` records
elsewhere (the committed 4B/9B fixtures were recorded under GPU
bf16-autocast, which the member-calibrated test gates account for).

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="pinned Vela 2.0 decoder package directory")
    parser.add_argument("--out", default=None, help="fixture output directory")
    parser.add_argument("--device", default="cpu", help="torch device (default: the CPU FP32 oracle)")
    return parser.parse_args(argv)


def long_span_body(model_name: str):
    filler = " ".join(f"Sentence {i} reports a routine delivery event." for i in range(220))
    return {
        "model": model_name,
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


def cases(model_name: str):
    named = [
        ("example-request", HERE / "example-request.json"),
        ("example-request-set-threshold", HERE / "example-request-set-threshold.json"),
        ("example-request-cjk", HERE / "example-request-cjk.json"),
        ("example-request-score", HERE / "example-request-score.json"),
    ]
    loaded = [
        (name, {**json.loads(path.read_text(encoding="utf-8")), "model": model_name})
        for name, path in named
    ]
    loaded.append(("long-span-windowed", long_span_body(model_name)))
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


def expected_head_calls(records, rows):
    """Mirror `run_batch`'s head dispatch order for one real batch.

    CandidateHead receives every q block in one stacked call, then each
    non-skipped span block runs in flattened record/block order. The row
    decomposition puts every q block in one logical row, so that stacked
    call still has one unambiguous fixture owner.
    """
    q_rows = {
        row
        for row, rec in zip(rows, records)
        if any(block["kind"] == "q" for block in rec["blocks"])
    }
    if len(q_rows) > 1:
        raise SystemExit(
            "one CandidateHead call spans several logical rows; the fixture "
            "schema cannot attribute it faithfully"
        )
    expected = [(next(iter(q_rows)), "q", None)] if q_rows else []
    for row, rec in zip(rows, records):
        for block in rec["blocks"]:
            if block["kind"] == "span" and not block.get("skip"):
                expected.append((row, "span", rec.get("head") or "router"))
    return expected


def execution_key(rec) -> str:
    """Stable identity for one rendered record across the metadata-only
    render pass and `_predict`'s real execution.

    `skip` is runtime dispatch metadata rather than token identity, so
    leave it out; the actual record still decides how many calls it makes.
    """
    core = {
        key: rec[key]
        for key in (
            "p_ids",
            "blocks",
            "n_tokens",
            "layout",
            "names",
            "w_offs",
            "w_pos",
            "span_over",
        )
        if key in rec
    }
    if "blocks" in core:
        core["blocks"] = [
            {key: value for key, value in block.items() if key != "skip"}
            for block in core["blocks"]
        ]
    return json.dumps(
        plain(core), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


class TracedRun:
    """Transparent `engine.run` proxy that attributes the real batches."""

    def __init__(self, delegate, expected_records, recorder):
        self.delegate = delegate
        self.recorder = recorder
        self.attributed: list[tuple[int, dict]] = []
        self.batches: list[list[int]] = []
        self.pending: dict[str, list[int]] = {}
        for index, rec in enumerate(expected_records):
            self.pending.setdefault(execution_key(rec), []).append(index)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def _claim(self, rec) -> int:
        key = execution_key(rec)
        candidates = self.pending.get(key, [])
        if not candidates:
            raise SystemExit(
                "the predict execution ran an unexpected rendered record; "
                "fixture row attribution is no longer faithful"
            )
        return candidates.pop(0)

    def __call__(self, records):
        records = list(records)
        rows = [self._claim(rec) for rec in records]
        expected = expected_head_calls(records, rows)
        before = len(self.recorder.calls)
        result = self.delegate(records)
        new_calls = self.recorder.calls[before:]
        if len(new_calls) != len(expected):
            raise SystemExit(
                f"a predict batch produced {len(new_calls)} head calls for "
                f"{len(expected)} expected calls"
            )
        for call, (_row, kind, head) in zip(new_calls, expected):
            if call["kind"] != kind or (head is not None and call.get("head") != head):
                raise SystemExit(
                    "the predict batch's head dispatch order no longer matches "
                    "the pinned run_batch contract"
                )
        self.batches.append(rows)
        self.attributed.extend(
            (row, call) for call, (row, _kind, _head) in zip(new_calls, expected)
        )
        return result

    def assert_consumed(self) -> None:
        remaining = sum(len(indices) for indices in self.pending.values())
        if remaining:
            raise SystemExit(
                f"the predict execution left {remaining} rendered record(s) unconsumed"
            )


def record_case(engine, req, rows, windows, recorder):
    """Record head tensors, logits and answers from one `_predict` call."""
    recorder.calls = []
    recorder.tensors = {}
    recorder.next_index = 0
    expected_records = [*rows, *windows]
    original_run = engine.run
    traced = TracedRun(original_run, expected_records, recorder)
    recorder._on = True
    try:
        engine.run = traced
        predict = plain(engine._predict(req)[0])
    finally:
        engine.run = original_run
        recorder._on = False
    traced.assert_consumed()
    if len(traced.attributed) != len(recorder.calls):
        raise SystemExit(
            "the predict execution made head calls outside an attributed run batch"
        )
    calls = [dict(call, row=row) for row, call in traced.attributed]
    return calls, dict(recorder.tensors), predict, traced.batches


def rows_of(engine, body):
    """The predict pipeline's own rows, rendered the way `predict_raw`
    runs them: the full renders carry the span block marked skipped for
    windowed rows (the runtime replaces it with the windows), and the
    windows render separately, in the runtime's own order."""
    from vela2_inference import system_one_to_predict

    req, plan = system_one_to_predict(body["state"], body["questions"])
    parts, qs, _thr = engine._to_generic(req)
    spans = [q for q in qs if q["type"] == "span"]
    rest = [q for q in qs if q["type"] != "span"]
    rows = [dict(parts=parts, questions=rest + spans[:1])] + [
        dict(parts=parts, questions=[s]) for s in spans[1:]
    ]
    rows = [r for r in rows if r["questions"]]
    preps = [engine._prep(r) for r in rows]
    full = [engine._render(p) for p in preps]
    for rec, p in zip(full, preps):
        rec["head"] = p["head"]
        rec["head_reason"] = p["head_reason"]
        rec["label_alias"] = plain(p.get("label_alias"))
    windows = []
    for k, p in enumerate(preps):
        window_rows = engine._window_rows(p)
        if window_rows:
            for b in full[k]["blocks"]:
                if b["kind"] == "span":
                    # The runtime replaces this block with the windows;
                    # it never runs the full-render span head.
                    b["skip"] = True
            for wrow in window_rows:
                wrec = engine._render(wrow)
                wrec["head"] = p["head"]
                windows.append(wrec)
    return req, plan, full, windows


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from models.vela.hybrid import verify_package

    package = Path(args.model)
    # The worker's own gate, before the package is imported: fixtures
    # freeze only what a pinned member produced, and the recorded
    # request carries that member's served name.
    member = verify_package(package)
    model_name = member.model_name
    label = model_name.rsplit("/", 1)[-1]
    sys.path.insert(0, str(package.resolve()))
    # The package gate rejects bytecode caches. Cover initialization too:
    # from_pretrained can import package modules lazily.
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from vela2_inference import Vela2

        engine = Vela2.from_pretrained(args.model, backend="torch", device=args.device)
        import modeling_vela2

        recorder = Recorder(modeling_vela2, engine.run.model)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    out = Path(args.out) if args.out else HERE.parents[1] / "tests" / "vela" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    for name, body in cases(model_name):
        req, _plan, rows, windows = rows_of(engine, body)
        # `_predict` owns sorting, budget decisions and actual batch
        # composition. Its one execution supplies the recorded inputs,
        # logits and calibrated answers together.
        calls, tensors, predict, execution_batches = record_case(
            engine, req, rows, windows, recorder
        )
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
            "execution_batches": execution_batches,
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
