#!/usr/bin/env python3
"""Freeze the head inputs/outputs of a pinned Vela 2.0 decoder member.

Records, per frozen case, exactly what
the two F32 heads consume and produce — CandidateHead (option-end
hidden states + the block query hidden) and SpanHeadV2 (word hidden
states + label-mean hiddens), on CPU FP32 with autocast off (the
heads' own setting, bit-deterministic) — so the Rust heads and the
scoring/decode mirror are checked against these without any GPU. Also
records the reference's calibrated predict answers for the same cases
(the scoring target). Hidden matrices go to a per-case safetensors
file; structure, small logits and answers go to JSON next to it. The
package must pass the worker's pin gate first; ``--device`` records
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


def head_calls_of(rec) -> int:
    """Head calls one rendered record makes: the stacked q call (if it
    has q blocks) plus its span block unless the runtime skipped it
    (windowed rows replace the full-render span with the windows)."""
    q = any(b["kind"] == "q" for b in rec["blocks"])
    span = any(b["kind"] == "span" and not b.get("skip") for b in rec["blocks"])
    return int(q) + int(span)


def run_batched(engine, recs, recorder):
    """Run records through the runtime's own `_run_all` batching — the
    same sort, budget and batch composition `predict_raw` uses — and
    attribute every recorded head call back to its record (a record
    makes at most one head call: the stacked q call, or the span call
    of a window)."""
    order = sorted(range(len(recs)), key=lambda k: recs[k]["n_tokens"])
    batches: list[list[int]] = []
    cur: list[int] = []
    for k in order:
        width = max([recs[j]["n_tokens"] for j in cur] + [recs[k]["n_tokens"]])
        if cur and (len(cur) + 1 > engine.bs or width * (len(cur) + 1) > engine.budget):
            batches.append(cur)
            cur = []
        cur.append(k)
    if cur:
        batches.append(cur)
    attributed: list[tuple[int, dict]] = []
    for batch in batches:
        before = len(recorder.calls)
        engine.run([recs[k] for k in batch])
        new_calls = recorder.calls[before:]
        # Records execute in batch order, each making head_calls_of(rec)
        # consecutive calls.
        expected: list[int] = []
        for k in batch:
            expected.extend([k] * head_calls_of(recs[k]))
        if len(new_calls) != len(expected):
            raise SystemExit(
                f"a batched run produced {len(new_calls)} head calls for "
                f"{len(expected)} records; the batching replication is wrong"
            )
        attributed.extend(zip(expected, new_calls))
    return attributed


def assert_consistent_recording(
    calls: list[dict],
    row_tensors: dict[str, "np.ndarray"],
    predict_calls: list[dict],
    predict_tensors: dict[str, "np.ndarray"],
    atol: float = 1e-4,
) -> None:
    """The predict pass must reproduce the recorded per-row head calls.

    Calls are matched by kind, head and input shapes, then the inputs
    and the logits are compared with a tolerance: the runtime's two
    passes are not bitwise stable (measured ~1e-6 render noise on CPU,
    more on a GPU recording), but a structural mismatch or a drift
    beyond the tolerance means the committed logits and the committed
    answers would come from diverging executions - the review's 76/373
    vs 79/377 fixture inconsistency.
    """

    def key_of(call: dict, tensors_by_name) -> tuple:
        shapes = tuple(
            np.ascontiguousarray(tensors_by_name[call[k]]).shape
            for k in ("cand", "query", "words", "labels")
            if k in call
        )
        return (call["kind"], call.get("head", ""), shapes)

    def arrays_of(call: dict, tensors_by_name) -> list:
        return [
            np.ascontiguousarray(tensors_by_name[call[k]])
            for k in ("cand", "query", "words", "labels")
            if k in call
        ]

    if len(calls) != len(predict_calls):
        raise SystemExit(
            f"the predict execution made {len(predict_calls)} head calls for "
            f"{len(calls)} recorded rows; the committed logits and answers "
            "would come from different executions"
        )
    remaining = list(predict_calls)
    for call in calls:
        want_key = key_of(call, row_tensors)
        match = next((i for i, c in enumerate(remaining) if key_of(c, predict_tensors) == want_key), None)
        if match is None:
            described = [key_of(c, predict_tensors) for c in predict_calls]
            raise SystemExit(
                "the predict execution made a head call set that does not "
                f"match the per-row pass; want {want_key}; predict calls: {described}"
            )
        predicted = remaining.pop(match)
        for got, ref in zip(arrays_of(predicted, predict_tensors), arrays_of(call, row_tensors)):
            if got.shape != ref.shape or not np.allclose(got, ref, atol=atol, rtol=1e-3):
                raise SystemExit(
                    "the predict execution's inputs diverge from the recorded "
                    f"ones (max |delta| {np.abs(got - ref).max():.3e})"
                )
        got = np.asarray(predicted["logits"], dtype=np.float32)
        ref = np.asarray(call["logits"], dtype=np.float32)
        if got.shape != ref.shape or not np.allclose(got, ref, atol=atol, rtol=1e-3):
            raise SystemExit(
                f"the predict execution's logits diverge from the recorded "
                f"ones (max |delta| {np.abs(got - ref).max():.3e}); the committed "
                "answers would not match the committed logits"
            )


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
        req, plan, rows, windows = rows_of(engine, body)
        # Mirror the predict execution: full renders (their span block
        # marked skipped for windowed rows) then the windows, both
        # through the runtime's own `_run_all` batching, so the
        # recorded logits come from the same batch composition the
        # reference runs — the review's 76/373 vs 79/377 fixture
        # inconsistency came from hand-driving records one by one.
        calls = []
        tensors: dict[str, np.ndarray] = {}
        recorder.calls = []
        recorder.tensors = {}
        recorder.next_index = 0
        recorder._on = True
        row_of_call: list[int] = []
        try:
            # Full renders run one record per call (trivial row
            # attribution); their inputs stay within the recording
            # tolerance of the predict pass's batched execution, which
            # the consistency check verifies. The windows run through
            # the runtime's own batching — the batch composition is
            # what the span aggregation is sensitive to.
            attributed = []
            for row, rec in enumerate(rows):
                before = len(recorder.calls)
                engine.run([rec])
                new_calls = recorder.calls[before:]
                expected = head_calls_of(rec)
                if len(new_calls) != expected:
                    raise SystemExit(
                        f"row {row} made {len(new_calls)} head calls, "
                        f"expected {expected}"
                    )
                attributed.extend((row, call) for call in new_calls)
            attributed += [
                # The windows are recorded as their own rows after the
                # main rows (the row index continues).
                (len(rows) + win, call)
                for win, call in run_batched(engine, windows, recorder)
            ]
            for row, call in attributed:
                row_of_call.append(row)
                calls.append(call)
                tensors.update(recorder.tensors)
                recorder.calls = []
            # The reference's calibrated answers, recorded in the same
            # session: the predict pass re-runs the same rows and windows
            # through the same batching, so its own recorded head calls
            # must reproduce the recorded calls above.
            saved_tensors = dict(tensors)
            recorded_count = len(recorder.calls)
            predict = plain(engine._predict(req)[0])
            predict_calls = recorder.calls[recorded_count:]
        finally:
            recorder._on = False
        assert_consistent_recording(calls, tensors, predict_calls, recorder.tensors)
        for call, row in zip(calls, row_of_call):
            call["row"] = row
        save_file(saved_tensors, str(out / f"head-input-{label}-{name}.safetensors"))
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
