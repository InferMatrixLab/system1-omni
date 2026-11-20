#!/usr/bin/env python3
"""Compare the Valen-Decision-1.5 worker against the checkpoint's own predict.

Runs each request through two paths on the same pinned checkpoint:

- reference: the released Transformers path, ``model.predict`` on a
  SystemOne request that carries the fixture's questions verbatim and its
  state as plain text or as a local image file;
- worker: the System1-Omni pipeline (protocol, preprocessing, executor,
  postprocessing).

Without ``--url`` the worker side is ``frontend.valen_decision_1_5.decide``
and both paths share one loaded model, so the comparison isolates input
construction, the readout and probability normalization. With ``--url`` the
worker side is an HTTP ``POST /v1/systemone`` to a running worker or Rust
frontend, and this process loads the checkpoint only for the reference.

The declared criteria are identical answer structure (question order, field
order, candidate order, chosen key, score legend), equal token accounting and
a tolerance of 1e-6 (``--tolerance``) on every probability, on the ``score``
expected level and on confidences. Requires a pinned checkpoint and a working
torch device; see recipe/valen_decision_1_5/reference.md.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import sys
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
PROBABILITY_TOLERANCE = 1e-6
CASES = {
    "image": "example-request.json",
    "text": "example-text-request.json",
    "noul": "example-noul-request.json",
    "score": "example-score-request.json",
    "mixed": "example-mixed-request.json",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", required=True, help="pinned Valen-Decision-1.5 checkpoint directory"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--case",
        action="append",
        choices=(*CASES, "all"),
        help="recipe fixture to run; repeatable (default: all)",
    )
    parser.add_argument(
        "--body",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="extra request body JSON file to run as case NAME; repeatable",
    )
    parser.add_argument(
        "--url",
        help="compare a running worker or frontend at this base URL instead of"
        " the in-process pipeline",
    )
    parser.add_argument("--tolerance", type=float, default=PROBABILITY_TOLERANCE)
    parser.add_argument(
        "--save", help="write every worker and reference response to this JSON file"
    )
    parser.add_argument(
        "--no-verify-weights",
        action="store_true",
        help="skip hashing the weight shards; code, config and tokenizer are always verified",
    )
    return parser.parse_args(argv)


def reference_request(body, media_dir):
    """The SystemOne request the released ``predict`` takes for a wire body.

    The questions pass through verbatim. Text state stays a plain string; an
    inline image is written to ``media_dir`` and referenced by file name.
    """

    state = body["state"]
    if isinstance(state, dict) and "image" in state:
        prefix, _, encoded = state["image"].partition(",")
        suffix = {"data:image/png;base64": ".png", "data:image/jpeg;base64": ".jpg"}[prefix]
        path = Path(media_dir) / f"reference{suffix}"
        path.write_bytes(base64.b64decode(encoded, validate=True))
        state = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": path.name}}],
                }
            ]
        }
    elif not isinstance(state, str):
        raise ValueError("a parity body needs plain-text or inline-image state")
    return {"state": state, "questions": copy.deepcopy(body["questions"])}


def _numbers(answer):
    """Split one answer into its discrete shape and its (group, value) floats."""

    shape = []
    numbers = []
    for field, value in answer.items():
        if field == "probabilities":
            shape.append((field, tuple(value)))
            numbers += [("probability", item) for item in value.values()]
        elif field == "noul":
            shape.append((field, None))
            numbers.append(("probability", value))
        elif field == "score":
            shape.append((field, None))
            numbers.append(("answer", value))
        elif field == "confidence":
            shape.append((field, None))
            numbers.append(("confidence", value))
        else:
            shape.append((field, json.dumps(value, ensure_ascii=False)))
    return shape, numbers


def compare_case(name, worker, reference, tolerance=PROBABILITY_TOLERANCE):
    """Compare one worker response with one reference response.

    Every non-numeric part of an answer must be identical, including field
    and candidate order. A non-finite number on either side fails the case
    before any tolerance verdict, because ``max`` keeps its previous finite
    maximum against a NaN difference. Returns (report lines, passed).
    """

    lines = [
        f"case: {name}",
        f"reference model: {reference['model']}",
        f"worker model: {worker['model']}",
    ]
    problems = []
    deltas = {"probability": 0.0, "answer": 0.0, "confidence": 0.0}
    seen = set()
    if list(worker["answers"]) != list(reference["answers"]):
        problems.append(
            f"question order {list(worker['answers'])} != {list(reference['answers'])}"
        )
    else:
        for qid, ref_answer in reference["answers"].items():
            shape, numbers = _numbers(worker["answers"][qid])
            ref_shape, ref_numbers = _numbers(ref_answer)
            if shape != ref_shape:
                problems.append(f"{qid}: answer differs outside its numeric fields")
                continue
            values = [value for _, value in numbers + ref_numbers]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values) or any(
                not math.isfinite(v) for v in values
            ):
                problems.append(f"{qid}: non-finite or non-numeric value")
                continue
            for (group, value), (_, ref_value) in zip(numbers, ref_numbers):
                seen.add(group)
                deltas[group] = max(deltas[group], abs(value - ref_value))

    usage = worker["usage"]
    ref_usage = reference["usage"]
    compute = worker["internal_usage"]["compute_tokens"]
    ref_compute = reference["internal_usage"]["compute_tokens"]
    tokens_exact = (
        usage == ref_usage and usage["output_tokens"] == 0 and compute == ref_compute
    )
    for field in ("input_tokens", "output_tokens"):
        sign = "==" if usage[field] == ref_usage[field] else "!="
        lines.append(f"{field}: {usage[field]} {sign} {ref_usage[field]}")
    lines.append(f"compute_tokens: {compute} {'==' if compute == ref_compute else '!='} {ref_compute}")
    labels = {
        "probability": "max probability delta",
        "answer": "max answer delta",
        "confidence": "confidence delta",
    }
    for group, label in labels.items():
        if group in seen:
            lines.append(f"{label}: {deltas[group]:.10e}")
    lines.append(
        "decision agreement: " + ("PASS" if not problems else "FAIL: " + "; ".join(problems))
    )
    within = True
    for group in labels:
        if group in seen:
            ok = deltas[group] <= tolerance
            within = within and ok
            lines.append(f"{group} tolerance <= {tolerance:g}: " + ("PASS" if ok else "FAIL"))
    lines.append("token accounting: " + ("PASS" if tokens_exact else "FAIL"))
    return lines, not problems and within and tokens_exact


def post(url, body):
    """POST one body to a running worker or frontend, bypassing HTTP proxies."""

    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        return json.loads(response.read())


def load_reference_model(checkpoint, device, dtype):
    """Load the checkpoint the way the released Transformers quickstart does."""

    import torch
    from transformers import AutoModel

    model = (
        AutoModel.from_pretrained(
            str(checkpoint.path),
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16 if dtype == "bf16" else torch.float32,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    return model


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from models.valen_decision_1_5.pinned import verify_checkpoint

    names = args.case or ["all"]
    if "all" in names:
        names = list(CASES)
    cases = [(name, HERE / CASES[name]) for name in names]
    for entry in args.body:
        name, separator, path = entry.partition("=")
        if not separator:
            raise SystemExit("--body takes NAME=PATH")
        cases.append((name, Path(path)))

    try:
        checkpoint = verify_checkpoint(args.checkpoint, weights=not args.no_verify_weights)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model_name = checkpoint.release.model_name
    print(f"checkpoint: {checkpoint.release.repo}@{checkpoint.release.revision}")
    print(f"device: {args.device}, dtype: {args.dtype}")
    print("worker path: " + (args.url or "in-process frontend.valen_decision_1_5.decide"))
    print()

    predict_kwargs = {}
    if args.url:
        model = load_reference_model(checkpoint, args.device, args.dtype)
        executor = None
    else:
        from frontend.valen_decision_1_5 import decide
        from models.valen_decision_1_5.engine import ValenDecisionExecutor
        from models.valen_decision_1_5.preprocess import ValenDecisionProcessor

        processor = ValenDecisionProcessor(checkpoint)
        executor = ValenDecisionExecutor(checkpoint, args.device, args.dtype)
        model = executor.model
        predict_kwargs["processor"] = processor.processor

    failed = 0
    responses = {}
    try:
        for name, path in cases:
            # The fixtures name the 0.8B id; a worker serves only its own size.
            body = dict(json.loads(path.read_text(encoding="utf-8")), model=model_name)
            if args.url:
                worker_response = post(args.url, body)
            else:
                worker_response = decide(json.dumps(body).encode("utf-8"), processor, executor)
            with TemporaryDirectory(prefix="valen-decision-parity-") as directory:
                reference = model.predict(
                    reference_request(body, directory), media_root=directory, **predict_kwargs
                )
            responses[name] = {"worker": worker_response, "reference": reference}
            lines, ok = compare_case(name, worker_response, reference, args.tolerance)
            print("\n".join(lines))
            print()
            failed += not ok
    finally:
        if executor is not None:
            executor.close()
    if args.save:
        Path(args.save).write_text(
            json.dumps(responses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(
        "compare_reference: ok"
        if not failed
        else f"compare_reference: {failed} of {len(cases)} FAILED"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
