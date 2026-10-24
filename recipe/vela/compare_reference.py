#!/usr/bin/env python3
"""Compare the Vela 2.0 0.3B worker pipeline against the pinned upstream server.

Runs each fixture request through two paths that share one loaded pinned
engine, so the comparison isolates this repository's admission and
orchestration from the checkpoint's own code:

- reference: the pinned package's own ``vela2_serve.py`` app, driven
  through FastAPI's test client exactly as a client would (routing, body
  parsing, whole-request 422/413 mapping included);
- worker: the production ``frontend.vela.decide`` pipeline (protocol
  envelope admission, pinned ``system_one_request`` execution, response
  finishing).

Both paths execute the same verified runtime, and the worker delegates
response construction to the same pinned code — so the criterion is the
charter's reference parity: exact match, including JSON key order, with
rejections agreeing on status and body (``{"detail": [{"loc", "msg",
"type"}]}``). ``--export`` additionally freezes the oracle fixture —
request, SystemOne response and the predict()-level answers — for the
future native encoder slices, and writes it only after a passing,
non-rejected comparison. Requires the pinned package, the reference
dependencies, and fastapi+uvicorn+httpx; see recipe/vela/reference.md.

The default fixture set is the recipe's four example requests plus
five synthetic rejections: an invalid span question (whole-request
422), a schema that alone exceeds the 8,192-token window (413), and
the three envelope 422s (missing field, extra field, wrong model
name) that freeze the worker's preamble admission against the pinned
runtime's. CPU FP32 is the oracle setting; GPU bf16-autocast is a
deployment option, not this tool's default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_NAME = "vllm-sr/Vela-2.0-0.3B"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="pinned Vela 2.0 0.3B package directory")
    parser.add_argument("--device", default="cpu", help="torch device (default: the CPU FP32 oracle)")
    parser.add_argument(
        "--fixture",
        action="append",
        default=None,
        help="request body file (repeatable; default: the recipe examples plus the "
        "synthetic invalid-span and over-limit rejections)",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="write the frozen oracle fixture (request, SystemOne response, predict-level "
        "answers per passing 200 case); only passing, non-rejected comparisons are written",
    )
    return parser.parse_args(argv)


def over_limit_body():
    """A schema that alone cannot fit the 8,192-token window.

    255 choice options with ~300-character descriptions: far past
    ``max_length`` before any state is rendered, which is exactly the
    upstream 413 condition (questions + schema alone; state text would be
    truncated, never 413'd).
    """

    criteria = {
        f"option_{index}": (
            f"a deliberately long description repeating filler text to overflow the "
            f"window for option {index}: " + "filler " * 40
        )
        for index in range(255)
    }
    return {
        "model": MODEL_NAME,
        "state": "short",
        "questions": {
            "big": {"type": "choice", "instructions": "Pick one.", "criteria": criteria}
        },
    }


def invalid_span_body():
    """A span question whose criteria is a list, not a label mapping."""

    return {
        "model": MODEL_NAME,
        "state": "short",
        "questions": {
            "bad": {
                "type": "span",
                "instructions": "Which spans?",
                "criteria": ["PERSON", "EMAIL_ADDRESS"],
            }
        },
    }


def missing_model_body():
    """An envelope missing the model field (the preamble's 'missing' 422)."""

    return {"state": "short", "questions": {"q": {"type": "noul", "instructions": "any?"}}}


def extra_field_body():
    """An envelope with a field the preamble does not permit ('extra_forbidden')."""

    return {
        "model": MODEL_NAME,
        "state": "short",
        "questions": {"q": {"type": "noul", "instructions": "any?"}},
        "images": [],
    }


def wrong_model_body():
    """Another member's name on this runtime ('value_error.model' 422)."""

    return {
        "model": "vllm-sr/Vela-2.0-0.8B",
        "state": "short",
        "questions": {"q": {"type": "noul", "instructions": "any?"}},
    }


def default_cases():
    return [
        ("example-request", HERE / "example-request.json"),
        ("example-request-set-threshold", HERE / "example-request-set-threshold.json"),
        ("example-request-cjk", HERE / "example-request-cjk.json"),
        ("example-request-score", HERE / "example-request-score.json"),
        ("invalid-span-422", invalid_span_body()),
        ("over-limit-413", over_limit_body()),
        ("missing-model-422", missing_model_body()),
        ("extra-field-422", extra_field_body()),
        ("wrong-model-422", wrong_model_body()),
    ]


def load_cases(paths):
    if not paths:
        cases = default_cases()
    else:
        cases = [(Path(path).name, path) for path in paths]
    return [
        (name, body if isinstance(body, dict) else json.loads(Path(body).read_text(encoding="utf-8")))
        for name, body in cases
    ]


def reference_response(client, body):
    """One request through the pinned upstream server app.

    Returns the (status, parsed-body) pair a client observes.
    """

    response = client.post("/v1/systemone", json=body)
    return response.status_code, response.json()


def worker_response(raw, executor, decide):
    """One request through the production worker pipeline.

    Returns (200, response) or the (status, body) rejection a client
    would observe.
    """

    try:
        return 200, decide(raw, executor)
    except Exception as exc:  # noqa: BLE001 - reported through the comparison
        status = getattr(exc, "status", 500)
        body = getattr(exc, "body", None)
        return status, body if body is not None else {"detail": str(exc)}


def same_result(worker, reference) -> bool:
    """Order-preserving equality of the two observed results.

    ``dict == dict`` ignores insertion order, but key order is part of
    the parity criterion; text dumped from each side keeps that order in
    the comparison.
    """

    return json.dumps(worker, ensure_ascii=False) == json.dumps(reference, ensure_ascii=False)


def compare_case(name, worker, reference):
    """Exact-match comparison of the two observed results.

    Returns (report lines, passed). Key order is part of the match:
    insertion order is the pinned runtime's answer/question order.
    """

    lines = [f"case: {name}", "reference path: pinned vela2_serve app (FastAPI test client)"]
    problems = []
    if same_result(worker, reference):
        status = worker[0]
        lines.append(f"both paths: HTTP {status}")
        if status == 200:
            answers = worker[1].get("answers", {})
            usage = worker[1].get("usage", {})
            lines.append(f"answers: {len(answers)} key(s) in pinned order")
            lines.append(f"input_tokens: {usage.get('input_tokens')} == reference")
        else:
            lines.append(f"rejection body: {json.dumps(worker[1], ensure_ascii=False)[:160]}")
        lines.append("exact match (status, body, key order): PASS")
        return lines, True
    problems.append("results differ")
    if worker[0] != reference[0]:
        problems.append(f"status {worker[0]} != {reference[0]}")
    lines.append(f"worker: HTTP {worker[0]}")
    lines.append(f"reference: HTTP {reference[0]}")
    lines.append("exact match (status, body, key order): FAIL: " + "; ".join(problems))
    return lines, False


def export_allowed(ok, worker):
    """The oracle freezes served 200 responses, never a rejection."""

    return ok and worker[0] == 200


def export_oracle(name, body, response, predict, path):
    """Freeze one passing case for the future native encoder slices."""

    oracle = {
        "case": name,
        "request": body,
        "system_one": response,
        "predict": predict,
    }
    Path(path).write_text(json.dumps(oracle, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return oracle


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from fastapi.testclient import TestClient

    from frontend.vela import decide
    from models.vela.engine import Vela2Executor

    executor = Vela2Executor(args.model, device=args.device)
    # Both runtime imports stay inside a no-bytecode window: the package
    # gate refuses existing caches, and vela2_serve is a fresh import that
    # would otherwise write the __pycache__ the next worker start rejects.
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import vela2_inference
        import vela2_serve
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode

    app = vela2_serve.create_app(executor.engine, model_name=executor.model_name, api_key=None)
    client = TestClient(app)

    exported = 0
    failed = 0
    try:
        for name, body in load_cases(args.fixture):
            raw = json.dumps(body).encode("utf-8")
            worker = worker_response(raw, executor, decide)
            reference = reference_response(client, body)
            lines, ok = compare_case(name, worker, reference)
            print("\n".join(lines))
            if not ok:
                failed += 1
            elif args.export and export_allowed(ok, worker):
                req, _plan = vela2_inference.system_one_to_predict(body["state"], body["questions"])
                target = Path(args.export)
                if target.is_dir():
                    target = target / f"oracle-{name}.json"
                export_oracle(name, body, worker[1], executor.engine.predict(req), target)
                exported += 1
            print()
    finally:
        executor.close()
    if args.export:
        print(f"oracle cases exported: {exported}" if exported else
              "oracle export skipped: no passing 200-response comparison to freeze")
    print("compare_reference: ok" if not failed else f"compare_reference: {failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
