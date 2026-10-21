#!/usr/bin/env python3
"""Record the pinned reference's end-to-end answers for the N4/N5 gates.

Serves the five parity fixtures — the text state of the N2 hidden fixture and
the example image request (both ``choice``), the example ``noul`` request,
the example ``score`` request, and one mixed-type body — through the
production reference worker pipeline (``frontend.valen.decide``: protocol,
preprocessing, the pinned executor with the unmerged FP32 language adapters
and the eager BF16 backbone, postprocessing) with a recording wrapper around
the executor, and records, per case:

- the exact request body served,
- the reference response (the parsed JSON the worker returns),
- the per-question FP32 logits the decision head produced, and
- the dual token accounting.

It also records a head probe from the same run: the reference's
``extract_features`` hidden rows (candidate ends + decision end, BF16 bits)
for the text branch and the FP32 logits its head computes on them — the
native CPU head is compared against those rows and logits directly.

The output is the committed fixture consumed by the env-gated Rust test
(``tests/valen/worker_gpu.rs``). Tolerances are measured first and then
declared via ``--tolerance-probability`` / ``--tolerance-confidence``
(``--head-tolerance`` for the probe, ``--tolerance-score`` for the score
questions' expected level); without them the fixture records ``null`` and the
Rust test refuses to pass until they are declared.

Usage (from the repository root, on the GPU host):

    PYTHONPATH=src .venv/bin/python recipe/valen/dump_choice_parity.py \
      --valen-source weights/Valen --checkpoint weights/Valen-Preview-0923 \
      --base weights/Qwen3.5-2B --device cuda

Exit status 0 means the fixture was written.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare_reference as reference  # noqa: E402  (the recipe's fixture helpers)

FIXTURE_FORMAT = "valen-choice-parity/2"
DEFAULT_FIXTURE_NAME = "choice-parity-reference.json"

TEXT_REQUEST = {
    "model": "valen-preview-0923",
    "state": "The QA log lists three runs: r1 passed in 12s, r2 failed on "
    "a timeout, r3 passed after a retry.",
    "questions": {
        "which_passed": {
            "type": "choice",
            "instructions": "Which runs passed?",
            "criteria": {
                "r1_only": "Only run r1 passed.",
                "r1_and_r3": "Runs r1 and r3 passed.",
            },
        }
    },
}

# The N5 mixed body: one question of each wire type over the example noul
# state, so the all-types chain (compile → execute → finish per kind) is one
# recorded round trip.
MIXED_REQUEST = {
    "model": "valen-preview-0923",
    "state": "The card was charged twice for one order.",
    "questions": {
        "move": {
            "type": "choice",
            "instructions": "Should the duplicate charge be voided?",
            "criteria": {
                "void": "Void the duplicate charge.",
                "keep": "Keep both charges.",
            },
        },
        "duplicate": {
            "type": "noul",
            "instructions": "Was the card charged twice for one order?",
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the billing error?",
            "criteria": [
                "No billing error: every charge is correct.",
                "Minor error: a single duplicated charge.",
                "Major error: repeated duplicated charges.",
            ],
        },
    },
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valen-source", required=True, help="pinned Valen source checkout")
    parser.add_argument("--checkpoint", required=True, help="Valen-Preview-0923 directory")
    parser.add_argument("--base", required=True, help="Qwen3.5-2B base directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--out",
        default=str(HERE / "fixtures" / DEFAULT_FIXTURE_NAME),
        help=f"output fixture path (default: fixtures/{DEFAULT_FIXTURE_NAME})",
    )
    parser.add_argument(
        "--tolerance-probability",
        type=float,
        default=None,
        help="declared absolute tolerance on per-candidate probabilities (native vs reference)",
    )
    parser.add_argument(
        "--tolerance-confidence",
        type=float,
        default=None,
        help="declared absolute tolerance on confidence scalars (native vs reference)",
    )
    parser.add_argument(
        "--head-tolerance",
        type=float,
        default=None,
        help="declared absolute tolerance for the CPU head probe against the recorded rows",
    )
    parser.add_argument(
        "--tolerance-score",
        type=float,
        default=None,
        help="declared absolute tolerance on score questions' expected level (native vs reference)",
    )
    parser.add_argument(
        "--declare-flip",
        action="append",
        default=[],
        metavar="CASE",
        help="declare a recorded answer flip for CASE (text/image): the reference's own "
        "decision margin is near a tie and the declared component tolerances close it; "
        "the declaration is recorded in the fixture and the Rust gate accepts exactly it",
    )
    return parser.parse_args(argv)


def bf16_bits(tensor) -> str:
    import torch

    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16, got {tensor.dtype}")
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy().tobytes().hex()


def main(argv):
    args = parse_args(argv)
    if args.dtype != "bf16":
        raise SystemExit("the native path is BF16; --dtype bf16 only")

    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from frontend.valen import _verify_source_revision, decide
    from models.valen.engine import (
        BASE_REVISION,
        CHECKPOINT_SHA256,
        REFERENCE_REVISION,
        ValenExecutor,
    )
    from models.valen.preprocess import ValenProcessor, prepare_request
    from models.valen.protocol import parse_request

    source = Path(args.valen_source).resolve()
    _verify_source_revision(source)
    reference._verify_clean_checkout(source)
    sys.path.insert(0, str(source))

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint = Path(args.checkpoint).resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    processor = ValenProcessor(
        Path(args.base), int(config["max_length"]), config.get("media_kwargs")
    )
    executor = ValenExecutor(checkpoint, Path(args.base), args.device, args.dtype)

    class Recording:
        """The production executor with its logits kept, per decide() call."""

        model_name = executor.model_name

        def __init__(self):
            self.rows = []

        def execute(self, compiled):
            output = executor.execute(compiled)
            self.rows.append(
                {
                    "logits": [list(map(float, row)) for row in output.logits],
                    "logical_tokens": output.logical_tokens,
                    "compute_tokens": output.compute_tokens,
                }
            )
            return output

        def warmup(self, compiled):
            return executor.warmup(compiled)

        def close(self):
            return executor.close()

    recording = Recording()

    image_request = json.loads((HERE / "example-request.json").read_text(encoding="utf-8"))
    noul_request = json.loads((HERE / "example-noul-request.json").read_text(encoding="utf-8"))
    score_request = json.loads((HERE / "example-score-request.json").read_text(encoding="utf-8"))
    cases = []
    for name, body in (
        ("text", TEXT_REQUEST),
        ("image", image_request),
        ("noul", noul_request),
        ("score", score_request),
        ("mixed", MIXED_REQUEST),
    ):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        response = decide(raw, processor, recording)
        row = recording.rows[-1]
        # The reference's own decision margin per question: |top - runner-up|
        # over the answer's probabilities — the flip-risk figure for every
        # question type (noul and score carry no discrete answer field, and a
        # mode flip moves only gated floats there).
        margins = {}
        for qid, answer in response["answers"].items():
            values = sorted(answer["probabilities"].values(), reverse=True)
            margins[qid] = values[0] - values[1] if len(values) > 1 else 1.0
        cases.append(
            {
                "name": name,
                "request": body,
                "response": response,
                "logits": row["logits"],
                "logical_tokens": row["logical_tokens"],
                "compute_tokens": row["compute_tokens"],
                "reference_decision_margins": margins,
            }
        )

    # The head probe: the reference's captured readout rows for the text
    # branch, plus the FP32 logits its own head computes on exactly them.
    with TemporaryDirectory(prefix="valen-probe-") as directory:
        request = parse_request(
            json.loads(json.dumps(TEXT_REQUEST, ensure_ascii=False)),
            special_tokens=processor.special_tokens,
        )
        prepared = prepare_request(request, Path(directory))
        compiled = processor.compile(prepared)
        with torch.no_grad():
            features = executor.model.extract_features(compiled.state.questions[0])
        rows = features[0][0]
        logits = executor.model.score_features(features)
        probe = {
            "rows_bf16_hex": bf16_bits(rows),
            "row_count": int(rows.shape[0]),
            "hidden_size": int(rows.shape[1]),
            "candidate_count": len(compiled.state.questions[0].branches[0].candidate_positions),
            "logits_f32": [float(v) for v in logits.tolist()],
        }

    device_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device
    capability = (
        ".".join(map(str, torch.cuda.get_device_capability(0)))
        if args.device.startswith("cuda")
        else None
    )
    fixture = {
        "format": FIXTURE_FORMAT,
        "recorded": {
            "date": subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "device": device_name,
            "compute_capability": capability,
            "torch": torch.__version__,
            "valen_revision": REFERENCE_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "base_revision": BASE_REVISION,
            "dtype": args.dtype,
            "tf32": False,
        },
        "cases": cases,
        "head_probe": probe,
        "tolerance": {
            "probability": args.tolerance_probability,
            "confidence": args.tolerance_confidence,
            "score": args.tolerance_score,
            "head_probe": args.head_tolerance,
        },
        "declared_flips": {},
    }
    for name in args.declare_flip:
        case = next(c for c in cases if c["name"] == name)
        fixture["declared_flips"][name] = {
            "note": (
                "the reference's own decision margin on this case is near a tie "
                f"({', '.join(f'{k}: {v:.6f}' for k, v in case['reference_decision_margins'].items())}); "
                "the declared component tolerances (vision tower 1.25 per N1, language "
                "hidden state 0.5 per N2) close it, so answer equality is not gatable "
                "there — the flip is recorded here and the Rust gate accepts exactly "
                "this declaration while still gating every probability and confidence "
                "delta"
            )
        }
    out = Path(args.out)
    out.write_text(json.dumps(fixture, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    for case in cases:
        print(
            f"{case['name']}: logits {case['logits']}, "
            f"logical {case['logical_tokens']}, compute {case['compute_tokens']}"
        )
    executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
