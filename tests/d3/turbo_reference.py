"""Opt-in fixed-package turbo oracle, preparation fixtures and HTTP baseline.

Run from the repository root with PYTHONPATH=src. No checkpoint or accelerator
is touched on import. The ``serve`` command uses the package's HTTP transport
and public AutoModel; it is comparison infrastructure, not another production
worker. ``freeze`` compares that oracle with the production worker adapter
before writing any fixture. Native execution is outside this tool's scope.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

TOLERANCE = 1e-6
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def numeric_delta(expected, actual, path="response"):
    """Exact structure/order/integers/strings; predeclared tolerance for floats."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and list(expected) == list(actual), f"{path}: fields/order"
        return max((numeric_delta(value, actual[key], f"{path}.{key}")
                    for key, value in expected.items()), default=0.0)
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), f"{path}: length"
        return max((numeric_delta(a, b, f"{path}[{i}]")
                    for i, (a, b) in enumerate(zip(expected, actual))), default=0.0)
    if isinstance(expected, float):
        assert isinstance(actual, (float, int)) and not isinstance(actual, bool), path
        assert math.isfinite(expected) and math.isfinite(actual), f"{path}: nonfinite"
        delta = abs(expected - actual)
        assert delta <= TOLERANCE, f"{path}: drift {delta} > {TOLERANCE}"
        return delta
    assert type(expected) is type(actual) and expected == actual, f"{path}: value"
    return 0.0


def compare_response(expected, actual):
    assert expected["usage"] == actual["usage"], "usage"
    for key, answer in expected["answers"].items():
        other = actual["answers"][key]
        if "error" in answer:
            assert answer == other, f"{key}: typed error"
        elif answer["type"] == "choice":
            assert answer["choice"] == other.get("choice"), f"{key}: decision"
        elif answer["type"] == "noul":
            assert (answer["noul"] >= 0.5) == (other.get("noul", -1) >= 0.5), f"{key}: decision"
    return numeric_delta(expected, actual)


def compare_fixture(expected, actual):
    if "request" in expected:
        assert expected["request"] == actual["request"], "request differs"
    assert expected["preparation"] == actual["preparation"], "preparation differs"
    numeric_delta(expected["probabilities"], actual["probabilities"], "probabilities")
    compare_response(expected["response"], actual["response"])


def questions():
    return {
        "z-route": {"type": "choice", "instructions": "Route the ticket.", "criteria": {
            "billing": "A billing or refund question", "technical": "A technical problem"}},
        "a-urgent": {"type": "noul", "instructions": "Does the state explicitly report urgency?"},
        "m-severity": {"type": "score", "instructions": "Rate the severity.",
                       "criteria": ["low", "medium", "high"]},
    }


def fixture_requests():
    """Synthetic cases; the generated checkerboards are owned by this test suite."""
    cases = {
        "text": {"state": "Please refund a duplicate charge. This is not urgent.", "questions": questions()},
        "json": {"state": {"ticket": "重复扣费", "urgent": True, "amount": 2,
                           "extra": [None, "<|im_start|>"]}, "questions": questions()},
        "typed-error": {"state": None, "questions": {
            "bad-choice": {"type": "choice", "criteria": {}},
            "valid-noul": {"type": "noul", "instructions": "Is the state empty?"}}},
        "over-limit": {"state": "token " * 32769, "questions": {
            "over": {"type": "noul", "instructions": "Is there text?"}}},
    }
    for suffix, mime in (("png", "png"), ("jpg", "jpeg")):
        data = base64.b64encode((FIXTURES / f"turbo-scene.{suffix}").read_bytes()).decode("ascii")
        cases[f"image-{suffix}"] = {
            "state": {"scene": "A synthetic black and white checkerboard."},
            "questions": questions(), "images": [f"data:image/{mime};base64,{data}"],
        }
    return cases


def metadata(model):
    import torch
    from models.d3.turbo import MANIFEST_SHA256, REFERENCE_REPO, REFERENCE_REVISION

    runtime = model.runtime
    return {
        "package": REFERENCE_REPO, "revision": REFERENCE_REVISION,
        "manifest_sha256": MANIFEST_SHA256, "attention_mode": runtime.attention_mode,
        "state_once": runtime.state_once is not None, "readout_dtype": runtime.readout_dtype,
        "max_input_tokens": model.max_input_tokens, "tolerance": TOLERANCE,
        "device": torch.cuda.get_device_name() if runtime.device.type == "cuda" else str(runtime.device),
        "compute_capability": list(torch.cuda.get_device_capability()) if runtime.device.type == "cuda" else None,
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "transformers", "flash-linear-attention", "numpy", "pillow")},
        "kernels": sys.modules[type(runtime).__module__].kernel_report(),
    }


def tensor_record(tensor, name, arrays):
    array = tensor.detach().cpu().contiguous().numpy()
    arrays[name] = array
    return {"array": name, "dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def snapshot(executor, body, name, arrays):
    """Capture fixed runtime preparation and public answers, without changing it."""
    import torch
    from frontend.d3 import decide

    model, runtime = executor.model, executor.runtime
    images = runtime.load_images(body.get("images") or [], strict=True)
    prepared = runtime.prepare(body.get("state"), body["questions"], images)
    prep = {"keys": prepared.keys, "errors": prepared.errors, "state_ends": prepared.states,
            "legal_codes": {}, "rows": {}}
    for key, question in prepared.questions.items():
        prep["legal_codes"][key] = {"codes": runtime.codes[:len(question.keys)],
                                    "answer_keys": list(question.keys), "type": question.kind}
    for key in prepared.runnable:
        text = prepared.texts[key] if images else runtime.text(body.get("state"), prepared.questions[key].rendered)
        if images:
            encoded = runtime.processor(text=[text], images=images, return_tensors="pt")
            ids = encoded["input_ids"][0].tolist()
            with torch.inference_mode():
                _, positions = runtime.image_embeddings({k: v.to(runtime.device) for k, v in encoded.items()})
        else:
            ids = prepared.sequences[key]
            positions = torch.arange(len(ids)).view(1, 1, -1).expand(3, 1, -1)
            encoded = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        end = prepared.states[key]
        tensors = {k: tensor_record(v, f"{name}/{key}/{k}", arrays) for k, v in encoded.items()}
        tensors["full_row_position_ids"] = tensor_record(positions, f"{name}/{key}/full_row_position_ids", arrays)
        # This is the bundled state-once continuation's actual position rule:
        # state positions from the full row, then text steps after its last token.
        state_positions = positions[:, :, :end]
        next_step = state_positions[0, 0, -1] + 1
        question_positions = (next_step + torch.arange(len(ids) - end, device=positions.device)).view(1, 1, -1).expand(3, 1, -1)
        tensors["state_position_ids"] = tensor_record(state_positions, f"{name}/{key}/state_position_ids", arrays)
        tensors["question_position_ids"] = tensor_record(question_positions, f"{name}/{key}/question_position_ids", arrays)
        prep["rows"][key] = {"prompt": text, "token_ids": ids, "state_tokens": ids[:end],
                            "question_tokens": ids[end:], "readout_position": len(ids) - 1,
                            "tensors": tensors}
    probabilities, tokens = runtime.run(prepared)
    direct = model.system_one(state=body.get("state"), questions=body["questions"], images=images)
    assert runtime.respond(prepared, probabilities, tokens) == direct, "prepared/public oracle differ"
    actual = decide(json.dumps(body, ensure_ascii=False).encode("utf-8"), executor)
    compare_response(direct, actual)
    descriptor = {"state_repeat": {"text": "token ", "count": 32769}, "questions": body["questions"]} if name == "over-limit" else body
    return {"request": descriptor, "preparation": prep, "probabilities": probabilities, "response": direct}


def freeze(args):
    from models.d3.turbo import D3TurboExecutor

    arrays = {}
    executor = D3TurboExecutor(args.model, device=args.device, verify=args.verify)
    try:
        executor.warmup()
        document = {"schema": "d3-turbo-oracle/1", "metadata": metadata(executor.model), "cases": {}}
        document["provenance"] = {
            "source_revision": args.source_revision,
            "generator": "tests/d3/turbo_reference.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generation": "D3_STATE_ONCE=1 (or 0) PYTHONPATH=src python tests/d3/turbo_reference.py freeze --model <pinned-package> --source-revision <full-source-SHA> --output <fixture.json>",
            "image_origin": "Synthetic 64x64 checkerboard generated by this test suite; no external asset.",
        }
        for name, body in fixture_requests().items():
            document["cases"][name] = snapshot(executor, body, name, arrays)
        if args.check:
            expected = json.loads(Path(args.check).read_text())
            for field in ("package", "revision", "manifest_sha256", "state_once", "attention_mode"):
                assert expected["metadata"][field] == document["metadata"][field], field
            assert list(expected["cases"]) == list(document["cases"]), "case order"
            for name, value in expected["cases"].items():
                compare_fixture(value, document["cases"][name])
        # Commit the pair only after all comparisons passed; a failing run never
        # overwrites an existing fixture. Tensor bytes are preserved in the NPZ.
        if args.output:
            import numpy as np

            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            tensor_path = target.with_suffix(".npz")
            temporary_tensor = tensor_path.with_suffix(".npz.tmp")
            with temporary_tensor.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            document["tensor_file"] = tensor_path.name
            document["tensor_file_sha256"] = hashlib.sha256(temporary_tensor.read_bytes()).hexdigest()
            temporary_json = target.with_suffix(target.suffix + ".tmp")
            temporary_json.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            temporary_tensor.replace(tensor_path)
            temporary_json.replace(target)
        print(json.dumps({"cases": len(document["cases"]), "tensor_arrays": len(arrays),
                          "checked": bool(args.check), "metadata": document["metadata"]}))
    finally:
        executor.close()


def serve(args):
    from models.d3.engine import put_first_on_sys_path
    from models.d3.turbo import verify_package

    root = Path(args.model).resolve()
    verify_package(root, verify=args.verify)
    put_first_on_sys_path(root)
    sys.dont_write_bytecode = True
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(root), trust_remote_code=True, device=args.device,
                                      batch_size=8, verify=args.verify, local_files_only=True)
    if args.preparation_manifest:
        if not args.prepared_output:
            raise ValueError("--preparation-manifest requires --prepared-output")
        rows = []
        for line in Path(args.preparation_manifest).read_text().splitlines():
            row = json.loads(line)
            body = row["request"]
            prepared = model.runtime.prepare(body.get("state"), body["questions"])
            assert not prepared.errors and len(prepared.keys) == 1, "Public231 must contain valid single-question text requests"
            row["prompt_tokens"] = sum(len(ids) for ids in prepared.sequences.values())
            rows.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
        Path(args.prepared_output).write_text("\n".join(rows) + "\n")
    model.runtime.warmup()
    ready = model.system_one(state="Ready.", questions={"ready": {"type": "noul", "instructions": "Is this a readiness check?"}})
    assert "error" not in ready["answers"]["ready"]
    transport = importlib.import_module("d3_server")

    class Oracle(transport.Decider):
        def answer(self, body):
            call = self.arguments(body)
            with self.turn:
                return model.system_one(**call)

    server = transport.SystemOneServer(("127.0.0.1", args.port), Oracle(model.runtime))
    if args.record:
        Path(args.record).write_text(json.dumps(metadata(model), indent=2) + "\n")
    print("OFFICIAL_AUTOMODEL_READY", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--verify", default="fast", choices=("fast", "full"))
    commands.choices["freeze"].add_argument("--output")
    commands.choices["freeze"].add_argument("--check")
    commands.choices["freeze"].add_argument("--source-revision", required=True,
        help="full repository SHA deployed for this fixture generation/check")
    commands.choices["serve"].add_argument("--port", type=int, default=8000)
    commands.choices["serve"].add_argument("--record")
    commands.choices["serve"].add_argument("--preparation-manifest")
    commands.choices["serve"].add_argument("--prepared-output")
    args = parser.parse_args()
    if args.command == "freeze" and (len(args.source_revision) != 40 or any(c not in "0123456789abcdef" for c in args.source_revision)):
        parser.error("--source-revision must be a full lowercase hexadecimal commit SHA")
    {"freeze": freeze, "serve": serve}[args.command](args)


if __name__ == "__main__":
    main()
