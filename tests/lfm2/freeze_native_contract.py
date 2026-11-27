"""Extract small Rust replay fixtures from the approved Q12 captures, with no forward."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recipe/lfm2"))
import _diagnostics as D
import _harness as H

PUBLIC_HASH = "1e8786bb3ffb29f56ffb55090e0dc3206ab4dd7ec0fd2994cb597cfc185206f4"
INPUT_HASH = "ca4a3723ad5d3fd80ebfba3a6908d3c283a0783b83fe15328fddb622742b23bd"
VERIFY_HASH = "908afa213373916f8ceae149b53a3db3a548d4e0c44e09fa94407f80940fe104"
PROFILE = json.loads((ROOT / "src/models/lfm2/native/profile.json").read_bytes())


def load_capture(path, digest):
    assert H.sha256_file(path) == digest, path
    return json.loads(path.read_bytes())


def fragment(fragments, text, ids):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    existing = fragments.setdefault(digest, ids)
    assert existing == ids
    return digest


def public_cases(fragments, capture, frozen_inputs):
    cases = []
    assert (capture["requests"], capture["questions"], capture["candidate_branches"]) == (5, 6, 266)
    for run, inputs in zip(capture["cases"], frozen_inputs, strict=True):
        assert run["case"] == inputs["case"] and run["request"] == inputs["request"]
        questions = []
        answers = {}
        for recorded, plan in zip(run["questions"], inputs["questions"], strict=True):
            assert recorded["qid"] == plan["qid"]
            prefix = recorded["prefix_token_ids"]
            assert prefix == plan["prefix_ids"]
            branches = []
            for branch, expected, score in zip(
                    recorded["branches"], plan["branches"], recorded["oracle_scores"], strict=True):
                suffix = branch["suffix_ids"]
                ids = branch["value_ids"]
                assert (branch["value"], suffix, ids) == (
                    expected["value"], expected["suffix_ids"], expected["value_ids"])
                assert branch["value"] == score["value"] and ids == score["token_ids"]
                start = len(prefix) + len(suffix) - 1
                assert score["score_logit_positions"] == list(range(start, start + len(ids)))
                assert branch.get("first_score_input_position", start) == start and score["newline_included"]
                assert branch.get("value_text", score["value_text"]) == score["value_text"]
                branches.append(fragment(fragments, score["value_text"], ids))
            fields = [{
                "name": "answer",
                "suffix": fragment(fragments, '  "answer": ', branch["suffix_ids"]),
                "values": branches,
                "scores": [item["log_likelihood"] for item in recorded["oracle_scores"]],
            }]
            assert recorded["input_tokens"] == plan["input_tokens"]
            questions.append({
                "id": plan["qid"], "plan": {
                    "prefix": fragment(fragments, plan["prompt"], prefix),
                    "fields": fields, "input_tokens": plan["input_tokens"],
                },
                "batch8_scores": [item["log_likelihood"]
                                  for item in recorded["batches"]["8"]["result"]["scores"]["answer"]],
            })
            answers[plan["qid"]] = recorded["oracle_response"]
        worker = run["actual_worker_response_batch8"]
        assert worker["usage"] == {"input_tokens": inputs["input_tokens"], "output_tokens": 0}
        cases.append({
            "id": run["case"], "request": run["request"], "questions": questions,
            "oracle_response": {"model": worker["model"], "answers": answers, "usage": worker["usage"]},
            "batch8_response": worker,
        })
    return cases


def internal_cases(fragments, report, checkpoint):
    assert H.sha256_file(checkpoint / "tokenizer.json") == PROFILE["checkpoint"]["tokenizer.json"]
    for name in ("tokenizer_config.json", "config.json", "chat_template.jinja"):
        assert H.sha256_file(checkpoint / name) == PROFILE["checkpoint"][name]

    # Use the exact Python prompt/encoding methods, without constructing a model.
    module = H.load_target_engine(H.default_engine_dir())
    assert H.sha256_file(Path(module.__file__)) == (
        "2ee59f35c84641587d73107f1297f5790febb1144a0b818410e583efdffff521")
    engine = module.Engine.__new__(module.Engine)
    engine.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    cases = []
    results = report["sections"]["score_parity"]["results"]
    assert len(results) == 17
    for (case_id, schema, state, _), run in zip(D.CASES + D.STRESS_CASES, results, strict=True):
        assert case_id == run["id"]
        prompt = engine.prompt(state, schema)
        prefix = engine.encode(prompt)
        assert H.sha256_json(prefix) == run["prefix_token_ids_sha256"]
        assert len(prefix) == run["oracle"]["prompt_tokens"]
        fields = []
        rows = []
        selected = {}
        for name, values in H.field_specs(schema):
            suffix = "  " + json.dumps(name, ensure_ascii=False) + ": "
            suffix_ids = engine.encode(suffix)
            value_fragments = []
            scores = []
            for value, actual in zip(values, run["oracle"]["scores"][name], strict=True):
                text = json.dumps(value, ensure_ascii=False) + "\n"
                ids = engine.encode(text)
                assert actual["value"] == value and actual["token_ids"] == ids
                rows.append(suffix_ids + ids)
                value_fragments.append(fragment(fragments, text, ids))
                scores.append(actual["log_likelihood"])
            selected[name] = values[max(range(len(values)), key=scores.__getitem__)]
            fields.append({"name": name, "suffix": fragment(fragments, suffix, suffix_ids),
                           "values": value_fragments, "scores": scores})
        assert len(rows) == run["candidate_count"]
        assert selected == run["reference"]["selected"]
        for batch in run["batches"].values():
            expected = [[prefix]]
            size = batch["candidate_batch_size"]
            for i in range(0, len(rows), size):
                chunk = rows[i:i + size]
                width = max(map(len, chunk))
                expected.append([row + [engine.tokenizer.pad_token_id] * (width - len(row)) for row in chunk])
            assert H.sha256_json(expected) == batch["forward_input_ids_sha256"]
            assert selected == batch["selected"]
        cases.append({
            "id": case_id, "state": state, "schema": schema, "selected": selected,
            "plan": {
                "prefix": fragment(fragments, prompt, prefix), "fields": fields,
                "input_tokens": len(prefix) + sum(map(len, rows)),
            },
        })
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path, help="directory containing the frozen Q12 runs")
    parser.add_argument("checkpoint", type=Path, help="local pinned tokenizer assets")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.capture_root
    fragments = {}
    inputs = load_capture(root / "fp32-public/runs/public-inputs.json", INPUT_HASH)
    capture = load_capture(root / "fp32-public/runs/public-oracle.json", PUBLIC_HASH)
    report = load_capture(root / "runs/fp32.verify.json", VERIFY_HASH)
    public = public_cases(fragments, capture, inputs)
    internal = internal_cases(fragments, report, args.checkpoint)
    fixture = {"schema": "lfm2-q12-replay-v1",
               "public_input_sha256": INPUT_HASH, "public_capture_sha256": PUBLIC_HASH,
               "verifier_capture_sha256": VERIFY_HASH,
               "public": public, "internal": internal, "fragments": fragments}
    def compact(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    # Keep each independent case and token fragment on its own line.
    lines = []
    for key, value in fixture.items():
        encoded = compact(value)
        if isinstance(value, list):
            encoded = "[\n    " + ",\n    ".join(map(compact, value)) + "\n  ]"
        elif isinstance(value, dict):
            entries = [compact(name) + ": " + compact(ids) for name, ids in value.items()]
            encoded = "{\n    " + ",\n    ".join(entries) + "\n  }"
        lines.append("  " + compact(key) + ": " + encoded)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("{\n" + ",\n".join(lines) + "\n}\n")


if __name__ == "__main__":
    main()
