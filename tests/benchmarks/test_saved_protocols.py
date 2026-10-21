"""Online collection and saved replay must use the same declared wire contract."""

import asyncio
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx
from test_bench import CASES, response_for, run_args

from benchmarks import bench

VALIDATION = "probabilities-4dp-model-score-v1"
LEGACY_VALIDATION = "systemone-4dp-score-abs1e-4-v1"


def score_case(target=0.88, count=2):
    return {
        "id": "rounded-score",
        "request": {
            "state": "A saved CPU response fixture",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "Choose a level",
                    "criteria": [str(i) for i in range(count)],
                }
            },
        },
        "expected": {"q": target},
    }


def score_response(value, probabilities):
    return {
        "answers": {
            "q": {"type": "score", "score": value, "probabilities": probabilities}
        }
    }


def jev_response(case):
    question = next(iter(case["request"]["questions"].values()))
    options = [f"{key}: {value}" for key, value in question["criteria"].items()]
    return {
        "kind": "choice",
        "effective_kind": "choice",
        "options": options,
        "probabilities": [1.0, 0.0],
        "choice_index": 0,
        "choice": options[0],
    }


def replace_config(path, change):
    config = json.loads((path / "config.json").read_text())
    change(config)
    bench.write_json(path / "config.json", config)
    state = json.loads((path / "completion.json").read_text())
    state["config_sha256"] = bench.digest(path / "config.json")
    bench.write_json(path / "completion.json", state)


class SavedProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, root, cases, payload, protocol="systemone", decimals=4):
        args = run_args(root)
        args.manifest = root / "manifest.jsonl"
        args.manifest.write_text("".join(json.dumps(case) + "\n" for case in cases))
        args.protocol = protocol
        args.score_decimals = decimals
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        )
        with (
            patch.object(bench.httpx, "AsyncClient", return_value=client),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(await bench.run(args, cases), 0)
        return args.output

    def assert_cli_summary(self, path, expected):
        result = subprocess.run(
            [sys.executable, str(Path(bench.__file__)), "summary", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), expected)

    async def test_protocol_and_precision_roundtrip_through_warmup_and_cli(self):
        for protocol in ("systemone", "jev-vl"):
            for decimals in (2, 3, 4):
                with (
                    self.subTest(protocol=protocol, decimals=decimals),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    scalar = round(0.8766, decimals)
                    case = score_case(scalar) if protocol == "systemone" else CASES[0]
                    payload = (
                        score_response(scalar, {"0": 0.1234, "1": 0.8766})
                        if protocol == "systemone"
                        else jev_response(case)
                    )
                    path = await self.collect(
                        Path(directory), [case], payload, protocol, decimals
                    )
                    config = json.loads((path / "config.json").read_text())
                    self.assertEqual(config["protocol"], protocol)
                    self.assertEqual(config["score_decimals"], decimals)
                    self.assertEqual(config["validation_protocol"], VALIDATION)
                    warmup = json.loads((path / "warmup.json").read_text())
                    records = [
                        json.loads(row)
                        for row in (path / "responses.jsonl").read_text().splitlines()
                    ]
                    for record in [*warmup, *records]:
                        self.assertIsNone(record["error"])
                        self.assertEqual(
                            record["response_sha256"],
                            bench.response_digest(record["response"]),
                        )
                    summary = json.loads((path / "summary.json").read_text())
                    self.assertTrue(summary["evidence_complete"])
                    self.assertEqual(summary["successful_requests"], 1)
                    self.assertEqual(bench.summarize_saved(path), summary)
                    self.assert_cli_summary(path, summary)

    async def test_unknown_saved_contracts_are_rejected_after_config_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = await self.collect(
                Path(directory), [CASES[0]], response_for(CASES[0])
            )
            config = json.loads((path / "config.json").read_text())
            mutations = (
                ("protocol", "unknown"),
                ("protocol", None),
                ("validation_protocol", "future-v999"),
                ("validation_protocol", None),
                ("score_decimals", 9),
                ("score_decimals", "2"),
                ("score_decimals", 2.0),
                ("score_decimals", True),
                ("score_decimals", None),
            )
            for key, value in mutations:
                with self.subTest(key=key, value=value):
                    replace_config(
                        path,
                        lambda saved, key=key, value=value: saved.update(
                            config | {key: value}
                        ),
                    )
                    with self.assertRaisesRegex(
                        ValueError, "protocol|precision|score_decimals|validation"
                    ):
                        bench.summarize_saved(path)

    async def test_missing_contract_fields_keep_legacy_systemone_four_decimal_defaults(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = await self.collect(
                Path(directory), [CASES[0]], response_for(CASES[0])
            )
            replace_config(
                path,
                lambda config: [
                    config.pop(key)
                    for key in ("protocol", "score_decimals", "validation_protocol")
                ],
            )
            summary = bench.summarize_saved(path)
            self.assertTrue(summary["evidence_complete"])
            self.assertEqual(summary["failed_requests"], 0)
            self.assert_cli_summary(path, summary)

    async def test_legacy_score_failure_is_not_reinterpreted_by_the_new_rounding_rule(
        self,
    ):
        case = score_case(2.003, 4)
        payload = score_response(
            2.003, {"0": 0.0337, "1": 0.0974, "2": 0.701, "3": 0.1678}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = await self.collect(Path(directory), [case], payload)
            records = [
                json.loads(row)
                for row in (path / "responses.jsonl").read_text().splitlines()
            ]
            records[0].update(
                answers={},
                error="score must equal the expected level",
                error_kind="invalid_response",
            )
            (path / "responses.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            state = json.loads((path / "completion.json").read_text())
            state["responses_sha256"] = bench.digest(path / "responses.jsonl")
            bench.write_json(path / "completion.json", state)
            replace_config(
                path,
                lambda config: [
                    config.pop(key)
                    for key in ("protocol", "score_decimals", "validation_protocol")
                ],
            )
            summary = bench.summarize_saved(path)
            self.assertTrue(summary["evidence_complete"])
            self.assertEqual(summary["errors"], {"invalid_response": 1})
            self.assertEqual(summary["successful_requests"], 0)
            self.assert_cli_summary(path, summary)
            replace_config(
                path, lambda config: config.update(validation_protocol=VALIDATION)
            )
            with self.assertRaisesRegex(ValueError, "saved validation result disagree"):
                bench.summarize_saved(path)

    async def test_legacy_contract_cannot_claim_another_precision_or_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = await self.collect(
                Path(directory), [CASES[0]], response_for(CASES[0])
            )
            config = json.loads((path / "config.json").read_text())
            for change in ({"protocol": "jev-vl"}, {"score_decimals": 2}):
                with self.subTest(change=change):
                    replace_config(
                        path,
                        lambda saved, change=change: saved.update(
                            config | {"validation_protocol": LEGACY_VALIDATION} | change
                        ),
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "legacy|protocol|precision|score_decimals|validation",
                    ):
                        bench.summarize_saved(path)

    async def test_raw_hash_and_torn_snapshot_checks_apply_to_both_protocols(self):
        for protocol in ("systemone", "jev-vl"):
            with (
                self.subTest(protocol=protocol),
                tempfile.TemporaryDirectory() as directory,
            ):
                payload = (
                    response_for(CASES[0])
                    if protocol == "systemone"
                    else jev_response(CASES[0])
                )
                path = await self.collect(
                    Path(directory), [CASES[0]], payload, protocol
                )
                responses = (path / "responses.jsonl").read_bytes()
                state_bytes = (path / "completion.json").read_bytes()
                records = [json.loads(row) for row in responses.decode().splitlines()]
                records[0]["response"] += " "
                (path / "responses.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
                state = json.loads(state_bytes)
                state["responses_sha256"] = bench.digest(path / "responses.jsonl")
                bench.write_json(path / "completion.json", state)
                with self.assertRaisesRegex(ValueError, "raw response checksum"):
                    bench.summarize_saved(path)
                (path / "responses.jsonl").write_bytes(responses + b'{"id":')
                (path / "completion.json").write_bytes(state_bytes)
                with self.assertRaisesRegex(ValueError, "saved responses checksum"):
                    bench.summarize_saved(path)

    async def test_interrupted_dual_protocol_runs_recompute_without_claiming_completion(
        self,
    ):
        for protocol in ("systemone", "jev-vl"):
            with (
                self.subTest(protocol=protocol),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                cases = [CASES[0], {**CASES[0], "id": "uncompleted-choice"}]
                args = run_args(root)
                args.protocol = protocol
                args.score_decimals = 4
                args.manifest = root / "manifest.jsonl"
                args.manifest.write_text(
                    "".join(json.dumps(case) + "\n" for case in cases)
                )
                waiting = asyncio.Event()
                calls = 0

                async def handler(request, protocol=protocol, waiting=waiting):
                    nonlocal calls
                    calls += 1
                    if calls < 3:
                        payload = (
                            response_for(CASES[0])
                            if protocol == "systemone"
                            else jev_response(CASES[0])
                        )
                        return httpx.Response(200, json=payload)
                    waiting.set()
                    await asyncio.Event().wait()

                client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                with (
                    patch.object(bench.httpx, "AsyncClient", return_value=client),
                    redirect_stdout(io.StringIO()),
                ):
                    task = asyncio.create_task(bench.run(args, cases))
                    await asyncio.wait_for(waiting.wait(), 2)
                    task.cancel()
                    self.assertEqual(await task, 1)
                summary = json.loads((args.output / "summary.json").read_text())
                self.assertFalse(summary["evidence_complete"])
                self.assertEqual(summary["failed_requests"], 0)
                self.assertEqual(summary["incomplete_requests"], 1)
                self.assertEqual(summary, bench.summarize_saved(args.output))
                self.assert_cli_summary(args.output, summary)


def choice_case(case_id):
    return {
        "id": case_id,
        "request": {
            "state": "A saved CPU state fixture",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Pick one",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
        },
        "expected": {"q": "a"},
    }


def multi_question_case():
    case = choice_case("two-questions")
    case["request"]["questions"]["r"] = {
        "type": "choice",
        "instructions": "Pick another",
        "criteria": {"a": "A", "b": "B"},
    }
    case["expected"]["r"] = "a"
    return case


class WirePreflightTests(unittest.IsolatedAsyncioTestCase):
    def jev_args(self, root):
        args = run_args(root)
        args.manifest = root / "manifest.jsonl"
        args.protocol = "jev-vl"
        return args

    async def assert_rejected_before_any_artifact(self, args, cases):
        with self.assertRaisesRegex(
            ValueError, "incompatible with the jev-vl wire protocol"
        ):
            await bench.run(args, cases)
        self.assertFalse(args.output.exists())

    async def test_incompatible_cases_are_rejected_before_any_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.jev_args(root)
            args.manifest.write_text(json.dumps(choice_case("ok")) + "\n")
            await self.assert_rejected_before_any_artifact(
                args, [choice_case("ok"), multi_question_case(), score_case()]
            )

    async def test_incompatible_warmup_manifest_is_rejected_before_any_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.jev_args(root)
            args.manifest.write_text(json.dumps(choice_case("ok")) + "\n")
            args.warmup_manifest = root / "warmup.jsonl"
            args.warmup_manifest.write_text(json.dumps(multi_question_case()) + "\n")
            await self.assert_rejected_before_any_artifact(args, [choice_case("ok")])


class AdapterContractTests(unittest.TestCase):
    def test_systemone_missing_or_wrong_answer_type_still_fails(self):
        case = CASES[0]
        qid = next(iter(case["request"]["questions"]))
        for value in (None, "score"):
            payload = copy.deepcopy(response_for(case))
            if value is None:
                payload["answers"][qid].pop("type")
            else:
                payload["answers"][qid]["type"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.wire_answers(case, payload, "test", "systemone")

    def test_jev_normalization_preserves_the_requested_type_for_strict_validation(self):
        case = CASES[0]
        result = bench.wire_answers(case, jev_response(case), "test", "jev-vl")
        self.assertEqual(result, bench.read_answers(case, response_for(case)))
        for field in ("kind", "effective_kind"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                bench.wire_answers(
                    case, jev_response(case) | {field: "score"}, "test", "jev-vl"
                )


if __name__ == "__main__":
    unittest.main()
