"""Response archives must retain and validate the bytes received by HTTPX."""

import base64
import copy
import hashlib
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
from test_saved_protocols import jev_response, replace_config

from benchmarks import bench

FORMAT = "httpx-content-base64-utf8-v1"


def body_for(protocol, note="original"):
    payload = (
        response_for(CASES[0]) if protocol == "systemone" else jev_response(CASES[0])
    )
    return json.dumps(payload | {"note": note}, ensure_ascii=False).encode("utf-8")


class ResponseBytesTests(unittest.IsolatedAsyncioTestCase):
    async def collect(
        self, root, raw, protocol="systemone", status=200, warmup=0, headers=None
    ):
        args = run_args(root)
        args.protocol = protocol
        args.warmup = warmup
        args.manifest = root / "manifest.jsonl"
        args.manifest.write_text(json.dumps(CASES[0]) + "\n")

        def handler(request):
            if raw is None:
                raise httpx.ConnectError(
                    "controlled transport failure", request=request
                )
            return httpx.Response(status, content=raw, headers=headers)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with (
            patch.object(bench.httpx, "AsyncClient", return_value=client),
            redirect_stdout(io.StringIO()),
        ):
            if warmup:
                with self.assertRaisesRegex(ValueError, "readiness/warmup failed"):
                    await bench.run(args, [CASES[0]])
            else:
                await bench.run(args, [CASES[0]])
        return args.output

    def records(self, path):
        return [
            json.loads(line)
            for line in (path / "responses.jsonl").read_text().splitlines()
        ]

    def assert_original_bytes(self, record, raw):
        self.assertEqual(
            base64.b64decode(record["response_bytes_base64"], validate=True), raw
        )
        self.assertEqual(record["response_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(record["response"], raw.decode("utf-8", errors="replace"))

    def assert_cli_summary(self, path, expected):
        result = subprocess.run(
            [sys.executable, str(Path(bench.__file__)), "summary", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), expected)

    async def test_invalid_utf8_roundtrips_as_failed_response_in_both_protocols(self):
        for protocol in ("systemone", "jev-vl"):
            with (
                self.subTest(protocol=protocol),
                tempfile.TemporaryDirectory() as directory,
            ):
                raw = body_for(protocol).replace(b"original", b"\xff")
                path = await self.collect(Path(directory), raw, protocol)
                record = self.records(path)[0]
                self.assertEqual(record["error_kind"], "invalid_response")
                self.assertEqual(record["answers"], {})
                self.assert_original_bytes(record, raw)
                config = json.loads((path / "config.json").read_text())
                self.assertEqual(config["response_format"], FORMAT)
                summary = json.loads((path / "summary.json").read_text())
                self.assertEqual(summary["errors"], {"invalid_response": 1})
                self.assertEqual(summary["successful_requests"], 0)
                self.assertEqual(summary, bench.summarize_saved(path))
                self.assert_cli_summary(path, summary)

    async def test_utf8_decoding_ignores_a_conflicting_http_charset(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = body_for("systemone", "原始 café 😀")
            path = await self.collect(
                Path(directory),
                raw,
                headers={"content-type": "application/json; charset=latin-1"},
            )
            record = self.records(path)[0]
            self.assertIsNone(record["error_kind"])
            self.assert_original_bytes(record, raw)
            self.assertEqual(bench.summarize_saved(path)["successful_requests"], 1)

    async def test_empty_body_bom_and_unusual_http_statuses_preserve_bytes(self):
        for raw, status, kind in (
            (b"", 200, "invalid_response"),
            (b"\xef\xbb\xbf" + body_for("systemone"), 200, "invalid_response"),
            (b"\xff", 600, "http_600"),
            (b"\xfe", 999, "http_999"),
        ):
            with (
                self.subTest(status=status, raw=raw[:3]),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = await self.collect(Path(directory), raw, status=status)
                record = self.records(path)[0]
                self.assertEqual(record["error_kind"], kind)
                self.assert_original_bytes(record, raw)
                self.assertEqual(bench.summarize_saved(path)["errors"], {kind: 1})

    async def test_no_http_body_is_distinct_from_an_empty_body(self):
        with tempfile.TemporaryDirectory() as directory:
            path = await self.collect(Path(directory), None)
            record = self.records(path)[0]
            self.assertEqual(record["error_kind"], "transport")
            for field in ("response", "response_bytes_base64", "response_sha256"):
                self.assertIsNone(record[field])
            self.assertEqual(bench.summarize_saved(path)["errors"], {"transport": 1})

    async def test_inner_byte_evidence_is_checked_after_outer_checksum_is_resigned(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = await self.collect(Path(directory), body_for("systemone"))
            records = self.records(path)
            mutations = (
                {"response_bytes_base64": "not*base64"},
                {"response_bytes_base64": 1},
                {"response_bytes_base64": None},
                {"response_sha256": hashlib.sha256(b"other").hexdigest()},
                {"response": records[0]["response"] + " "},
            )
            for change in mutations:
                with self.subTest(change=change):
                    changed = copy.deepcopy(records)
                    changed[0].update(change)
                    (path / "responses.jsonl").write_text(json.dumps(changed[0]) + "\n")
                    state = json.loads((path / "completion.json").read_text())
                    state["responses_sha256"] = bench.digest(path / "responses.jsonl")
                    bench.write_json(path / "completion.json", state)
                    with self.assertRaises(ValueError):
                        bench.summarize_saved(path)

    async def test_unknown_response_format_is_not_guessed(self):
        for value in ("future-format-v99", None, 1):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = await self.collect(Path(directory), body_for("systemone"))
                replace_config(
                    path,
                    lambda config, value=value: config.update(response_format=value),
                )
                with self.assertRaisesRegex(ValueError, "response|format"):
                    bench.summarize_saved(path)

    async def test_legacy_text_archive_keeps_its_replacement_decoding_semantics(self):
        for explicit in (False, True):
            with (
                self.subTest(explicit=explicit),
                tempfile.TemporaryDirectory() as directory,
            ):
                raw = body_for("systemone").replace(b"original", b"\xff")
                path = await self.collect(Path(directory), raw)
                record = self.records(path)[0]
                record.pop("response_bytes_base64", None)
                record["response"] = httpx.Response(200, content=raw).text
                record["response_sha256"] = hashlib.sha256(
                    record["response"].encode()
                ).hexdigest()
                record["error"] = record["error_kind"] = None
                record["answers"] = bench.wire_answers(
                    CASES[0], json.loads(record["response"]), "english", "systemone"
                )
                (path / "responses.jsonl").write_text(json.dumps(record) + "\n")
                state = json.loads((path / "completion.json").read_text())
                state["responses_sha256"] = bench.digest(path / "responses.jsonl")
                bench.write_json(path / "completion.json", state)

                def legacy_config(config, explicit=explicit):
                    config.pop("response_format", None)
                    if explicit:
                        config["response_format"] = "httpx-text-v1"

                replace_config(path, legacy_config)
                before = {
                    name: (path / name).read_bytes()
                    for name in ("config.json", "responses.jsonl")
                }
                summary = bench.summarize_saved(path)
                self.assertEqual(summary["successful_requests"], 1)
                self.assertEqual(summary["failed_requests"], 0)
                self.assert_cli_summary(path, summary)
                self.assertEqual(
                    before, {name: (path / name).read_bytes() for name in before}
                )

    async def test_failed_warmup_retains_the_original_bytes_and_unattempted_denominator(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            raw = body_for("systemone").replace(b"original", b"\xff")
            path = await self.collect(Path(directory), raw, warmup=1)
            record = json.loads((path / "warmup.json").read_text())[0]
            self.assertEqual(record["error_kind"], "invalid_response")
            self.assert_original_bytes(record, raw)
            self.assertFalse((path / "responses.jsonl").exists())
            summary = bench.summarize_saved(path)
            self.assertEqual(summary["attempted_requests"], 0)
            self.assertEqual(summary["unattempted_requests"], 1)
            self.assertFalse(summary["evidence_complete"])
            self.assert_cli_summary(path, summary)


if __name__ == "__main__":
    unittest.main()
