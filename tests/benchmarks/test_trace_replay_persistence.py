"""CPU-only replay persistence regressions using a local HTTP server and mock transport."""

import argparse
import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

TRACES = Path(__file__).resolve().parents[2] / "benchmarks/traces"
sys.path.insert(0, str(TRACES))
import replay  # noqa: E402
import tracefile  # noqa: E402


def request(group, step, state, **extra):
    return {"type": "request", "id": f"{group}:{step}", "group": group, "step": step,
            "request": {"state": state, "questions": {}}, **extra}


def write_trace(path, requests):
    tracefile.write(path, [{"type": "header", "format": tracefile.FORMAT,
                           "capture": "converted"}, *requests])


def responses(out):
    return [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]


class ReplayProcessTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.arrivals = []
        self.held = {name: threading.Event() for name in ("hold-a", "hold-b")}
        self.release = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state = body["state"]
                owner.arrivals.append(state)
                if state in owner.held:
                    owner.held[state].set()
                    owner.release.wait(15)
                raw = json.dumps({"echo": state, "metadata": {"prefix_cache": {"enabled": True}}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *unused):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.release.set)
        self.trace, self.out = self.root / "trace.jsonl", self.root / "out"

    def command(self, *flags):
        return [sys.executable, str(TRACES / "replay.py"), str(self.trace), "--url",
                f"http://127.0.0.1:{self.server.server_port}", "--think-scale", "0",
                "--out", str(self.out), *flags]

    def start(self, *flags):
        process = subprocess.Popen(self.command(*flags), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

        self.addCleanup(cleanup)
        return process

    def check_interruption(self, sequential):
        planned = [request("a", 0, "done"), request("a", 1, "hold-a"),
                   request("a", 2, "later-a"), request("b", 0, "hold-b"),
                   request("b", 1, "later-b")]
        write_trace(self.trace, planned)
        process = self.start("--sequential" if sequential else "--together")
        self.assertTrue(self.held["hold-a"].wait(10), "second request never arrived")
        if not sequential:
            self.assertTrue(self.held["hold-b"].wait(10), "concurrent episode never arrived")
        # Capture the live file before cancellation; completion must not depend on exit.
        live = responses(self.out) if (self.out / "responses.jsonl").exists() else []
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        self.assertNotEqual(process.returncode, 0, (stdout, stderr))
        self.assertEqual([row["id"] for row in live], ["a:0"])
        config = json.loads((self.out / "config.json").read_text())
        self.assertCountEqual(config["planned_ids"], [row["id"] for row in planned])
        rows = responses(self.out)
        self.assertCountEqual([row["id"] for row in rows], [row["id"] for row in planned])
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["a:0"]["state"], "completed")
        self.assertEqual(json.loads(base64.b64decode(by_id["a:0"]["body"]))["echo"], "done")
        self.assertEqual(by_id["a:1"]["state"], "incomplete")
        self.assertEqual(by_id["b:0"]["state"], "unattempted" if sequential else "incomplete")
        for name in ("a:2", "b:1"):
            self.assertEqual(by_id[name]["state"], "unattempted")
            self.assertIsNone(by_id[name]["sent_s"])
        for row in rows:
            if row["state"] != "completed":
                for key in ("status", "latency_ms", "missed", "prefix_cache"):
                    self.assertIsNone(row[key], (row, key))
        summary = json.loads((self.out / "summary.json").read_text())
        expected = {"requests": 5, "ok": 1, "failed": 0, "completed": 1,
                    "incomplete": 1 if sequential else 2, "unattempted": 3 if sequential else 2,
                    "stop_reason": "interrupted"}
        for key, value in expected.items():
            self.assertEqual(summary[key], value, key)
        self.assertEqual(summary["latency_ms"]["mean"], by_id["a:0"]["latency_ms"])
        self.assertEqual(summary["shared_prefix_requests"], 1)

    @unittest.skipUnless(sys.platform != "win32", "requires POSIX SIGINT")
    def test_sigint_preserves_sequential_progress(self):
        self.check_interruption(sequential=True)

    @unittest.skipUnless(sys.platform != "win32", "requires POSIX SIGINT")
    def test_sigint_preserves_all_concurrent_episodes(self):
        self.check_interruption(sequential=False)

    def test_existing_directory_is_rejected_without_modifying_any_bytes_or_sending(self):
        write_trace(self.trace, [request("a", 0, "done")])
        for name in (None, "config.json", "responses.jsonl", "summary.json"):
            with self.subTest(existing=name):
                self.out = self.root / (name or "empty")
                self.out.mkdir()
                if name:
                    (self.out / name).write_bytes(b"original evidence\n")
                before = {p.name: p.read_bytes() for p in self.out.iterdir()}
                result = subprocess.run(self.command(), capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("already exists", result.stderr)
                self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, before)
        self.assertEqual(self.arrivals, [])

    def test_second_process_cannot_reuse_an_active_run_directory(self):
        write_trace(self.trace, [request("a", 0, "hold-a")])
        first = self.start()
        self.assertTrue(self.held["hold-a"].wait(10))
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        second = subprocess.run(self.command(), capture_output=True, text=True, timeout=10)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, before)
        self.assertEqual(self.arrivals, ["hold-a"])
        self.release.set()
        stdout, stderr = first.communicate(timeout=10)
        self.assertEqual(first.returncode, 0, (stdout, stderr))
        self.assertEqual(len(responses(self.out)), 1)


class ReplayFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.args = argparse.Namespace(trace=self.root / "trace.jsonl", out=self.root / "out",
                                       url="http://local.test", model=None, think_scale=0,
                                       together=True, sequential=False, skip_retries=False,
                                       deadline_s=None, timeout=1)

    async def test_internal_error_drains_other_episodes_before_saving_summary(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def handle(request):
            state = json.loads(request.content)["state"]
            if state == "crash":
                await started.wait()
                raise RuntimeError("local transport failed")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        write_trace(self.args.trace, [request("a", 0, "crash"), request("b", 0, "blocked"),
                                      request("b", 1, "later")])
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with patch.object(replay.httpx, "AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "local transport failed"):
                await replay.replay(self.args)
        self.assertTrue(cancelled.is_set())
        rows = responses(self.args.out)
        self.assertEqual(sorted(row["state"] for row in rows), ["incomplete", "incomplete", "unattempted"])
        summary = json.loads((self.args.out / "summary.json").read_text())
        self.assertEqual(summary["stop_reason"], "error")
        self.assertEqual((summary["ok"], summary["failed"], summary["incomplete"], summary["unattempted"]), (0, 0, 2, 1))

    async def test_non_object_cache_metadata_does_not_lose_the_summary(self):
        raw = b'{"metadata":{"prefix_cache":true}}'
        write_trace(self.args.trace, [request("a", 0, "done")])
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=raw)))
        with patch.object(replay.httpx, "AsyncClient", return_value=client):
            await replay.replay(self.args)
        rows = responses(self.args.out)
        self.assertEqual(base64.b64decode(rows[0]["body"]), raw)
        summary = json.loads((self.args.out / "summary.json").read_text())
        self.assertEqual((summary["ok"], summary["shared_prefix_requests"]), (1, 0))

    async def test_http_errors_remain_completed_failures_and_skipped_retries_are_not_planned(self):
        async def handle(req):
            if json.loads(req.content)["state"] == "timeout":
                raise httpx.ReadTimeout("timed out", request=req)
            return httpx.Response(503, content=b"unavailable")

        retry = dict(request("a", 0, "skip"), id="retry", attempt=1, retry_of="a:0")
        write_trace(self.args.trace, [request("a", 0, "http-error"), retry, request("a", 1, "timeout")])
        self.args.skip_retries = True
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with patch.object(replay.httpx, "AsyncClient", return_value=client):
            await replay.replay(self.args)
        rows = responses(self.args.out)
        self.assertEqual([row["id"] for row in rows], ["a:0", "a:1"])
        self.assertTrue(all(row["state"] == "completed" for row in rows))
        self.assertEqual(base64.b64decode(rows[0]["body"]), b"unavailable")
        self.assertIn("ReadTimeout", rows[1]["error"])
        summary = json.loads((self.args.out / "summary.json").read_text())
        self.assertEqual((summary["requests"], summary["completed"], summary["failed"], summary["ok"]), (2, 2, 2, 0))
        self.assertEqual(summary["stop_reason"], "completed")
        self.assertIsNone(summary["latency_ms"]["mean"])


if __name__ == "__main__":
    unittest.main()
