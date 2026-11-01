"""Decision traces: format checks, metrics on synthetic token ids, and a paced local replay."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

TRACES = Path(__file__).resolve().parents[2] / "benchmarks/traces"
sys.path.insert(0, str(TRACES))
import metrics  # noqa: E402
import replay  # noqa: E402
import tracefile  # noqa: E402

HEADER = {"type": "header", "format": tracefile.FORMAT, "capture": "converted"}


def request(group, step, state, attempt=0, retry_of=None, **extra):
    return {"type": "request", "id": f"{group}:s{step}:a{attempt}", "group": group, "step": step,
            "attempt": attempt, "retry_of": retry_of,
            "request": {"state": state, "questions": {"q": {"type": "noul", "instructions": "Done?"}}},
            **extra}


class FormatTests(unittest.TestCase):
    def test_a_valid_trace_reads_back(self):
        records = [HEADER, request("e", 0, "a"), request("e", 0, "a", 1, "e:s0:a0"),
                   request("e", 1, "b"), {"type": "episode_end", "group": "e", "steps": 2}]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "t.jsonl"
            tracefile.write(path, records)
            header, requests, ends = tracefile.read(path)
        self.assertEqual([r["id"] for r in requests], ["e:s0:a0", "e:s0:a1", "e:s1:a0"])
        self.assertEqual(list(ends), ["e"])

    def test_malformed_traces_name_the_problem(self):
        cases = {
            "starts with one header": [request("e", 0, "a")],
            "duplicate id": [HEADER, request("e", 0, "a"), request("e", 0, "a")],
            "must increase": [HEADER, request("e", 1, "a"), request("e", 0, "b")],
            "retries name": [HEADER, request("e", 0, "a"), request("e", 0, "a", 1)],
            "already ended": [HEADER, request("e", 0, "a"), {"type": "episode_end", "group": "e"},
                              request("e", 1, "b")],
            "state and questions": [HEADER, {**request("e", 0, "a"), "request": {"state": "a"}}],
            "nonnegative number": [HEADER, request("e", 0, "a", think_ms=-1)],
        }
        for message, records in cases.items():
            with self.assertRaisesRegex(ValueError, message):
                tracefile.check(records)

    def test_ticks_convert_to_one_episode_without_timing(self):
        ticks = [json.dumps({"state": {"page": i}, "questions": {"q": {"type": "noul", "instructions": "x"}},
                             "answers": {"q": "true"}, "model": "m"}) for i in range(3)]
        records = tracefile.from_ticks(ticks, "ep", {"repo": "r"})
        _, requests, ends = tracefile.check(records)
        self.assertEqual([r["step"] for r in requests], [0, 1, 2])
        self.assertTrue(all(r["think_ms"] is None for r in requests))
        self.assertEqual(ends["ep"]["steps"], 3)


class Layout:
    """Token ids straight from the request: the state's list, then one row per candidate."""

    def rows(self, request):
        return [[request["state"] + c for c in q["candidates"]] for q in request["questions"].values()]

    def state(self, request):
        return request["state"]


def tokens_request(group, step, state, candidates, attempt=0, retry_of=None, images=(), **extra):
    record = request(group, step, state, attempt, retry_of, images=list(images), **extra)
    record["request"]["questions"] = {"q": {"candidates": candidates}}
    return record


class MetricTests(unittest.TestCase):
    def test_the_planner_follows_prefix_rs(self):
        state = list(range(200))
        groups = [[state + [900] * 100 + [1, 2], state + [900] * 100 + [1, 3], state + [900] * 100 + [4]],
                  [state + [901] * 10 + [5]]]
        # the shared 200 tokens floor to 192; the first question's 100 more reach 256
        self.assertEqual(metrics.plan(groups), (200, 192, [256, 192], 3 * 192 + 2 * (256 - 192)))
        self.assertEqual(metrics.plan([[state]]), (0, 0, [0], 0))
        self.assertEqual(metrics.plan([[[1], [1]]]), (0, 0, [0], 0))

    def test_cross_request_reuse_counts_chunks_once_per_request(self):
        a, b = list(range(300)), list(range(130)) + [-1] * 170
        image = {"sha256": "x", "pixels_sha256": "p"}
        requests = [
            tokens_request("e", 0, a, [[1], [2]], images=[image]),
            tokens_request("e", 1, b, [[1], [2]], images=[image]),
            tokens_request("e", 1, b, [[1], [2]], 1, "e:s1:a0", images=[image]),
            tokens_request("e", 2, a, [[1], [2]], images=[image]),
        ]
        rows = metrics.measure(requests, Layout(), last_k=8)
        self.assertEqual([r["local_saved"] for r in rows], [256, 256, 256, 256])
        # step 1 shares 130 tokens with step 0: two chunks
        self.assertEqual((rows[1]["prev_lcp64"], rows[1]["cross_saved_prev"]), (128, 128))
        self.assertTrue(rows[1]["images_seen_prev"] and not rows[1]["state_seen_before"])
        self.assertEqual(rows[1]["near_duplicate"], round(2 * 130 / 600, 4))
        # the retry is not compared; step 2 repeats step 0's state
        self.assertNotIn("prev_lcp64", rows[2])
        self.assertEqual((rows[3]["prev_lcp64"], rows[3]["cross_saved_best"]), (128, 256))
        self.assertTrue(rows[3]["state_seen_before"] and not rows[3]["state_seen_prev"])
        self.assertEqual(rows[3]["near_duplicate"], 1.0)
        summary = metrics.summarize(rows)
        self.assertEqual((summary["requests"], summary["retries"]), (4, 1))
        self.assertEqual((summary["images"], summary["images_seen_prev"]), (3, 0.6667))

    def test_refused_and_failed_requests_are_not_measured(self):
        a = list(range(100))
        requests = [
            tokens_request("e", 0, a, [[1], [2]], status=422),
            tokens_request("e", 1, a, [[1], [2]], notes={"error": "CancelledError: "}),
            tokens_request("e", 2, a, [[1], [2]], status=200),
            tokens_request("e", 3, a, [[1], [2]]),  # converted ticks carry no status
        ]
        self.assertEqual([r["step"] for r in metrics.measure(requests, Layout())], [2, 3])
        self.assertEqual(metrics.summarize([])["questions_per_request"], None)
        with self.assertRaises(ValueError):
            metrics.measure(requests, Layout(), last_k=0)

    def test_open_jev_prompts_follow_the_worker_fixture(self):
        # golden prompts from Open-Jev @ 3308a15's request compiler, as the Rust worker tests use them
        cases = json.loads((TRACES.parents[1] / "tests/open_jev/data/contract.json").read_text())
        for case in cases:
            self.assertEqual(metrics.open_jev_prompts(case["request"]), case["prompts"])
        self.assertIsNone(metrics.open_jev_prompts({"state": "x", "questions": {"q": {"type": "choice"}}}))
        self.assertIsNone(metrics.open_jev_prompts({"state": "x", "questions": {}}))

    def test_cua_s1_prompts_follow_the_worker(self):
        layout = metrics.CuaS1.__new__(metrics.CuaS1)
        layout.contract = metrics.cua_s1_contract()
        choice = {"type": "choice", "instructions": "go", "criteria": {"a": 'Say "hi"\n', "b": None}}
        prompt = layout.prompts({"state": {"a": [1.5, "é"]}, "questions": {"q": choice}})[0]
        self.assertTrue(prompt.startswith(f"<|im_start|>system\n{layout.contract.SYSTEM_PROMPT}<|im_end|>\n"))
        self.assertIn("<|im_start|>user\nGoal: go\n\nApp: Cua Driver\n", prompt)
        self.assertIn('Accessibility tree:\n{"a": [1.5, "é"]}\n', prompt)
        self.assertTrue(prompt.endswith(
            'Options:\nA. Decision "Say \\"hi\\"\\n" -> select\nB. Decision "b" -> select\n\n'
            "Answer with a single letter.<|im_end|>\n<|im_start|>assistant\n<think>\n"))
        # an empty goal leaves the line out; a request with a Noul is refused whole
        empty = layout.prompts({"state": "s", "questions": {"q": dict(choice, instructions="")}})[0]
        self.assertIn("<|im_start|>user\nApp: Cua Driver\n", empty)
        self.assertIsNone(layout.prompts({"state": "s", "questions": {
            "q": choice, "n": {"type": "noul", "instructions": "x"}}}))


class CacheTests(unittest.TestCase):
    def test_a_cache_skips_what_earlier_prompts_cover_once_per_span(self):
        a = list(range(300))
        # one row, no plan: a cached earlier prompt covers 256 of its tokens
        self.assertEqual(metrics.cached([[a + [1]]], 0, [0], [a + [2]]), 256)
        # two questions sharing 300 tokens: the request prefix (256) is computed once and fully cached;
        # the rows' rests start at 256 and the cache covers nothing more
        groups = [[a + [1]], [a + [2]]]
        _, p, ends, saved = metrics.plan(groups)
        self.assertEqual((p, saved), (256, 256))
        self.assertEqual(metrics.cached(groups, p, ends, [a + [3]]), 256)
        self.assertEqual(metrics.cached(groups, p, ends, []), 0)
        # a row keeps its last token, in whole chunks: a repeated 129-token row reuses 128,
        # a repeated 128-token row 64
        self.assertEqual(metrics.cached([[a[:129]]], 0, [0], [a]), 128)
        self.assertEqual(metrics.cached([[a[:128]]], 0, [0], [a[:128]]), 64)


class Server(BaseHTTPRequestHandler):
    arrivals = []
    slow = set()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Server.arrivals.append((time.perf_counter(), body["state"]))
        if body["state"] in Server.slow:
            time.sleep(0.5)
        raw = json.dumps({"answers": {}, "metadata": {"prefix_cache": {"enabled": True}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class ReplayTests(unittest.TestCase):
    def test_episodes_keep_order_and_think_time(self):
        Server.arrivals, Server.slow = [], {"b1"}
        server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        records = [HEADER]
        for group in "ab":
            for step in range(3):
                records.append(request(group, step, f"{group}{step}", think_ms=80, deadline_s=0.25))
        with tempfile.TemporaryDirectory() as root:
            trace, out = Path(root) / "t.jsonl", Path(root) / "out"
            tracefile.write(trace, records)
            subprocess.run([sys.executable, str(TRACES / "replay.py"), str(trace), "--url",
                            f"http://127.0.0.1:{server.server_port}", "--out", str(out)],
                           check=True, capture_output=True)
            summary = json.loads((out / "summary.json").read_text())
            responses = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
        self.assertEqual((summary["requests"], summary["ok"], summary["shared_prefix_requests"]), (6, 6, 6))
        self.assertEqual(summary["deadline_misses"], 1)
        self.assertEqual([r["id"] for r in responses if r["missed"]], ["b:s1:a0"])
        for group in "ab":
            times = [t for t, state in Server.arrivals if state.startswith(group)]
            states = [state for _, state in Server.arrivals if state.startswith(group)]
            self.assertEqual(states, [f"{group}{i}" for i in range(3)])
            self.assertTrue(all(later - earlier >= 0.075 for earlier, later in zip(times, times[1:])))

    def test_episodes_start_at_their_recorded_offsets_and_retries_can_be_dropped(self):
        records = [request("a", 0, "a0", t_send_ms=1000.0), request("a", 0, "a0", 1, "a:s0:a0", t_send_ms=1200.0),
                   request("b", 0, "b0", t_send_ms=2500.0)]
        groups = replay.episodes(records, skip_retries=True)
        self.assertEqual({g: [r["id"] for r in rs] for g, rs in groups.items()}, {"a": ["a:s0:a0"], "b": ["b:s0:a0"]})
        self.assertEqual(replay.offsets(groups, together=False), {"a": 0.0, "b": 1.5})
        self.assertEqual(replay.offsets(groups, together=True), {"a": 0.0, "b": 0.0})
        self.assertEqual(len(replay.episodes(records, skip_retries=False)["a"]), 2)

    def test_sequential_episodes_run_one_after_another(self):
        Server.arrivals, Server.slow = [], set()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        records = [HEADER] + [request(g, s, f"{g}{s}", think_ms=0) for g in "ab" for s in range(2)]
        with tempfile.TemporaryDirectory() as root:
            trace, out = Path(root) / "t.jsonl", Path(root) / "out"
            tracefile.write(trace, records)
            subprocess.run([sys.executable, str(TRACES / "replay.py"), str(trace), "--url",
                            f"http://127.0.0.1:{server.server_port}", "--sequential", "--deadline-s", "60",
                            "--out", str(out)], check=True, capture_output=True)
            summary = json.loads((out / "summary.json").read_text())
        self.assertEqual([state for _, state in Server.arrivals], ["a0", "a1", "b0", "b1"])
        self.assertEqual((summary["answered_with_deadline"], summary["deadline_misses"]), (4, 0))


FAKE_S1A = {
    "s1a/__init__.py": "",
    "s1a/console.py": "",
    "s1a/decision_models/__init__.py": "",
    "s1a/decision_models/wire.py": """
class JevDecisionsClient:
    _timeout_s = 5.0

    async def decide(self, body):
        if body["state"] == "fail":
            raise RuntimeError("decisions endpoint returned HTTP 422: too long")
        return {"answers": {"q": {"type": "noul", "noul": 0.5}}}, 3
""",
    "s1a/tool/__init__.py": "",
    "s1a/tool/loop.py": """
import types
from s1a.decision_models.wire import JevDecisionsClient


async def run_episode(spec, env, *, seed, model_name="jev"):
    client = JevDecisionsClient()
    for step in range(env):
        await client.decide({"state": f"{seed}-{step}", "questions": {"q": {"type": "noul"}}})
    return types.SimpleNamespace(extra={"result_type": "timeout"} if seed else {}, error=None)
""",
    "s1a/entry.py": """
import asyncio, sys, types


def s1a():
    from s1a.tool.loop import run_episode  # bound after the recorder's patch, as s1a.tool.series does
    from s1a.decision_models.wire import JevDecisionsClient
    spec = types.SimpleNamespace(name=sys.argv[2])

    async def main():
        for seed in range(int(sys.argv[4])):
            await run_episode(spec, 2, seed=seed)
        try:
            await JevDecisionsClient().decide({"state": "fail", "questions": {}})
        except RuntimeError:
            pass

    asyncio.run(main())
    sys.exit(0)
""",
}


class RecorderTests(unittest.TestCase):
    def test_the_recorder_wraps_decisions_and_episodes_from_outside(self):
        with tempfile.TemporaryDirectory() as root:
            for name, text in FAKE_S1A.items():
                path = Path(root) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            out = Path(root) / "trace.jsonl"
            subprocess.run([sys.executable, str(TRACES / "record_s1a.py"), str(out), "--commit", "abc",
                            "--group", "outside", "--", "run", "fake", "--episodes", "2"],
                           check=True, capture_output=True, env={"PYTHONPATH": root, "PATH": ""})
            header, requests, ends = tracefile.read(out)
        self.assertEqual(header["source"]["front"], "run fake")
        self.assertEqual([r["id"] for r in requests],
                         ["fake-seed0-0:s0:a0", "fake-seed0-0:s1:a0", "fake-seed1-1:s0:a0", "fake-seed1-1:s1:a0",
                          "outside:s0:a0"])
        self.assertEqual([r["request"]["state"] for r in requests[:4]], ["0-0", "0-1", "1-0", "1-1"])
        self.assertIsNone(requests[0]["think_ms"])
        self.assertGreaterEqual(requests[1]["think_ms"], 0)
        self.assertEqual((requests[1]["status"], requests[1]["deadline_s"]), (200, 5.0))
        self.assertEqual((requests[4]["status"], requests[4]["notes"]["error"]),
                         (422, "RuntimeError: decisions endpoint returned HTTP 422: too long"))
        self.assertEqual({g: e["status"] for g, e in ends.items()},
                         {"fake-seed0-0": "done", "fake-seed1-1": "timeout", "outside": None})

    def test_the_samples_are_valid_traces(self):
        for path in (TRACES / "samples").glob("*.trace.jsonl"):
            header, requests, _ = tracefile.read(path)
            self.assertTrue(header["content"]["license"], path)
            self.assertTrue(all(r["status"] == 200 for r in requests), path)


if __name__ == "__main__":
    unittest.main()
