"""CPU checks for the hk2 nightly harness. No GPU, weights, or network."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from benchmarks import bench
from benchmarks.nightly import run_nightly as nightly

SMOKE = Path(bench.__file__).with_name("smoke.jsonl")
PINS = {
    "open-jev-9b": (
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "47e966881e489511c0c7f5633a9e1960a676a551",
    ),
    "open-jev-27b-v1.1": (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "28cf73067d5b337860bbef3c85b8b82ba8730956",
    ),
}


def quiet(argv):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = nightly.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def without_nightly_env():
    kept = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OMNI_NIGHTLY_")
    }
    return patch.dict(os.environ, kept, clear=True)


def write_export(repo, model):
    path = repo / model["weights"]
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"not-a-real-tensor")
    (path / nightly.EXPORT_NAME).write_text(
        json.dumps(
            {
                "format": nightly.EXPORT_FORMAT,
                "model_id": model["export_model_id"],
                "base_revision": model["base_revision"],
                "checkpoint_revision": model["checkpoint_revision"],
            }
        ),
        encoding="utf-8",
    )


def write_binaries(repo):
    directory = repo / "target" / "release"
    directory.mkdir(parents=True)
    for name in ("omni-open-jev-native", "omni-jev"):
        (directory / name).write_bytes(b"#!/bin/sh\nexit 0\n")


def public231(path):
    template = bench.load_cases(SMOKE)
    lines = []
    for index in range(nightly.PUBLIC231_REQUESTS):
        case = json.loads(json.dumps(template[index % len(template)]))
        case["id"] = f"public231-{index}"
        lines.append(json.dumps(case))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class NightlyConfigTests(unittest.TestCase):
    def test_default_models_match_the_hk2_decision(self):
        config = nightly.load_config(None)
        self.assertEqual(
            [model["id"] for model in nightly.selected_models(config, False)],
            ["open-jev-9b", "open-jev-27b-v1.1"],
        )
        self.assertEqual(
            [model["id"] for model in nightly.selected_models(config, True)],
            [
                "open-jev-9b",
                "open-jev-27b-v1.1",
                "intern-decision-0.8b",
                "intern-decision-2b",
                "intern-decision-4b",
            ],
        )
        by_id = {model["id"]: model for model in config["models"]}
        for model_id, (export_id, base, checkpoint) in PINS.items():
            self.assertEqual(by_id[model_id]["export_model_id"], export_id)
            self.assertEqual(by_id[model_id]["base_revision"], base)
            self.assertEqual(by_id[model_id]["checkpoint_revision"], checkpoint)
            self.assertEqual(by_id[model_id]["launch"], "open-jev-native")
            self.assertTrue(by_id[model_id]["role"] == "headline")
        for model_id, hub in (
            ("intern-decision-0.8b", "internlm/Intern-Decision-0.8B"),
            ("intern-decision-2b", "internlm/Intern-Decision-2B"),
            ("intern-decision-4b", "internlm/Intern-Decision-4B"),
        ):
            self.assertEqual(by_id[model_id]["hub_id"], hub)
            self.assertEqual(by_id[model_id]["launch"], "attach-only")
            self.assertEqual(by_id[model_id]["cadence"], "optional")
        blob = json.dumps(config).lower()
        for banned in ("d3", "vela", "valen", "decider", "laya"):
            self.assertNotIn(banned, blob)
        ports = [
            model["worker_port"] for model in config["models"] if "worker_port" in model
        ]
        ports += [
            model["frontend_port"]
            for model in config["models"]
            if "frontend_port" in model
        ]
        self.assertEqual(len(ports), len(set(ports)))

    def test_excluded_models_are_rejected(self):
        for model_id in ("d3", "vela-7b", "valen", "decider-2b"):
            with self.assertRaises(nightly.NightlyError) as caught:
                nightly.assert_allowed(
                    {"id": model_id, "launch": "attach-only", "cadence": "optional"}
                )
            self.assertIn("错误", str(caught.exception))

    def test_shell_search_order_matches_python(self):
        shell = (nightly.HERE / "run_nightly.sh").read_text(encoding="utf-8")
        positions = [shell.index(item) for item in nightly.CONVENTIONAL_REPOS]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(os.access(nightly.HERE / "run_nightly.sh", os.X_OK))

    def test_conventional_repo_prefers_the_first_checkout_that_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            first = home / nightly.CONVENTIONAL_REPOS[0]
            second = home / nightly.CONVENTIONAL_REPOS[1]
            marker = second / "benchmarks" / "nightly" / "run_nightly.py"
            marker.parent.mkdir(parents=True)
            marker.write_text("second\n", encoding="utf-8")
            self.assertEqual(
                nightly.select_conventional_repo(home, Path.exists), second
            )
            other = first / "benchmarks" / "nightly" / "run_nightly.py"
            other.parent.mkdir(parents=True)
            other.write_text("first\n", encoding="utf-8")
            self.assertEqual(nightly.select_conventional_repo(home, Path.exists), first)

    def test_self_check(self):
        code, stdout, stderr = quiet(["--self-check"])
        self.assertEqual(code, 0, stderr)
        self.assertIn("self-check ok", stdout)

    def test_results_directory_is_ignored(self):
        text = (nightly.HERE / "results" / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*", text.splitlines())
        self.assertIn("!.gitignore", text)


class NightlyPreflightTests(unittest.TestCase):
    def test_missing_weights_exit_bilingual_without_sleep_or_bench(self):
        with tempfile.TemporaryDirectory() as directory, without_nightly_env():
            repo = Path(directory)
            with (
                patch.object(
                    nightly.time, "sleep", side_effect=AssertionError("slept")
                ),
                patch.object(nightly, "run_bench", side_effect=AssertionError("bench")),
                patch.object(nightly, "execute", side_effect=AssertionError("execute")),
            ):
                code, _stdout, stderr = quiet(
                    [
                        "--repo",
                        str(repo),
                        "--dry-run",
                        "--no-update",
                        "--no-lock",
                        "--manifest",
                        str(SMOKE),
                        "--workload",
                        "smoke",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("export_merged", stderr)
        self.assertIn("download", stderr)
        self.assertIn("错误", stderr)
        self.assertIn("recipe/open_jev/native.md", stderr)

    def test_intern_without_endpoint_fails_before_any_wait(self):
        config = nightly.load_config(None)
        with tempfile.TemporaryDirectory() as directory, without_nightly_env():
            repo = Path(directory)
            for model in nightly.selected_models(config, True):
                if model["launch"] == "open-jev-native":
                    write_export(repo, model)
            write_binaries(repo)
            with patch.object(
                nightly.time, "sleep", side_effect=AssertionError("slept")
            ):
                code, _stdout, stderr = quiet(
                    [
                        "--repo",
                        str(repo),
                        "--dry-run",
                        "--no-update",
                        "--no-lock",
                        "--intern",
                        "--manifest",
                        str(SMOKE),
                        "--workload",
                        "smoke",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("OMNI_NIGHTLY_ENDPOINT_INTERN_0_8B", stderr)
        self.assertIn("错误", stderr)
        self.assertIn("不启动官方 HF", stderr)

    def test_dry_run_lists_headline_launch_without_intern(self):
        config = nightly.load_config(None)
        with tempfile.TemporaryDirectory() as directory, without_nightly_env():
            repo = Path(directory)
            for model in nightly.selected_models(config, False):
                write_export(repo, model)
            write_binaries(repo)
            manifest = repo / "public231.jsonl"
            public231(manifest)
            with patch.dict(
                os.environ, {"CUA_S1_GRAPH": "1", "OMNI_NIGHTLY_HOST": "hk2"}
            ):
                code, stdout, stderr = quiet(
                    [
                        "--repo",
                        str(repo),
                        "--dry-run",
                        "--no-update",
                        "--no-lock",
                        "--manifest",
                        str(manifest),
                        "--regression-concurrency",
                        "8,16",
                    ]
                )
        self.assertEqual(code, 0, stderr)
        plan = json.loads(stdout)
        self.assertEqual(
            [row["id"] for row in plan["models"]], ["open-jev-9b", "open-jev-27b-v1.1"]
        )
        self.assertFalse(plan["intern"])
        self.assertTrue(plan["headline"])
        self.assertTrue(plan["packing_probe"])
        self.assertEqual(plan["regression_concurrency"], [8, 16])
        self.assertEqual(plan["manifest_requests"], 231)
        for row in plan["models"]:
            self.assertEqual(row["mode"], "launch")
            self.assertFalse(row["launch"]["cuda_graph"])
            self.assertTrue(row["launch"]["endpoint"].endswith("/v1/systemone"))
            self.assertIn("weights/open-jev-", row["launch"]["open_jev_model"])
        self.assertIn("Do not compare", plan["clock_note"])

    def test_public231_must_have_231_requests(self):
        with tempfile.TemporaryDirectory() as directory, without_nightly_env():
            repo = Path(directory)
            manifest = repo / "public231.jsonl"
            manifest.write_text(SMOKE.read_text(encoding="utf-8"), encoding="utf-8")
            for model in nightly.selected_models(nightly.load_config(None), False):
                write_export(repo, model)
            write_binaries(repo)
            code, _stdout, stderr = quiet(
                [
                    "--repo",
                    str(repo),
                    "--dry-run",
                    "--no-update",
                    "--no-lock",
                    "--manifest",
                    str(manifest),
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("231", stderr)
        self.assertIn("错误", stderr)

    def test_c1_is_not_a_regression_gate(self):
        with self.assertRaises(nightly.NightlyError) as caught:
            nightly.parse_concurrency_list("1,8")
        self.assertIn("headline", str(caught.exception))
        self.assertIn("错误", str(caught.exception))

    def test_dirty_checkout_does_not_pull(self):
        calls = []

        def fake(_repo, *args, timeout):
            calls.append(args)
            self.assertGreater(timeout, 0)
            if args == ("status", "--porcelain"):
                return " M README.md"
            raise AssertionError(args)

        with patch.object(nightly, "git", fake):
            with self.assertRaises(nightly.NightlyError) as caught:
                nightly.sync_checkout(Path("."), "main")
        self.assertEqual(calls, [("status", "--porcelain")])
        self.assertIn("专用克隆", str(caught.exception))

    def test_lock_does_not_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nightly.lock"
            held = nightly.acquire_lock(path)
            try:
                with self.assertRaises(nightly.NightlyError) as caught:
                    nightly.acquire_lock(path)
            finally:
                held.close()
        self.assertIn("不会等待", str(caught.exception))


class NightlySummaryTests(unittest.TestCase):
    def test_percentiles_match_bench_py(self):
        self.assertEqual(nightly.percentiles([]), (None, None))
        p50, p95 = nightly.percentiles([10, 30, 20, 40])
        self.assertEqual(p50, 25)
        self.assertEqual(p95, 40)

    def test_regression_and_packing_are_not_headline_numbers(self):
        model = nightly.selected_models(nightly.load_config(None), False)[1]
        pooled = {
            "p50_ms": 80.0,
            "p95_ms": 120.0,
            "sample_count": 4,
            "planned_requests": 4,
            "successful_requests": 4,
            "failed_requests": 0,
            "incomplete_requests": 0,
            "unattempted_requests": 0,
            "evidence_complete": True,
            "rounds": [],
        }
        headline = nightly.arm_record(
            model,
            pooled,
            "public231",
            1,
            "headline",
            "http://127.0.0.1/v1/systemone",
            True,
            "abc",
            nightly.note_for("headline"),
        )
        regression = nightly.arm_record(
            model,
            pooled,
            "public231",
            8,
            "regression-only",
            "http://127.0.0.1/v1/systemone",
            True,
            "abc",
            nightly.REGRESSION_NOTE,
        )
        self.assertTrue(headline["headline"])
        self.assertTrue(headline["leaderboard_eligible"])
        self.assertFalse(regression["headline"])
        self.assertFalse(regression["leaderboard_eligible"])
        self.assertIn("war-report", regression["note"])
        started = datetime(2026, 10, 24, 2, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        report = nightly.assemble_report(
            started=started,
            finished=started,
            host="hk2",
            operator="hsliu",
            sha="abc123",
            ref="main",
            dirty=False,
            gpu_info={
                "gpu": "L20X",
                "gpu_ids": "GPU-1",
                "driver": "1",
                "cuda": "12",
                "compute_cap": "9.0",
            },
            workload="public231",
            manifest_sha="abc",
            manifest_requests=231,
            arms=[headline, regression],
            packing=nightly.packing_record(pooled, "def", 90.0, "2026-10-23"),
            failures=[],
            issue=None,
        )
        self.assertEqual(report["headline_p50_ms"], {"open-jev-27b-v1.1": 80.0})
        self.assertNotIn("8", report["headline_p50_ms"])
        self.assertFalse(report["comparable_to_h800_historical_board"])
        self.assertEqual(report["packing_probe"]["delta_p50_ms_vs_previous"], -10.0)
        self.assertFalse(report["packing_probe"]["headline"])
        text = nightly.markdown_report(report)
        self.assertIn("每日主指标", text)
        self.assertIn("不是 headline", text)
        self.assertIn(nightly.REGRESSION_NOTE, text)
        digest = nightly.issue_digest(report)
        self.assertEqual(digest["headline_p50_ms"], {"open-jev-27b-v1.1": 80.0})
        self.assertEqual(digest["regression_only"][0]["concurrency"], 8)
        self.assertEqual(set(nightly.REPORT_KEYS), set(report))

    def test_recent_table_and_dated_file_are_not_overwritten(self):
        started = datetime(2026, 10, 24, 2, 5, tzinfo=ZoneInfo("Asia/Shanghai"))
        report = nightly.assemble_report(
            started=started,
            finished=started,
            host="hk2",
            operator="hsliu",
            sha="abc",
            ref="main",
            dirty=False,
            gpu_info={
                "gpu": "L20X",
                "gpu_ids": "GPU-1",
                "driver": "1",
                "cuda": "12",
                "compute_cap": "9.0",
            },
            workload="public231",
            manifest_sha="abc",
            manifest_requests=231,
            arms=[],
            packing=None,
            failures=[],
            issue=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = nightly.write_results(root, report)
            second = nightly.write_results(root, report)
            self.assertNotEqual(first, second)
            self.assertTrue((root / "latest.json").is_file())
            recent = (root / "recent.md").read_text(encoding="utf-8")
            self.assertIn("open-jev-9b", recent)
            self.assertIn("regression-only", recent)
            saved = json.loads((root / "latest.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["illustrative"])

    def test_previous_packing_ignores_a_different_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nightly.write_json(
                root / "latest.json",
                {
                    "illustrative": False,
                    "date": "2026-10-23",
                    "packing_probe": {"manifest_sha256": "old", "p50_ms": 3},
                },
            )
            self.assertEqual(nightly.previous_packing(root, "new"), (None, None))
            self.assertEqual(nightly.previous_packing(root, "old"), (3.0, "2026-10-23"))

    def test_issue_parser_and_comment_command(self):
        number, slug = nightly.split_issue(
            "https://github.com/InferMatrixLab/system1-omni/issues/12"
        )
        self.assertEqual((number, slug), ("12", "InferMatrixLab/system1-omni"))
        self.assertEqual(nightly.split_issue("15")[0], "15")
        with self.assertRaises(nightly.NightlyError):
            nightly.split_issue("0")
        with self.assertRaises(nightly.NightlyError):
            nightly.split_issue("TODO")
        with tempfile.TemporaryDirectory() as directory:
            body = Path(directory) / "comment.md"
            body.write_text("hello\n", encoding="utf-8")
            seen = {}

            def runner(command, **_kwargs):
                seen["command"] = command

                class Result:
                    returncode = 0
                    stderr = ""
                    stdout = ""

                return Result()

            posted = nightly.post_issue(
                Path(directory),
                "https://github.com/InferMatrixLab/system1-omni/issues/15",
                body,
                runner=runner,
            )
            self.assertEqual(posted[0], "15")
            self.assertEqual(seen["command"][0:3], ["gh", "issue", "comment"])
            self.assertIn(str(body), seen["command"])

    def test_launch_spec_leaves_cuda_graphs_off(self):
        config = nightly.load_config(None)
        model = nightly.selected_models(config, False)[0]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CUA_S1_GRAPH": "1"}),
        ):
            spec = nightly.launch_spec(model, Path(directory), "0")
        self.assertNotIn("CUA_S1_GRAPH", spec["worker_env"])
        self.assertEqual(spec["worker_env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertTrue(spec["worker_argv"][0].endswith("omni-open-jev-native"))
        self.assertEqual(
            spec["frontend_env"]["OMNI_JEV_BACKEND_URL"], "http://127.0.0.1:18090"
        )

    def test_attach_plan_rejects_the_wrong_open_jev_model(self):
        model = nightly.selected_models(nightly.load_config(None), False)[0]
        env = {model["endpoint_env"]: "http://127.0.0.1:8080/v1/systemone"}
        with self.assertRaises(nightly.NightlyError) as caught:
            nightly.plan_service(
                model,
                Path("."),
                env,
                probe=lambda _endpoint: {"ok": True, "model": "Qwen/Qwen3.8-27B"},
            )
        self.assertIn("错误", str(caught.exception))
        plan = nightly.plan_service(
            model,
            Path("."),
            env,
            probe=lambda _endpoint: {"ok": True, "model": "Qwen/Qwen3.5-9B"},
        )
        self.assertEqual(plan["mode"], "attach")

    def test_endpoint_rejects_credentials(self):
        with self.assertRaises(nightly.NightlyError):
            nightly.check_endpoint("http://user:secret@127.0.0.1/v1/systemone")
        with self.assertRaises(nightly.NightlyError):
            nightly.check_endpoint("http://127.0.0.1/health")

    def test_smoke_workload_is_not_a_headline(self):
        model = nightly.selected_models(nightly.load_config(None), False)[0]
        self.assertEqual(nightly.role_for(model, "smoke", 1), "smoke")
        self.assertEqual(
            nightly.classify_workload(Path("public231.jsonl"), False), "public231"
        )
        self.assertEqual(nightly.classify_workload(SMOKE, False), "smoke")

    def test_run_bench_calls_the_existing_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "smoke.jsonl"
            manifest.write_text(SMOKE.read_text(encoding="utf-8"), encoding="utf-8")
            metadata = root / "metadata.json"
            metadata.write_text("{}\n", encoding="utf-8")
            seen = {}

            def runner(command, **kwargs):
                seen["command"] = command
                seen["cwd"] = kwargs.get("cwd")
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                (output / "summary.json").write_text(
                    json.dumps(
                        {
                            "evidence_complete": True,
                            "failed_requests": 0,
                            "planned_requests": 4,
                            "successful_requests": 4,
                            "incomplete_requests": 0,
                            "unattempted_requests": 0,
                            "successful_latency_p50_ms": 10,
                            "successful_latency_p95_ms": 12,
                        }
                    ),
                    encoding="utf-8",
                )
                (output / "responses.jsonl").write_text(
                    json.dumps({"state": "completed", "error": None, "latency_ms": 10})
                    + "\n",
                    encoding="utf-8",
                )

                class Result:
                    returncode = 0
                    stdout = "{}\n"
                    stderr = ""

                return Result()

            with patch.object(nightly.subprocess, "run", runner):
                result = nightly.run_bench(
                    nightly.REPO_ROOT,
                    manifest,
                    "http://127.0.0.1:8080/v1/systemone",
                    {"alias": "open-jev-9b", "score_decimals": 4},
                    "measured",
                    1,
                    5,
                    120,
                    metadata,
                    root / "out",
                    4,
                )
            command = seen["command"]
            self.assertEqual(
                command[1], str(nightly.REPO_ROOT / "benchmarks" / "bench.py")
            )
            self.assertEqual(command[2], "run")
            self.assertIn("--endpoint", command)
            self.assertIn("http://127.0.0.1:8080/v1/systemone", command)
            self.assertEqual(command[command.index("--concurrency") + 1], "1")
            self.assertEqual(seen["cwd"], nightly.REPO_ROOT)
            self.assertTrue(result["ok"])
            self.assertEqual(nightly.pool_runs([result])["p50_ms"], 10)
        cases = bench.load_cases(nightly.HERE / "packing_probe.jsonl")
        self.assertEqual(len(cases), 4)
        script = (nightly.HERE / "run_nightly.sh").read_text(encoding="utf-8")
        self.assertIn("flock -n", script)
        self.assertNotIn("flock -w", script)


if __name__ == "__main__":
    unittest.main()
