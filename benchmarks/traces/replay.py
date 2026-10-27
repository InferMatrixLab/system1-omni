#!/usr/bin/env python3
"""Replay a decision trace against a running server, keeping each episode's order and pace.

Each episode is one client: it sends its requests one at a time, in trace order, and
waits `think_ms * --think-scale` after each response before the next, as the agent did.
Episodes start at their recorded offsets (or together) and run concurrently. Unlike
`bench.py`, nothing is compared with expected answers: the output is per-request latency,
deadline misses and the worker's `prefix_cache` metadata.

    python benchmarks/traces/replay.py TRACE --url http://127.0.0.1:8000 --out DIR
"""

import argparse
import asyncio
import base64
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tracefile  # noqa: E402
from benchmarks.bench import RESPONSE_FORMAT, digest, write_json  # noqa: E402


def episodes(requests, skip_retries):
    out = {}
    for record in requests:
        if skip_retries and record.get("attempt", 0):
            continue
        out.setdefault(record["group"], []).append(record)
    return out


def offsets(groups, together):
    """Seconds after the replay starts at which each episode sends its first request."""
    starts = {g: rs[0].get("t_send_ms") for g, rs in groups.items()}
    if together or any(v is None for v in starts.values()):
        return {g: 0.0 for g in groups}
    first = min(starts.values())
    return {g: (v - first) / 1000 for g, v in starts.items()}


async def run_episode(client, url, model, records, delay, scale, clock, results, save):
    await asyncio.sleep(delay)
    for i, record in enumerate(records):
        if i:
            await asyncio.sleep((record.get("think_ms") or 0) * scale / 1000)
        body = dict(record["request"])
        if model:
            body["model"] = model
        sent = time.perf_counter()
        result = results[record["id"]]
        result.update(state="incomplete", sent_s=round(sent - clock, 6))
        try:
            error = None
            try:
                response = await client.post(url, json=body)
                status, content = response.status_code, response.content
            except httpx.HTTPError as e:
                status, content, error = None, b"", f"{type(e).__name__}: {e}"
            latency = (time.perf_counter() - sent) * 1000
            prefix_cache = None
            if status == 200:
                try:
                    prefix_cache = json.loads(content).get("metadata", {}).get("prefix_cache")
                except (ValueError, AttributeError):
                    pass
            deadline = result["deadline_s"]
            result.update(
                state="completed", latency_ms=round(latency, 3), status=status, error=error,
                missed=latency > deadline * 1000 if status == 200 and deadline is not None else None,
                prefix_cache=prefix_cache, body=base64.b64encode(content).decode(),
            )
        except asyncio.CancelledError:
            result["error"] = "request cancelled before its response completed"
            raise
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save(result)


def summarize(records, elapsed):
    """Latency of answered requests, with `bench.py`'s median and nearest-rank p95."""
    completed = [r for r in records if r["state"] == "completed"]
    ok = [r for r in completed if r["status"] == 200]
    latencies = sorted(r["latency_ms"] for r in ok)
    with_deadline = [r for r in ok if r["missed"] is not None]
    shared = [r for r in ok if isinstance(r["prefix_cache"], dict) and r["prefix_cache"].get("enabled")]
    return {
        "requests": len(records), "ok": len(ok), "failed": len(completed) - len(ok),
        "completed": len(completed),
        "incomplete": sum(r["state"] == "incomplete" for r in records),
        "unattempted": sum(r["state"] == "unattempted" for r in records),
        "elapsed_s": round(elapsed, 3),
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
            "max": latencies[-1] if latencies else None,
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
        },
        "deadline_misses": sum(r["missed"] for r in with_deadline),
        "answered_with_deadline": len(with_deadline),
        "shared_prefix_requests": len(shared),
    }


async def replay(args):
    _, requests, _ = tracefile.read(args.trace)
    groups = episodes(requests, args.skip_retries)
    starts = offsets(groups, args.together)
    url = args.url.rstrip("/") + "/v1/systemone"
    results = {
        r["id"]: {
            "id": r["id"], "group": r["group"], "step": r["step"], "attempt": r.get("attempt", 0),
            "state": "unattempted", "sent_s": None, "latency_ms": None, "status": None,
            "error": None, "deadline_s": args.deadline_s if args.deadline_s is not None else r.get("deadline_s"),
            "missed": None, "prefix_cache": None, "body": "",
        }
        for group in groups.values() for r in group
    }
    config = {
        "trace": str(args.trace), "trace_sha256": digest(args.trace), "url": args.url,
        "model": args.model, "think_scale": args.think_scale, "together": args.together,
        "sequential": args.sequential, "deadline_s": args.deadline_s,
        "skip_retries": args.skip_retries, "timeout_s": args.timeout,
        "response_format": RESPONSE_FORMAT, "python": platform.python_version(),
        "httpx": httpx.__version__, "planned_ids": list(results),
    }
    # mkdir claims the whole run atomically, including against another replay process.
    # Even an empty existing directory may belong to a run that has just started.
    args.out.mkdir(parents=True)
    write_json(args.out / "config.json", config)
    saved = set()
    clock = time.perf_counter()
    stop_reason = "error"
    with (args.out / "responses.jsonl").open("x", encoding="utf-8") as responses:
        def save(result):
            responses.write(json.dumps(result, allow_nan=False) + "\n")
            responses.flush()
            saved.add(result["id"])

        limits = httpx.Limits(max_connections=len(groups), max_keepalive_connections=len(groups))
        try:
            async with httpx.AsyncClient(timeout=args.timeout, trust_env=False, limits=limits) as client:
                tasks = []
                try:
                    if args.sequential:
                        for records in groups.values():
                            await run_episode(client, url, args.model, records, 0.0, args.think_scale,
                                              clock, results, save)
                    else:
                        tasks = [asyncio.create_task(run_episode(
                            client, url, args.model, records, starts[g], args.think_scale,
                            clock, results, save,
                        )) for g, records in groups.items()]
                        await asyncio.gather(*tasks)
                finally:
                    # gather propagates one failure without stopping its siblings. Drain
                    # every episode before closing the client or finalizing the evidence.
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            stop_reason = "completed"
        except asyncio.CancelledError:
            stop_reason = "interrupted"
            raise
        finally:
            for result in results.values():
                if result["id"] not in saved:
                    save(result)
            summary = summarize(list(results.values()), time.perf_counter() - clock)
            summary["stop_reason"] = stop_reason
            write_json(args.out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", help="model alias added to every request")
    parser.add_argument("--think-scale", type=float, default=1.0, help="0 sends each step right after the last")
    parser.add_argument("--together", action="store_true", help="start every episode at once")
    parser.add_argument("--sequential", action="store_true", help="run the episodes one after another")
    parser.add_argument("--deadline-s", type=float, help="count misses against this deadline, not the trace's")
    parser.add_argument("--skip-retries", action="store_true", help="send first attempts only")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.think_scale < 0:
        parser.error("--think-scale must be nonnegative")
    if args.sequential and args.together:
        parser.error("--sequential and --together exclude each other")
    try:
        summary = asyncio.run(replay(args))
    except FileExistsError:
        parser.error(f"output directory already exists: {args.out}; choose a new --out directory")
    except KeyboardInterrupt:
        print(f"Replay interrupted; check {args.out} for partial results", file=sys.stderr)
        raise SystemExit(130)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
