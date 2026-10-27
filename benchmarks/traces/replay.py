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


async def run_episode(client, url, model, records, delay, scale, clock, sink, deadline_s=None):
    await asyncio.sleep(delay)
    for i, record in enumerate(records):
        if i:
            await asyncio.sleep((record.get("think_ms") or 0) * scale / 1000)
        body = dict(record["request"])
        if model:
            body["model"] = model
        sent = time.perf_counter()
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
        deadline = deadline_s if deadline_s is not None else record.get("deadline_s")
        answered = status == 200
        sink.append({
            "id": record["id"], "group": record["group"], "step": record["step"],
            "attempt": record.get("attempt", 0), "sent_s": round(sent - clock, 6),
            "latency_ms": round(latency, 3), "status": status, "error": error,
            "deadline_s": deadline,
            "missed": latency > deadline * 1000 if answered and deadline is not None else None,
            "prefix_cache": prefix_cache, "body": base64.b64encode(content).decode(),
        })


def summarize(records, elapsed):
    """Latency of answered requests, with `bench.py`'s median and nearest-rank p95."""
    ok = [r for r in records if r["status"] == 200]
    latencies = sorted(r["latency_ms"] for r in ok)
    with_deadline = [r for r in ok if r["missed"] is not None]
    shared = [r for r in ok if (r["prefix_cache"] or {}).get("enabled")]
    return {
        "requests": len(records), "ok": len(ok), "failed": len(records) - len(ok),
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
    sink = []
    clock = time.perf_counter()
    limits = httpx.Limits(max_connections=len(groups), max_keepalive_connections=len(groups))
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False, limits=limits) as client:
        if args.sequential:
            for records in groups.values():
                await run_episode(client, url, args.model, records, 0.0, args.think_scale, clock, sink, args.deadline_s)
        else:
            await asyncio.gather(*(
                run_episode(client, url, args.model, records, starts[g], args.think_scale, clock, sink, args.deadline_s)
                for g, records in groups.items()
            ))
    return sink, time.perf_counter() - clock


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
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "config.json", {
        "trace": str(args.trace), "trace_sha256": digest(args.trace), "url": args.url,
        "model": args.model, "think_scale": args.think_scale, "together": args.together,
        "sequential": args.sequential, "deadline_s": args.deadline_s,
        "skip_retries": args.skip_retries, "timeout_s": args.timeout,
        "response_format": RESPONSE_FORMAT, "python": platform.python_version(),
        "httpx": httpx.__version__,
    })
    records, elapsed = asyncio.run(replay(args))
    records.sort(key=lambda r: r["sent_s"])
    (args.out / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = summarize(records, elapsed)
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
