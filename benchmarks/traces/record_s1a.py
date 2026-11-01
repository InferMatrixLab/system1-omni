#!/usr/bin/env python3
"""Record System1-Agents' decision requests as an s1-trace/1 file, without changing its source.

    python record_s1a.py OUT.jsonl --commit SHA -- run blackjack --model jev --rethink off --episodes 20
    python record_s1a.py OUT.jsonl --commit SHA --group flights-1 -- run flights --model jev
    python record_s1a.py OUT.jsonl --commit SHA -- -m evals.recovery.live --model jev --repeat 1

Wraps `JevDecisionsClient.decide`, the HTTP client behind `--model jev` (the request body as
sent, the payload, timing; retries inside one decision are not separate records), and
`s1a.tool.loop.run_episode` (episode boundaries). Run it with System1-Agents' environment
and `TYPESAFE_API_URL` pointing at the decision server, from System1-Agents' checkout with
its virtual environment's Python. Episodes must run one at a time, as `s1a run` does.
"""

import argparse
import contextvars
import json
import os
import re
import sys
import time

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("out")
parser.add_argument("--commit", required=True, help="System1-Agents commit")
parser.add_argument("--server-commit", help="system1-omni commit of the decision server")
parser.add_argument("--worker", help="decision worker, e.g. open_jev")
parser.add_argument("--group", default="run", help="episode name for requests outside the tool loop")
if "--" not in sys.argv:
    parser.error("put the s1a arguments after --")
split = sys.argv.index("--")
args = parser.parse_args(sys.argv[1:split])
s1a_args = sys.argv[split + 1:]
if not s1a_args or s1a_args == ["-m"]:
    parser.error("name the s1a command, or -m and a module, after --")

from s1a import console  # noqa: E402,F401  (first, as s1a.entry does: routes logs before openjiuwen loads)
from s1a.decision_models import wire  # noqa: E402
from s1a.tool import loop  # noqa: E402

out = open(args.out, "w", buffering=1)
clock = time.perf_counter()
episode = contextvars.ContextVar("episode", default=args.group)
steps, last_recv, started = {}, {}, set()


def now_ms():
    return round((time.perf_counter() - clock) * 1000, 3)


def emit(record):
    out.write(json.dumps(record, ensure_ascii=False) + "\n")


emit({
    "type": "header", "format": "s1-trace/1",
    "source": {"repo": "ThinkFlowLab/system1-agents", "commit": args.commit, "front": " ".join(s1a_args[:2]),
               "argv": s1a_args},
    "server": {"repo": "InferMatrixLab/system1-omni", "commit": args.server_commit, "worker": args.worker,
               "url": os.environ.get("TYPESAFE_API_URL"), "model": os.environ.get("TYPESAFE_MODEL")},
    "capture": "client:transport", "clock": "ms since the recorder started (perf_counter)",
    "content": {}, "images": {"store": None},
})

decide = wire.JevDecisionsClient.decide


async def recorded_decide(self, body):
    group = episode.get()
    step = steps.get(group, 0)
    steps[group] = step + 1
    started.add(group)
    t_send = now_ms()
    record = {
        "type": "request", "id": f"{group}:s{step}:a0", "group": group, "step": step, "attempt": 0,
        "retry_of": None, "request": body, "images": [], "t_send_ms": t_send, "t_recv_ms": None,
        "think_ms": round(t_send - last_recv[group], 3) if group in last_recv else None,
        "deadline_s": self._timeout_s, "status": None, "response": None, "notes": {},
    }
    try:
        payload, latency_ms = await decide(self, body)
        record.update(status=200, response=payload, notes={"client_latency_ms": latency_ms})
        return payload, latency_ms
    except BaseException as e:  # a timed-out episode cancels the decision in flight
        record["notes"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        status = re.search(r"HTTP (\d{3})", str(e))
        record["status"] = int(status.group(1)) if status else None
        raise
    finally:
        record["t_recv_ms"] = last_recv[group] = now_ms()
        emit(record)


wire.JevDecisionsClient.decide = recorded_decide
run_episode = loop.run_episode
episodes = iter(range(10**9))


async def recorded_run_episode(spec, env, *, seed, **kwargs):
    group = f"{spec.name}-seed{seed}-{next(episodes)}"
    token = episode.set(group)
    status = "error"
    try:
        result = await run_episode(spec, env, seed=seed, **kwargs)
        status = "error" if result.error else (result.extra.get("result_type") or "done")
        return result
    finally:
        episode.reset(token)
        if group in started:
            emit({"type": "episode_end", "group": group, "status": str(status), "steps": steps[group]})


loop.run_episode = recorded_run_episode  # before s1a.tool.series imports it by name

try:
    if s1a_args[:1] == ["-m"]:
        import runpy

        sys.argv = [s1a_args[1], *s1a_args[2:]]
        runpy.run_module(s1a_args[1], run_name="__main__", alter_sys=True)
    else:
        from s1a import entry

        sys.argv = ["s1a", *s1a_args]
        entry.s1a()
finally:
    if args.group in started:
        emit({"type": "episode_end", "group": args.group, "status": None, "steps": steps[args.group]})
    out.close()
