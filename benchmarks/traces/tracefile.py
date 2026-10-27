#!/usr/bin/env python3
"""Read, check and write decision traces (`s1-trace/1`), and convert recorded requests.

A trace is JSONL: one `header`, then `request` records (one per attempt) and optional
`episode_end` records. See README.md in this directory.
"""

import argparse
import hashlib
import json
from pathlib import Path

FORMAT = "s1-trace/1"
CAPTURES = {"client:transport", "converted"}  # record_s1a.py; from_ticks


def canonical(value):
    """Bytes of a JSON value with its key order kept, as the request was sent."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def state_digest(request, images=()):
    """Identity of what a request asks about: its state bytes and its images' pixels."""
    h = hashlib.sha256(canonical(request.get("state")))
    for image in images:
        h.update(image["pixels_sha256"].encode())
    return h.hexdigest()


def check(records):
    """Return (header, requests, ends) or raise ValueError naming the first problem."""
    if not records or records[0].get("type") != "header":
        raise ValueError("a trace starts with one header record")
    header = records[0]
    if header.get("format") != FORMAT:
        raise ValueError(f"header format must be {FORMAT}")
    if header.get("capture") not in CAPTURES:
        raise ValueError(f"header capture must be one of {sorted(CAPTURES)}")
    requests, ends, ids, last = [], {}, set(), {}
    for number, record in enumerate(records[1:], start=2):
        kind = record.get("type")
        where = f"record {number}"
        if kind == "request":
            for key in ("id", "group"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError(f"{where}: {key} must be a nonempty string")
            if record["id"] in ids:
                raise ValueError(f"{where}: duplicate id {record['id']}")
            if record["group"] in ends:
                raise ValueError(f"{where}: episode {record['group']} already ended")
            step, attempt = record.get("step"), record.get("attempt", 0)
            if not isinstance(step, int) or step < 0 or not isinstance(attempt, int) or attempt < 0:
                raise ValueError(f"{where}: step and attempt must be nonnegative integers")
            if (step, attempt) <= last.get(record["group"], (-1, -1)):
                raise ValueError(f"{where}: steps and attempts must increase within an episode")
            retry_of = record.get("retry_of")
            if (attempt == 0) != (retry_of is None) or (retry_of is not None and retry_of not in ids):
                raise ValueError(f"{where}: retries name an earlier id, first attempts none")
            request = record.get("request")
            if not isinstance(request, dict) or "state" not in request or "questions" not in request:
                raise ValueError(f"{where}: request must hold state and questions")
            for image in record.get("images", []):
                if not all(isinstance(image.get(k), str) for k in ("sha256", "pixels_sha256")):
                    raise ValueError(f"{where}: images need sha256 and pixels_sha256")
            for key in ("t_send_ms", "t_recv_ms", "think_ms"):
                value = record.get(key)
                if value is not None and (not isinstance(value, (int, float)) or value < 0):
                    raise ValueError(f"{where}: {key} must be a nonnegative number or null")
            ids.add(record["id"])
            last[record["group"]] = (step, attempt)
            requests.append(record)
        elif kind == "episode_end":
            group = record.get("group")
            if group not in last or group in ends:
                raise ValueError(f"{where}: episode_end needs an open episode")
            ends[group] = record
        else:
            raise ValueError(f"{where}: unknown record type {kind!r}")
    if not requests:
        raise ValueError("a trace needs at least one request")
    return header, requests, ends


def read(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return check(records)


def write(path, records):
    check(records)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def from_ticks(lines, episode, source, model=None):
    """Records from System1-Agents' saved browser ticks: one JSON object per decision with
    `state`, `questions` and optionally `answers` and `model`. Ticks carry no timing."""
    records = [{
        "type": "header", "format": FORMAT, "source": source,
        "server": {"model": model}, "capture": "converted",
        "clock": None, "content": {}, "images": {"store": None},
    }]
    for step, line in enumerate(lines):
        tick = json.loads(line)
        request = {"state": tick["state"], "questions": tick["questions"]}
        records.append({
            "type": "request", "id": f"{episode}:s{step}:a0", "group": episode,
            "step": step, "attempt": 0, "retry_of": None, "request": request,
            "images": [], "t_send_ms": None, "t_recv_ms": None, "think_ms": None,
            "deadline_s": None, "status": None,
            "response": {"answers": tick["answers"]} if "answers" in tick else None,
            "notes": {"model": tick["model"]} if "model" in tick else {},
        })
    records.append({"type": "episode_end", "group": episode, "status": None, "steps": len(lines)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="check a trace and print its counts")
    checking.add_argument("trace", type=Path)
    converting = commands.add_parser("from-ticks", help="convert saved System1-Agents ticks")
    converting.add_argument("ticks", type=Path)
    converting.add_argument("out", type=Path)
    converting.add_argument("--episode", required=True)
    converting.add_argument("--source", required=True, help="JSON object naming repo, commit and front")
    converting.add_argument("--model", help="the decision model that answered the ticks")
    args = parser.parse_args()
    if args.command == "check":
        header, requests, ends = read(args.trace)
        groups = {r["group"] for r in requests}
        print(json.dumps({"requests": len(requests), "episodes": len(groups), "ended": len(ends)}))
    else:
        lines = [line for line in args.ticks.read_text().splitlines() if line.strip()]
        write(args.out, from_ticks(lines, args.episode, json.loads(args.source), args.model))
        print(f"wrote {len(lines)} requests to {args.out}")


if __name__ == "__main__":
    main()
