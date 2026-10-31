#!/usr/bin/env python3
"""Measure repeated work in a decision trace, as one worker would tokenize it.

For each request the worker would serve: questions, sequences and tokens; the tokens
Decider's request-local planner (`src/models/decider/native/src/prefix.rs`) would not
recompute; the tokens a cache of earlier prompts would skip on top of that, in 64-token
chunks; whether its state or images were seen before. The summary adds these up into the
share of prefill tokens each kind of reuse could skip.

    python benchmarks/traces/metrics.py TRACE --layout open_jev --export weights/open-jev-9b-merged --out DIR
"""

import argparse
import difflib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracefile as traces  # noqa: E402

CHUNK = 64
ROOT = Path(__file__).resolve().parents[2]


def render(value):
    """Open-Jev's rendering: text as is, anything else as Python's sorted json.dumps."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def open_jev_prompts(request):
    """Candidate prompts per question, as `src/models/open_jev/native/src/contract.rs` builds
    them; None for a malformed request. It does not repeat all of the worker's checks."""
    try:
        state = render(request["state"])
        prompts = []
        for question in request["questions"].values():
            instructions = render(question["instructions"])
            criteria = question.get("criteria")
            kind = question["type"]
            if kind == "noul" and criteria is not None:
                instructions += f"\nYes means: {render(criteria['true'])}\nNo means: {render(criteria['false'])}"
            head = f"Context:\n{state}\n\nQuestion: {instructions}\n"
            if kind == "noul":
                prompts.append([head + "Is the answer to this question yes? Answer Yes or No."])
                continue
            if kind == "choice":
                options = [name if value is None else f"{name}: {render(value)}" for name, value in criteria.items()]
            elif kind == "score":
                options = [render(level) for level in criteria]
            else:
                return None
            prompts.append([
                f"{head}Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
                for option in options
            ])
        return prompts if prompts and all(prompts) else None
    except (KeyError, TypeError, AttributeError):
        return None


class OpenJev:
    """The native Open-Jev worker: one sequence per candidate, from its merged export."""

    name = "open_jev"

    def __init__(self, export):
        from tokenizers import Tokenizer

        manifest = json.loads((Path(export) / "open_jev_export.json").read_text())
        self.prefix, self.suffix = manifest["chat_prefix"], manifest["chat_suffix"]
        self.max_length = manifest["max_length"]
        self.tokenizer = Tokenizer.from_file(str(Path(export) / "tokenizer.json"))

    def encode(self, text, special=True):
        return self.tokenizer.encode(text, add_special_tokens=special).ids

    def rows(self, request):
        """Token ids per question, then per candidate; None when the worker would refuse it."""
        prompts = open_jev_prompts(request)
        if prompts is None:
            return None
        rows = [[self.encode(f"{self.prefix}{p.strip()}{self.suffix}") for p in q] for q in prompts]
        return rows if all(len(r) <= self.max_length for q in rows for r in q) else None

    def state(self, request):
        return self.encode(render(request["state"]), special=False)


def cua_s1_contract():
    """The Cua-S1 worker's request mapping and prompts (stdlib only)."""
    spec = importlib.util.spec_from_file_location("cua_s1_contract", ROOT / "src/models/cua_s1/text/contract.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


class CuaS1:
    """The native Cua-S1 text worker: one sequence per Choice question, its goal before the state.
    It runs no prefix planner; its "request-local" share is what the planner would skip."""

    name = "cua_s1"

    def __init__(self, export):
        from tokenizers import Tokenizer

        self.contract = cua_s1_contract()
        self.tokenizer = Tokenizer.from_file(str(Path(export) / "tokenizer.json"))

    def prompts(self, request):
        """The chat text `chat_text` renders per question; None when the worker would refuse it."""
        try:
            state, questions = self.contract.map_request(dict(request, model=self.contract.MODEL_NAME))
        except self.contract.RequestError:
            return None
        out = []
        for question in questions:
            system, user = (m["content"] for m in self.contract.build_messages(state, question))
            out.append(f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                       "<|im_start|>assistant\n<think>\n")
        return out

    max_tokens = 16384  # MAX_PROMPT_TOKENS in src/models/cua_s1/native/src/processing.rs

    def rows(self, request):
        prompts = self.prompts(request)
        if prompts is None:
            return None
        rows = [[self.tokenizer.encode(p, add_special_tokens=False).ids] for p in prompts]
        return rows if all(len(q[0]) <= self.max_tokens for q in rows) else None

    def state(self, request):
        value = request["state"]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return self.tokenizer.encode(text, add_special_tokens=False).ids


def common(rows, skip):
    """`prefix.rs` `common`: tokens all rows share from `skip`, keeping one per row."""
    limit = min(len(r) for r in rows) - 1
    first, n = rows[0], skip
    while n < limit and all(r[n] == first[n] for r in rows):
        n += 1
    return max(n - skip, 0)


def plan(groups):
    """`PrefixPlan::new` on rows grouped by question: (request prefix, its 64-token floor, each group's
    floored prefix end, saved tokens). Without a plan the floors are 0 and nothing is saved."""
    rows = [r for g in groups for r in g]
    if len(rows) < 2 or any(not r for r in rows):
        return 0, 0, [0] * len(groups), 0
    request = common(rows, 0)
    p = request // CHUNK * CHUNK
    saved = (len(rows) - 1) * p
    ends = []
    for group in groups:
        prefix = common(group, request) if len(group) > 1 else 0
        ends.append((request + prefix) // CHUNK * CHUNK)
        saved += (len(group) - 1) * (ends[-1] - p)
    if not saved:
        return request, 0, [0] * len(groups), 0
    return request, p, ends, saved


def worth_auto(saved, total):
    return saved >= 4096 and saved >= -(-total // 3)


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cached(groups, p, ends, earlier):
    """Tokens a cache of earlier prompts, restored at 64-token boundaries, would skip on top of the plan:
    each span the plan computes once (the request prefix, each group prefix, each row's rest) skips what
    a cached prompt already covers. A row keeps at least its last token."""
    if not earlier:
        return 0
    cover = [[max(lcp(row, e) for e in earlier) // CHUNK * CHUNK for row in group] for group in groups]
    saved = min(max(c for g in cover for c in g), p)
    for group, end, c in zip(groups, ends, cover):
        if end > p:
            saved += max(0, min(max(c), end) - p)
        start = end if end else p
        saved += sum(max(0, min(ci, (len(row) - 1) // CHUNK * CHUNK) - start) for row, ci in zip(group, c))
    return saved


def served(record):
    """Whether the recorded server answered: a recorded failure is not measured."""
    return record.get("status") in (None, 200) and "error" not in (record.get("notes") or {})


def measure(requests, layout, last_k=8):
    """One row per request the worker would serve, in trace order."""
    if last_k < 1:
        raise ValueError("last_k must be at least 1")
    out, history, arrivals = [], {}, []
    for record in requests:
        groups = layout.rows(record["request"]) if served(record) else None
        if not groups:
            continue
        first = groups[0][0]
        total = sum(len(r) for g in groups for r in g)
        request_prefix, p, ends, saved = plan(groups)
        state = traces.state_digest(record["request"], record.get("images", []))
        state_ids = layout.state(record["request"])
        images = [i["pixels_sha256"] for i in record.get("images", [])]
        earlier = history.setdefault(record["group"], [])
        row = {
            "id": record["id"], "group": record["group"], "step": record["step"],
            "attempt": record.get("attempt", 0),
            "questions": len(groups), "sequences": sum(len(g) for g in groups),
            "tokens": total, "request_prefix": request_prefix, "local_saved": saved,
            "auto": worth_auto(saved, total),
        }
        if row["attempt"] == 0:  # a retry resends the same bytes; it is not compared
            previous = earlier[-1] if earlier else None
            window = earlier[-last_k:]
            near = max((difflib.SequenceMatcher(None, e["state_ids"], state_ids, autojunk=False).ratio()
                        for e in window), default=0.0)
            row.update({
                "prev_lcp64": lcp(previous["first"], first) // CHUNK * CHUNK if previous else 0,
                "cross_saved_prev": cached(groups, p, ends, previous["rows"] if previous else []),
                "cross_saved_best": cached(groups, p, ends, [r for e in window for r in e["rows"]]),
                # what a server-side cache of the last K requests sees, whatever episode they came from
                "cross_saved_server": cached(groups, p, ends, [r for e in arrivals[-last_k:] for r in e["rows"]]),
                "state_seen_prev": bool(previous and previous["state"] == state),
                "state_seen_before": any(e["state"] == state for e in earlier),
                "near_duplicate": round(near, 4),
                "images": len(images),
                "images_seen_prev": sum(i in (previous["images"] if previous else ()) for i in images),
                "images_seen_before": sum(any(i in e["images"] for e in earlier) for i in images),
            })
            earlier.append({"first": first, "state": state, "state_ids": state_ids, "images": images,
                            "rows": [r for g in groups for r in g]})
            arrivals.append(earlier[-1])
        out.append(row)
    return out


def summarize(rows):
    firsts = [r for r in rows if r["attempt"] == 0]
    total = sum(r["tokens"] for r in rows)
    local = sum(r["local_saved"] for r in rows)
    images = sum(r["images"] for r in firsts)

    def share(n, d):
        return round(n / d, 4) if d else None

    def mean(values):
        return statistics.mean(values) if values else None

    return {
        "requests": len(rows), "retries": len(rows) - len(firsts),
        "episodes": len({r["group"] for r in rows}),
        "questions_per_request": mean([r["questions"] for r in rows]),
        "sequences_per_request": mean([r["sequences"] for r in rows]),
        "tokens": total,
        "skipped_share": {
            "request_local": share(local, total),
            "plus_cache_of_previous_request": share(local + sum(r["cross_saved_prev"] for r in firsts), total),
            "plus_cache_of_last_k_requests": share(local + sum(r["cross_saved_best"] for r in firsts), total),
            "plus_server_cache_of_last_k": share(local + sum(r["cross_saved_server"] for r in firsts), total),
        },
        "auto_requests": sum(r["auto"] for r in rows),
        "state_seen_prev": share(sum(r["state_seen_prev"] for r in firsts), len(firsts)),
        "state_seen_before": share(sum(r["state_seen_before"] for r in firsts), len(firsts)),
        "near_duplicate_0_9": share(sum(r["near_duplicate"] >= 0.9 for r in firsts), len(firsts)),
        "images": images,
        "images_seen_prev": share(sum(r["images_seen_prev"] for r in firsts), images),
        "images_seen_before": share(sum(r["images_seen_before"] for r in firsts), images),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--layout", choices=("open_jev", "cua_s1"), default="open_jev")
    parser.add_argument("--export", type=Path, required=True,
                        help="the worker's export directory (its tokenizer.json; Open-Jev also its manifest)")
    parser.add_argument("--last-k", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.last_k < 1:
        parser.error("--last-k must be at least 1")
    _, requests, _ = traces.read(args.trace)
    layout = (OpenJev if args.layout == "open_jev" else CuaS1)(args.export)
    rows = measure(requests, layout, args.last_k)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"trace": str(args.trace), "layout": layout.name, "last_k": args.last_k,
               "unserved_requests": len(requests) - len(rows), **summarize(rows)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
