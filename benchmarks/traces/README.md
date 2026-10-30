# Decision traces

Tools for #150's first section: record what agents actually send, measure how much of
it repeats, and replay it at the pace the agent sent it.

- [`tracefile.py`](tracefile.py) checks a trace and converts saved System1-Agents ticks.
- [`record_s1a.py`](record_s1a.py) records System1-Agents' decisions as a trace without
  changing its source.
- [`metrics.py`](metrics.py) measures repeated work as one worker tokenizes it.
- [`replay.py`](replay.py) replays a trace against a running server, one client per
  episode, keeping its order and think time.
- [`samples/`](samples/) holds two small traces whose content may be shared.
- The [historical report from 2026-10-25](https://github.com/InferMatrixLab/system1-omni/blob/c18e9e7c634412e623cfa0f18aed25809c3d75a0/benchmarks/traces/results-2026-10-25.md)
  remains available at the original revision for [PR #185](https://github.com/InferMatrixLab/system1-omni/pull/185).
  It is author-reported experiment evidence, not a maintained benchmark input or a
  validation of the current revision.

They need Python 3.11+ and `httpx==0.28.1`; `metrics.py` also needs `tokenizers`.

## Format `s1-trace/1`

JSONL with three record types:

- **header**, first and once: `format`, `source` (repo, commit, front, argv), `server`
  (repo, commit, worker, url, model), `capture`, `clock`, `content` (provenance,
  license, redaction) and `images.store`. `capture` is `client:transport` for
  `record_s1a.py` and `converted` for `tracefile.py from-ticks`.
- **request**, one per attempt: `id`, `group` (the episode), `step`, `attempt`,
  `retry_of`, `request` (the body as sent, key order kept, inline images replaced by
  `{"$image": "<sha256>"}`), `images` (`sha256`, `pixels_sha256`, size, media type),
  `t_send_ms`, `t_recv_ms`, `think_ms` (time since the previous response in the
  episode), `deadline_s`, `status`, `response`, and free `notes` (a failure's error).
- **episode_end**: `group`, `status`, `steps`.

`id`, `group` and `request` match the manifest rows of System1-Agents' proposed
changed-image replay (ThinkFlowLab/system1-agents#72). Steps and attempts increase within
an episode; a retry names the id it repeats, so identical resent bytes are not counted as
a repeated state. Token ids are not stored: they depend on each worker's rendering and
tokenizer.

## Recording System1-Agents

`record_s1a.py` wraps `JevDecisionsClient.decide`, the HTTP client behind `--model jev`,
and the tool loop's `run_episode`. Each run of the tool loop is an episode; other fronts
take `--group`, one episode per invocation. Run it with System1-Agents' Python, from its
checkout, against a decision server:

```sh
TYPESAFE_API_KEY= TYPESAFE_API_URL=http://127.0.0.1:8000/v1/systemone TYPESAFE_MODEL=jev-latest \
OPENROUTER_API_KEY=local .venv/bin/python /path/to/benchmarks/traces/record_s1a.py \
  blackjack.trace.jsonl --commit "$(git rev-parse HEAD)" -- run blackjack --model jev --rethink off --episodes 20
```

A failed decision keeps its HTTP status when the error names one.

## Metrics

`metrics.py --layout open_jev|cua_s1 --export <worker export>` renders and tokenizes each
request as that native worker does. Open-Jev has one sequence per candidate with the
state first; its prompts match the worker's contract fixture. Cua-S1 has one sequence per
Choice question with the question's goal first, built by
`src/models/cua_s1/text/contract.py`. Requests the trace records as failed, malformed
requests, prompts over the worker's token limit, and anything Cua-S1's request mapping
rejects are counted as unserved and not measured. Per request:

- questions, sequences and their tokens;
- tokens the request-local planner would not recompute (the rule of Decider's
  `src/models/decider/native/src/prefix.rs`, 64-token chunks). Cua-S1 runs no planner, so
  for it this share is hypothetical;
- tokens a cache of earlier prompts would skip on top of that: each span the plan computes
  once (the request prefix, each group prefix, each sequence's rest) skips whatever an
  earlier sequence already covers, in whole 64-token chunks, keeping each sequence's last
  token. Earlier means the previous request of the episode, the last K of the episode, or
  the last K the server received from any episode;
- whether the state (its bytes and its images' pixels) was seen at the previous step or
  earlier, the best token similarity of its state to the last K states, and how many of
  its images were seen before.

The summary turns these into the share of prefill tokens each kind of reuse could skip.

## Replay

```sh
python benchmarks/traces/replay.py trace.jsonl --url http://127.0.0.1:8000 --out out/
```

Episodes start at their recorded offsets and run concurrently; `--together` starts them
at once and `--sequential` runs them one after another. Offsets come from one recorder
clock, so replay traces concatenated from several recordings with `--sequential`. `--think-scale 0` sends each step
as soon as the last returns, `--skip-retries` drops repeated attempts, and
`--deadline-s` counts misses against a deadline other than the recorded one. Latency and
misses count answered requests only. The output has `config.json` (with the trace's
checksum), `responses.jsonl` (latency, status, deadline miss, `prefix_cache`, base64
body) and `summary.json` (median and nearest-rank p95, as `bench.py` reports them).

`--out` must name a new directory; an existing directory, even an empty one, is
rejected before sending requests or modifying its contents. `config.json` lists
all planned request IDs after `--skip-retries` filtering. Each finished attempt is
appended and flushed to `responses.jsonl` immediately, in completion order; use
`sent_s` to recover send order. On Ctrl-C or a replay error, all episode tasks stop
before the files are finalized, and the process exits unsuccessfully. Every
planned request then has a row with one of these states:

- `completed`: an HTTP response or transport error was observed;
- `incomplete`: it was attempted but interrupted before its response completed;
- `unattempted`: it was still waiting for its episode start or next step.

Incomplete and unattempted rows have no response latency or deadline result.
The summary separates `completed`, `incomplete`, and `unattempted`; `failed`
counts completed HTTP/transport failures, and `ok` counts completed HTTP 200
responses. It also records `stop_reason` (`completed`, `interrupted`, or `error`).
Latency/deadline statistics exclude unfinished work, while `requests` retains the
full planned denominator. A forced kill or machine failure cannot run finalization:
keep the already-flushed responses and config as partial evidence, not a completed
run. A flush is not a power-loss durability guarantee.

## Data

Traces of real web pages copy third-party content. Keep them outside this repository
and retain raw run output and derived reports in PR/CI evidence or durable artifact
archives. The source tree should contain maintained replay inputs or fixtures from
local fixtures, self-hosted pages or sets whose license allows it.
