# hk2 nightly GPU benchmark

这是 hk2 本机 nightly，不是 GitHub Actions，也不会在托管 runner 上跑 GPU。
每天 02:00（Asia/Shanghai）默认只测 **Open-Jev-9B** 与 **Open-Jev-27B-v1.1** 的 Public231 文本、HTTP 闭环、并发 C1。
Intern-Decision-0.8B / 2B / 4B 默认关闭。C8/C16 和 packing probe 不是 headline。
缺少 `download` + `export_merged` 合并权重时，进程立刻以非 0 退出，中英文说明原因，不会挂起。

This is a local nightly for the hk2 GPU host (4×L20X). It does not replace CI.
The job reuses [`benchmarks/bench.py`](../bench.py). It does not contain a second
HTTP client. D3, Vela and other unmerged model RFCs are rejected.

Primary operator: `hsliu`. The wrapper looks for a checkout at
`~/repos/ThinkFlowLab/system1-omni`, then
`~/repos/InferMatrixLab/system1-omni`, unless `OMNI_NIGHTLY_REPO` is set or the
script itself lives in the checkout.

## What a night runs

| Arm | When | Dashboard role |
| --- | --- | --- |
| Open-Jev-9B, Omni native | every night | headline, Public231, C1 |
| Open-Jev-27B-v1.1, Omni native | every night | headline, Public231, C1 |
| 27B packing probe (4 multi-candidate requests) | every night, after a successful 27B headline | `packing_probe`, not headline |
| Intern-Decision-0.8B / 2B / 4B, official HF | `OMNI_NIGHTLY_INTERN=1` or `--intern` | `optional`. Same five-model board when present; not required |
| C8 / C16 | `OMNI_NIGHTLY_REGRESSION_CONCURRENCY=8,16` | `regression-only`. Not a war-report number |

Each headline model gets one smoke feasibility pass (readiness only), then one
measured Public231 traversal at C1 with five warmups. Set `OMNI_NIGHTLY_REPEATS`
to 2 or 3 to match the historical three-traversal budget. The harness pools
successful `latency_ms` samples: p50 is `statistics.median`, p95 is nearest rank.
Per-round files stay in the raw directory.

The full [prefill packing campaign](../prefill_batching/README.md) does not fit
this window. The probe only checks that 27B still answers a four-request
multi-candidate manifest. Its `delta_p50_ms_vs_previous` is milliseconds against
the previous probe (negative is faster). It is not the H200 percent reduction.

## Install

Python 3.11+ and `benchmarks/requirements.txt` (`httpx==0.28.1`). The Open-Jev
worker is the release binary from [the native recipe](../../recipe/open_jev/native.md),
not this virtualenv.

Use a dedicated clone. The default job fast-forwards `main` and refuses a dirty
tree. If `HEAD` moves, the process re-execs so the new harness handles the run.

Cron, 02:00 Asia/Shanghai. `CRON_TZ` needs cronie (or another cron that honors
it). The summary date always uses `Asia/Shanghai` even when the host clock is
UTC; the schedule itself follows cron. Prefer the systemd timer below when
`CRON_TZ` is unsupported. Point the path at whichever checkout exists:

```cron
CRON_TZ=Asia/Shanghai
0 2 * * * /bin/sh /home/hsliu/repos/InferMatrixLab/system1-omni/benchmarks/nightly/run_nightly.sh >> /home/hsliu/experiments/system1-omni-nightly/cron.log 2>&1
```

`flock -n` makes a second start exit immediately.

systemd, user `hsliu`:

```ini
# /etc/systemd/system/system1-omni-nightly.service
[Service]
Type=oneshot
User=hsliu
ExecStart=/bin/sh /home/hsliu/repos/InferMatrixLab/system1-omni/benchmarks/nightly/run_nightly.sh

# /etc/systemd/system/system1-omni-nightly.timer
[Timer]
OnCalendar=*-*-* 02:00:00
Timezone=Asia/Shanghai
Persistent=true
```

Before enabling the timer:

```sh
python3 benchmarks/nightly/run_nightly.py --self-check
python3 benchmarks/nightly/run_nightly.py --dry-run
```

`--self-check` validates the config, packing probe and schema example. It does
not need weights. `--dry-run` checks weights, binaries and the Public231
manifest, prints the plan, and does not start a server or call `bench.py`.

## Weights and binaries

Both headline models need a merged export. A plain Hub download is not enough.
Follow [recipe/open_jev/native.md](../../recipe/open_jev/native.md):

- `weights/open-jev-9b-merged/open_jev_export.json`
- `weights/open-jev-27b-merged/open_jev_export.json`

The marker format must be `open-jev-text-merged/1`, the pinned revisions must
match the recipe, and a non-empty `*.safetensors` shard must exist. Otherwise
the process exits 2. The message names `download` and `export_merged` in
English and Chinese.

Launch also needs `target/release/omni-open-jev-native` and `target/release/omni-jev`.
Build them with the native recipe, or set `OMNI_NIGHTLY_BUILD=1` together with
`OMNI_NIGHTLY_CUDA_ARCH` (the device compute capability, for example `90`).
This harness does not guess the arch. `build.sh` itself defaults to 89 when the
arch argument is omitted; the historical packing note's scheduler label "L20X"
was an H200 (sm_90) on that host. Read `nvidia-smi` on hk2 and set the arch
from that device before building.

The worker is started with `CUA_S1_GRAPH` unset. Default device is
`OMNI_NIGHTLY_CUDA_VISIBLE_DEVICES=0` (else `CUDA_VISIBLE_DEVICES`, else `0`).
Models run one after another on that device. If it already holds more than
`OMNI_NIGHTLY_MAX_GPU_USED_MIB` (default 2048) the launch is refused. `-1`
disables the check. An endpoint that is already healthy is attached and does
not start a second worker:

```sh
OMNI_NIGHTLY_ENDPOINT_OPEN_JEV_9B=http://127.0.0.1:8080/v1/systemone
OMNI_NIGHTLY_ENDPOINT_OPEN_JEV_27B=http://127.0.0.1:8082/v1/systemone
```

`/health` must report the export base id (`Qwen/Qwen3.5-9B` or
`Qwen/Qwen3.8-27B`). A down endpoint exits. It is not polled for the model-load
timeout.

## Intern-Decision

Leave this off for the nightly. Weekly, or whenever the official HF servers are
already up:

```sh
OMNI_NIGHTLY_INTERN=1
OMNI_NIGHTLY_ENDPOINT_INTERN_0_8B=http://127.0.0.1:8090/v1/systemone
OMNI_NIGHTLY_ENDPOINT_INTERN_2B=http://127.0.0.1:8092/v1/systemone
OMNI_NIGHTLY_ENDPOINT_INTERN_4B=http://127.0.0.1:8094/v1/systemone
```

Hub ids are `internlm/Intern-Decision-0.8B`, `internlm/Intern-Decision-2B` and
`internlm/Intern-Decision-4B`. This repo does not launch those servers. The
`model` alias sent by `bench.py` defaults to `intern-decision-0.8b`,
`intern-decision-2b` and `intern-decision-4b`. Change `alias` in a copied
config (`OMNI_NIGHTLY_CONFIG`) if the server expects another name.
`score_decimals` stays at 4, which is `bench.py`'s default. Native Decider in
this repo is a different server and uses two decimals. If Intern warmup fails
on score rounding, set `score_decimals` in that config; do not guess it here.

## Public231 manifest

The headline file is not in git. Put the frozen JSONL at:

```text
~/experiments/system1-omni-nightly/manifests/public231.jsonl
```

The name must contain `public231`, or pass `--workload public231`. The file
must contain 231 requests. Optional pin: `OMNI_NIGHTLY_MANIFEST_SHA256`.
The two-model export hashed in the leaderboard notes is
`00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3`.
That is a file checksum, not the canonical dataset hash. `--smoke` runs
`benchmarks/smoke.jsonl` and marks every arm `smoke`, so it cannot enter
`headline_p50_ms`.

## Where results go

| Path | Contents |
| --- | --- |
| `benchmarks/nightly/results/<date>.json` | small summary. Gitignored |
| `benchmarks/nightly/results/latest.json` | newest summary |
| `benchmarks/nightly/results/recent.md` | rolling table |
| `~/experiments/system1-omni-nightly/<run-id>/` | raw `bench.py` archives and server logs |
| `~/experiments/system1-omni-nightly/summaries/` | copy of the small JSON |

A second run the same day does not overwrite `<date>.json`. `latest.json` moves
forward. Do not commit raw latency dumps. [`example-latest.json`](example-latest.json)
is a schema sample (`"illustrative": true`), not a measurement.

Exit 0: every headline arm completed with zero failures, and the issue comment
succeeded when one was requested. Exit 2: preflight (weights, manifest, intern
endpoint, dirty tree, lock, binaries). Exit 1: a bench arm, timeout or issue
comment failed after the local summary was written. Bench invocations are killed
at two hours. Model load waits at most `OMNI_NIGHTLY_READY_TIMEOUT` seconds
(default 900).

## Dashboard

Read `benchmarks/nightly/results/latest.json`, schema
`system1-omni-nightly/1`.

```sh
jq '{date, git_sha, gpu, ok, headline_p50_ms, headline_p95_ms, packing: .packing_probe.delta_p50_ms_vs_previous}' \
  benchmarks/nightly/results/latest.json
```

Day-over-day movement is the difference of `headline_p50_ms["open-jev-9b"]` and
`headline_p50_ms["open-jev-27b-v1.1"]` between dated files. Negative means
faster. Plot a model only when that arm has `leaderboard_eligible: true`.
Top-level `ok` is false when any arm, the packing probe, or the issue comment
failed. A failed probe does not remove a completed headline arm from
`headline_p50_ms`. `recent.md` is the same series for a person.

Ignore, for the headline series:

- `role: regression-only` (C8/C16). Present in `issue_digest.regression_only` so it stays visible and unlabeled as a win.
- `packing_probe`. Use `delta_p50_ms_vs_previous` only as a cheap regression hint.
- `role: smoke` and `role: custom`.
- `role: optional` when the question is "did the required nightly pair move?". Include those arms only on the five-model board, and only on days they ran.

`comparable_to_h800_historical_board` is false. The clock is
`post_to_body_json_and_schema_validation`. The [H800 board](../latency-leaderboard.md)
used an earlier body-receipt clock on different hardware.

## Tracking issue

Maintainers: open one issue, for example "hk2 nightly GPU summaries", and set
`OMNI_NIGHTLY_ISSUE` to that number (or the issue URL) on the host. Each run
appends a comment whose fenced JSON is `issue_digest`: date, git SHA, GPU,
`headline_p50_ms`, `headline_p95_ms`, packing probe and regression-only rows.
The bot digest can read that thread without SSH. Until the variable is set, the
harness writes local files and skips the comment. This repository does not pick
an issue number. There is no default.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMNI_NIGHTLY_REPO` | detected checkout | Repo to run |
| `OMNI_NIGHTLY_REF` | `main` | Branch to fast-forward |
| `OMNI_NIGHTLY_UPDATE` | `1` | Pull when the tree is clean |
| `OMNI_NIGHTLY_MANIFEST` | `~/experiments/.../public231.jsonl` | Headline JSONL |
| `OMNI_NIGHTLY_WORKLOAD` | from the file name | `public231`, `smoke` or `custom` |
| `OMNI_NIGHTLY_INTERN` | off | Also run the three HF arms |
| `OMNI_NIGHTLY_REGRESSION_CONCURRENCY` | empty | Comma-separated integers ≥ 2 |
| `OMNI_NIGHTLY_REPEATS` | `1` | Measured C1 repeats, max 3 |
| `OMNI_NIGHTLY_WARMUP` | `5` | Measured warmups per repeat |
| `OMNI_NIGHTLY_TIMEOUT` | `120` | Per-request HTTP timeout, seconds |
| `OMNI_NIGHTLY_READY_TIMEOUT` | `900` | Model-load wait, seconds |
| `OMNI_NIGHTLY_PACKING` | `1` | 27B probe after Public231 |
| `OMNI_NIGHTLY_BUILD` | `0` | Rebuild CUDA and the workers |
| `OMNI_NIGHTLY_CUDA_ARCH` | unset | Required when building |
| `OMNI_NIGHTLY_CUDA_VISIBLE_DEVICES` | `0` | Device passed to the worker |
| `OMNI_NIGHTLY_MAX_GPU_USED_MIB` | `2048` | Refuse to launch above this |
| `OMNI_NIGHTLY_RESULTS` | `benchmarks/nightly/results` | Summary directory |
| `OMNI_NIGHTLY_RAW` | `~/experiments/system1-omni-nightly` | Raw archives |
| `OMNI_NIGHTLY_ISSUE` | unset | Tracking issue number or URL |
| `OMNI_NIGHTLY_HOST` | hostname | Recorded host, set `hk2` |
| `OMNI_NIGHTLY_LOCK` | `~/experiments/system1-omni-nightly/nightly.lock` | Non-blocking lock |
| `OMNI_JEV_TEST_TOKEN` | unset | Bearer token for `bench.py`. Not written to summaries |

Copied config: `OMNI_NIGHTLY_CONFIG` replaces [`config.example.json`](config.example.json).
Keep the same model ids. Unknown launch modes and excluded model ids fail closed.
