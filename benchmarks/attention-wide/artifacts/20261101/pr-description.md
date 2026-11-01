# PR: 128-query attention tile on sm_89 / sm_120

## Required human contributor answers

The answers below are transcribed from the maintainer's own instructions during
the 2026-11-01 session, at the maintainer's explicit request ("你来填，填完以后
开draft pr"). Each answer cites the message it comes from; nothing here is an
agent-invented rationale.

1. **What outcome are you trying to achieve?**
   A CUDA kernel improvement in this repository that is actually verified on the
   hardware we can measure, and a compliant evidence package for it. Source:
   "查看仓库中的cuda算子，激活cuda开发skill，看看有没有能优化的" and, after an
   unverified laptop result, "在l20上面测符合要求吗" / "只能用l20，h200我们没有权限" /
   "在sm120上跑一下" / "同机同协议的 A/B 记录 + PR 里的 RSI 阶段/备选方案说明 这是什么，你来完成一下".

2. **What is the highest-leverage variable for that outcome?**
   Occupancy and K/V reuse of the causal attention kernel on long prefill
   windows. The kernel is capped at one block per SM by 67,584 bytes of dynamic
   shared memory, so the query tile alone decides warps per SM (4 versus 8) and
   how many query rows share one K/V tile (64 versus 128). Source: the profile
   request "看看有没有能优化的", then the measured 1 block/SM, 4 warps result that
   motivated the exploration.

3. **How will this work move that variable rather than create busywork?**
   It changes only that tile: a 128-query, 256-thread instantiation with the same
   32-key tiles, the same online-softmax recurrence and the same epilogue, so the
   outputs stay byte-identical. It is enabled only on the two measured
   capabilities and only for windows of at least 640 rows; the shared-prefix
   kernel, the 64-query kernel and every other architecture are untouched.
   Source: "同机同协议的 A/B 记录 … 你来完成一下" is the maintainer's acceptance
   framing, and the scope follows the measured data (512 rows is 4.9% slower on
   L20).

4. **Who is the customer or user, and what do they actually need?**
   Native CUDA consumers of `libqwen3_5_cuda.so` in this repo that run on
   L20-class (sm_89) hosts and on consumer Blackwell (sm_120) workstations: they
   need lower long-prompt attention latency with unchanged decision outputs.
   Source: the maintainer's constraint "只能用l20，h200我们没有权限" identifies the
   L20 deployment class, and "在sm120上跑一下" adds the local sm_120 workstation.

**After answering the four questions: is a PR needed at all to achieve this
outcome?** Yes. Source: the explicit instruction "填完以后开draft pr".

**Bug fix?** N/A. This is a performance change with byte-identical output.

## Purpose

Long causal attention windows run at four warps per SM because the block's
67,584 bytes of shared memory hold one block. A 128-query, 256-thread block with
the same 32-key tiles raises that to eight warps per SM and reads each K/V tile
once for 128 queries instead of 64, at 101,376 bytes, the sm_89/sm_120 opt-in
limit, still one block per SM.

A query row keeps the same key-tile walk, softmax recurrence and epilogue, so
every measured output is byte-identical to the 64-query block; only which block
computes a row changes.

Scope:

- Enabled for the causal full pass and 128-aligned prefix windows of at least
  640 query rows on sm_89 and sm_120, measured on one L20 and one RTX 5060
  Laptop.
- `cs1_attention_gated_shared`, the batch entry points and every other
  architecture keep 64 queries. `cs1_attention_gated_prefix` uses the wider tile
  only when `q_base % 128 == 0`.
- H200/sm_90 and end-to-end serving are unverified; the dispatch keeps them on
  64 queries until measured.

## Test Plan

- Frozen protocol, controls and budget: `benchmarks/attention-wide/artifacts/20261101/plan.json`.
- Kernel arms, two trials per device: `harness/attn_record.cu`.
- Public entry point, two trials per device, with the base and candidate library
  swapped between arms: `harness/attn_lib_record.cu`.
- Prefix windows against the full pass: `harness/attn_prefix_check.cu`.
- Parity: byte comparison of both arms and both revisions at every length.
- Memory safety: `compute-sanitizer --tool memcheck --error-exitcode 1`.
- Kernel tests: `CUA_S1_CUDA_LIB=<candidate> cargo test --release -p omni-qwen3-5-native --test kernels -- --ignored --test-threads=1 attention`.
- Aggregate rebuild: `python3 benchmarks/attention-wide/artifacts/20261101/harness/summarize.py`.

## Test Result

System1-Omni version / commit: base `5fa5179d57f9267bb0d20162004d8ce26352aa9d`
plus the branch commit.

Public `cs1_attention_gated`, medians of two trials, old -> new:

| tokens | NVIDIA L20 (sm_89, 92 SMs) | RTX 5060 Laptop (sm_120, 26 SMs) |
| ---: | --- | --- |
| 512 | 0.1300 -> 0.1296 ms (-0.3%) | 0.7720 -> 0.7720 ms (0.0%) |
| 640 | 0.1747 -> 0.1698 (-2.8%) | 1.0478 -> 0.8279 (-21.0%) |
| 1,024 | 0.3323 -> 0.3144 (-5.4%) | 2.2597 -> 1.8092 (-19.9%) |
| 2,048 | 0.9859 -> 0.8998 (-8.7%) | 6.9033 -> 5.5222 (-20.0%) |
| 4,096 | 3.4010 -> 2.9960 (-11.9%) | 23.9185 -> 19.4021 (-18.9%) |
| 8,192 | 12.0261 -> 10.4388 (-13.2%) | 87.1708 -> 71.5387 (-17.9%) |

Kernel arms alone: L20 -2.2% at 640 rows to -13.8% at 8,192, and +4.9% at 512
rows (why the threshold is 640); sm_120 -13.0% to -22.4% from 512 rows up.

Fidelity on both devices: byte-identical outputs at every measured length in
both trials; prefix windows at q_base 64/128/1,024 match the full pass;
compute-sanitizer reports 0 errors at 640, 1,024, 4,096 and 8,192 rows; the
attention kernel tests pass with the wider tile active, and the full ignored
suite is 26 passed with 1 failure that only reports the unset
`QWEN3_5_CHECKPOINT`.

## Demo / evidence

`benchmarks/attention-wide/README.md` (record), `artifacts/20261101/plan.json`
(frozen protocol), `summary.json` (aggregates, p10/p90, per-trial deltas),
`results_*.jsonl` and `lib_*.jsonl` (every batch), `parity_*.log`,
`sanitizer_*.log`, and the frozen harness sources. No HTTP, Public231 C1 or
whole-model measurement is included; the numbers are kernel and entry-point
latency on two named devices.

## Self-review

- **RSI phase:** kernel (occupancy and K/V reuse in one kernel; no scheduling,
  launch, or parallelism-across-devices change).
- **Primary change:** one extra kernel instantiation with a 128-query,
  256-thread block plus a launch-time choice of tile; the existing 64-query
  kernel is unchanged and still used below the threshold, off the two measured
  capabilities, and for unaligned prefix windows.
- **Alternatives considered:** two blocks per SM via a swizzled layout and
  16-key tiles (numerics change, not taken); 16-key tiles with the padded layout
  (103,424 bytes for two blocks, over the 102,400-byte SM budget); a 192-query
  tile (135,168 bytes, over the opt-in limit); lowering registers to fit a
  second block (shared memory, not registers, is the limit); enabling the wider
  tile at every length (512 rows is 4.9% slower on L20).
- **Changed size:** 2 source files plus the measurement record.
- **Validation by changed area:** CUDA backend tests, sanitizer, kernel parity,
  and the attention kernel tests listed above; CPU-only CI covers the Rust
  crates unchanged here.
- **Still unverified:** H200/sm_90, the shared-prefix kernel, HTTP/Public231 C1,
  and whole-model parity on the changed tile. Keep this PR in draft until the
  owner's separate yes.
