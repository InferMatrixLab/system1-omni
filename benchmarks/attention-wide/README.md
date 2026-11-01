# 128-query attention tile

The full pass (causal or bidirectional) and 128-aligned causal prefix windows of
at least 640 query rows use a 128-query, 256-thread block instead of 64 queries
and 128 threads on sm_89 and sm_120. Each query row keeps the same 32-key tile
walk and epilogue, so the outputs are byte-identical; only occupancy and K/V
reuse change. Code:
[`attention.cu`](../../src/backends/cuda/qwen3_5/attention.cu).

## Why

Both tiles stay at one block per SM (67,584 and 101,376 bytes of shared memory,
the latter the sm_89/sm_120 opt-in limit), so the query tile decides warps per
SM: four for 64 queries, eight for 128, which also halves K/V reads per query.

| device | tile | regs/thread | smem/block | warps/SM |
| --- | --- | ---: | ---: | ---: |
| L20 (sm_89) | 64 x 128 | 244 | 67,584 | 4 |
| L20 (sm_89) | 128 x 256 | 244 | 101,376 | 8 |
| RTX 5060 Laptop (sm_120) | 64 x 128 | 228 | 67,584 | 4 |
| RTX 5060 Laptop (sm_120) | 128 x 256 | 222 | 101,376 | 8 |

## Protocol

Frozen before the runs in [`plan.json`](plan.json): one variable (the launch
tile), seeded incompressible bfloat16 inputs, Qwen3.5-27B attention shape
(Hq=24, Hk=4, head dim 256), one excluded warmup pair per length, then
interleaved A B B A, 10 repetitions, each sample the mean of 5 launches between
two CUDA events, medians per length, two independent trials per device. Kernel
and public-entry-point latency only, not HTTP.

## Results

Public `cs1_attention_gated`, old -> new, medians of the two trials:

| tokens | L20 (sm_89) | RTX 5060 Laptop (sm_120) |
| ---: | --- | --- |
| 512 | 0.1300 -> 0.1296 ms (-0.3%) | 0.7720 -> 0.7720 ms (0.0%) |
| 640 | 0.1747 -> 0.1698 (-2.8%) | 1.0478 -> 0.8279 (-21.0%) |
| 1,024 | 0.3323 -> 0.3144 (-5.4%) | 2.2597 -> 1.8092 (-19.9%) |
| 2,048 | 0.9859 -> 0.8998 (-8.7%) | 6.9033 -> 5.5222 (-20.0%) |
| 4,096 | 3.4010 -> 2.9960 (-11.9%) | 23.9185 -> 19.4021 (-18.9%) |
| 8,192 | 12.0261 -> 10.4388 (-13.2%) | 87.1708 -> 71.5387 (-17.9%) |

Kernel arms alone: L20 -2.2% at 640 rows to -13.8% at 8,192, and +4.9% at 512
rows, which is why the tile is gated at 640; sm_120 -13.0% to -22.4% from 512
rows up. Per-trial medians and delta ranges are in [`results.json`](results.json).

## Fidelity

- Byte-identical outputs at every measured length in both trials on both
  devices, and prefix windows at q_base 64/128/1,024 match the full pass.
- `compute-sanitizer --tool memcheck`: 0 errors at 640, 1,024, 4,096 and 8,192
  rows on both devices.
- Attention kernel tests pass with the wider tile active, including
  `windowed_attention_matches_full_pass_bit_for_bit` and the float64 reference
  at 700 and 2,048 rows.

## Limits

- Measured on one L20 and one RTX 5060 Laptop. `H200`/sm_90 is unverified and
  stays on the 64-query tile.
- `cs1_attention_gated_shared` keeps 64 queries; `cs1_attention_gated_prefix`
  uses the wider tile only when `q_base % 128 == 0`. The bidirectional entry
  point shares `launch<>` and widens the same way (it always has `q_base = 0`).
- No HTTP or Public231 C1 measurement; a serving-level claim needs the full-C1
  protocol on the same named hardware as its baseline.

## Reproduction

The per-batch records from both runs are in commit `a3bb8c4` on this branch;
this tree keeps the frozen protocol and the per-trial results. Build each
revision for the target (`build.sh <out> 89` on CUDA 12.1, `120` on CUDA 12.9)
and follow the protocol above.
