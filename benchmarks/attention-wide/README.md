# 128-query attention tile

The causal attention full pass and 128-aligned prefix windows of at least 640
query rows run a 128-query, 256-thread block instead of 64 queries and 128
threads on sm_89 and sm_120. A query row still walks the same 32-key tiles in
the same order, so the outputs are byte-identical; only occupancy and K/V reuse
change.

Kernel code and its unit description are in
[`src/backends/cuda/qwen3_5/attention.cu`](../../src/backends/cuda/qwen3_5/attention.cu)
and the [backend README](../../src/backends/cuda/qwen3_5/README.md). This
directory is the measurement record for that change; it is not a maintained
benchmark suite.

## Why

Both tiles need 101,376 bytes or less of dynamic shared memory, the sm_89 and
sm_120 opt-in limit, so each stays at one block per SM. The query tile is what
decides how many warps that one block holds:

| device | tile | regs/thread | smem/block | blocks/SM | warps/SM |
| --- | --- | ---: | ---: | ---: | ---: |
| L20 (sm_89) | 64 x 128 | 244 | 67,584 | 1 | 4 |
| L20 (sm_89) | 128 x 256 | 244 | 101,376 | 1 | 8 |
| RTX 5060 Laptop (sm_120) | 64 x 128 | 228 | 67,584 | 1 | 4 |
| RTX 5060 Laptop (sm_120) | 128 x 256 | 222 | 101,376 | 1 | 8 |

The wider tile also reads each K/V tile once for 128 queries instead of 64. On
L20 the narrower tile still hides memory latency through block count, so the
wider one only wins from 640 query rows; on the 26-SM sm_120 it wins from the
shortest measured length.

## Protocol

Frozen before the recorded runs in [`artifacts/20261101/plan.json`](artifacts/20261101/plan.json):

- Isolated variable: `flash_kernel<64,128>` versus `flash_kernel<128,256>`. Same
  key tiles, order, online-softmax recurrence, epilogue, inputs, buffers, stream.
- Inputs: seeded (mt19937 `20261101`) incompressible bfloat16 bytes; Qwen3.5-27B
  full-attention shape Hq=24, Hk=4, head dim 256, ldv=Hk*head_dim.
- Timing: interleaved A B B A, 10 repetitions per length, each sample the mean of
  5 launches between two CUDA events, medians reported; two independent processes
  (trials) per device, run back to back. No warm cache drops, no clock changes.
- Budget: one excluded warmup pair per length; the budget above and nothing else.
- Measurements are kernel and public-entry-point latency only. They are not HTTP
  or end-to-end numbers.

Each device was measured through the same program built for its own
architecture: `nvcc -O3 -std=c++17`, `arch=compute_89,code=sm_89` on CUDA 12.1
(L20, driver 580.126.20) and `-arch=sm_120` on CUDA 12.9 (RTX 5060 Laptop,
driver 575.57.08).

## Results

Medians of the two trials; `range` is the smaller..larger per-trial delta.
Negative is faster. 512 rows are shown to document the threshold: on L20 the
wider tile is 5% slower there, so it is not used below 640 rows.

### Public `cs1_attention_gated` (the shipped dispatch)

| tokens | L20 old -> new (ms) | delta | RTX 5060 old -> new (ms) | delta |
| ---: | --- | ---: | --- | ---: |
| 512 | 0.1300 -> 0.1296 | -0.3% | 0.7720 -> 0.7720 | 0.0% |
| 640 | 0.1747 -> 0.1698 | -2.8% | 1.0478 -> 0.8279 | -21.0% |
| 1,024 | 0.3323 -> 0.3144 | -5.4% | 2.2597 -> 1.8092 | -19.9% |
| 2,048 | 0.9859 -> 0.8998 | -8.7% | 6.9033 -> 5.5222 | -20.0% |
| 4,096 | 3.4010 -> 2.9960 | -11.9% | 23.9185 -> 19.4021 | -18.9% |
| 8,192 | 12.0261 -> 10.4388 | -13.2% | 87.1708 -> 71.5387 | -17.9% |

Trial-to-trial delta ranges: L20 640 rows -2.62..-3.00%, 8,192 rows
-13.18..-13.22%; sm_120 640 rows -20.65..-21.32%, 8,192 rows -17.67..-18.20%.

### Kernel-level arms (both tiles in one process)

| tokens | L20 delta | RTX 5060 delta |
| ---: | ---: | ---: |
| 512 | +4.9% | -16.6% |
| 640 | -2.2% | -18.3% |
| 768 | -2.2% | -13.0% |
| 1,024 | -4.8% | -19.1% |
| 1,536 | -7.0% | -22.4% |
| 2,048 | -8.5% | -18.5% |
| 3,072 | -11.4% | -19.9% |
| 4,096 | -11.9% | -18.1% |
| 6,144 | -13.2% | -18.0% |
| 8,192 | -13.8% | -17.5% |

Full per-trial values, sample counts and p10/p90 are in
[`summary.json`](artifacts/20261101/summary.json); every raw batch is in
`results_*.jsonl` and `lib_*.jsonl`.

## Fidelity

- Byte-identical old versus new output at every measured length and in both
  trials on both devices (12 comparisons per device, `parity_*.log`).
- Prefix windows at q_base 64, 128 and 1,024 match the full pass byte for byte
  (`parity_prefix_*.log`); the 128- and 1,024-row cases use the wider tile.
- `compute-sanitizer --tool memcheck` reports 0 errors at 640, 1,024, 4,096 and
  8,192 rows on both devices (`sanitizer_*.log`).
- `tests/qwen3_5/kernels.rs` attention tests pass with the wider tile active,
  including `windowed_attention_matches_full_pass_bit_for_bit` and the
  float64 reference at 700 and 2,048 rows. The full ignored suite is 26 passed
  and 1 failed only because `QWEN3_5_CHECKPOINT` was unset.

## Alternatives considered

| alternative | outcome |
| --- | --- |
| Keep 64 queries and fit two blocks per SM by removing the row padding (XOR-swizzled shared layout) | Needs a new ldmatrix swizzle and a 16-key tile to get under the 50 KiB that two blocks would need; a 16-key tile changes the softmax reduction order, so it was not taken. Not measured. |
| 16-key tiles with the padded layout | Two blocks need 2 x (50,688 + 1,024) = 103,424 bytes, above the 102,400-byte SM budget. Rejected by arithmetic. |
| 192-query tile | (192 + 64) x 264 x 2 = 135,168 bytes, above the 101,376-byte opt-in limit. Impossible. |
| Lower registers to fit a second block | The limit is shared memory, not registers (244 registers still allow two blocks). No effect. |
| 128-query tile everywhere | 512 rows is 4.9% slower on L20, so the tile is gated at 640 rows. |

## Limits

- Measured only on one L20 (sm_89, 92 SMs) and one RTX 5060 Laptop (sm_120,
  26 SMs). H200/sm_90 has different shared memory and SM counts and is
  **unverified**; the dispatch keeps it on 64 queries until it is measured.
- The shared-prefix kernel (`cs1_attention_gated_shared`) and the batch entry
  points are unchanged and stay at 64 queries. `cs1_attention_gated_prefix`
  uses the wider tile only when `q_base % 128 == 0`.
- No HTTP or Public231 C1 measurement is included. A serving-level claim would
  need the full-C1 protocol on the same named hardware as its baseline.

## Reproduction

The measurement programs were local scratch, not part of this change: the
repository's CUDA guidance asks one-off optimizations not to add a per-PR
microbenchmark, so only the protocol and the raw results are recorded here. The
protocol is fully specified above; any program that follows it reproduces the
numbers, and `summarize.py` recomputes `summary.json` from the raw files.

```sh
# Libraries for each revision (same sources, one attention.cu difference):
build.sh $WORK/lib_cand <arch>     # candidate tree
build.sh $WORK/lib_base <arch>     # base_git_sha attention.cu

# Each arm: 1 excluded warmup pair per length, then interleaved A B B A,
# 10 repetitions, each sample the mean of 5 launches between two CUDA events,
# medians per length; run the whole program twice as two trials.

# Recompute summary.json from the recorded raw files:
python3 benchmarks/attention-wide/artifacts/20261101/summarize.py \
  benchmarks/attention-wide/artifacts/20261101
```

`<arch>` is 89 on CUDA 12.1 (L20) and 120 on CUDA 12.9 (RTX 5060 Laptop). The
library-level arms swap `libqwen3_5_cuda.so` in the loader directory between
revisions and record the same protocol through `cs1_attention_gated`.
