# Open-Jev-9B validation

## Reference comparison, 2026-10-05

The native worker serving the Open-Jev-9B export was compared with the
[Open-Jev reference](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16)
at `3308a15` (uncached) on 253 requests, both with 9B's trained head and saved
temperature (1.8969118766347646). Every request had the same input token count
on both sides.

| Requests | Questions | Same decision | Largest probability difference | Questions above 0.01 |
| --- | ---: | ---: | ---: | ---: |
| M1 to M3: generated `choice` and `score` requests and Open-Jev's examples (22) | 87 | 87 | 0.041 | 5 |
| M4: JevBench single-candidate `noul` tasks (74) | 74 | 74 | 0.068 | 4 |
| M5: the other public JevBench tasks, 3 to 6 candidates (157) | 157 | 155 | 0.104 | 25 |

Both changed decisions are close calls on both sides. One is a four-way choice
whose top two options are at 0.403/0.390 native and 0.397/0.400 in the
reference; the other is a three-way choice split almost evenly between two
options, 0.52/0.48 native and 0.44/0.56 in the reference. The largest difference
is on a four-way choice of about 2,240 tokens per candidate. Differences grow
with prompt length: across JevBench, the mean of each question's largest
difference is 0.012 above 2,000 tokens per candidate and 0.003 below 500.

The final-norm hidden state at each candidate's last token, the input to the
scalar head, was also compared for all 2,157 candidates:

| Requests | Candidates | Relative L2: median | p99 | largest | Head output: median | p99 | largest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 to M3 | 1,382 | 0.0093 | 0.020 | 0.027 | 0.024 | 0.18 | 0.36 |
| M4 | 74 | 0.014 | 0.036 | 0.050 | 0.051 | 0.43 | 0.87 |
| M5 | 701 | 0.011 | 0.028 | 0.063 | 0.038 | 0.39 | 1.49 |
| All | 2,157 | 0.0096 | 0.023 | 0.063 | 0.028 | 0.21 | 1.49 |

The relative L2 difference is ‖native − reference‖ / ‖reference‖ per candidate.
The head output is the scalar before temperature, computed from each side's
hidden state with the same FP64 head. These differences come from two different
implementations: the native worker runs the LoRA adapter merged into BF16
weights on its CUDA kernels, while the reference applies the adapter unmerged
through PEFT, uses the PyTorch Gated DeltaNet path and pads candidates into
batches of 8. The median relative L2 difference is 0.009 to 0.010 in each of
three prompt-length ranges (below 500 tokens per candidate, 500 to 2,000, and
above 2,000), while the tail of the head output difference grows with length
(p99 0.18 below 500 tokens, 0.38 above 2,000), as do the probability
differences above.

These results are for 9B on one GPU. They make no claim about 27B.

### Workload

- **M1:** one `choice` question with 2, 8, 32, 128 or 255 candidates, over
  states of about 256, 1,024 or 3,072 tokens built from the text of Open-Jev's
  example states (15 requests).
- **M2:** the four `examples/workflows` requests and `examples/community/support_28`
  at the reference revision, 7 to 50 candidates each (5 requests).
- **M3:** one `score` question with 5 or 10 levels over a 1,024-token state
  (2 requests).
- **M4 and M5:** the 231 public tasks of
  [JevBench](https://github.com/fstandhartinger/jevbench) at
  `f8ce71361165846101d02ebc83ad44e47ae44fc3`, one question each: the 74 `noul`
  tasks used in the [H200 validation](validation.md), and the 157 tasks with 3 to
  6 candidates. JevBench's license allows publishing aggregate results only, so
  requests and responses are not included.

### Controls and reproduction

- One NVIDIA RTX 6000 Ada (compute capability 8.9, 48 GB, 300 W limit), CUDA 13.2,
  BF16, one request at a time. The card ran near 1 GHz at its power limit during
  the long requests; timing is not part of this comparison.
- Native: this change's Rust sources and `Cargo.lock` on `main` `7f39ac4`
  (unchanged since the measurement), with the CUDA library built by
  `src/backends/cuda/qwen3_5/build.sh <dir> 89`, eager execution
  (`CUA_S1_GRAPH` unset), requests over HTTP to the worker. The worker used
  15,874 MiB of device memory after warmup and 16,672 MiB after all requests.
- Export: [`export_merged.py`](export_merged.py) on the pinned base and checkpoint
  in the [native recipe](native.md), with the default 4096-token limit. It
  produced the same bytes as the export used for these measurements, in all 10
  files.
- Reference: `jev.serving.load_predictor` on the 9B checkpoint package, batch
  size 8, prefix cache off, with Torch 2.14.0+cu130, Transformers 5.10.2,
  PEFT 0.19.1 and Accelerate 1.13.0. Flash Linear Attention is not installed,
  so Gated DeltaNet runs on the PyTorch path, as in the H200 HF baseline.
- Hidden states: on the native side, `Model::forward` with the worker's own
  prompt rendering and tokenization; on the reference side, the input to its
  scalar head, recorded in a separate run with the same predictor settings.
  Through each side's head, the native hidden states reproduce the worker's
  probabilities and the recorded reference ones reproduce the compared reference
  probabilities, both to within 1e-15.

Request manifests, raw responses, hidden-state dumps and scripts are kept
locally, outside this repository.

## Shared prefixes, 2026-10-25

The same 253 requests ran through five workers, one request at a time:
`main`, and this worker with no switch, with `OPEN_JEV_FIXED=1`, with
`OPEN_JEV_PREFIX=1` and with `OPEN_JEV_PREFIX=auto`. Each request was sent twice
in a row; times are the faster of the two, measured at the client.

**Responses.** With no switch, all 253 responses equal `main`'s apart from
timing. With `OPEN_JEV_PREFIX=1`, the 95 requests with at least one reusable
64-token chunk ran on shared prefixes, and all 253 responses equal those with
`OPEN_JEV_FIXED=1` apart from timing and `prefix_cache`. With `auto`, the 46
requests it shared equal the fixed responses and the other 207 equal those with
no switch.

**Fixed against default GEMM algorithms.** Against the default path, and both
against the reference from the comparison above:

| Requests | Questions | Fixed vs default: changed decisions | largest difference | Against the reference, default / fixed: changed decisions | largest difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 to M3 (22) | 87 | 0 | 0.035 | 0 / 0 | 0.041 / 0.042 |
| M4 (74) | 74 | 0 | 0.098 | 0 / 0 | 0.068 / 0.030 |
| M5 (157) | 157 | 1 | 0.064 | 2 / 1 | 0.104 / 0.141 |

The changed decision is the three-way near tie described above: 0.52/0.48 with
the default algorithms, 0.46/0.54 with the fixed ones and 0.44/0.56 in the
reference. The fixed path's largest difference from the reference is on the same
four-way choice as the default path's. Against JevBench's labels, the reference
answers 165 of the 231 tasks correctly, the default path 163, fixed and shared
164, and `auto` 163.

**Time.** Sums of the per-request times, in seconds:

| Requests | Input tokens | Not recomputed when shared | `main` | No switch | Fixed | Shared | `auto` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 to M3 (22) | 1,966,702 | 95% | 305.3 | 308.7 | 371.1 | 35.5 | 37.9 |
| M4 (74) | 31,802 | 0 | 5.76 | 5.74 | 6.44 | 6.16 | 5.69 |
| M5 (157) | 464,274 | 69% | 80.0 | 79.8 | 92.0 | 34.3 | 42.0 |
| M5 requests over 8,192 input tokens (30) | 347,670 | 75% | 54.9 | 54.8 | 65.2 | 17.5 | 17.5 |

Single requests: 255 candidates over a 3,072-token state take 6.7 s shared
instead of 122.5 s, 32 candidates over a 1,024-token state 0.86 s instead of
5.3 s, and `support_28` 1.06 s instead of 1.88 s. The fixed algorithms alone add
12 to 20% to these sums. Requests with nothing to share pay that under
`OPEN_JEV_PREFIX=1` (M4: 6.16 s instead of 5.74 s); `auto` keeps them on the
default path. Its threshold also leaves some requests unshared that would gain:
M2's requests of 2,600 to 5,500 tokens run 2 to 3 times faster shared, but save
fewer than 4,096 tokens.

With no switch, the worker takes as long as `main`. Sent alternately to both
(A B B A, `main` at `27176cb`), the 253 requests took 393.5 s against 392.8 s,
with a median per-request ratio of 1.001 (0.99 to 1.02 from the 5th to the 95th
percentile), and identical responses. In the five-way run `main` and no switch ran
one after the other, so their 1% gap there, up to 11% on the first requests with
many short candidates, is drift between runs.

### Controls

- Same GPU, export, CUDA build and HTTP setup as above. The five-way run used an
  earlier state of this change on `main` `7b4c194`. Its later changes (the
  `CUA_S1_GRAPH` check, tests and documentation) and `main` up to `51f80ad` leave
  the responses with no switch, shared and `auto` unchanged.
- On 2026-10-26, on the same GPU, the `prefix` and `prefill_batch` GPU tests and
  `fixed_gemm_rows_do_not_depend_on_m` pass at `01c84c6` and again at `2a19ed2`, which
  adds the startup check of fixed GEMM rows. On this export the check takes 0.37 s
  at startup.
- Device memory after warmup, and after the 22 M1 to M3 requests and the five
  longest JevBench tasks: 15,874 and 16,674 MiB with no switch, 15,906 and
  16,884 MiB with `OPEN_JEV_PREFIX=1`. The difference is the fixed-algorithm GEMM
  workspace (32 MiB, allocated at warmup) and the prefix states.
