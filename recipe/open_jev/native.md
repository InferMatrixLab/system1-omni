# Open-Jev native text worker

The worker owns request compilation, tokenization, candidate scoring and typed
responses in Rust. It uses the native CUDA prefill implementation introduced in
[PR #19](https://github.com/ThinkFlowLab/system1-omni/pull/19), shared with Cua-S1
under [`src/models/qwen3_5/native/`](../../src/models/qwen3_5/native/).
Python is required only to prepare the merged checkpoint.

The worker serves Open-Jev-27B-v1.1 or Open-Jev-9B. Both checkpoints use the
same method, prompt format and head design, each with its own trained head; the
export's `model_id` selects the pinned revisions, the accepted request model
names and the expected backbone dimensions.

It supports `choice` (1–255 candidates), `score` (2–10 levels), and `noul`
(yes/no). Each candidate has an independent prompt; the last hidden state goes
through Open-Jev's trained FP32 scalar head. Noul uses logits `[0, score]`.
The saved calibration temperature is applied before normalizing each complete
question. There is no autoregressive generation. Structured state and descriptions
use Open-Jev's sorted JSON rendering; question and candidate order is preserved.

## Prepare the checkpoint

Use the reference dependencies from
[Open-Jev @ 3308a15](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16):
PyTorch 2.8 or newer, Transformers 5.10.2, PEFT 0.19.1, Accelerate 1.13.0,
and safetensors. An optional `kernels` installation must be compatible with that
Transformers release. Run these commands from the repository root:

```sh
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir weights/Qwen3.8-27B
hf download ZefanCai/Open-Jev-27B-v1.1 \
  --revision 28cf73067d5b337860bbef3c85b8b82ba8730956 \
  --include 'package/checkpoint/*' --local-dir weights/Open-Jev-27B-v1.1
CUDA_VISIBLE_DEVICES='' python recipe/open_jev/export_merged.py \
  --base weights/Qwen3.8-27B \
  --checkpoint weights/Open-Jev-27B-v1.1/package/checkpoint \
  --out weights/open-jev-27b-merged
```

For Open-Jev-9B, download its pinned base and checkpoint and export them the
same way:

```sh
hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir weights/Qwen3.5-9B
hf download ZefanCai/Open-Jev-9B \
  --revision 47e966881e489511c0c7f5633a9e1960a676a551 \
  --include 'package/checkpoint/*' --local-dir weights/Open-Jev-9B
CUDA_VISIBLE_DEVICES='' python recipe/open_jev/export_merged.py \
  --base weights/Qwen3.5-9B \
  --checkpoint weights/Open-Jev-9B/package/checkpoint \
  --out weights/open-jev-9b-merged
```

CPU export needs roughly 110 GB of RAM and 52 GB of output storage for 27B,
and about 20 GB of RAM and 16 GB of storage for 9B. It merges LoRA in BF16 and
saves the trained head, temperature, and single-user chat template in
`open_jev_export.json`. The worker refuses a plain base checkpoint
or an incomplete export. The saved limit defaults to 4096 tokens per candidate;
`--max-length` may raise it to 16384. Oversize prompts fail before inference.

## Build and serve

The CUDA kernels require compute capability 8.0 or newer. The current build
target below is Ada (`89`); pass your GPU's compute capability explicitly.
The CUDA shared library and Rust workers must be rebuilt together for ABI
version 9, which includes shared vision, CUDA Graph, canonical prefix continuation,
fixed-GEMM selection and device-feature validation.
The 27B worker also requires the additive packed-GEMM and batched-GDN
symbols from this checkout; rebuild the worker and library together.

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-open-jev-native -p omni-jev
OPEN_JEV_MODEL=weights/open-jev-27b-merged RAYON_NUM_THREADS=4 \
  target/release/omni-open-jev-native
```

For 9B, set `OPEN_JEV_MODEL=weights/open-jev-9b-merged`.

`OPEN_JEV_HOST` and `OPEN_JEV_PORT` default to `127.0.0.1` and `8000`.
`OPEN_JEV_CUDA_LIB` overrides the default library next to the executable.
For 27B, `RAYON_NUM_THREADS=4` bounds the tokenizer's thread pool in the example;
select a thread count appropriate to the worker's CPU allocation. Tokenization
processes at most 16 candidates from one question at a time. Single candidates,
configured padding and the 9B checkpoint keep per-candidate encoding.
The worker loads all text weights onto visible CUDA device 0, performs a real
warmup inference, then exposes `/health` and `/v1/systemone`.
Use a reservation before any GPU command on hosts with a GPU scheduler.

In another terminal, start the existing Rust frontend:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @recipe/open_jev/example-request.json
```

The worker accepts the loaded checkpoint's base name (`Qwen/Qwen3.8-27B` or
`Qwen/Qwen3.5-9B`), `open-jev`, `jev-latest`, and its own alias
(`open-jev-27b-v1.1` or `open-jev-9b`); other names get HTTP 422. The response
and `/health` model is the base name, matching Open-Jev. The checkpoint aliases,
error wording and metadata differ from the reference service.
Requests are bounded to 4 MiB, 4096 questions, and 65536 candidate sequences.

## Validation and optimization scope

Tests and fixtures live in the repository-level `tests/` tree: Open-Jev's typed
contract and tokenizer cases are in
[`tests/open_jev/`](../../tests/open_jev/), and shared Qwen JSON, configuration
and CUDA reference tests are in [`tests/qwen3_5/`](../../tests/qwen3_5/).
The default suites below run on CPU without downloading model weights:

```sh
cargo test --locked -p omni-open-jev-native -p omni-qwen3-5-native
cargo test --locked -p omni-jev --test frontend
```

The frontend mock-worker API coverage is tracked in
[issue #46](https://github.com/ThinkFlowLab/system1-omni/issues/46) and
[PR #58](https://github.com/ThinkFlowLab/system1-omni/pull/58). Checkpoint tokenizer
and CUDA kernel tests are opt-in; the latter require a GPU reservation:

```sh
# Tokenizer cases against either merged export, on CPU:
OPEN_JEV_MODEL=weights/open-jev-9b-merged \
  cargo test --locked -p omni-open-jev-native --lib -- --ignored
# Inside a GPU reservation, after building the library:
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-qwen3-5-native --test kernels -- --ignored

# Full-checkpoint packing, ordering, graph replay and prefix-buffer reuse checks:
# Run inside the reservation; repeat with CUA_S1_GRAPH=0 for eager execution.
OPEN_JEV_MODEL=$PWD/weights/open-jev-27b-merged \
OPEN_JEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so CUA_S1_GRAPH=1 \
  cargo test --release --locked -p omni-open-jev-native --test prefill_batch \
  -- --ignored --test-threads=1
```

CPU golden fixtures come from Open-Jev's request compiler and response formatter
at the revision above. Kernel tests compare attention and Gated DeltaNet with
float64 references and require exact BF16 equality between fused attention gating
and a separate gate pass. Residual RMSNorm and packed SiLU are checked against
rounded references, including odd widths and unaligned pointers. The shared
kernel tests retain PR #19's
`CUA_S1_CUDA_LIB` environment variable.

The worker reuses PR #19's fused norm, activation, QK/RoPE and chunked Gated
DeltaNet operations. Attention's sigmoid gate is fused into its output epilogue,
preserving both BF16 rounding points and removing one launch and one output
read/write pass per full-attention layer (16 layers on 27B, 8 on 9B). Residual
RMSNorm keeps thread values in registers at widths 2560/5120; 9B's width 4096
uses the general kernel. MLP SiLU uses
16-byte BF16 loads/stores when width, stride and pointers permit it, retaining
both BF16 rounding points; other layouts use the scalar path.

This recipe leaves `CUA_S1_GRAPH` unset and uses eager prefill. For
Open-Jev-27B-v1.1, candidates within a request are packed in prepared order, up
to 16 sequences and 4096 total tokens per group; longer prompts execute alone
without truncation. Input and gate/up
projections share GEMMs. Output/down projections combine consecutive candidates
only when their original cuBLASLt algorithms are identical and the combined shape
passes its workspace check. Unsupported shapes retain separate calls. This keeps
the original algorithm and reduction configuration; numerical validation is still
required for each supported hardware/toolchain combination. Batched GDN launches
keep independent state and chunk-rounded workspace for every sequence. Attention,
positions and convolution also remain independent. Calibration still uses every
candidate in its question.
The [H200 packing comparison](../../benchmarks/prefill_batching/README.md)
records latency, exact output checks and frozen controls.
That historical comparison covers the earlier packing implementation on H200
(sm_90). The newer projection/GDN paths have exact-output regression coverage on
an SM90 device; their execution on other CUDA architectures remains unverified.
Open-Jev-9B runs one forward pass per candidate: packing is not validated for it.

For one question with 2–16 candidates all longer than 2048 tokens, the worker can
share an exact token prefix within compatible candidate groups. The prefix must
be at least 512 tokens, aligned to 64 tokens, with a nonempty suffix for every
candidate. Compatibility checks retain the complete prompt's algorithms for every
prefix/suffix projection. A declined group uses complete forwards. Group outputs
are restored to the original candidate order before normalization. The model
retains allocation capacity for one prefix, up to the loaded maximum length;
every request invalidates and rewrites its numeric state. It does not cache request
contents across requests. The response reports executed reuse through
`metadata.prefix_cache`:

| Field | Meaning in this worker |
| --- | --- |
| `policy` | `default`: no explicit prefix-policy override; compatible long candidates may still share automatically. `fixed`, `shared` or `auto` with the switches below. |
| `enabled` | Whether this request actually reused a prefix after compatibility checks. |
| `scope` | `within_question` when sharing ran within one question, `across_questions` when `OPEN_JEV_PREFIX` shared the request prefix across questions, otherwise `none`. |
| `saved_tokens` | Prefix tokens whose repeated computation was avoided, summed over executed compatible groups. |

Each compatible group of `n` candidates with a prefix of `p` tokens saves
`(n - 1) * p` token computations. Singletons save zero; separate groups each
compute their own prefix. For three candidates with a 3072-token prefix, all
sharing saves 6144 tokens, while two sharing and one falling back saves only
3072. Splitting or packing the suffixes does not change this prefix count.
These four fields are the complete prefix metadata contract; `mode` and
`shared_tokens` are not emitted. `usage.input_tokens` still counts every complete
candidate's input. No sharing returns `enabled=false`, `scope=none` and
`saved_tokens=0`. This worker does not implement the explicit fixed/shared/auto
policy switches or cross-question reuse proposed in
[PR #179](https://github.com/InferMatrixLab/system1-omni/pull/179).

The default 27B optimization is validated against the original default
complete-prompt execution with the same weights, inputs and temperature:
maximum absolute candidate-probability and ordinal-score error at most 0.001,
zero Choice changes, zero Noul flips at the `>= 0.5` threshold, and zero request
failures. These are comparison-test gates, not a second baseline inference on
every served request. Retaining the original projection algorithms and falling
back when incompatible remains the execution policy. Equality to an alternative
fixed-algorithm path alone does not establish these default-path gates. The
accepted historical latency tradeoff below does not relax numerical validation.

Compatible suffixes can also share compact storage and projection calls, bounded
to 16 sequences and 4096 suffix tokens per group. Attention reads immutable prefix
keys/values separately; convolution reads the prefix's three retained rows; each
linear-attention sequence starts from its own copy of the group state. If the
optional compact-suffix symbols or an algorithm check are unavailable, suffixes
use the original continuation path.

The 27B opt-in also uses vector convolution for aligned layouts, wider work
assignment for Gated DeltaNet preparation, and independent-column output tiling.
These preserve each accumulator's arithmetic and BF16 rounding boundaries. For
one ordinary text execution unit of 512–16384 tokens, only the final position
needs the last layer's feed-forward network and final norms. The model skips
unused rows only when both projections accept one row with their original
complete-prompt algorithms. Prefix/suffix, multimodal, bidirectional, unmerged,
reference and incompatible paths retain their complete computation.

Packed projection plans and retained prefix/suffix plans each have a separate
512-entry host LRU cache. These hold shape/algorithm descriptions, not token values.
GPU tests cover replay of an older graph after plan eviction, changed request
contents and prefix-buffer growth/shrinkage. Batched GDN adds approximately
96 MiB of bounded workspace for this 27B configuration; prefix storage grows with
the largest shared prefix encountered. Include both allocations in memory
measurements and distinguish first traversal from warmed latency.

Set `CUA_S1_GRAPH=1` on the worker to enable CUDA Graph replay. The
27B packed text path retains at most 256 graphs; other modes retain 64. Keys are
ordered sequence token lengths. Graph-enabled packed text initially reserves
scratch for at least 4096 rows, rounded up in 1024-row increments. Longer supported
inputs can still grow the buffer; growth synchronizes work and clears graphs
whose addresses would become invalid. A new shape first runs an eager forward
to initialize its plans and captures the layer loop for later replay.
This adds cost for new lengths, so graph mode remains opt-in. The earlier
single-prompt graph comparison with Open-Jev-27B-v1.1 on the 74-case H200 workload
measured mean HTTP latency 2.03% below eager execution after all lengths were
warmed. These historical numbers describe that earlier implementation. Tokenization, transfers
and the CPU scalar head remain outside the graph. Cross-request prefix-content caching, GEMM autotuning,
quantization and multimodal inference are not implemented; sharing across questions is
opt-in (below). The
[H200 validation](validation.md) reports Open-Jev-27B-v1.1 results for 74
single-candidate requests, including probability differences and timing
variability. It does not establish general accuracy parity or a speedup over
OpenJev-Fast; the author's B300 results use different hardware and workloads.
The [Open-Jev-9B validation](validation-9b.md) compares the 9B worker with the
reference on one RTX 6000 Ada; it makes no claim about 27B.

### Cumulative 27B experiment and accepted tradeoff

The October 22–24, 2026 experiment compared frozen baseline
`d577bb824a5dbf0f5e385cf829bb6b9025dabf8d` with cumulative candidate P19
(patch SHA-256 `5bb08cafffcf1825c1467de36705ae78baf07e0f7b24662c67ed7237b6504799`).
P19 predates integration with the current shared executor; its measurements are
historical evidence, not measurements of every later checkout.

The workload was 231 fixed public JevBench requests (139 Choice, 74 Noul,
18 Score), single-request concurrency through the Rust frontend, with complete
response receipt, parsing and validation inside the client timer. Twelve balanced
blocks compared baseline, parent P18 and P19; each service start had one separate
feasibility traversal and eight measured traversals. Baseline used eager execution;
both candidates enabled Graph replay. Across all three arms, 66,528 responses
were timed and 8,316 were feasibility responses. No observations were excluded.

| Historical metric | Baseline | P19 | Change |
| --- | ---: | ---: | ---: |
| Mean full-request latency | 216.31 ms | 123.02 ms | -43.13% |
| Median full-request latency | 61.04 ms | 44.67 ms | -26.82% |
| 95th-percentile full-request latency | 1081.56 ms | 471.11 ms | -56.44% |
| Noul, 513–2048 tokens, 95th percentile | 116.32 ms | 119.77 ms | **+2.97%** |
| Sampled whole-invocation peak device memory | 50,949 MiB | 52,035 MiB | +1086 MiB |

Device: driver-reported **NVIDIA L20X**, measured compute capability **9.0**,
132 streaming multiprocessors, about 150.1 GB visible memory; the conflicting
name/properties do not establish a commercial model. CUDA 13.0.88, driver
570.133.20, BF16 merged weights and the unchanged FP32 scalar head were used.
Memory was sampled approximately every 0.5 seconds, including loading and warmup;
brief peaks may be missed. Loading is outside the request timer.

All 74,844 historical responses had zero observed probability/Score drift,
decision flips and request failures. The six-request Noul group above regressed
in all 12 blocks; its paired approximate 95% interval was +2.32% to +3.70%.
The contributor accepts this specific tradeoff for submission. The original
2% all-group guard did not pass: four additional tail checks were inconclusive
(baseline: Score 129–512; parent: Noul 1–128, Score 1–128 and Score 129–512).
They remain inconclusive, and the conditional final synthetic timing stage was
not run. Earlier candidate synthetic passes do not validate P19's missing stage.
Multi-request concurrency, other architectures and other complete checkpoints
remain unverified for these optimizations. Source reconstruction, current-head
checks and exact reproduction/evidence records belong to the accompanying PR.

## Optional request-local shared prefixes

Each candidate prompt repeats the chat prefix, the state and its question, so the
candidates of one request share long token prefixes. Without a switch, Open-Jev-27B-v1.1
already shares long candidates' prefixes within one question, with the complete
prompt's algorithms (above). The switches below share across questions too, on both
checkpoints, with the fixed GEMM algorithms. `OPEN_JEV_PREFIX=1` plans them
with the Qwen crate's request-local planner
([`prefix.rs`](../../src/models/qwen3_5/native/src/prefix.rs)), which Decider also
uses: the prefix all candidates share, then a longer prefix for each question's
candidates. Each prefix runs once through the shared executor's `forward_shared`
([architecture](../../docs/architecture.md)) and every candidate continues from its
state. Prefixes end at their last multiple of 64 tokens and the rest runs in each
candidate. The prefix-state buffers persist, but their contents are valid only within
one request; there is no cross-request cache. A request with less than one reusable
64-token chunk runs its candidates one at a time with the same fixed GEMM algorithms.

Shared execution keeps one GEMM algorithm per weight shape. On GPUs where that keeps
rows independent of the row count, each candidate's score is bit-identical to
`OPEN_JEV_FIXED=1`, which runs every candidate as its own full pass with those
algorithms; both round differently from the default path. With `OPEN_JEV_PREFIX`
set to `1` or `auto`, the worker checks this at startup for the model's own
projection shapes, with row counts like those of Decider's
[fixed-algorithm check](../decider/validation.md#deploying-fixed-and-shared-modes):
each subset of 4,096 random rows, run alone, must equal the same rows of the full
call byte for byte, or the worker refuses to start. The check briefly takes about
0.35 GiB of device memory on 9B. It covers only the loaded model's shapes, so
Decider's check is still the one to run when qualifying a new GPU. If the backend
rejects the fixed algorithm for a shape, the startup check fails; at row counts the
check did not try, and with `OPEN_JEV_FIXED=1`, the request fails with HTTP 500 like
any other inference error. It never falls back to another algorithm.
`OPEN_JEV_PREFIX=auto` shares only when the plan saves at least 4096 tokens and at
least a third of the request's tokens, as `DECIDER_PREFIX=auto` does; other requests
take the default path, including its sharing within one question. Shared candidates run one at a time and are not packed. The
switches default off, cannot be combined, and require `CUA_S1_GRAPH` off.

```sh
OPEN_JEV_PREFIX=auto OPEN_JEV_MODEL=weights/open-jev-9b-merged \
  target/release/omni-open-jev-native
```

With a switch set, `metadata.prefix_cache` reports the switch as `policy` (`fixed`,
`shared` or `auto`) and `scope` `across_questions` when more than one question shared
the request prefix; the fields are otherwise as in the table above. The
[Open-Jev-9B validation](validation-9b.md#shared-prefixes-2026-10-25)
has the measurements; shared prefixes have not been run on Open-Jev-27B-v1.1. The
GPU test compares shared and fixed scores on a two-question request:

```sh
OPEN_JEV_MODEL=$PWD/weights/open-jev-9b-merged \
OPEN_JEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-open-jev-native --test prefix -- --ignored
```
