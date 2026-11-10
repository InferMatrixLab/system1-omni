# Open-Jev

The [native Rust/CUDA worker](../../../recipe/open_jev/native.md) serves
Open-Jev-27B-v1.1 on its pinned Qwen3.8-27B backbone or Open-Jev-9B on its
pinned Qwen3.5-9B backbone, with the merged LoRA adapter, trained scalar decision
head, and saved calibration temperature. Both checkpoints use the same method,
prompt format and head design, each with its own trained head; the export's
`model_id` selects the checkpoint's pinned revisions, request model names and
backbone dimensions. It supports choice, ordinal score, and yes/no text
decisions through the existing Rust frontend.

The request compiler and response formulas follow
[Open-Jev @ 3308a15](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16).
The CUDA prefill implementation is shared with Cua-S1 under
[`../qwen3_5/native/`](../qwen3_5/native/). See the recipe for preparation,
numerical limitations, validation and optimization scope.

The Rust contract is adapted from Open-Jev's MIT-licensed code; its copyright
and license are retained in [`native/LICENSE.open-jev`](native/LICENSE.open-jev).
No model weights are distributed here.

## Integration boundary

The native worker follows the [architecture contracts](../../../docs/architecture.md)
with independent [processing](native/src/processing.rs) and
[executor](native/src/executor.rs) modules. Preparation returns token IDs grouped
by question and candidate, plus the response context for identity/order,
calibration, usage, and metadata. By default, preparation tokenizes up to 16 candidates
from one question in parallel when the tokenizer has no padding; singleton and
padded inputs retain per-candidate encoding. Token IDs, candidate order and the
first validation error are preserved. `RAYON_NUM_THREADS` controls tokenizer
parallelism. Batched encoding skips unused character offsets. The executor owns the trained head and returns one FP32 scalar per
candidate plus whether prefix sharing actually ran. Response finishing adds the `noul` false baseline
and applies calibrated normalization across each complete question. The worker
retains request-wide model locking and independent sequence semantics, with
the scalar head on the CPU after CUDA prefill. For Open-Jev-27B-v1.1, its
[batch adapter](native/src/batching.rs) packs at most 16 candidates and 4096
tokens per group, in prepared order;
longer individual prompts execute alone without truncation. Input and gate/up
projections share packed GEMMs. Output/down projections can combine consecutive
candidates when their original cuBLASLt algorithms match and the combined shape
fits the existing workspace; otherwise they retain separate calls. Batched GDN
launches process at most 16 sequences with separate state and workspace. Attention,
positions and convolution also reset at each sequence boundary.

A single question with 2–16 candidates, each longer than 2048 tokens, can share
an exact token prefix of at least 512 tokens, rounded down to a multiple of 64.
Every candidate keeps a nonempty suffix. The executor groups candidates only
when all projection algorithms from their complete forwards remain compatible
with the prefix and suffix shapes. Each group captures its own prefix state;
singletons use complete forwards, and all results return to their original indices
before question-wide normalization. The model retains one bounded prefix-buffer
capacity, invalidates its numeric contents after each request, and captures new
contents for every group. This is within-question reuse, with no reuse of request
contents across requests. Response `metadata.prefix_cache` separates configured
`policy` (currently `default`) from actual `enabled`/`scope` and `saved_tokens`.
Savings count avoided prefix computations after compatibility grouping, with
zero for singleton fallbacks; suffix packing adds no prefix savings. These four
fields are the complete contract; `mode` and `shared_tokens` are not emitted.
Input usage still counts complete prompts. See the
[native recipe](../../../recipe/open_jev/native.md#validation-and-optimization-scope)
for field semantics and the unchanged default-path numerical validation gates.
Results are regrouped before question normalization.
Open-Jev-9B packs candidates and batches queued requests by default. Set
`OPEN_JEV_PACK=0` to restore serial tokenization, serial request scheduling and
eager per-candidate forwards, without prefix reuse or packed backend capabilities.
This is an execution-strategy fallback, not a restoration of every historical
kernel implementation. See the [rollback recipe](../../../recipe/open_jev/native.md#9b-packing-rollback)
for configuration constraints and validation commands.
Optionally, the executor plans the candidates' shared token prefixes with the shared
Qwen planner and runs each prefix once per request through `forward_shared`, or runs
every candidate with the same fixed GEMM algorithms as a control; see
[optional request-local shared prefixes](../../../recipe/open_jev/native.md#optional-request-local-shared-prefixes).
The engine owns a [shared runtime scheduler](../../runtime/README.md): default 9B
uses cross-request batching; rollback, explicit prefix modes and 27B admit complete
requests serially. Shared queue capacity limits remain planned. `/health` reports
HTTP 503 with `status: failed` after a batch worker exits or the model mutex is
poisoned. A busy worker and ordinary request errors do not mark it failed; recovery
from permanent failure requires restarting the process.


The experimental 9B layerwise prefix tree can batch the independent suffix
attention queries into one CUDA launch per layer. Each candidate keeps its
original query blocks, causal key order, softmax accumulation and BF16 rounding;
only independent candidates execute together. Prefix K/V is immutable for that
layer and every suffix writes a disjoint output range. The optional
`cs1_attention_gated_shared_batch` entry consumes host lengths before returning
and passes descriptors by value, including under graph capture. Libraries
without the entry retain the separate candidate launches. The model continues
to own scratch and stream lifetimes; this adds no request cache or device state.

The same optional batching covers ordinary and pointwise-compacted 9B causal
sequences: all candidates first prepare their separate Q/K/gate rows, then one
`cs1_attention_gated_batch` launch computes independent attention windows.
Sequence boundaries, original projection plans and output ordering remain
unchanged. Single sequences, reference numerics, bidirectional attention and
unvalidated adapter modes keep their previous entry points. Older CUDA
libraries without the optional symbol also retain separate launches.

For opted-in prefix projections on the measured device profile (CUDA-reported
`NVIDIA L20X`, SM 9.0, 132 multiprocessors, `cublasLtGetVersion() == 130101`), the original algorithm
must still support the actual row count. Within the already validated compatible
arithmetic profiles, the runtime can then select the heuristic for that actual
count. Unknown or split-K profiles and other hardware retain the original
algorithm. Candidate grouping and fallback eligibility stay unchanged; full-row
bitwise comparisons, changed-input graph replay and model output checks cover
the choice. cuBLASLt does not guarantee arithmetic equivalence across algorithms.
