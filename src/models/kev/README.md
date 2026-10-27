# Kev-4B native text decisions

Kev serves `choice`, `noul`, and `score` with its own processor and FP32 pointer
head over the shared Qwen3.5 CUDA backbone. The [preparation recipe](../../../recipe/kev/README.md)
remains independently reviewable; the [native recipe](../../../recipe/kev/native.md)
covers the complete eager worker. Other sizes and prefix reuse are deferred.
Full-model numerical acceptance is pending; CPU processing and isolated head
validation do not establish end-to-end parity.

## Pinned artifacts

| Artifact | Immutable revision |
| --- | --- |
| [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b/tree/139fdd94f1b6a6ad80cc15e08fcb99cac885a101), adapter and head | `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` |
| [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base/tree/1001bb4d826a52d1f399e183466143f4da7b741b) | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| [Kev reference code](https://github.com/jaredpalmer/kev/tree/2074cc44256200bb37540103fa4572901faa10f3) | `2074cc44256200bb37540103fa4572901faa10f3` |

All are Apache-2.0; the adapted head's license is retained in
[`native/LICENSE.kev`](native/LICENSE.kev). No model weights are committed.
The source hash in the fixture generator covers the complete pinned `kev/model.py`.

## Export contract

[`export.py`](../../../recipe/kev/export.py) verifies SHA-256 for every required
base/checkpoint file **before deserializing `head.pt`**, which uses PyTorch's
weights-only CPU loader. It merges all 496 FP32 LoRA tensors into 248 language
projections as `BF16(FP32(W) + 2 * (B @ A))`. This includes all five GDN projection
types: `in_proj_qkv`, `in_proj_a`, `in_proj_z`, `in_proj_b` and `out_proj`.
Missing pairs or unused targets fail the export. The output excludes vision/MTP
weights and preserves the original language tensor names for the shared Qwen loader.
Other language tensors are converted to BF16, including originally FP32 norms.

The four pointer-head tensors remain FP32: `q.weight`/`k.weight` are `[256, 2560]`
and their biases are `[256]`. The manifest records temperature
`2.406050072164233`, dimensions, delimiter IDs, exact input hashes, every merged
target and all output hashes. It is written after the artifacts, and the private
staging directory is renamed only on success. Existing outputs are refused.

The tokenizer comes from the **base**: `<|fim_prefix|>` = 248060 (state),
`<|fim_middle|>` = 248061 (question), `<|box_start|>` = 248049 (option start),
`<|box_end|>` = 248050 (option readout), `<|fim_suffix|>` = 248062 (decision
readout). The adapter's older added-token metadata is not copied.

## Executor and postprocessing boundary

`head::PointerHead` owns the trained CPU weights and accepts one final-normalized
decision hidden state plus 1–255 option hidden states, each with 2560 FP32 values
for this checkpoint. It computes `k(h_option) dot q(h_decide) / sqrt(256)` in FP32,
including both learned biases, and preserves candidate order. Inputs, shapes,
weights and results must be finite. Returned logits own their storage.
`postprocess::probabilities` divides these logits by the saved temperature and
normalizes across the **whole question**. This follows the eval-mode reference;
temperature must not be applied twice. Projection and calibration have separate
modules even though both currently run on CPU.

## Request flow and ownership

`http::decide` admits a bounded request and prepares it on a blocking CPU task.
`contract::compile` follows the pinned `kev/api.py`: recursive text rendering,
insertion order, Choice `name: description`, Noul `no`/`yes` candidates and ordered
Score levels. `processing::Processor` owns the base tokenizer, rewrites caller
`<|name|>` spellings to `<¦name¦>`, and encodes each fragment separately. Each
prepared question owns one complete causal row and its candidate/decision indices.
Question positions restart at the state length. No padding or truncation is used.

`Engine` owns a shared `SerialScheduler` and one mutex-protected `Executor`.
One admitted execution unit is a whole request. Questions run sequentially;
`Model::forward_multimodal_readout` consumes a text-only input with identical
T/H/W positions and returns final-normalized candidate and decision rows. The
executor owns CUDA weights/scratch and the FP32 pointer head; only owned logits
leave it. Every question starts with clean GDN state. The stream is synchronized
before execution resources and admission permits can be released, including on
errors. Cancelled callers cannot free buffers while dispatched GPU work uses them.

`ResponseContext` retains original question/candidate mapping and the tokenizer.
Postprocessing calibrates once, normalizes across each entire question, chooses
the first candidate on ties, and restores the reference's confidence, legends,
four-decimal scalars, and billing token counts. State counts once in input usage;
compute cost separately counts each complete row. The native recipe documents
stricter context/request limits and deliberate HTTP differences.

The export loader verifies every input pin in `recipe/kev/pins.json`, the complete
merged target inventory, all seven output hashes, shard references, head dimensions
and base tokenizer markers before CUDA loading. Startup runs all three question
types before binding its HTTP listener. Shared Qwen/kernel/runtime code is reused
without adding a Kev-specific backend or scheduling policy.

## Validation and remaining gates

Tests live under [`tests/kev/`](../../../tests/kev/). Ordinary Rust tests cover
learned biases, scale, candidate ordering, one/255 candidates, malformed weights,
hidden states and temperature. CPU Python tests cover pinned-input rejection,
no overwrite, all five GDN target names, one-rounding merge arithmetic, incomplete
pairs, delimiter IDs and head metadata.

The opt-in **CPU** test loads the real exported head and compares 1, 2, 7, 31 and
255 synthetic hidden-state candidates with the actual pinned upstream
`PointerHead` class (PyTorch 2.8.0 CPU). Logit tolerance is
`1e-4 + 1e-5 * abs(reference)`; probability absolute tolerance is `1e-5`, declared
before comparison. The small committed fixture contains expected outputs and
provenance, not weights; its deterministic binary-fraction inputs are regenerated
by the generator and Rust test.

The maintained `tests/kev/data/processing.json` oracle is generated by the
hash-verified official API and `encode`/`rows_of` functions. CPU checks compare
rendering and answers; the opt-in tokenizer check compares every token and readout,
plus input/output usage. The native capture and HTTP harness are explicit GPU
checks, never run by ordinary workspace tests.

Full-model acceptance is fixed at maximum probability error 0.01, Choice agreement
for reference top-two margins above 0.02, and Noul agreement at threshold 0.5 when
the reference distance exceeds 0.01. The comparator reports all near-boundary
flips and separately compares unmerged official FP32, exported BF16 PyTorch and
native execution. Failure never widens these limits. See the native recipe for
commands and current validation status. No performance claim is made.
