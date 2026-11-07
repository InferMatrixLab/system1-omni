# Kev-4B preparation and pointer head

First independent slice of [#27](https://github.com/InferMatrixLab/system1-omni/issues/27):
a pinned CPU exporter and a Rust FP32 pointer head. **There is no Kev worker or
end-to-end inference in this slice.** The [recipe](../../../recipe/kev/README.md)
reproduces preparation and head validation. Other Kev sizes are not supported.

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

The future processor must reproduce `encode`/`rows_of`: Qwen's hybrid layers require
one causal row containing state plus one question's complete options. Each option
is read at its closing marker, and the decision is read at the final marker.
Questions must not leak into one another's GDN state. The reference rewrites
caller-supplied `<|name|>` to `<¦name¦>` before tokenization. No HTTP mapping,
prompt compiler, token accounting, request budgets, scheduler or device ownership
is implemented here. Those remain explicit requirements for the worker slice.

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

Passing export/head checks does **not** establish BF16 merge decision parity,
backbone parity, real prompt/token parity, native serving, GPU support or a speedup.
The worker follow-up must qualify BF16 merge versus the official unmerged/FP32
reference, pin full runtime dependencies, reuse selected-position Qwen readouts
(already carried in [#129](https://github.com/InferMatrixLab/system1-omni/pull/129)),
implement model-specific processing, and validate readiness plus direct/frontend
inference for `choice`, `score` and `noul`. Prefix reuse is a later measured change.
