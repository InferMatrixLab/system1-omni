# Spec 04 — Vela task-head ownership boundary

Status: draft, local-only. RFC Phase 4, scoped as a **boundary spec**:
it fixes what the shared core guarantees to heads and what stays
Vela-owned, and sizes the first head work. The full worker
integration (wire reuse, serving, recipes) is a charter slice that
consumes this; nothing here touches `omni-modernbert`.

## Problem

The Vela readouts are the model's value: the 0.3B "both" readout
(cosine + MLP summed, proj dim 256), the word-unit span readout
(normalize(W_t·LN(h[first sub-word])) · normalize(W_e·LN(h[E])) /
τ_span), per-member calibration (τ, τ_span, thresholds from the
member's own `calibration.json`). The RFC froze that none of this
enters the ModernBERT core, and that heads load through the model's
own path. Without a written contract, head work drifts into the core
or quietly redefines the encoder's output obligations.

## Solution

### The core→head contract (all the heads may assume)

1. Final hidden states at **requested token positions** of the last
   encoder layer, after final norm, in f32 — nothing else (no
   per-layer hidden, no pre-norm states; RFC §5.4).
2. The position set is the head's own declaration (markers, word
   first-sub-words, [E] anchor), resolved before the encoder forward;
   the encoder does not interpret positions.
3. `StructureIdentity` of the profile (the heads' weights are
   geometry-dependent: proj 256 × hidden 768) is available for
   load-time compatibility checks.

### Vela-owned (in `src/models/vela/native`, beside spec 03's loader)

- **Head tensors** (0.3B: the readout/span parameters inside
  `model.safetensors`; hybrids later side-load `broad_head.
  safetensors` — that pattern lands with the hybrid slice): loaded
  F32 exact-shape in the vela crate's own path, omnijev
  `heads.rs`-style (`src/models/omnijev/native/src/heads.rs:121-240`
  prior art), with sha256 pinning per file once a second file exists
  (cua_s1 manifest pattern). Missing/extra/shape/dtype failures list
  both sides — same discipline as the encoder inventory.
- **Readout computation**: F32 on GPU or CPU (first slice: CPU over
  the downloaded hiddens — the reference disables autocast inside
  heads, matching), cosine+MLP summed logits, τ/τ_span tempering,
  span word readout, threshold application (strict `>`,
  per-question override echo) — semantics are the frozen reference
  contract (`src/models/vela/postprocess.py` + the pinned package);
  this spec adds no new semantics, only the native implementation
  surface.
- **Calibration/threshold loading**: per-member
  `calibration.json`, never cross-copied (charter §2 rule); loaded
  beside the checkpoint, validated against the member's identity.

### Explicitly forbidden (fence the review against)

- Any head tensor, temperature, threshold, or readout concept in
  `omni-modernbert`; any encoder IR field that exists only for a
  head; any head reading encoder internals other than contract item
  1; any MultimodalInput-adjacent surface (text/token_ids only).

## Evidence & tests

1. CPU: head-tensor inventory tests (synthetic fixture; missing/
   extra/duplicate/shape/dtype rejections).
2. CPU: readout parity on **recorded hidden fixtures** — run the
   pinned package on the spec-03 prompt set, dump final hidden +
   the package's own head outputs; the native head over the same
   hiddens must reproduce the outputs within the same tolerance
   discipline (exact where the math is exact; float-tolerance where
   the package itself computes in f32 — declared with the fixtures).
   This pins head numerics independently of the encoder (valen
   seam-C precedent).
3. Negative: a core change that alters the final-hidden contract
   (dtype, norm placement, position semantics) must fail these
   fixtures — the fence is testable, not aspirational.

## Acceptance

- Heads load, compute, and match the package on the hidden fixtures;
  `omni-modernbert` diff is empty across the whole head PR.
- The boundary survives review: any core-file touch in the head
   implementation is a spec violation by construction.

## Note on the hybrid members

The hybrid heads (CandidateHead, SpanHeadV2 ×2, set_bias; F32,
side-file `broad_head.safetensors`) obey this same boundary when the
hybrid native slice lands after spec 05 — same contract against the
Qwen backbone's final hidden at requested positions, same model-local
loading. No forward work is taken here beyond keeping the contract
wording backbone-agnostic.
