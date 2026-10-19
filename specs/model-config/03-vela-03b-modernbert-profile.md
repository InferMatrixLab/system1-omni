# Spec 03 — Vela 0.3B as the second ModernBERT profile

Status: draft, local-only. RFC Phase 3: wire the pinned
`vllm-sr/Vela-2.0-0.3B@d6f03aa9` encoder into the extracted core as a
**profile, not a new family** (frozen decision: "Vela 0.3B 是
ModernBERT 的第二个 profile，不是独立 encoder family"). Deliverable
per the RFC: native profile parity with the frozen reference contract
on the encoder surface (final hidden at requested positions).

## Problem

The reference worker (commit `2812144`) froze the behavioral contract
but executes the upstream Python package. A native profile needs: the
0.3B structure accepted through the family layer (it differs from
Laya in every dimension — see README table), its all-F32 checkpoint
mapped and validated, YaRN-scaled rotary tables the current bundle
does not produce, and a CUDA bundle whose kernels are generated for
1024-hidden geometry only (`model_ops.cu` hardcodes 1024/1028;
TileLang GEMMs are instantiated per N/K). None of this exists; doing
it inside Laya would recreate the coupling Phase 2 just removed.

## Solution

### 1. Consumer crate

New Rust crate `src/models/vela/native` (package `omni-vela-native`)
depending on `omni-modernbert` and `omni-cuda` (feature-gated like
`omni-laya`). It owns, in this spec: the profile registration, the
model-local weight loading (RFC §7 ownership), rope-table
preparation, and the encoder execution against the regenerated
bundle. Task heads (spec 04), worker/serving surfaces (charter
slices), and the hybrid (Qwen-backboned) members are explicitly not
here.

### 2. Profile registration and adapter wiring

`ModernBertConfigAdapter` (spec 01) already parses the 0.3B
`encoder_config`. This spec registers `vela-2.0-0.3b`: structure
validated by the family invariants, then checked against the
**regenerated bundle's** `BackendLimits` (hidden 768 instantiated;
b/l envelope; rope policy = yarn allowed; dtype policy below). The
derived schedule (22 kinds, full at i%3==0) must be cross-checked
once against the pinned package's encoder construction and recorded
as a fixture — `global_attn_every_n_layers` is read as data, never
re-derived from habit.

### 3. Weight loading (model-local, no WeightContract)

The pinned docstring fixes the naming discipline: parameter names are
exactly the training checkpoint's, `model.safetensors` strict-loads,
encoder = stock `ModernBertModel` under `encoder.*`. That is the same
canonical ModernBERT naming Laya uses, minus `type_emb`, at 768
geometry. The vela crate's `Weights::load`:

- maps `encoder.embeddings.tok_embeddings.weight`,
  `encoder.layers.{i}.{attn_norm,attn.Wqkv,attn.Wo,mlp.Wi,mlp.Wo,
  mlp_norm}`, `encoder.final_norm.weight` (external == internal for
  the encoder half);
- validates shape against `DerivedGeometry` (vocab 256008 · hidden
  768; qkv [2304, 768]; Wi [2304, 768]; Wo [768, 1152]; …) and dtype
  (F32 source), with missing/extra/duplicate failure listing both
  sets — the laya `validate_names` discipline, re-owned here;
- applies the profile's **storage policy**: GEMM-class tensors cast
  to bf16 at upload; norms and embeddings kept f32/f16 per class —
  mirroring Laya's `storage_dtype` classes. The F32→bf16 rounding is
  a declared numerics choice carried in `StructureIdentity`; its
  parity cost is measured, not assumed (below).

First implementation step before any wiring: census the pinned
checkpoint's tensor inventory and dtypes from the safetensors header
and record it as the fixture this loader must satisfy (names are
evidence, not assumption — the local survey copy carries code+config
only, not weights).

### 4. Rotary tables (YaRN ×4)

The only genuinely new encoder math. Served positions stay ≤ 8192 =
`original_max_position_embeddings`, but YaRN still rescales the
frequency basis (ramp between `beta_fast` 32 / `beta_slow` 1, factor
4, truncate beyond 32 768) — the tables are **not** plain
theta-160000 tables even inside the original window. Plan:

- compute cos/sin (rotary_half = 32 per kind; both kinds share theta
  160000 and the scaling) CPU-side in f32 from the IR's `YarnScaling`
  at load time — deterministic, no extra bundle artifact;
- ground them against the pinned package: run the package's own
  rotary path on CPU/FP32, dump tables at sampled positions (0, 1,
  mid, 8191/8192 boundary, 16 384, 32 767) into fixtures, and gate
  the Rust generator on those fixtures. Any attention-scale factor
  the package applies with YaRN (mscale-style) must be discovered
  from the pinned `transformers` version's rotary implementation and
  either baked into the tables or passed to the attention kernel —
  recorded as part of the fixture evidence, not guessed.

### 5. Kernel bundle regeneration + the ABI decision

The one kernel-scope item in the whole series (everything else is
Rust):

- **TileLang instantiations** for 768-hidden geometry: gemm
  (N, K) ∈ {(2304, 768), (768, 768), (1152, 768)} + geglu F=1152 +
  down (768, 1152); attention base kernels already take (b, l) at
  launch — verify the 12-head / Dh=64 / window-128 instantiations
  accept l up to 8192 (shared-mem tiling bounds), adding bucket
  specializations (`attn_full_l8192`-style, resolved like today's
  `_b1_l512` variants) if the generic path does not hold;
- **`model_ops.cu` de-hardcoding**: `embed_norm` / `gather_norm` /
  `add_type` take D as a runtime argument instead of `1024`;
  `laya_linear`'s `n ≤ 1024` guard becomes a limits-derived bound;
  `action_features`'s 1028/1024 stays Laya-task-local;
- **ABI**: this changes kernel sources and the exported bundle ⇒
  bump `laya.backend.json` `abi_version` 1 → 2 and rebuild. Blast
  radius is the laya backend's consumers only (Laya + this crate) —
  unlike the Qwen ABI, no sibling coordination. A separate vela
  backend manifest reusing the same sources was considered and
  rejected as default: one source tree, one honest ABI ledger. Rust
  -only changes elsewhere in this series do not bump anything (RFC
  §8 rule, restated in spec 05 for Qwen).

### 6. Runtime envelope

First slice mirrors the reference worker's execution shape:
serialized single request, one encoder forward per rendered sequence,
final hidden gathered at the marker/requested positions (the RFC's
position-level readout; no per-layer extraction). Registered limits:
b ∈ {1, 2, 4}, l buckets {512 … 8192} (l % 16), vocab bound from the
structure. Workspace sizing via the core geometry formulas; graph
capture per (b, l) bucket reuses the Laya workspace-cache pattern if
it pays for itself — optional, evidence-gated.

### 7. Parity harness (seam B)

Extend the `recipe/vela` harness pattern: drive the **pinned
package** (CPU, FP32, the charter oracle) to dump final hidden at
requested positions for a fixture prompt set (reusing the reference
worker's frozen example/rejection fixtures where they exercise the
encoder); run the native profile over the same token ids; report
max/rms deltas. Tolerance gates are declared at implementation,
anchored to the upstream noise floor (≈2.9e-3 max logit, charter §3)
— the bf16 GEMM storage policy's cost shows up here and is reported,
not averaged away. A device-gated variant runs the native side on
GPU with hardware reported (seam D).

## Evidence & tests

1. CPU: adapter→IR→gate accept for `vela-2.0-0.3b`; reject matrix
   (hidden 1024 against the 768-instantiated limits, schedule edit,
   yarn factor drift) with field diffs.
2. CPU: weight-mapping tests against a synthetic safetensors fixture
   built from the recorded census (missing/extra/duplicate/shape/
   dtype each fail loudly); storage-policy class assignment table.
3. CPU: rope-table fixtures vs the pinned package at the sampled
   positions (§4).
4. CPU: plan fixture for the vela profile (22 layers, kind sequence,
   no type step) — same golden-fixture discipline as spec 02.
5. GPU (device-gated): encoder forward on real weights; parity
   report vs the CPU-FP32 oracle per §7; two consecutive runs
   identical.

## Acceptance

- The 0.3B encoder runs natively with its own profile through the
  shared core; Laya still passes everything (spec 02's fixtures
  untouched) — the two profiles share the core and nothing else.
- Parity report recorded with tested hardware; failures or
  unexercised checks disclosed per repo convention.

## Out of scope (fences)

Vela task heads (spec 04 — they only consume the final-hidden
contract), the native worker/serving/frontend slice, ONNX options,
the hybrid members (they ride spec 05 + charter slice 4), any change
to Laya's numerics or bundle behavior beyond the shared-bundle
regeneration (Laya's own instantiations and outputs must remain
bit-identical — verified by spec 02's GPU regression re-running on
the ABI-2 bundle).

## Risks

- **YaRN subtleties** (mscale, ramp conventions differing between the
  pinned transformers version and our generator) — mitigated by
  fixture-grounding against the package, not the HF docs.
- **Generic attention kernel failing at l=8192** — bucket
  specializations are the planned fallback; if even those are
  unsound, the profile's first slice can register l ≤ 4096 while the
  windowing plan (charter §3's ≤1800-token windows) covers the 8192
  budget — a limits decision, not a family change.
- **bf16 storage rounding** moving low-margin answers — measured by
  the parity harness before any tolerance is frozen; switching the
  policy is a `StructureIdentity` change, visible by construction.
