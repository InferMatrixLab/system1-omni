# Spec 01 — ModernBERT family IR, source adapters, capability gate

Status: draft, local-only (branch `model/configurable-base-model`).
Implements the first half of RFC Phase 2: the typed family layer that
today is scattered across `laya/src/config.rs` ensure-blocks,
`model.rs` constants, and `weights.rs` inventory.

## Problem

`src/models/laya/src/config.rs:31-90` reads raw JSON, normalizes the
family, and enforces the one supported profile in a single function;
`src/models/laya/src/model.rs:11` (`const D: usize = 1024`),
`Workspace::sizes` (`:137-162`, hardcoded 2624/4096/2048/1028/256) and
`prepare_encoder` (`:303-513`, `0..28`, `i % 3`) repeat the same
constants as code; `weights.rs:95-163` repeats them again as shapes.
RFC §3 problems 1–3: parsing, geometry and naming are coupled, so a
second ModernBERT profile (Vela 0.3B, frozen as RFC Phase 3) can only
be added by copying a hardcoded loader. There is also nowhere to put
a capability decision that is checkable without CUDA.

## Solution

A new pure-Rust crate `src/models/modernbert` (package
`omni-modernbert`, registered in the root workspace) containing the
family layer with **no dependency on `omni-cuda`, libloading, or any
device pointer** (RFC §5.1 boundary rule):

```text
src/models/modernbert/src/
├── structure.rs   # ModernBertStructure + provenance + family invariants
├── adapter.rs     # LayaBundleAdapter, ModernBertConfigAdapter
├── geometry.rs    # DerivedGeometry: tensor shapes, widths, scratch inputs
├── gate.rs        # CapabilityProfile / backend limits / StructureIdentity
└── plan.rs        # named-kernel encoder plan builder (pure data)
```

No consumer changes in this spec: Laya keeps compiling exactly as
today; the crate lands with its own CPU tests and two recorded
structure fixtures (Laya profile, Vela 0.3B profile). Spec 02 performs
the Laya migration onto it.

## Design

### `ModernBertStructure` (structure.rs)

One type for the whole family; every field carries provenance:

```rust
pub struct ModernBertStructure {
    pub hidden: usize, pub intermediate: usize, pub num_layers: usize,
    pub activation: Activation,            // Gelu (only family member today)
    pub norm_eps: f32,
    pub attention: AttentionSpec,          // heads; head_dim derived; bias=false
    pub schedule: Vec<AttentionKind>,      // normalized Full/Sliding array
    pub local_window: usize,
    pub rope: PerKindRope,                 // theta per kind + optional YarnScaling
    pub vocab: usize,
    pub type_embedding_rows: Option<usize>,
    pub max_positions: usize,
    pub provenance: ProvenanceMap,         // field → (source_path | default | derived)
}
```

Family invariants enforced at construction (fail with the field and
actual value, never silently):

- `heads * head_dim == hidden` (holds for both profiles: 16·64=1024,
  12·64=768; `head_dim` is derived unless the source states it);
- `schedule.len() == num_layers`, kinds ∈ {Full, Sliding};
- `local_window > 0`, `intermediate > 0`, `vocab > 0`,
  `norm_eps > 0`, theta > 0 and finite per kind;
- `activation == Gelu`, `attention_bias == mlp_bias == norm_bias ==
  false` (family constants — a source that sets them is a different
  family, rejected here);
- if YarnScaling present: `factor > 1`, `original_max_positions > 0`.

`mlp.Wi = [2·intermediate, hidden]` is **derived** (RFC §6: the 5248
in the Laya checkpoint is not a semantic field).

### Source adapters (adapter.rs)

The model crate picks its adapter explicitly; each adapter validates
its own source markers (README decision 2):

- `LayaBundleAdapter` — reads `rl_agent_config.json` (only to reject
  on missing/invalid, agent fields stay Laya-side) and
  `encoder/config.json`: requires `model_type == "modernbert"`,
  `hidden_activation == "gelu"`, bias flags false; reads the explicit
  `layer_types` array verbatim into `schedule` (provenance:
  `layer_types[i]`); reads both `rope_parameters[kind].rope_theta`.
- `ModernBertConfigAdapter` — reads any HF-ModernBERT-style config
  object (the Vela 0.3B case: the `encoder_config` member of the
  vela2-unified root; requires inner `model_type ==
  "modernbert"`): **derives** `schedule` from
  `global_attn_every_n_layers` (full at i % n == 0 — provenance marks
  it derived with the source field name), reads per-kind thetas
  (`global_rope_theta` / `local_rope_theta`), maps `rope_scaling`
  (`rope_type == "yarn"`, `factor`, `original_max_position_embeddings`)
  into `YarnScaling`, and treats `position_embedding_type ==
  "sans_pos"` and absent type-vocab as `type_embedding_rows: None`.

Both record `source.config_files` in the provenance map (RFC §5.2).
Unknown top-level fields are ignored but listed in the provenance
summary (explicit policy, no silent misread — the adapter must never
accept a `max_length`-style recipe limit as a structure field; the
Vela root's `max_length: 8192` is worker config, not encoder
geometry).

### DerivedGeometry (geometry.rs)

Pure functions of the structure: `qkv = [3·hidden, hidden]`,
`attn_out = [hidden, hidden]`, `mlp.Wi = [2·intermediate, hidden]`,
`mlp.Wo = [hidden, intermediate]`, `rotary_half = head_dim / 2` (32
for both profiles), rope table bytes `max_positions_exercised ·
rotary_half · 4`, and the encoder scratch size formula parameterized
by (b, l, structure) — replacing `Workspace::sizes`' hardcoded
`m * D * 6` / `m * 2624 * 2` arithmetic with structure-derived terms
(task-specific buffers stay with the model, spec 02).

### Capability gate (gate.rs)

Two levels, both CPU-only, both answering with actual-vs-required:

- `BackendLimits` — what one CUDA bundle supports: instantiated hidden
  sizes, `(b, l)` envelope (today: b power-of-two ≤ 16, 16 ≤ l ≤ 512,
  l % 16 == 0), rope policy (plain-theta tables only), dtype policy.
- `RegisteredProfile` — a named, frozen structure+limits pair. First
  registration: `laya-0.3.20` (the exact constants of
  `config.rs:50-79` as data). Acceptance = family invariants ∧
  backend limits ∧ (if a registered profile is requested) structure
  equality with a field-diff on mismatch.

`StructureIdentity` (RFC §8): a digest over `schema_version`, family
tag, canonical structure serialization, and the backend/ABI tag.
Defined here, consumed by cache/identity decisions later; no caller
yet — landing it now is cheap and fixes the format before two
profiles exist.

### Plan builder (plan.rs)

`encoder_plan(&structure, opts) -> Vec<PlanStep>` reproduces the
encoder half of today's `prepare_encoder` sequence (embed → per-layer
qkv/rope/attn/out/addln/geglu/down/addln → final norm → optional
type) emitting **named kernel names + argument roles** as pure data —
no `omni_cuda::Kernel` resolution, which stays with the consumer
(Laya keeps its `SelectedKernel` resolver). This keeps the core crate
CUDA-free while making the execution order fixture-testable.

## Evidence & tests (all CPU, in-crate)

1. Adapter provenance matrix: present/missing/type-error/defaulted
   fields for both adapters; derived-schedule case (22 entries, 8
   full) provenance-checked.
2. Invariant rejections: each violated invariant fails with the field
   and actual value.
3. Gate matrix: `laya-0.3.20` fixture accepted; each single-field
   mutation (hidden 768 against the ABI-1 limits, theta drift,
   schedule edit, window 127) rejected with a diff naming both
   values; raw-JSON inputs never touch CUDA (crate has no such dep —
   structurally guaranteed).
4. `StructureIdentity`: same config → same digest; any one field
   changed → different digest.
5. Plan fixtures: `encoder_plan` on the Laya structure produces the
   exact step sequence of `prepare_encoder`'s encoder half (recorded
   as a golden fixture; spec 02 makes the live comparison).

## Acceptance

- `omni-modernbert` compiles with zero device dependencies and lands
  in the workspace; no existing crate changes behavior; full existing
  test suite stays green.
- The two structure fixtures (Laya, Vela 0.3B) round-trip through
  their adapters with complete provenance, and the gate's
  accept/reject matrix is recorded as tests.

## Open items (resolved by later specs, noted here)

- The regenerated-bundle limits for the 768-hidden profile (spec 03)
  register as a second `BackendLimits` + profile entry; the gate API
  must make that a data addition, not a code change.
- `schema_version` starts at 1; bump rules recorded in structure.rs
  docs when the first bump happens.
