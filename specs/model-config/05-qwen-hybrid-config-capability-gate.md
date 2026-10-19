# Spec 05 — Qwen hybrid config/capability migration

Status: draft, local-only. RFC Phase 5: separate source parsing,
family normalization, and kernel capability inside the existing
`src/models/qwen3_5/native` family crate — the vehicle the Vela
hybrid members' native route rides (charter: "0.8B native rides the
Phase-5 outputs"). **No gate constant, kernel, or ABI change**: the
frozen ABI-6 reject set (RFC §5.3) is expressed as data, not widened.

## Problem

`Config::load` (`src/models/qwen3_5/native/src/model.rs:49-180`)
simultaneously reads JSON, normalizes the family, and enforces
kernel capability via inline `ensure!`s (silu, output gate, no bias,
conv kernel 4, head dims 256/128/128, rotary_half 32, interleaved
mrope summing to 32, heads divisibility, hidden % 8) — RFC §2.1's
"固定能力边界". Consequences: capability decisions are not checkable
without constructing a `Config` from a directory; the Vela hybrid
consumer's three known trips (config location `backbone_config`,
tensor prefix `backbone.`, per-tensor mixed dtype — research 019,
gate table C1/W1/W2) have no seam to plug into without forking the
loader; and `PrefixState` sizing (full K/V per full layer, F32
[H,K,V] GDN state + conv tail, `model.rs:557-588`) is implicit in
`alloc_prefix` rather than derived from structure where it can be
CPU-tested.

## Solution

All inside the family crate (no new crate — the family owns its IR):

1. **`QwenHybridStructure` IR**: today's `Config` public fields plus
   the ordered `layer_kinds` (Full/GDN) with provenance, split per
   RFC §6 (attention: heads/kv_heads/head_dim/rotary_half/rope_theta/
   mrope_section/output-gate; GDN: lin_k/lin_v heads and dims, conv
   kernel; common: hidden/intermediate/layers/eps/max_positions/
   activation). Constructed only by a source adapter.
2. **Source adapter**: exactly today's reading — root or
   `text_config` (`model.rs:55`), `layer_types` strings with
   length == `num_hidden_layers`. It parses and records provenance;
   it does not judge capability.
3. **`QwenAbi6` capability gate**: the complete current ensure-set
   as typed checks over the IR, each rejection reporting actual vs
   required. Content-frozen: the accepted set after this spec equals
   the accepted set before it (fixture-proven below).
4. **`Config::load` facade**: adapter → invariants → gate → project
   onto the unchanged public `Config` fields (shadow-normalize,
   RFC §8 migration rule). All five consumers (qwen workers,
   jev_vl, cua_s1, open_jev, omnijev) compile and behave identically
   — public API untouched, ABI 6 untouched, no rebuild required.
5. **`PrefixLayout` derived geometry**: per full-attention layer
   (post-prep K/raw V bytes at head_dim 256) and per GDN layer (F32
   [H,K,V] recurrent state + three conv-tail rows at conv kernel 4),
   computed from the IR and CPU-testable; `alloc_prefix` consumes
   the derived layout (identical numbers, now asserted by tests
   instead of implied). The 64-token boundary and `max_positions`
   cap stay in `alloc_prefix` as today.
6. **Loader seams for the hybrid consumer** (used by Vela's crate,
   charter ticket 030 ownership — nothing Vela-specific lives here):
   - `Config::from_structure(ir)` — a constructor accepting an
     already-parsed structure, so the Vela source adapter can
     extract `backbone_config` from its bundle and feed the family
     gate directly;
   - `Weights::load` gains optional **prefix candidates** (default:
     the current three, `model.rs:273-281`) and a **per-tensor
     dtype policy** (default: uniform BF16, today's `ensure!` at
     `model.rs:291-295`). The Vela consumer passes the `backbone.`
     candidate and the faithful mixed policy (Linear+conv1d BF16,
     embed/norms/A_log/dt_bias F32, cast at upload) — per-tensor
     faithful loading per the charter; export re-save stays the
     recorded fallback. Rust-only, no C symbol or layout change ⇒
     **no ABI bump** (RFC §8 rule).
   - `StructureIdentity` (the spec-01 pattern, family tag `qwen3_5`):
     normalized schema version, structure digest, weight-mapping
     digest, dtype policy, backend, ABI 6. Prefix-cache keys/records
     carry it; any change forces re-capture instead of reuse
     (jev_vl's cross-request cache is the consumer to audit).

## Evidence & tests (all CPU unless noted)

1. **Shadow equivalence**: every existing config fixture —
   `tests/qwen3_5/config.rs` + `tests/qwen3_5/data/config.json` and
   each consumer's pinned configs — produces byte-equal public
   `Config` fields through the facade (recorded before/after).
2. **Gate matrix**: the frozen reject set (head_dim ≠ 256, lin dims
   ≠ 128, conv ≠ 4, non-silu, output gate off, bias on, non-default
   rope, non-interleaved mrope, sections ≠ 32 sum, bad max positions,
   heads % kv, lin_v % lin_k, hidden % 8) each rejected with
   actual-vs-required; plus the accept side using the three Vela
   hybrid backbone geometries as fixtures (0.8B: 1024/3584/24 layers
   18+6, 8+2 heads, 16/16 GDN; 4B: 2560/9216/32, 16+4, 16/32; 9B:
   4096/12288/32, 16+4, 16/32 — research 019 §2 table, which already
   established all three pass every gate). Raw config inputs never
   approach CUDA: gate tests construct IR directly (seam C).
3. **PrefixLayout**: derived sizes equal the current implicit
   allocations for the fixture geometries (unit-asserted against
   recorded numbers per size).
4. **Consumer regression**: jev_vl / cua_s1 / open_jev / omnijev test
   suites green; `PrefixState` capture/continue device-gated tests
   (existing) re-run where hardware allows (seam D).
5. Loader-seam tests: mixed-policy loading on a synthetic sharded
   fixture (BF16 Linear class + F32 norm class) with missing/extra/
   shape/dtype rejections; prefix-candidate selection including the
   `backbone.` case excluding in-shard head tensors by prefix.

## Acceptance

- Behavior-preserving by construction and by fixtures: no consumer
  change, no gate widening, ABI 6 and all kernel contracts untouched.
- The three Vela hybrid geometries pass the gate as pure CPU tests —
  the Phase-5 output the hybrid native slice (charter step ④)
  consumes with only Vela-side source-adapter and loader work left.

## Out of scope

The Vela hybrid consumer crate itself (charter slice 4: execution
plan over `alloc_prefix` capture/continue, windowing, heads), any
arbitrary Full/GDN schedule opening (README decision 3), M-RoPE or
modal behavior changes (opaque per RFC §5.2), prefix-state disk
persistence, and anything touching `src/backends/cuda/qwen3_5`.

## Risks

- **Facade drift** (a field quietly re-derived differently) —
  shadow-equivalence fixtures are the guard; run them first in the
  PR.
- **Co-editing window**: the family crate has live sibling PRs
  (charter §"Co-edit surface"); this refactor lands as a pure
  internal move with zero API delta to minimize rebase pain, and
  coordinates timing if an ABI-window PR is in flight (no ABI
  interaction here, but `model.rs` is shared ground).
