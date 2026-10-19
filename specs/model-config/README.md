# Configurable base-model specs (RFC implementation, Phases 2–5)

Specs for implementing the configurable base model defined by
`docs/rfc/model-parameter-configuration.md` (Final). This is charter
step ③ of the Vela 2.0 integration order: the 0.3B reference worker
(step ①) landed as `2812144`; the RFC implementation is the
prerequisite every native Vela slice waits on — the 0.3B ModernBERT
native profile rides Phases 2–4, the 0.8B/4B/9B Qwen-hybrid native
route rides Phase 5.

**Local and untracked-of-record** until promoted: this series is
committed on branch `model/configurable-base-model` only; promotion to
tracker issues happens spec by spec after the user approves, same rule
as `specs/valen/`.

## Index

| Spec | Title | RFC phase | Depends on |
| --- | --- | --- | --- |
| [01](01-modernbert-family-ir.md) | ModernBERT family IR, source adapters, capability gate | 2 (first half) | — |
| [02](02-laya-encoder-core-extraction.md) | Laya native on the extracted encoder core | 2 (second half) | 01 |
| [03](03-vela-03b-modernbert-profile.md) | Vela 0.3B as the second ModernBERT profile | 3 | 02 |
| [04](04-vela-task-heads-boundary.md) | Vela task-head ownership boundary | 4 | 03 |
| [05](05-qwen-hybrid-config-capability-gate.md) | Qwen hybrid config/capability migration | 5 | 01 (pattern); independent of 02–04 |

PR order on this branch: 01 → 02 → 03 → 04, with 05 parallel after 01.
RFC Phase 6 (unified facade sweep + release evidence) is a follow-on
after the Vela slices consume these outputs; it is deliberately not
spec'd here.

## What this series does NOT do

Frozen-boundary fence (RFC §11, charter §8) — no spec below may:
introduce a WeightContract/WeightResolver (weight mapping stays in each
model's own `load_weight`/`Weights::load`), move task heads or their
tensors into a shared core, extend `MultimodalInput` (text/token_ids
only), promise per-layer hidden extraction (final hidden at requested
positions only, reusing #129/#134), reopen the Laya-baseline or
ModernBERT-profile decisions, or change any Qwen gate constant /
kernel / ABI (ABI 6 stays).

Also out: the Vela workers themselves (HTTP, frontend, recipes —
charter slices that consume these outputs), ONNX serving options,
quantization, and any Qwen-hybrid native executor changes.

## Decisions taken on the RFC's "Not yet specified" list

The RFC left four execution-stage questions open (§"Not yet
specified"); this series takes these stances:

1. **Normalized schema = Rust-internal IR only.** No operator-facing
   manifest file is introduced; `schema_version` stays an internal
   field of the IR. Revisit only if a non-Rust consumer appears.
2. **`family` is chosen explicitly by the adapter the model crate
   constructs** — never inferred silently from `model_type`. Each
   adapter validates its own source markers (`model_type` etc.) and
   produces a family IR; a mismatch fails with source evidence.
3. **Qwen keeps the frozen ABI-6 capability set** (RFC §5.3): no
   arbitrary Full/GDN schedule opening in this series. The gate
   expresses today's reject set as data; widening it is a future,
   benchmark-backed decision.
4. **Bundle layout and profile fields** are defined in spec 03 (the
   only profile that needs a new bundle); `StructureIdentity` is
   defined once in spec 01 and consumed wherever identity matters.

## The two profiles this series makes configurable (evidence)

Both are ModernBERT-family encoders with identical naming discipline
(HF ModernBERT tensor names); every number below is read from source,
not assumed:

| Field | Laya profile (`rl_agent_config.json` + `encoder/config.json`) | Vela 0.3B profile (`encoder_config` in the pinned repo config) |
| --- | --- | --- |
| hidden / intermediate | 1024 / 2624 | 768 / 1152 |
| layers / heads / head_dim | 28 / 16 / 64 (derived) | 22 / 12 / 64 (derived) |
| schedule | explicit `layer_types` array, full at i%3==0 | `global_attn_every_n_layers: 3` (derived array, full at i%3==0 → 8 full / 14 sliding) |
| local window | 128 | 128 |
| RoPE | plain theta, full 160000 / sliding 10000 (two `rope_parameters` entries) | both kinds 160000 + `rope_scaling` yarn (factor 4.0, original_max 8192, truncate) |
| vocab | 50368 | 256008 |
| type embedding | `type_emb.weight` [3, 1024] | none |
| norm eps / activation / bias | 1e-5 / gelu / all false | 1e-5 / gelu / all false |
| max positions exercised | 512 (bundle rope tables, workspace cap) | 8192 served (`max_length`; tables to 32768) |
| checkpoint dtype | mixed per class (f16 embed, bf16 GEMM class, f32 norms) | all F32, single `model.safetensors`, strict load |

Vela 0.3B facts from the pinned copy at `d6f03aa9`
(`modeling_vela2.py` `_build`: `m.encoder = ModernBertModel(ecfg)`,
stock HF naming under `encoder.*`; parameter names "exactly those of
the training checkpoint, so `model.safetensors` loads with
strict=True").

## Why the kernels are in scope (the one real capability gap)

The shared CUDA bundle (`src/backends/cuda/kernels/`, manifest
`laya.backend.json`, ABI 1) is **generated per geometry**: the
TileLang templates take `N, K` (GEMM) and `B, L, H, Dh, window`
(attention) as compile-time parameters, and `kernels/model_ops.cu`
hardcodes 1024/1028/2048 (`embed_norm` ids[r]*1024, `add_type` 1024,
`action_features` 1028, `laya_linear` guard n≤1024). Laya's 768-hidden
sibling therefore needs bundle regeneration plus de-hardcoding — a
kernel-source change, hence an explicit ABI decision (spec 03), which
is exactly the "configurable ≠ arbitrary" boundary the RFC freezes.
Everything else about the Vela profile is adapter/mapping/planning
work in Rust.

## Test seams

Mirrors `specs/valen/` (no new seams):

- **Seam C — CPU golden fixtures** is the main evidence channel for
  01/02/05: adapter provenance, IR invariants, gate accept/reject
  matrices, serialized execution-plan fixtures, derived prefix-layout
  sizes — all without a GPU.
- **Seam B — parity harness** (spec 03): native encoder final hidden
  at requested positions vs the pinned 0.3B package (CPU, FP32
  oracle), tolerance declared at implementation anchored by the
  upstream noise floor (charter §3).
- **Seam D — device-gated GPU regression**: existing
  `tests/cuda/laya_runtime.rs` stays green through the refactor;
  spec 03 adds a Vela-profile encoder smoke.

## Standing conventions

Inherited verbatim from `specs/valen/README.md`: repo-facing docs and
code comments carry no process narrative (no spec numbers, tickets, or
decision-status references in `src/`, `tests/`, `recipe/`); no
trailing `# …` comments; evidence blocks phrased as measured facts.
The specs themselves (this directory) are local planning documents and
may reference the RFC and wayfinder assets.

## Provenance of the facts embedded in these specs

Working tree @ `2812144` (branch `model/vela2-reference-serving`):
`src/models/laya/src/{config,model,weights}.rs`,
`src/models/qwen3_5/native/src/model.rs`,
`src/backends/cuda/kernels/{model_ops.cu,laya_tilelang.py}`,
`src/backends/cuda/laya.backend.json`, `tests/laya/*`,
`src/models/omnijev/native/src/heads.rs`. Pinned upstream copies:
`/tmp/vela-survey/Vela-2.0-0.3B` (`d6f03aa9`; config.json,
modeling_vela2.py). Research baselines: `wayfinder/research/017-*`
(upstream survey), `019-*` (hybrid configs vs native gates — the
gate×verdict table and per-size geometry), charter
`wayfinder/drafts/vela-integration-charter.md` (locked order,
ownership, tolerance anchors). Line numbers drift; the facts were
verified at planning time.
