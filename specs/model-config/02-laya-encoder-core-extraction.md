# Spec 02 — Laya native on the extracted encoder core

Status: draft, local-only. Second half of RFC Phase 2: migrate the
Laya native path onto spec 01's core with **zero behavior change**,
keeping Laya as the running proof that the extracted core serves a
real profile (RFC Phase-2 deliverable: "Laya 仍能以原 profile 运行").

## Problem

After spec 01 the family layer exists but Laya still executes its own
copies: `prepare_encoder` hardcodes the 28-layer / i%3 / D=1024 plan
(`model.rs:303-513`), `Workspace::sizes` hardcodes
2624/4096/2048/1028/256 (`:137-162`), `infer` hardcodes vocab 50368
(`:679`), and `checkpoint_tensors()` hardcodes the inventory
(`weights.rs:95-163`). The Laya agent task head (the `head.layers.0..2`
loop, scorer, act_head) is interleaved with the encoder plan in one
function — exactly the backbone/task coupling the RFC forbids
extending.

## Solution

Laya becomes a consumer of `omni-modernbert`; its public surface
(`Config::load`, `Model::load*`, `infer`, `ModelOutput`), its CUDA
bundle, and its ABI stay untouched. Migration is internal:

1. **Config facade** (`config.rs`): `Config::load(dir)` keeps its
   signature and public fields (`AgentConfig`, `EncoderConfig`).
   Internally: `LayaBundleAdapter::read(dir)` → `ModernBertStructure`
   → `laya-0.3.20` profile validation via the gate (the ensure-set of
   `config.rs:46-79` moves into the registered profile) → projection
   back onto the old public fields. Agent-config parsing and its
   validations (max_len 512 / head_max_len 192 / head_layers 2,
   temperatures) stay in the Laya crate verbatim — they are Laya task
   bundle, not encoder fields (RFC §2.4).
2. **Plan split** (`model.rs`): `prepare_encoder` decomposes into
   - `modernbert::plan::encoder_plan(&structure, rope_variant)` —
     the embed → 28 layers → final-norm → type steps, produced from
     the IR (spec 01 §plan builder); the `original_rope` flag remains
     a Laya serving option passed through as a plan option;
   - a Laya-owned `head_plan()` — the `head.layers.{0,1}` loop and
     everything after the encoder (type consume, scorer/act_head
     launches stay in `infer` as today).
   The two plans concatenate into the same `Vec<Step>` the executor
   consumes; `Step`/`Argument`/`SelectedKernel` stay in Laya (they
   are the CUDA-facing resolution layer, not family data).
3. **Workspace**: encoder scratch sizes come from
   `geometry.rs` formulas; the task-side buffers (head ffn 4·D, head
   rows 2048, action vocab 1028, action hidden 256, marker caps) are
   sized by a Laya-side layout function appended to the encoder
   layout. The single-`Workspace` struct and the graph-cache/
   eager-fallback logic (`model.rs:198-244, 706-768`) are unchanged.
4. **Weight inventory**: `checkpoint_tensors()` = core-generated
   encoder inventory (canonical names
   `encoder.embeddings.tok_embeddings.weight`,
   `encoder.layers.{i}.{attn.Wqkv,attn.Wo,mlp.Wi,mlp.Wo}.weight`,
   norms, `encoder.final_norm.weight`, `type_emb.weight`; shapes from
   `DerivedGeometry`) **concatenated with** the Laya task inventory
   (head.layers.{0,1}, scorer, act_head, temperature —
   `weights.rs:123-161` stays verbatim). `validate_names` still runs
   over the union inside Laya's own load path (RFC §7: the model's
   loader owns external→internal mapping and missing/extra checks;
   Laya's mapping is the identity, which is precisely why it is the
   reuse baseline).
5. **Runtime bounds**: `infer`'s `*i < 50368` check reads
   `structure.vocab`; the `(b, l)` envelope comes from the profile's
   backend limits instead of the inline `ensure!`s (`model.rs:676-709`
   keeps its packing checks — they are request validation, not
   structure).

Unchanged on purpose: `storage_dtype` (the per-class f16/bf16/f32
storage policy is Laya's load policy, owned here), the bundle's rope
`.f32` tables and their 512×32 check (replaced by a
limits-consistency check, same numbers), `LAYA_VERIFY_WEIGHTS`
read-back, dump hooks, `serve.rs`/`executor.rs` — no serving-surface
change of any kind.

## Evidence & tests

1. **Plan bit-identity fixture**: serialize the full
   `Vec<Step>` (kernel name, argument kinds, counts) for the Laya
   profile before and after the refactor; assert equality. Recorded
   once as a golden fixture so any later core change that alters
   Laya's execution order fails CPU CI (seam C).
2. Existing suites stay green unchanged: `tests/laya/{checkpoint,
   weights, preprocess, packing, decision, batch, model}.rs` — same
   fixtures, same expectations.
3. Device-gated (seam D): `tests/cuda/laya_runtime.rs` and the
   graph-capture path re-run on available hardware; parity with the
   pre-refactor outputs on the existing fixtures (bit-identical
   expected: same kernels, same order, same bytes).
4. Negative tests: the profile gate's reject matrix from spec 01 is
   exercised through the real `Config::load` (mutated fixture configs
   produce the same refusals as today, now with field-diff errors).

## Acceptance

- `cargo test` (CPU) fully green with no fixture edits; the plan
   fixture proves execution-order identity; GPU regression recorded
   when hardware is available, reported with hardware line per repo
   convention.
- `git grep` shows no remaining profile constants in
   `src/models/laya/src/` outside the registered-profile data and the
   task-head code path (D=1024 survives only where it is genuinely
   task geometry, e.g. head ffn, act_head dims — each tagged with the
   task bundle in review).
- ABI 1 unchanged; `laya.backend.json` untouched; no rebuild of the
   CUDA bundle required by this spec.

## Risks

- **Plan reorder by accident** (e.g. weight-pointer resolution
  interplay) — mitigated by the plan fixture and unchanged
  `upload_order` discipline.
- **Workspace layout drift** changing graph-cache shape keys — the
  cache keys on `(b, l)`, and total sizes must equal the old formula
  for the Laya profile (asserted in a unit test against the recorded
  numbers).
