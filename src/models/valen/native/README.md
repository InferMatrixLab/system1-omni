# Valen-Preview-0923 native worker

The native `/v1/systemone` worker for Valen-Preview-0923 (spec 03): the
reference worker's HTTP contract, compiler-equivalent request processing, and
eager execution on the shared Qwen3.5 native backends — no Python or PyTorch
in the serving path. Startup, examples, validation scope and the declared
parity gates live in [`recipe/valen/native.md`](../../../../recipe/valen/native.md);
the reference worker's contract is documented in [`../README.md`](../README.md).

## Layout

| Module | Role |
| --- | --- |
| `wire` | the JSON body decode with the reference's exact rejection bodies |
| `protocol` | `/v1/systemone` wire validation (limits, acceptance order, 4xx bodies) |
| `preprocess` | question records, chat-template-equivalent state rendering, image grid |
| `compile` | the pinned compiler: branch suffixes, candidate/decision position capture, token accounting |
| `text` | unmerged FP32 language-LoRA loading and export-policy validation (N2) |
| `head` | the CPU FP32 decision head from `head.safetensors` |
| `engine` | execution orchestration: text/image paths, mrope positions, readout |
| `finish` | response finishing with `json.dumps` byte parity (CPython float repr) |
| `server` | the HTTP surface: routes, gate order, error bodies, serial admission |
| `main.rs` | the `omni-valen-native` binary: warmup before the listener binds |

## Served scope

All three question types are served end-to-end (text and single-image
states): `choice` (1–255 named candidates), `noul` (fixed `true`/`false`,
optional custom wording compiles as the equivalent choice record and still
answers `type: noul` with P(true)), and `score` (2–10 ordered levels, one
isolated branch per level, cross-branch softmax → expected level + legend +
concentration confidence). The types mix freely in one request.

## Tests

- CPU, no GPU: `tests/valen/native.rs`, `native_tokenization.rs` (N3),
  `worker.rs` (the N4 contract suite + the N5 noul/score finisher byte
  parity and mixed-type round trips), plus the frontend integration
  `tests/frontend/native_worker.rs`.
- Env-gated GPU: `tests/valen/valen_vision.rs` (N1), `tests/valen/text.rs`,
  `text_reference.rs` (N2), and `worker_gpu.rs` (the end-to-end parity gate
  for all three question types; run serially on one GPU — `--test-threads 1`
  — each test loads the model).
