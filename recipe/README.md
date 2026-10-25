# Recipes

For a first real decision, follow the [complete CPU walkthrough](../docs/getting-started.md)
([中文](../docs/getting-started.zh.md)) and its [recorded LAYA demo](laya/validation.md).

- [Laya text worker](laya/README.md): start the external Python worker, connect the
  Rust frontend and compare direct and proxied responses.
- [Laya native CUDA worker](laya/native/README.md): build the Hopper bundle and
  serve English text decisions with Rust and CUDA.
- [Laya on Apple Silicon](laya/apple-silicon.md): serve Laya on the Mac GPU with the Laya
  worker, put the frontend in front of it and run the benchmarks.
- [Cua-S1 4B 0.2 text worker](cua_s1/text.md): download the pinned weights, start
  the worker and connect the Rust frontend.
- [Valen-Preview-0923 reference worker](valen/reference.md): prepare the pinned
  Preview checkpoint and Qwen base, start the text-or-image `choice` worker and verify the frontend.
- [Cua-S1 4B 0.2 native text worker](cua_s1/native.md): build the CUDA library and
  the Rust worker, export BF16 weights with the original FP32 adapter and start
  the worker.
- [Decider-2B v11 native text worker](decider/README.md): load verified pinned
  weights, build the Rust/CUDA worker and configure optional batching, Graph,
  prefix reuse, request/token capacity and HTTP deadlines;
  [validation](decider/validation.md).
- [Cua-S1 native screenshot worker](cua_s1/native_screenshot.md): export the
  multimodal language checkpoint and serve one-image decisions with Rust/CUDA.
- [Open-Jev native text worker](open_jev/native.md): export the merged text
  backbone and trained decision head of Open-Jev-27B-v1.1 or Open-Jev-9B, then
  serve with Rust and CUDA. [Open-Jev-9B validation](open_jev/validation-9b.md)
  compares the 9B worker with the reference.
- [JEMM native worker](jemm/README.md): pinned text/image decisions and export;
  known multi-image numerical limitation is documented in the model contract.
- [JEV-27B-VL experimental worker](jev_vl/README.md): native text and preencoded
  image decisions; bounded H800 validation with explicit deployment limits.
- [OmniJev-4B v1.1 preparation](omnijev/README.md): export the checkpoint and
  regenerate the CPU fixtures; the native worker is not available yet.
- [CLM behind the frontend](clm/README.md): run CLM's own server behind the frontend on
  CPU with a stub encoder, and what the response comparison has to allow for.
- [pplx-decider reference worker](pplx_decider/README.md): pin
  perplexity-ai/pplx-decider-v1.1-27b, serve `choice`/`noul`/`score` through
  the checkpoint runtime, and measure Public231 concurrency 1 only on the
  recorded sm_89 device. Head `1ba07d1` has a reported L20X primary; the
  unpadded-mask head's delta is not measured in this tree.
- [D3 reference worker](d3/reference.md): download the pinned vllm-sr/d3 package,
  verify its runtime manifest, and serve `choice`/`noul`/`score` decisions over
  text, image and video state through the Python worker and the Rust frontend
  (text serving and parity validated on A100; media budgets and caps in the recipe).

- [LFM2.5-350M choice worker](lfm2/README.md): score text decisions with independent
  questions, hybrid cache branching and configurable candidate batches.

Recipes contain setup, launch commands and examples. Reusable implementation code
belongs under `src/`.

[`compare_with_backend.py`](compare_with_backend.py) checks that the frontend returns what
the worker returned, for any recipe; [`test_compare_with_backend.py`](test_compare_with_backend.py)
covers it without a model.
