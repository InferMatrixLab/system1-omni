# pplx-decider reference worker

Serves [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
at revision `14c9d90137962d300f3de95633f7b6535e80c5ce` through
[`src/frontend/pplx_decider.py`](../../src/frontend/pplx_decider.py) and the
existing Rust frontend. The contract is
[`src/models/pplx_decider/README.md`](../../src/models/pplx_decider/README.md).

Run the commands from the repository root. `release-manifest.json` records
the 11 weight shards plus `readout.safetensors` as 52,173,413,928 bytes
(about 48.6 GiB). The card asks for roughly 49 GiB of weights plus working
memory. Plan for that much free disk. A reported load on 1× lab L20X
(~140 GiB, sm_89) was BF16 resident ~50333 MiB with no OOM. Lab GPUs
labeled L20X are the H200 target. The kept primary is the remeasure of
head `f968e576`: 231/231, p50 94.077 ms, p95 300.200 ms, warmup 5,
concurrency 1, stock Public231 with `thinking.enabled=false`, clock
`successful_latency`. `/health` reported `linear_attention=fla-chunk`,
`flashinfer=null`, and `compute_capability=9.0`.

## CPU checks

Python 3.12 and Pillow. No weights.

```sh
python3 -m pip install pytest pillow
PYTHONPATH=src python3 -m pytest tests/pplx_decider -q
PYTHONPATH=src python3 -m compileall -q src/models/pplx_decider src/frontend/pplx_decider.py tests/pplx_decider
```

After a full download, this checks the port against the checkpoint's own
prompt and answer functions. It still does not load the backbone:

```sh
CHECKPOINT=weights/pplx-decider-v1.1-27b
PYTHONPATH=src python3 recipe/pplx_decider/check_contract.py "$CHECKPOINT"
```

## Download and start

Install the checkpoint's pinned `requirements.txt` into the environment
that will run the worker (torch 2.14.0, transformers 5.17.0, and the other
pinned lines). Do not point `--device cpu` at a latency run: the checkpoint
then loads the backbone in float32.

```sh
python3 -m pip install huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "perplexity-ai/pplx-decider-v1.1-27b",
    revision="14c9d90137962d300f3de95633f7b6535e80c5ce",
    local_dir="weights/pplx-decider-v1.1-27b",
)
PY
python3 -m pip install -r weights/pplx-decider-v1.1-27b/requirements.txt
PYTHONPATH=src python3 recipe/pplx_decider/install_causal_conv1d.py
PYTHONPATH=src python3 -m frontend.pplx_decider \
  --model weights/pplx-decider-v1.1-27b \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000
```

`/health` is ready only after the hash gate, the weight load, the
causal-conv check, the FLA chunk check, the unpadded full-attention
install, and, on CUDA, the dynamic-compile warmup. It reports `model`, `max_input_tokens` (8192), `modalities`,
`attention_mode`, `unpadded_full_attention_layers`, `causal_conv1d`, `fla`,
`flashinfer`, `flashinfer_status`, `linear_attention`, `sdpa`, `mlp`,
`fused_gate_up_layers`, `compile`, `compute_capability`, and `gpu_name`.
A null operator field is an explicit partial, not an omitted key. The
pinned `config.json` lists 16 `full_attention` layers and 64 text layers,
so a normal load reports 16 unpadded layers. On CUDA, `causal_conv1d` is
`1.7.0`, `fla` is `0.5.2`, `sdpa` is `default`, `mlp` is `default`,
`fused_gate_up_layers` is 0, `compile` is `dynamic-tail`,
`compile_min_tokens` is 512, and `compile_median` is `isolated`.
After warmup, `compile_last_length` is 136. CPU reports
`sdpa=default`, `mlp=default`, `compile=default`, and
`compile_min_tokens=0`. Capability 8.9 reports
`linear_attention` `fla-chunk`, `flashinfer` null, and
`flashinfer_status=not-selected`, and it is ready. Capability 9.0 with
`flashinfer-python` 0.6.16.post3 reports `flashinfer=0.6.16.post3` and, by
default, `flashinfer_status=held` and `linear_attention=fla-chunk`.
`PPLX_GDN_PREFILL=short` is the rejected length dispatch: one sequence of
at most 256 tokens uses FlashInfer, and health then reports
`linear_attention=flashinfer-short+fla-chunk` with status `bound`. The
same capability without that package, or when the module does not import,
stays on `fla-chunk` with status `absent` or `import-failed` and is still
ready. The kept primary is `fla-chunk` at capability 9.0 with the
checkpoint SDPA order and the checkpoint's separate gate and up GEMMs.
The checkpoint leaves cuDNN SDPA off. `PPLX_SDPA=cudnn-first` and
`PPLX_MLP=fused-gate-up` are refused. Always-on dynamic compile and the
plain 512-token split both cut p95 and raised p50. CUDA still compiles
the language-model forward only when the sequence is longer than 512
tokens. A shorter sequence calls the original eager function with the
Dynamo frame hook suspended and the CUDA math flags from before
`torch.compile` restored. Text-only warmup lengths 768, 1024, and 2048
run before `/health`, then one 136-token forward must take the eager
route.
`PPLX_COMPILE=default` leaves every length eager.
`PPLX_COMPILE=dynamic` is the rejected always-on compile. Inductor is
not bit-exact with eager. A device name of
`NVIDIA L20X` is accepted at either capability. Startup raises if
causal_conv1d or FLA is missing, the conv cubin has no image for the
visible GPU, the loaded FlashInfer version is not 0.6.16.post3,
`PPLX_GDN_PREFILL` is an unknown value, `PPLX_SDPA` or `PPLX_MLP` is set
to anything but `default`, `PPLX_COMPILE` is an unknown value, the dynamic
compile records no frames, or Transformers still bound `F.conv1d` or the
PyTorch chunk reference. That last case is what logs `falling back to its
reference PyTorch implementation` on the first forward. It does not report
a filesystem path. CuteDSL and CUDA graphs are not used. `/health` stays
down through the compile, so a long pause before ready is that compile.

`install_causal_conv1d.py` builds [causal-conv1d 1.7.0](https://pypi.org/project/causal-conv1d/1.7.0/)
against the installed torch. That release emits sm_90 when nvcc is 11.8
or newer and skips sm_89. The script inserts a missing
`arch=compute_90,code=sm_90` line and an `arch=compute_89,code=sm_89`
line, and refuses to build if either cubin is still absent. It needs
nvcc. This tree did not compile it.

The kept primary did not install FlashInfer, left cuDNN SDPA off, and
left the gate and up GEMMs separate. The next candidate does not change
GDN, SDPA order, or those GEMM shapes. Leave `PPLX_GDN_PREFILL`,
`PPLX_SDPA`, `PPLX_MLP`, and `PPLX_COMPILE` unset. An already installed
pin stays `held` on FLA chunk. Startup refuses `PPLX_SDPA=cudnn-first`
and `PPLX_MLP=fused-gate-up`. CUDA then reports `sdpa=default`,
`mlp=default`, `fused_gate_up_layers=0`, `compile=dynamic-tail`,
`compile_min_tokens=512`, and `compile_median=isolated`.
`PPLX_COMPILE=default` reproduces the eager
primary forward. `PPLX_COMPILE=dynamic` is the rejected always-on
compile from `e36b32ed`. `PPLX_GDN_PREFILL=short` is the rejected ≤256
FlashInfer dispatch, not this candidate. `install_flashinfer.py` remains
optional. Wait for `/health` before the benchmark; the long-sequence
compile runs first. A new C1 is inconclusive until it reports p50 and
p95 against 94.077 / 300.200, plus `choice_acc`, `noul_acc`, and
`score_mae`.

```sh
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/pplx_decider/example-request.json
```

The Rust frontend, from the same root, only forwards. Its inference timeout
is 60 seconds and is separate from the worker.

```sh
cargo build --release --locked -p omni-jev
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

`--device cuda` is the evaluated BF16 path (`noncausal_full_attention`, last
token, temperature `1.0087417621345625`). Image inputs use the checkpoint
default cap of 2048×2048 pixels, which is wider than the 512×512 training
cap. Pass `--max-image-pixels 262144` only when reproducing that training
cap, and record it. Text Public231 does not send images.

## Public231 concurrency 1

Head `1ba07d15b11a8c0796657e4a10b7ba3e62354421` was measured on 1× lab L20X
(sm_89, ~140 GiB). Head `f968e576aa733a0e6e4b2bc46bb11035e7662c8f` was
remeasured on a lab GPU labeled L20X and is the kept primary. That
remeasure reported `compute_capability=9.0`. A load failure is a startup
failure, not a latency. H800 board numbers use a different clock and stay
on that board. The next candidate is compared with `f968e576` p50 94.077 ms
and p95 300.200 ms, not with a new invented target.

Reported load: BF16 resident ~50333 MiB, no OOM. Warmup 5, concurrency 1,
clock `successful_latency` / `post_to_body_json_and_schema_validation`.
Heads `1ba07d1` and `b6fa9d0` return HTTP 422
`unknown request fields: ['thinking']` for any `thinking` field. Their
measured workload, where one exists, is the stripped no-thinking twin of
the same 231 rows. This head admits `thinking` only when the value is
`{"enabled": false}` and ignores it. Other values return HTTP 422
`thinking.enabled must be false`. The checkpoint template stays
`enable_thinking=False`.

| Round | Successful | p50 | p95 |
| --- | --- | --- | --- |
| `1ba07d15` run1, primary | 231/231 | 98.777 ms | 341.202 ms |
| `1ba07d15` run2 | 231/231 | 97.921 ms | 340.804 ms |
| `1ba07d15` run3 | 231/231 | 98.177 ms | 343.953 ms |
| `f968e576` earlier paste | 231/231 | 98.037 ms | 302.569 ms |
| `f968e576` remeasure, primary | 231/231 | 94.077 ms | 300.200 ms |

The primary row is the later lab remeasure of the same worker. Protocol:
warmup 5, concurrency 1, stock Public231 with `thinking.enabled=false`,
metric `successful_latency`. Against `1ba07d15` run1, p50 is −4.700 ms
(−4.76%) and p95 is −41.002 ms (−12.02%). `/health` reported
`causal_conv1d=1.7.0`, `fla=0.5.2`, `flashinfer=null`,
`linear_attention=fla-chunk`, and `compute_capability=9.0`. The gain is
the FLA chunk path on a capability-9.0 device. FlashInfer was not bound.
The earlier 98.037 / 302.569 paste is not the primary. Run2, run3, resident
memory, and a decision-disagreement count were not in the remeasure.
The next tip has to beat 94.077 / 300.200 on both p50 and p95 in a new C1.
`3c72f02` did not: p50 improved and p95 did not. `a50b9aa`, `0dcaa0ac`,
and `8b4ee326` did not either: both percentiles were worse. `f968e576`
stays the primary.

The two-model Public231 export hashes to
`00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3`
([latency leaderboard](../../benchmarks/latency-leaderboard.md)). That hash
is not the canonical task-record hash. The `1ba07d1` measurement used the
stripped twin: the same 231 rows with `thinking` removed. Its SHA-256 was
not in the latency report. A run of this head may point `PUBLIC231_JSONL`
at that twin or at stock Public231 whose `thinking` value is
`{"enabled": false}`. Record the digest `validate` prints. Stop if the row
count is not 231, or if any `thinking` value is not `{"enabled": false}`.

Clock, quantiles and the three-traversal budget are the existing runner in
[benchmarks/README.md](../../benchmarks/README.md): client latency includes
body read, JSON and schema checks; p50 is the median; p95 is nearest rank
`x[ceil(0.95 n) - 1]`; warmups stay out of the percentiles. The historical
H800 board uses a different clock and does not receive this run.

1. Record `nvidia-smi` name, compute capability, driver, CUDA version and
   memory before load. Keep the device index. Omit host names, accounts,
   home directories and scheduler names.
2. Copy `recipe/pplx_decider/c1-metadata.example.json` and replace every
   `FILL` value. A file that still contains `FILL` is not evidence.
   `precision` stays `bf16`. `runtime_revision` is `git rev-parse HEAD`.
3. Start the worker with `--device cuda` and wait until `/health` returns
   `"status": "ready"`. Record memory after load. If the process dies or
   CUDA reports out of memory, stop and publish that startup failure with
   the memory figures. Do not quantize, offload, or switch devices to
   obtain a number.
4. Install `benchmarks/requirements.txt`. From the repository root, with
   the worker still up and the manifest path in `PUBLIC231_JSONL`:

```sh
python3 benchmarks/bench.py validate "$PUBLIC231_JSONL"
python3 benchmarks/bench.py run "$PUBLIC231_JSONL" \
  --endpoint http://127.0.0.1:8000/v1/systemone \
  --model jev-latest \
  --metadata "$RUN_DIR/c1-metadata.json" \
  --phase feasibility \
  --concurrency 1 --warmup 5 --timeout 60 \
  --output "$RUN_DIR/feasibility"
```

5. Leave the same process running. Three measured traversals, each in a
   new output directory:

```sh
python3 benchmarks/bench.py run "$PUBLIC231_JSONL" \
  --endpoint http://127.0.0.1:8000/v1/systemone \
  --model jev-latest \
  --metadata "$RUN_DIR/c1-metadata.json" \
  --phase measured \
  --concurrency 1 --warmup 5 --timeout 60 \
  --output "$RUN_DIR/c1-round-1"
```

Repeat for `c1-round-2` and `c1-round-3`. Then
`python3 benchmarks/bench.py summary` on each directory. Pool latency only
across successful completed requests from the three measured rounds, 693
samples when every request succeeds. Report planned, attempted, successful,
failed and incomplete counts. Keep failure latencies out of p50 and p95.
Requests that hit the 60 second client timeout are failures. Do not retry
them and do not raise the timeout after seeing the times.

6. A second arm through `http://127.0.0.1:8080/v1/systemone` is frontend
   time only. Keep it in different output directories. The Rust frontend's
   own 60 second cap still applies. Do not average it into the worker p50.
7. The kept primary is the `f968e576` remeasure above. `3c72f02` bound
   FlashInfer on every sequence and failed the gate: p50 88.808 ms, p95
   300.751 ms. `a50b9aa` used FlashInfer only for one sequence of at most
   256 tokens and failed: p50 95.446 ms, p95 300.871 ms.
   `0dcaa0ac` preferred cuDNN SDPA and failed: p50 138.501 ms, p95
   332.859 ms. `8b4ee326` fused the text MLP gate and up GEMMs and
   failed: p50 102.437 ms, p95 304.970 ms. `e36b32ed` compiled every
   language-model forward and failed the both-percentiles gate: p50
   97.751 ms, p95 263.124 ms. p95 improved. `7e3b1be` compiled sequences
   longer than 512 tokens and failed the same gate: p50 100.037 ms, p95
   257.084 ms. p95 improved again and p50 was worse than always-on.
   Do not try another FlashInfer length cutoff, another SDPA order, the
   wider GEMM, or always-on `PPLX_COMPILE=dynamic`. The next candidate
   keeps the 512 split, FLA chunk, the checkpoint SDPA order, and the
   separate gate and up GEMMs. Sequences of 512 tokens or fewer call the
   original eager function with Dynamo suspended and the pre-compile
   CUDA flags restored. Startup runs a 136-token forward after the long
   warmup and refuses to become ready if that forward is not detected
   as 136 tokens on the eager route. Unset `PPLX_SDPA`, `PPLX_MLP`,
   `PPLX_COMPILE`, and `PPLX_GDN_PREFILL`. `/health` must show
   `sdpa=default`, `mlp=default`, `fused_gate_up_layers=0`,
   `compile=dynamic-tail`, `compile_min_tokens=512`,
   `compile_median=isolated`, `compile_last_length=136`,
   `linear_attention=fla-chunk`, and `flashinfer_status=held` before the
   run counts. Also record `compile_eager_forwards`,
   `compile_compiled_forwards`, and `compile_median_frame_delta`. A new
   Public231 C1 has to beat p50 94.077 ms and p95 300.200 ms before that
   candidate replaces the primary. Record `choice_acc`, `noul_acc`, and
   `score_mae`. Until that paste, the candidate is inconclusive.

`benchmarks/smoke.jsonl` checks the client, not this model and not Public231.

## Optimization log

| Date | Hypothesis | Measurement | p50 / p95 delta |
| --- | --- | --- | --- |
| 2026-10-24 | No kernel change. Asked whether the BF16 resident set fits a 48 GB sm_89 card. | No CUDA device in the implementation environment. Samples: 0. | none |
| 2026-10-25 | Baseline at `1ba07d15b11a8c0796657e4a10b7ba3e62354421`. The checkpoint supplies an all-ones full-attention mask, so SDPA stays on the masked math kernel. | 1× lab L20X, sm_89, ~140 GiB. BF16 resident ~50333 MiB, no OOM. Warmup 5, concurrency 1, clock `successful_latency` / `post_to_body_json_and_schema_validation`. Stripped no-thinking twin. run1 231/231, p50 98.777 ms, p95 341.202 ms (primary). run2 231/231, p50 97.921 ms, p95 340.804 ms. run3 231/231, p50 98.177 ms, p95 343.953 ms. | baseline |
| 2026-10-25 | Omit the explicit full-attention mask when every padding bit is true, and set full-attention `is_causal` false, so Transformers 5.17 can take the grouped-query SDPA kernel (`head_dim` 256). Padded batches keep the bool mask. Linear attention is unchanged. Flash and math are not bit-exact. | No CUDA device in this environment. Head `b6fa9d0`. Samples: 0. | not measured |
| 2026-10-25 | Admit stock Public231 `thinking` only when it is `{"enabled": false}`. No kernel change. | No CUDA device in this environment. Head `778609f`. Samples: 0. Baseline primary stays `1ba07d1` run1. | not measured |
| 2026-10-25 | In-process profile of the `1ba07d1` primary. About 136 tokens at p50: total ~89 ms, forward CUDA 87.7 ms (~98%), prepare 1.4 ms, parse/post/answer <0.3 ms. No decode. Short-sequence floor ~88–91 ms. A ~2676-token prompt forwarded in ~327 ms and drives p95. Sinks named by the profile: the 27B HF backbone forward; short-sequence launch and HF SDPA plus GDN with no CUDA graphs; missing `causal_conv1d` (Transformers fallback warning, generic depthwise conv in the trace). | Owner profile of the primary run. Not a new C1 sample. | n/a |
| 2026-10-25 | Install `causal-conv1d` 1.7.0 and add an sm_89 cubin the upstream gencode omits. CUDA startup refuses a missing package, a cubin without the visible GPU arch, or a Qwen conv still bound to `F.conv1d`. The package path is not bit-exact with that fallback. | No CUDA device in this environment. Samples: 0. `778609f` is still unverified. | not measured |
| 2026-10-25 | Classification C1 follows the sm_89 vLLM/SGLang mapping: FLA/Triton chunk plus `causal-conv1d`. The installer inserts the sm_89 cubin upstream omits and still refuses `F.conv1d`. `/health` reports `linear_attention=fla-chunk`. FlashInfer, CuteDSL, and CUDA graphs are not on this path. Unpadded full-attention masks and `thinking.enabled=false` are already in the branch. | No L20X in this environment. Samples: 0. Remeasure against primary 98.777 / 341.202 when the host is back. | not measured |
| 2026-10-25 | Lab L20X-labeled GPUs are the H200 target. Keep sm_90 and sm_89 cubins. Capability 9.0 binds FlashInfer `chunk_gated_delta_rule` (FLA L2 norm, float32 gate, `use_cp` false, one sequence) when `flashinfer-python` 0.6.16.post3 imports. Capability 8.9, and capability 9.0 without that package, stay ready on `fla-chunk` plus `causal_conv1d`. `/health` reports `flashinfer`, `linear_attention`, `compute_capability`, and `gpu_name`. Ready is not limited to capability 9.0. | No CUDA device in this environment. Samples: 0. No latency claim. | not measured |
| 2026-10-25 | Keep head `f968e576`. The whole head, including unpadded full attention, `thinking.enabled=false`, `causal-conv1d` 1.7.0, and FLA chunk, was verified on the lab L20X-labeled GPU. `/health` reported `linear_attention=fla-chunk`, `causal_conv1d=1.7.0`, `fla=0.5.2`, `flashinfer=null`. FlashInfer was not the bind under test. | Owner-verified Public231 C1, 231/231, p50 98.037 ms, p95 302.569 ms, versus `1ba07d15` run1 p50 98.777 ms, p95 341.202 ms. One pasted pair. Decision disagreements, extra rounds, capability, and resident memory were not in the paste. | p50 −0.74 ms (−0.75%), flat. p95 −38.6 ms (−11.32%). Superseded as primary by the remeasure below. |
| 2026-10-25 | Remeasure the same worker `f968e576` on the lab L20X-labeled GPU. No new kernel. Stock Public231 with `thinking.enabled=false`, warmup 5, concurrency 1, metric `successful_latency`. `/health`: `causal_conv1d=1.7.0`, `fla=0.5.2`, `flashinfer=null`, `linear_attention=fla-chunk`, `compute_capability=9.0`. | 231/231, p50 94.077 ms, p95 300.200 ms versus `1ba07d15` run1 98.777 / 341.202. | p50 −4.700 ms (−4.76%), p95 −41.002 ms (−12.02%). New primary. |
| 2026-10-25 | One lever: bind FlashInfer 0.6.16.post3 for every GDN prefill on capability 9.0. Tip `3c72f02682616c58fdbb38e7fcdf320af20c33df`. | Lab L20X-labeled host. Warmup 5, concurrency 1, stock Public231, `thinking.enabled=false`. 231/231, p50 88.808 ms, p95 300.751 ms. `/health`: `causal_conv1d=1.7.0`, `fla=0.5.2`, `flashinfer=0.6.16.post3`, `flashinfer_status=bound`, `linear_attention=flashinfer+fla+causal_conv1d`, `compute_capability=9.0`. Quality on this tip only: `choice_acc=0.9065`, `noul_acc=0.8784`, `score_mae=0.2035`. Primary quality was not in the paste. | p50 −5.269 ms (−5.60%). p95 +0.551 ms (+0.18%). Gate failed. Primary stays `f968e576`. |
| 2026-10-25 | Different lever: FlashInfer only when the forward is one sequence of at most 256 tokens. Longer sequences and multi-sequence forwards stay on FLA chunk. Tip `a50b9aaa9f7ee546c6cfba5c231fff6dd3ff9716`. 256 is between the profiled median length of about 136 tokens and the profiled tail example of about 2676 tokens. It is not a fitted cutoff. | Public231 C1, 231/231, p50 95.446 ms, p95 300.871 ms. `/health`: `linear_attention=flashinfer-short+fla-chunk`, `flashinfer=0.6.16.post3`, `flashinfer_status=bound`, `causal_conv1d=1.7.0`, `fla=0.5.2`, `compute_capability=9.0`. Quality: `choice_acc=0.9065`, `noul_acc=0.8784`. `score_mae` was not in the paste. Warmup, concurrency, and the latency metric were not restated in the paste. | p50 +1.369 ms (+1.46%). p95 +0.671 ms (+0.22%). Gate failed. Primary stays `f968e576`. |
| 2026-10-25 | Different lever, not another length cutoff. GDN stays on FLA chunk even when the FlashInfer pin is installed (`flashinfer_status=held`). CUDA forwards prefer cuDNN SDPA, then flash, mem-efficient, and math. Tip `0dcaa0ac12187a8881237739773c145d11906cbc`. | Public231 C1, 231/231, p50 138.501 ms, p95 332.859 ms. `/health`: `linear_attention=fla-chunk`, `flashinfer=0.6.16.post3`, `flashinfer_status=held`, `sdpa=cudnn-first`, `compute_capability=9.0`. Quality: `choice_acc=0.9065`, `noul_acc=0.8784`, `score_mae=0.2050`. Warmup, concurrency, and the latency metric were not restated in the paste. | p50 +44.424 ms (+47.22%). p95 +32.659 ms (+10.88%). Gate failed. Primary stays `f968e576`. Do not pursue cuDNN-first without a new hypothesis. |
| 2026-10-25 | Different lever, not SDPA order and not a short FlashInfer cutoff. GDN stays on FLA chunk. SDPA stays on the checkpoint order. CUDA fuses each text MLP `gate_proj` and `up_proj` into one GEMM. Tip `8b4ee32604be6945d797d4d066a2e0d74111fb60`. | Public231 C1, 231/231, p50 102.437 ms, p95 304.970 ms. `/health`: `linear_attention=fla-chunk`, `flashinfer=0.6.16.post3`, `flashinfer_status=held`, `sdpa=default`, `mlp=fused-gate-up`, `fused_gate_up_layers=64`, `compute_capability=9.0`. Quality: `choice_acc=0.9065`, `noul_acc=0.8784`, `score_mae=0.2027`. Warmup, concurrency, and the latency metric were not restated in the paste. | p50 +8.360 ms (+8.89%). p95 +4.770 ms (+1.59%). Gate failed. Primary stays `f968e576`. |
| 2026-10-25 | Different lever. The forward is still almost the whole request after the GDN, SDPA-order, and wider-GEMM trials, and the wider GEMM was slower, so GEMM shapes stay as in the checkpoint. Compile the language-model forward with a dynamic sequence length, and warm lengths 32, 256, and 1024 before `/health`. `PPLX_COMPILE=default` stays eager. Inductor is not bit-exact. CUDA graphs are not selected. Tip `e36b32ed9da69b7aec3aaac810614e9efcb47001`. | Public231 C1, 231/231, p50 97.751 ms, p95 263.124 ms. `/health`: `compile=dynamic`, `mlp=default`, `sdpa=default`, `fused_gate_up_layers=0`, `linear_attention=fla-chunk`, `flashinfer=0.6.16.post3`, `flashinfer_status=held`. Quality: `choice_acc≈0.906`, `noul_acc≈0.878`, `score_mae≈0.203`. Ready about 75s (compile+load). Warmup, concurrency, the latency metric, and compute capability were not restated in the paste. | p50 +3.674 ms (+3.91%). p95 −37.076 ms (−12.35%). Gate failed because both percentiles must improve. Primary stays `f968e576`. p95 improved. Do not abandon compile. |
| 2026-10-25 | Different lever, a compile variant. Always-on dynamic compile cut p95 and raised p50, so sequences of 512 tokens or fewer stay on the eager forward and longer sequences use the compiled forward. 512 sits between the profiled median of about 136 tokens and the profiled tail example of about 2676 tokens. It is not a fitted cutoff and not a FlashInfer policy. Warmup lengths 768, 1024, and 2048 run before `/health`. GDN stays on FLA chunk. SDPA and the gate and up GEMMs stay as in the checkpoint. Tip `7e3b1be19eb9bb41d118bae558a0c0eef6561e16`. | Public231 C1, 231/231, p50 100.037 ms, p95 257.084 ms. Prior always-on `e36b32ed` was p50 97.751 ms, p95 263.124 ms. `/health`: `compile=dynamic-tail`, `compile_min_tokens=512`, `mlp=default`, `sdpa=default`, `fused_gate_up_layers=0`, `linear_attention=fla-chunk`, `flashinfer_status=held`. Quality: `choice_acc=0.906`, `noul_acc=0.878`, `score_mae=0.200`. Warmup, concurrency, the latency metric, the FlashInfer version, and compute capability were not restated in the paste. | p50 +5.960 ms (+6.34%). p95 −43.116 ms (−14.36%). Gate failed. Primary stays `f968e576`. p95 improved further. p50 was worse than always-on. |
| 2026-10-25 | Different lever, same 512 split. The plain split did not keep the median on the primary eager path. A sequence of 512 tokens or fewer calls the original forward with the Dynamo frame hook suspended and the CUDA math flags captured before `torch.compile` restored. After the long warmup, one 136-token forward must be detected as 136 on the eager route or startup fails. Longer sequences stay compiled. GDN, SDPA, and the gate and up GEMMs stay as in the checkpoint. | Not run. Samples: 0. | inconclusive |

## Limits

Image parity is unverified. The kept Public231 primary is the `f968e576`
remeasure in the optimization log: p50 94.077 ms, p95 300.200 ms, on
`fla-chunk` at compute capability 9.0. The all-sequence FlashInfer bind
`3c72f02` is not the primary: p50 88.808 ms, p95 300.751 ms. The
short-sequence FlashInfer bind `a50b9aa` is not the primary: p50 95.446 ms,
p95 300.871 ms. cuDNN-first SDPA `0dcaa0ac` is not the primary: p50
138.501 ms, p95 332.859 ms. Fused gate-up `8b4ee326` is not the primary:
p50 102.437 ms, p95 304.970 ms. Always-on dynamic compile `e36b32ed`
is not the primary: p50 97.751 ms, p95 263.124 ms. The 512-token compile
split `7e3b1be` is not the primary: p50 100.037 ms, p95 257.084 ms.
Decision agreement against `1ba07d1` was not in the primary paste. The
isolated-median compile candidate has no sample.
The shared Qwen CUDA backend is not used. PR #140 is a separate Rust port of this same contract;
it does not serve, and its JSON parser rewrites integers outside i64/u64.
This worker keeps those integers because it uses Python's parser and the
checkpoint's `json.dumps`.
