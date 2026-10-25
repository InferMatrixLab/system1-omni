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
labeled L20X are the H200 target. The kept primary is head
`4b99b730289efc8637f0940cf0079ee3ca4fe61f`: 231/231, p50 88.942 ms,
p95 295.797 ms. Versus the prior primary `ffae182` 92.844 / 296.458
that is p50 −4.20% (−3.902 ms) and p95 −0.22% (−0.661 ms). Warmup,
concurrency, and the latency metric were not restated in that paste.
`/health` reported `compile=default`, `compile_median=eager`,
`sdpa=default`, `mlp=default`, `fused_gate_up_layers=0`,
`linear_attention=flashinfer+fla+causal_conv1d`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`, and
`cu_seqlens_slots=48`. Quality: `choice_acc=0.9065` (139),
`noul_acc=0.8784` (74), `score_mae=0.2035` (18). `17a7b120` kept that
ring and also rotated 48 FlashInfer state workspaces. It missed the
gate: 231/231, p50 89.165 ms, p95 295.723 ms, +0.25% (+0.223 ms) /
−0.02% (−0.074 ms). Quality on that paste: `choice_acc=0.906` (139),
`noul_acc=0.878` (74), `score_mae=0.203` (18). Primary stays
`4b99b73`. `3afb472` fused every residual RMSNorm and missed the
gate: 231/231, p50 93.935 ms, p95 274.902 ms, +5.61% (+4.993 ms) /
−7.06% (−20.895 ms). Quality: `choice_acc=0.9065` (139),
`noul_acc=0.8784` (74), `score_mae=0.2052` (18). Primary stays
`4b99b73`. `705ae089` kept the FLA kernel only above 512 tokens and
missed the same way: 231/231, p50 94.604 ms, p95 274.835 ms, +6.37%
(+5.662 ms) / −7.09% (−20.962 ms). Quality: `choice_acc=0.906` (139),
`noul_acc=0.878` (74), `score_mae=0.205` (18). `/health` reported
`rms_norm=fla-long`, `fused_rms_norm_layers=129`,
`fused_rms_norm_min_tokens=512`, `cu_seqlens_slots=48`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`,
`compile_median=eager`, and no `gdn_state_slots`. Primary stays
`4b99b73`. `a6f4d9a` passed FlashInfer `use_cp=auto` and missed both
percentiles: 231/231, p50 112.294 ms, p95 299.419 ms, +26.26%
(+23.352 ms) / +1.22% (+3.622 ms). Quality: `choice_acc=0.906` (139),
`noul_acc=0.878` (74), `score_mae=0.204` (18). `/health` reported
`gdn_use_cp=auto`, `gdn_cp_heuristic=true`, `rms_norm=eager`,
`fused_rms_norm_layers=0`, `cu_seqlens_slots=48`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`,
`compile_median=eager`, and no `gdn_state_slots`. The heuristic was
true, so the CP kernel ran. Primary stays `4b99b73`. `3ecdf5a`
skipped the q/k repeat and crashed in warmup: 0/231,
`stop_reason=warmup_failed`, p50 and p95 none, quality N/A. The
gate is HOLD_PRIMARY. `/health` had already installed the lever:
`gdn_layout=gva`, `gva_layers=48`, `gdn_use_cp` false,
`gdn_cp_heuristic` null, `rms_norm=eager`,
`fused_rms_norm_layers=0`, `cu_seqlens_slots=48`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`,
`compile=default`, `compile_median=eager`. Warmup `easy-intent-00`
returned HTTP 500. FLA's L2 norm views q before the wrapper packs
it, and the unrepeated reshape is not contiguous. Primary stays
`4b99b73`. The unverified lever is the same GVA path with q and k
packed before that view (`gdn_qk_contiguous` true).

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
`flashinfer`, `flashinfer_status`, `linear_attention`, `cu_seqlens_slots`,
`rms_norm`, `fused_rms_norm_layers`, `fused_rms_norm_min_tokens`,
`gdn_use_cp`, `gdn_cp_heuristic`, `gdn_layout`, `gva_layers`, `gdn_qk_contiguous`, `sdpa`, `mlp`, `fused_gate_up_layers`, `compile`,
`compute_capability`, and `gpu_name`.
A null operator field is an explicit partial, not an omitted key. The
pinned `config.json` lists 16 `full_attention` layers and 64 text layers,
so a normal load reports 16 unpadded layers. On CUDA, `causal_conv1d` is
`1.7.0`, `fla` is `0.5.2`, `sdpa` is `default`, `mlp` is `default`,
`fused_gate_up_layers` is 0, `compile` is `default`,
`compile_min_tokens` is 0, and `compile_median` is `eager`. CPU reports
`sdpa=default`, `mlp=default`, `compile=default`,
`compile_min_tokens=0`, `compile_median=eager`, `rms_norm=eager`,
`fused_rms_norm_layers=0`, `fused_rms_norm_min_tokens=0`,
`gdn_use_cp=off`, `gdn_cp_heuristic` null, `gdn_layout=repeat`,
`gva_layers=0`, and `gdn_qk_contiguous` false. CUDA, including
capability 8.9, leaves every residual RMSNorm eager and reports
`rms_norm=eager`, `fused_rms_norm_layers=0`, and
`fused_rms_norm_min_tokens=0`. Always-on `rms_norm=fla` is the rejected
`3afb472` tip. `rms_norm=fla-long` is the rejected `705ae089` tip.
Capability 8.9 reports
`linear_attention` `fla-chunk`, `flashinfer` null,
`flashinfer_status=not-selected`, `cu_seqlens_slots=0`,
`gdn_use_cp=off`, `gdn_layout=repeat`, `gva_layers=0`,
`gdn_qk_contiguous` false, and it is ready.
Capability 9.0 with `flashinfer-python` 0.6.16.post3 reports
`flashinfer=0.6.16.post3`, `flashinfer_status=bound`,
`linear_attention=flashinfer+fla+causal_conv1d`,
`cu_seqlens_slots=48`, `gdn_use_cp` false, `gdn_cp_heuristic` null,
`gdn_layout=gva`, `gva_layers=48`, and `gdn_qk_contiguous` true.
The JSON boolean false is not
the string `auto` and not `off`. The wrapper keeps 48 distinct
`cu_seqlens` tensors per device and length. FlashInfer allocates its
own state workspace. `PPLX_GDN_PREFILL=fla` reproduces the prior
`f968e576` chunk and reports `flashinfer_status=held`,
`cu_seqlens_slots=0`, `gdn_use_cp=off`, `gdn_layout=repeat`,
`gva_layers=0`, and `gdn_qk_contiguous` false.
`PPLX_GDN_PREFILL=short` is the rejected length
dispatch: one sequence of at most 256 tokens uses FlashInfer, and
health then reports `linear_attention=flashinfer-short+fla-chunk`
with status `bound`, `cu_seqlens_slots=48`, `gdn_layout=repeat`,
`gva_layers=0`, and `gdn_qk_contiguous` false. Long sequences on that dispatch stay on FLA, which
still needs the q/k repeat, so GVA is not installed. The same
capability without that package, or when the module does not import,
stays on `fla-chunk` with status `absent` or `import-failed`,
`cu_seqlens_slots=0`, `gdn_layout=repeat`, `gva_layers=0`,
`gdn_qk_contiguous` false, and is
still ready. The kept primary rotates the `cu_seqlens` ring, passes
`use_cp` false, repeats q and k, and leaves every residual RMSNorm
eager. The code in this tree is the next candidate: the same ring,
the same eager norms, and `use_cp` false, with the q/k repeat removed
on the 48 linear layers and those heads packed before FLA's L2-norm
view. `3ecdf5a` crashed because that view ran on the unpacked reshape.
FlashInfer 0.6.16's prefill accepts the unrepeated shape as grouped
value attention. It still uses the checkpoint SDPA
order and the checkpoint's separate gate and up GEMMs. Query and key
norms stay eager.
The checkpoint leaves cuDNN SDPA off. `PPLX_SDPA=cudnn-first` and
`PPLX_MLP=fused-gate-up` are refused. Compile stays off.
`PPLX_COMPILE=dynamic` and `PPLX_COMPILE=dynamic-tail` are rejected
opt-ins. Inductor is not bit-exact with eager. A device name of
`NVIDIA L20X` is accepted at either capability. Startup raises if
causal_conv1d or FLA is missing, the conv cubin has no image for the
visible GPU, the loaded FlashInfer version is not 0.6.16.post3,
`PPLX_GDN_PREFILL` is an unknown value, `PPLX_SDPA` or `PPLX_MLP` is set
to anything but `default`, `PPLX_COMPILE` is an unknown value, a rejected compile opt-in records no
frames, or Transformers still bound `F.conv1d` or the PyTorch chunk
reference. That last case is what logs `falling back to its reference
PyTorch implementation` on the first forward. It does not report a
filesystem path. CuteDSL and CUDA graphs are not used. The default path
does not compile, so ready is not waiting on Inductor.

`install_causal_conv1d.py` builds [causal-conv1d 1.7.0](https://pypi.org/project/causal-conv1d/1.7.0/)
against the installed torch. That release emits sm_90 when nvcc is 11.8
or newer and skips sm_89. The script inserts a missing
`arch=compute_90,code=sm_90` line and an `arch=compute_89,code=sm_89`
line, and refuses to build if either cubin is still absent. It needs
nvcc. This tree did not compile it.

The kept primary installed FlashInfer, rotates 48 `cu_seqlens` buffers,
left cuDNN SDPA off, left the gate and up GEMMs separate, left the
residual RMSNorms eager, and left the forward eager. Always-on FLA
residual RMSNorm, the 512-token gate, and `use_cp=auto` missed the
gate. `3ecdf5a` passed unrepeated q and k and crashed in warmup
because FLA's L2 norm views them before they are packed. The next
candidate changes that order: q and k are contiguous before the
view. `use_cp` stays false. Leave `PPLX_GDN_PREFILL`, `PPLX_SDPA`,
`PPLX_MLP`, and `PPLX_COMPILE` unset. Startup refuses
`PPLX_SDPA=cudnn-first` and `PPLX_MLP=fused-gate-up`. CUDA then
reports `sdpa=default`, `mlp=default`, `fused_gate_up_layers=0`,
`compile=default`, `compile_median=eager`,
`linear_attention=flashinfer+fla+causal_conv1d`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`,
`cu_seqlens_slots=48`, `rms_norm=eager`,
`fused_rms_norm_layers=0`, `fused_rms_norm_min_tokens=0`,
`gdn_use_cp` false, `gdn_cp_heuristic` null, `gdn_layout=gva`,
`gva_layers=48`, and `gdn_qk_contiguous` true. There is no
`gdn_state_slots` field. `gdn_qk_contiguous` false means the crashed
tip's unpack is what would run. `rms_norm=fla`
and `rms_norm=fla-long` are the rejected tips. `gdn_use_cp=auto` is
the rejected `a6f4d9a` tip.
`PPLX_GDN_PREFILL=fla` reproduces the prior `f968e576` chunk.
`PPLX_COMPILE=dynamic` and `PPLX_COMPILE=dynamic-tail` are the rejected
compile tips. `PPLX_GDN_PREFILL=short` is the rejected ≤256 dispatch.
`install_flashinfer.py` is required for this candidate. A new C1 is
inconclusive until it reports p50 and p95 strictly under 88.942 /
295.797, plus `choice_acc`, `noul_acc`, and `score_mae`.

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
remeasured on a lab GPU labeled L20X and is the prior primary. That
remeasure reported `compute_capability=9.0`. Head
`ffae182cc80eaeb023f5e3501fb2dd36b1c65004` is the prior primary. Head
`4b99b730289efc8637f0940cf0079ee3ca4fe61f` is the kept primary. A load
failure is a startup failure, not a latency. H800 board numbers use a
different clock and stay on that board. The next candidate is compared
with `4b99b73` p50 88.942 ms and p95 295.797 ms, not with a new invented
target.

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
| `f968e576` remeasure | 231/231 | 94.077 ms | 300.200 ms |
| `ffae182` reused `cu_seqlens` | 231/231 | 92.844 ms | 296.458 ms |
| `4b99b73` 48-slot `cu_seqlens`, primary | 231/231 | 88.942 ms | 295.797 ms |
| `17a7b120` 48-slot state workspace, rejected | 231/231 | 89.165 ms | 295.723 ms |
| `3afb472` FLA residual RMSNorm ×129, rejected | 231/231 | 93.935 ms | 274.902 ms |
| `705ae089` FLA residual RMSNorm above 512, rejected | 231/231 | 94.604 ms | 274.835 ms |
| `a6f4d9a` `use_cp=auto`, rejected | 231/231 | 112.294 ms | 299.419 ms |
| `3ecdf5a` GVA, warmup failed | 0/231 | none | none |

The primary row is `4b99b73`. Versus the prior primary `ffae182`
92.844 / 296.458, p50 is −4.20% (−3.902 ms) and p95 is −0.22%
(−0.661 ms). `/health` reported `compile=default`,
`compile_median=eager`, `sdpa=default`, `mlp=default`,
`fused_gate_up_layers=0`, `linear_attention=flashinfer+fla+causal_conv1d`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`, and
`cu_seqlens_slots=48`. Quality: `choice_acc=0.9065` (139),
`noul_acc=0.8784` (74), `score_mae=0.2035` (18). Warmup, concurrency,
the latency metric, and compute capability were not restated in that
paste. `ffae182` versus `f968e576` 94.077 / 300.200 was p50 −1.31% and
p95 −1.25%. The `f968e576` remeasure stays in the table: protocol
warmup 5, concurrency 1, stock Public231 with `thinking.enabled=false`,
metric `successful_latency`, `/health` `linear_attention=fla-chunk`,
`flashinfer=null`, `compute_capability=9.0`. Against `1ba07d15` run1 that
remeasure is p50 −4.700 ms (−4.76%) and p95 −41.002 ms (−12.02%). The
earlier 98.037 / 302.569 paste is not a primary. `17a7b120` missed
the gate against `4b99b73`: p50 89.165 ms (+0.25%, +0.223 ms), p95
295.723 ms (−0.02%, −0.074 ms). Quality on that paste:
`choice_acc=0.906` (139), `noul_acc=0.878` (74), `score_mae=0.203`
(18). `3afb472` missed the gate against `4b99b73`: p50 93.935 ms
(+5.61%, +4.993 ms), p95 274.902 ms (−7.06%, −20.895 ms). Quality:
`choice_acc=0.9065` (139), `noul_acc=0.8784` (74), `score_mae=0.2052`
(18). `/health` reported `rms_norm=fla`, `fused_rms_norm_layers=129`,
`cu_seqlens_slots=48`, `gdn_state_slots=None`, `flashinfer_status=bound`,
`flashinfer=0.6.16.post3`, `compile=default`, and `compile_median=eager`.
It is not the primary. `705ae089` missed the gate against `4b99b73`:
p50 94.604 ms (+6.37%, +5.662 ms), p95 274.835 ms (−7.09%, −20.962 ms).
Quality: `choice_acc=0.906` (139), `noul_acc=0.878` (74),
`score_mae=0.205` (18). `/health` reported `rms_norm=fla-long`,
`fused_rms_norm_layers=129`, `fused_rms_norm_min_tokens=512`,
`cu_seqlens_slots=48`, `flashinfer_status=bound`,
`flashinfer=0.6.16.post3`, `compile=default`, `compile_median=eager`,
and no `gdn_state_slots`. It is not the primary. `a6f4d9a` missed
the gate against `4b99b73`: p50 112.294 ms (+26.26%, +23.352 ms),
p95 299.419 ms (+1.22%, +3.622 ms). Quality: `choice_acc=0.906`
(139), `noul_acc=0.878` (74), `score_mae=0.204` (18). `/health`
reported `gdn_use_cp=auto`, `gdn_cp_heuristic=true`, `rms_norm=eager`,
`fused_rms_norm_layers=0`, `cu_seqlens_slots=48`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`,
`compile_median=eager`, and no `gdn_state_slots`. The heuristic was
true. It is not the primary. `3ecdf5a` installed GVA and crashed
before a measured sample: 0/231, `stop_reason=warmup_failed`, p50
and p95 none, quality N/A. Gate HOLD_PRIMARY against `4b99b73`.
`/health` reported `gdn_layout=gva`, `gva_layers=48`, `gdn_use_cp`
false, `gdn_cp_heuristic` null, `rms_norm=eager`,
`fused_rms_norm_layers=0`, `cu_seqlens_slots=48`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`,
`compile=default`, and `compile_median=eager`. Warmup
`easy-intent-00` returned HTTP 500 from FLA `l2norm_fwd` viewing a
non-contiguous q. It is not a latency result. `/health` for `17a7b120` reported `cu_seqlens_slots=48`, `gdn_state_slots=48`,
`flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`,
and `compile_median=eager`. It is not the primary. The next tip has to beat
88.942 / 295.797 on both p50 and p95. `3c72f02` beat an older p50 and
missed that older p95. `a50b9aa`, `0dcaa0ac`, and `8b4ee326` missed both
percentiles of the older primary. The three compile tips improved p95 and
raised p50. None of those is the primary.

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
7. The kept primary is `4b99b73` in the table above. The prior primary
   is `ffae182`. `3c72f02` bound
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
   `34ba5de` isolated that median path and failed the same gate: p50
   97.084 ms, p95 256.679 ms. The 136-token probe passed and added no
   compiled frames, so the remaining p50 gap is in-process compile.
   Do not run another compile split. `4b99b73` rotates 48 `cu_seqlens`
   tensors per device and length and passed the gate against `ffae182`:
   p50 88.942 ms, p95 295.797 ms. That is the kept primary.
   `17a7b120` rotated the FlashInfer state workspace and missed the
   gate: p50 89.165 ms, p95 295.723 ms. Do not retry a GDN or
   `cu_seqlens` ring sizing without a concrete reason both percentiles
   beat 88.942 / 295.797.
   `3afb472` fused every residual RMSNorm and missed the gate: p50
   93.935 ms, p95 274.902 ms. Do not run that always-on fusion again.
   `705ae089` kept the kernel only above 512 tokens and missed the
   gate: p50 94.604 ms, p95 274.835 ms. Do not ship another residual
   RMSNorm variant unless a new hypothesis explains a p50 under
   88.942 ms. `a6f4d9a` passed `use_cp=auto`. `/health` reported
   `gdn_use_cp=auto` and `gdn_cp_heuristic=true`, so the CP kernel
   ran on every layer. It missed both percentiles: p50 112.294 ms,
   p95 299.419 ms, +26.26% (+23.352 ms) / +1.22% (+3.622 ms).
   Quality: `choice_acc=0.906` (139), `noul_acc=0.878` (74),
   `score_mae=0.204` (18). Do not retry `use_cp=auto` or the CP
   heuristic without a new hypothesis that explains a p50 under
   88.942 ms. `3ecdf5a` installed that GVA path and crashed in
   warmup: 0/231, `stop_reason=warmup_failed`, no p50 or p95.
   Quality was N/A. `/health` showed the lever installed. FLA
   `l2norm_fwd` calls `x.view(-1, D)` before q was packed, and the
   unrepeated reshape is not contiguous. Do not rerun `3ecdf5a`.
   The next candidate keeps the same ring, eager norms, `use_cp`
   false, and unrepeated heads, and packs q and k before that view.
   Unset `PPLX_SDPA`, `PPLX_MLP`, `PPLX_COMPILE`, and
   `PPLX_GDN_PREFILL`. `/health` must show `sdpa=default`,
   `mlp=default`, `fused_gate_up_layers=0`, `compile=default`,
   `compile_median=eager`,
   `linear_attention=flashinfer+fla+causal_conv1d`,
   `flashinfer_status=bound`, `flashinfer=0.6.16.post3`,
   `cu_seqlens_slots=48`, `rms_norm=eager`,
   `fused_rms_norm_layers=0`, `fused_rms_norm_min_tokens=0`,
   `gdn_use_cp` false (JSON boolean, not the string `auto` or
   `off`), `gdn_cp_heuristic` null, `gdn_layout=gva`,
   `gva_layers=48`, and `gdn_qk_contiguous` true before the run
   counts. It must not report `gdn_state_slots`, `rms_norm=fla`,
   or `rms_norm=fla-long`. A true heuristic means the rejected CP
   lever is engaged. `gdn_layout=repeat`, `gva_layers=0`, or
   `gdn_qk_contiguous` false means this fix is not the running
   worker. Both p50 and p95 have to be strictly under 88.942 /
   295.797 before that candidate replaces the primary. Record
   `choice_acc`, `noul_acc`, and `score_mae`. Until that paste,
   the candidate is inconclusive.

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
| 2026-10-25 | Different lever, same 512 split. The plain split did not keep the median on the primary eager path. A sequence of 512 tokens or fewer calls the original forward with the Dynamo frame hook suspended and the CUDA math flags captured before `torch.compile` restored. After the long warmup, one 136-token forward must be detected as 136 on the eager route or startup fails. Longer sequences stay compiled. Tip `34ba5de861e0c278d5037090ee5519f78187bb16`. | Public231 C1, 231/231, p50 97.084 ms, p95 256.679 ms. `/health`: `compile=dynamic-tail`, `compile_min_tokens=512`, `compile_median=isolated`, `compile_last_length=136`, `compile_eager_forwards=1`, `compile_compiled_forwards=3`, `compile_median_frame_delta=0`, `sdpa=default`, `mlp=default`, `linear_attention=fla-chunk`, `flashinfer=0.6.16.post3`, `flashinfer_status=held`. The 136-token probe passed and the worker became ready. Quality: `choice_acc=0.9065` (139), `noul_acc=0.8784` (74), `score_mae=0.2003` (18). The eager and compiled counts are the startup probe, not the measured run. | p50 +3.007 ms (+3.20%). p95 −43.521 ms (−14.50%). Gate failed. Both must be strictly under 94.077 / 300.200. Primary stays `f968e576`. The median route was eager, and p50 was still slower. |
| 2026-10-25 | Different lever, not compile. All-sequence FlashInfer on `3c72f02` was p50 88.808 ms and p95 300.751 ms. The wrapper built a new CUDA `cu_seqlens` on every linear layer, and that allocation syncs. Reuse one tensor per device and length. The forward stays eager. SDPA and the gate and up GEMMs stay as in the checkpoint. Tip `ffae182cc80eaeb023f5e3501fb2dd36b1c65004`. | Public231 C1, 231/231, p50 92.844 ms, p95 296.458 ms. Prior primary `f968e576` was 94.077 / 300.200. `/health`: `compile=default`, `compile_median=eager`, `sdpa=default`, `mlp=default`, `fused_gate_up_layers=0`, `linear_attention=flashinfer+fla+causal_conv1d`, `flashinfer_status=bound`, `flashinfer=0.6.16.post3`. Quality: `choice_acc=0.9065`, `noul_acc=0.8784`, `score_mae=0.2035`, the same pair of accuracies and the same `score_mae` as `3c72f02`. Warmup, concurrency, the latency metric, and compute capability were not restated in the paste. | p50 −1.233 ms (−1.31%). p95 −3.742 ms (−1.25%). Gate passed. New primary. Versus `3c72f02` 88.808 / 300.751: p50 +4.036 ms, p95 −4.293 ms. |
| 2026-10-25 | Different lever, same all-sequence FlashInfer bind. The single reused tensor cleared the p95 miss and gave median back versus the uncached bind. Keep 48 distinct tensors per device and length, one per linear layer, and rotate. The first forward of a length allocates the ring. Later forwards of that length do not allocate. 48 is the text-stack linear-layer count (64 layers, 16 full attention), not a fitted cutoff. The forward stays eager. SDPA and the gate and up GEMMs stay as in the checkpoint. Tip `4b99b730289efc8637f0940cf0079ee3ca4fe61f`. | Public231 C1, 231/231, p50 88.942 ms, p95 295.797 ms. Prior primary `ffae182` was 92.844 / 296.458. `/health`: `compile=default`, `compile_median=eager`, `sdpa=default`, `mlp=default`, `fused_gate_up_layers=0`, `linear_attention=flashinfer+fla+causal_conv1d`, `flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `cu_seqlens_slots=48`. Quality: `choice_acc=0.9065` (139), `noul_acc=0.8784` (74), `score_mae=0.2035` (18). Warmup, concurrency, the latency metric, and compute capability were not restated in the paste. | p50 −3.902 ms (−4.20%). p95 −0.661 ms (−0.22%). Gate passed. New primary. |
| 2026-10-25 | Different lever, same all-sequence FlashInfer bind and the same `cu_seqlens` ring. FlashInfer 0.6.16 allocates a float32 recurrent-state workspace on every SM90 call even when the final state is discarded. Keep 48 uninitialized buffers per device and head shape, rotate them, and pass the tensor as `output_state`. The shape does not depend on sequence length. `torch.empty` is not a host-to-device copy, so the win can be smaller than the `cu_seqlens` ring. The forward stays eager. SDPA and the gate and up GEMMs stay as in the checkpoint. Leave `PPLX_GDN_PREFILL` unset. `/health` adds `gdn_state_slots=48`. Tip `17a7b120b8335254bc28b1ec77fd15fabc25a892`. | Public231 C1, 231/231, p50 89.165 ms, p95 295.723 ms. Primary `4b99b73` was 88.942 / 295.797. `/health`: `cu_seqlens_slots=48`, `gdn_state_slots=48`, `flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`, `compile_median=eager`. Quality: `choice_acc=0.906` (139), `noul_acc=0.878` (74), `score_mae=0.203` (18). SDPA, MLP, fused gate-up, and `linear_attention` were not restated in the paste. | p50 +0.223 ms (+0.25%). p95 −0.074 ms (−0.02%). Gate failed. Primary stays `4b99b73`. |
| 2026-10-25 | Different lever, not another GDN or `cu_seqlens` ring. The state workspace was not the host sync the `cu_seqlens` ring removed, and pooling it raised p50. The residual RMSNorm is still the eager chain `rms(x.float()) * (1 + weight.float())`, twice per text layer plus the final norm (129 calls at hidden size 5120). One FLA `layer_norm_fwd` replaces that chain on every length. The fp32 scale is `1 + weight`, so the zero-centered affine stays. FLA 0.5.2's kernel above 512 features does not put sequence length in the autotune key. Query and key norms are head_dim 256 and stay eager. The gated norm stays eager. The wider gate-up GEMM stays refused. Compile stays off. The `cu_seqlens` ring stays. FlashInfer allocates its own state workspace. `/health` adds `rms_norm=fla` and `fused_rms_norm_layers=129`. Tip `3afb472d5569f00b12dc47ac52d6a68e233e4d51`. | Public231 C1, 231/231, p50 93.935 ms, p95 274.902 ms. Primary `4b99b73` was 88.942 / 295.797. `/health`: `rms_norm=fla`, `fused_rms_norm_layers=129`, `cu_seqlens_slots=48`, `gdn_state_slots=None`, `flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`, `compile_median=eager`. Quality: `choice_acc=0.9065` (139), `noul_acc=0.8784` (74), `score_mae=0.2052` (18). | p50 +4.993 ms (+5.61%). p95 −20.895 ms (−7.06%). Gate failed. Primary stays `4b99b73`. p95 improved. p50 regressed. |
| 2026-10-25 | Median-safe subset of the measured RMSNorm result, not another always-on fusion and not compile. `3afb472` raised p50 by 4.993 ms and cut p95 by 20.895 ms. The median is about 136 tokens and the tail example is about 2676, so the tax is the FLA kernel on short sequences. Sequences of 512 tokens or fewer call the saved eager `Qwen3_5RMSNorm` forward. Longer sequences call FLA `layer_norm_fwd` with the same fp32 `1 + weight` scale. 512 is between those profiled lengths and is not a fitted cutoff. The short path does not compile and does not change CUDA flags. Query and key norms stay eager. The `cu_seqlens` ring stays. `/health` reports `rms_norm=fla-long`, `fused_rms_norm_layers=129`, and `fused_rms_norm_min_tokens=512`. Tip `705ae089a9546f12471fafe8b6728cbced819854`. | Public231 C1, 231/231, p50 94.604 ms, p95 274.835 ms. Primary `4b99b73` was 88.942 / 295.797. `/health`: `rms_norm=fla-long`, `fused_rms_norm_layers=129`, `fused_rms_norm_min_tokens=512`, `cu_seqlens_slots=48`, `flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`, `compile_median=eager`, no `gdn_state_slots`. Quality: `choice_acc=0.906` (139), `noul_acc=0.878` (74), `score_mae=0.205` (18). | p50 +5.662 ms (+6.37%). p95 −20.962 ms (−7.09%). Gate failed. Primary stays `4b99b73`. The 512 threshold did not protect the median. |
| 2026-10-25 | Different lever, not another residual RMSNorm and not compile. `705ae089` matched the always-on shape, so both FLA residual-norm variants stay off. The non-CP SM90 kernel's host-side parallelism is one sequence times 48 value heads. FlashInfer 0.6.16 selects its CP kernel on HBM when that product times 2 is below the SM count. An H200 has 132 SMs and 96 is below 132, so every Public231 layer is in the low-parallelism region. The kept primary passes `use_cp` false. This candidate passes `auto` and leaves the CP chunk length to that kernel. The `cu_seqlens` ring stays. Residual RMSNorm stays eager. FlashInfer still allocates its own state workspace. SDPA and the gate and up GEMMs stay as in the checkpoint. `/health` reports `gdn_use_cp=auto`, `gdn_cp_heuristic=true`, and `rms_norm=eager`. Tip `a6f4d9a0773f2ce2f34cf0118e3792b9fbe253d7`. | Public231 C1, 231/231, p50 112.294 ms, p95 299.419 ms. Primary `4b99b73` was 88.942 / 295.797. `/health`: `gdn_use_cp=auto`, `gdn_cp_heuristic=true`, `rms_norm=eager`, `fused_rms_norm_layers=0`, `cu_seqlens_slots=48`, `flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`, `compile_median=eager`, no `gdn_state_slots`. Quality: `choice_acc=0.906` (139), `noul_acc=0.878` (74), `score_mae=0.204` (18). Warmup, concurrency, the latency metric, and compute capability were not restated in the paste. | p50 +23.352 ms (+26.26%). p95 +3.622 ms (+1.22%). Gate failed. Both percentiles worse. Primary stays `4b99b73`. |
| 2026-10-25 | Different lever, not another residual RMSNorm and not `use_cp`. `a6f4d9a` engaged the CP kernel on every layer and raised both percentiles, so `use_cp` stays false. The checkpoint `repeat_interleave`s q and k from 16 key heads to 48 value heads before the chunk kernel, two copies on each of the 48 linear layers. FlashInfer 0.6.16's prefill documents grouped value attention as `num_v_heads > num_q_heads = num_k_heads`: q and k stay at 16 heads, v and the gates stay at 48, and the output head count is 48. L2 norm is per head along the head dimension, so the unrepeated heads match a repeat followed by the same norm. This candidate replaces only the linear-layer forward on the all-sequence bind. The `cu_seqlens` ring stays. Residual RMSNorm stays eager. FlashInfer still allocates its own state workspace. Compile stays off. Full-attention layers are unchanged. `PPLX_GDN_PREFILL=short` does not install this path, because its long sequences stay on FLA and FLA needs the repeat. If the grouped kernel is slower than the expanded path, both percentiles can rise. The paste decides. `/health` reports `gdn_layout=gva`, `gva_layers=48`, `gdn_use_cp` false, `gdn_cp_heuristic` null, and `rms_norm=eager`. Tip `3ecdf5adb68ebbe0ca0e9c93ad1dae573f8e1f12`. | Public231 C1, 0/231, `stop_reason=warmup_failed`. p50 and p95 none. Quality N/A. Gate HOLD_PRIMARY against `4b99b73` 88.942 / 295.797. `/health` had installed the lever: `gdn_layout=gva`, `gva_layers=48`, `gdn_use_cp` false, `gdn_cp_heuristic` null, `rms_norm=eager`, `fused_rms_norm_layers=0`, `cu_seqlens_slots=48`, `flashinfer_status=bound`, `flashinfer=0.6.16.post3`, `compile=default`, `compile_median=eager`. Warmup `easy-intent-00` returned HTTP 500. `l2norm_fwd` views q before it is packed. | No measured samples. Not a latency result. Primary stays `4b99b73`. |
| 2026-10-25 | Same GVA lever, with the warmup crash fixed. `3ecdf5a` skipped the q/k repeat and died before a sample because FLA 0.5.2 `l2norm_fwd` calls `x.view(-1, D)` on the unrepeated reshape. `repeat_interleave` used to copy those heads into a contiguous tensor. This candidate calls `contiguous` on q and k before that view. An already packed tensor is not copied. The `cu_seqlens` ring stays. `use_cp` stays false. Residual RMSNorm stays eager. FlashInfer still allocates its own state workspace. Compile stays off. `/health` adds `gdn_qk_contiguous` true next to `gdn_layout=gva` and `gva_layers=48`. False means the crashed unpack is still what would run. | Not run. Samples: 0. | inconclusive |

## Limits

Image parity is unverified. The kept Public231 primary is `4b99b73` in
the optimization log: p50 88.942 ms, p95 295.797 ms, all-sequence
FlashInfer with a 48-slot `cu_seqlens` ring. The prior primary
`ffae182` is p50 92.844 ms, p95 296.458 ms, one reused `cu_seqlens`.
`f968e576` is p50 94.077 ms, p95 300.200 ms, on `fla-chunk` at compute
capability 9.0. Uncached all-sequence FlashInfer `3c72f02` is not the
primary: p50 88.808 ms, p95 300.751 ms. The short-sequence FlashInfer bind
`a50b9aa` is not the primary: p50 95.446 ms, p95 300.871 ms. cuDNN-first
SDPA `0dcaa0ac` is not the primary: p50 138.501 ms, p95 332.859 ms.
Fused gate-up `8b4ee326` is not the primary: p50 102.437 ms, p95
304.970 ms. Always-on dynamic compile `e36b32ed` is not the primary:
p50 97.751 ms, p95 263.124 ms. The 512-token compile split `7e3b1be`
is not the primary: p50 100.037 ms, p95 257.084 ms. Isolated-median
compile `34ba5de` is not the primary: p50 97.084 ms, p95 256.679 ms.
Decision agreement against `1ba07d1` was not in the primary paste. The
FlashInfer state-workspace ring `17a7b120` is not the primary: p50
89.165 ms, p95 295.723 ms. Always-on FLA residual RMSNorm `3afb472`
is not the primary: p50 93.935 ms, p95 274.902 ms. The length-gated
FLA residual RMSNorm `705ae089` is not the primary: p50 94.604 ms,
p95 274.835 ms. `use_cp=auto` `a6f4d9a` is not the primary: p50
112.294 ms, p95 299.419 ms. Grouped-value `3ecdf5a` is not a
latency result: 0/231, `stop_reason=warmup_failed`, p50 and p95
none. The packed-q/k candidate has no sample.
The shared Qwen CUDA backend is not used. PR #140 is a separate Rust port of this same contract;
it does not serve, and its JSON parser rewrites integers outside i64/u64.
This worker keeps those integers because it uses Python's parser and the
checkpoint's `json.dumps`.
