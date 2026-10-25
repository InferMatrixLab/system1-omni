# StartLux-Decision v1.1 4B Instant

Serve text `choice`, `noul`, and `score` from the pinned Instant 4B
checkpoint with the native worker `omni-startlux-native`. AutoThink and
Instant 27B are not part of this recipe.

Weights are **CC BY-NC 4.0**
(`startlux-models/StartLux-Decision-v1.1-4B-Instant`, revision
`416088f77cfce09633906a3675c7cd2e5dabcf10`, about 9.4 GB of BF16 files).
Download them on the machine that will serve. This repository does not
contain them. Inference code adapted from StartLux-Decision
`a99977d9332decc9fb27a2488b4a83b533b6cd5f` is Apache-2.0; see
[NOTICE](../../src/models/startlux/NOTICE).

GPU parity and latency are unverified here. The intended device is one
L20X (Ada, sm_89). [latency.md](latency.md) is the closed-loop measurement
to run after this worker starts on that GPU, including the operator
checklist. Record the bench metadata field `reservation` for the device
hold. This recipe does not add scheduler flags.

Commands below run from the repository root unless a path says otherwise.

## Build

The CUDA library and the worker must be built together. ABI stays 9.
`cs1_attention_gated_bidirectional` is an extra symbol; a library built
before it loads for causal workers and is refused by this one.

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-startlux-native -p omni-jev
```

`build.sh` needs the CUDA toolkit (`nvcc`) and writes
`target/release/libqwen3_5_cuda.so`. The second command builds the worker
and the Rust frontend. There is no merge/export step.

## Weights

```sh
hf download startlux-models/StartLux-Decision-v1.1-4B-Instant \
  --revision 416088f77cfce09633906a3675c7cd2e5dabcf10 \
  --local-dir "$STARTLUX_MODEL"
```

`STARTLUX_MODEL` is the folder that contains `config.json`,
`decision_config.json`, `tokenizer.json`, and the safetensors index. The
worker checks the Instant 4B shapes and refuses an AutoThink or 27B folder.

## Serve

```sh
STARTLUX_MODEL=/path/to/StartLux-Decision-v1.1-4B-Instant \
STARTLUX_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
STARTLUX_HOST=127.0.0.1 \
STARTLUX_PORT=8000 \
  ./target/release/omni-startlux-native
```

The process loads the checkpoint, runs one warmup decision, then listens.
Startup prints the CC BY-NC line. `GET /health` reports `ready`, the model
id, `prompt_attention: bidirectional`, the license, and the weights pin.

| Variable | Default | Meaning |
| --- | --- | --- |
| `STARTLUX_MODEL` | required | Instant 4B folder |
| `STARTLUX_CUDA_LIB` | `libqwen3_5_cuda.so` beside the binary | Shared CUDA library |
| `STARTLUX_HOST` | `127.0.0.1` | Listen address |
| `STARTLUX_PORT` | `8000` | Listen port, so `OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000` matches |
| `STARTLUX_MAX_MODEL_LEN` | `65536` | Reject longer prompts. `1..=262144` |
| `STARTLUX_NONE_CHECK` | on | `0` turns off the extra "None of the above" option |
| `STARTLUX_ALLOW_NONE` | off | `1` allows `choice: null` when that extra option wins |

The body limit is 32 MiB. A larger body is 413. The content type must be
`application/json`. Bad JSON is 400. Contract errors are 422 with
`error` and `detail`. The worker does not truncate.

Optional frontend, from the same root, against the worker above:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

## Requests

```sh
curl -sS http://127.0.0.1:8000/health

curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state":"I was charged twice.","questions":{"route":{"type":"choice","instructions":"What should happen?","criteria":{"refund":"Refund the charge","keep":"Keep the charge"}}}}'

curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state":"I was charged twice.","questions":{"ask":{"type":"noul","instructions":"Is a refund requested?"}}}'

curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state":"I was charged twice.","questions":{"severity":{"type":"score","instructions":"How severe is it?","criteria":["low","high"]}}}'
```

With the defaults, the choice answer includes `none_probability` and the
choice stays one of the listed options. `think_mode` omitted or `never` is
the Instant path. `think_mode` `auto` or `always` is 422.

A string `state` is copied unchanged. Any other JSON state follows pinned
`startlux_decide/prompt.py` (`a99977d`):
`json.dumps(annotate_indices(state), ensure_ascii=False)`. Separators are
`, ` and `: `. Lists of eight or more items gain `_index`. Non-ASCII stays
UTF-8 (`café`, `東京`); Python's default `ensure_ascii=True` escapes are not
the pin. Floats follow Python `repr`, so `1e-5` is `1e-05` and `1e16` is
`1e+16`, not serde_json's `0.00001` and `1e16`. Control characters still use
`\u00XX`. `cargo test --locked -p omni-startlux-native --test contract` locks
both the exponent and the non-ASCII cases.

## Checks without a GPU

```sh
cargo test --locked -p omni-startlux-native --test contract
```

That test does not load weights or CUDA. It does not establish numerical
parity.

## Limits

Text only. No AutoThink thought, no 27B, no images, no prefix cache, no
CUDA graph, no packed prompts. Two GDN parameters are stored FP32 upstream
and rounded to BF16 at load. Letter logits are a CPU dot of the tied
embedding and are unverified against vLLM. A causal vLLM server without the
StartLux bidirectional plugin is a different model; do not use it as the
Instant baseline. The baseline command is in [latency.md](latency.md).
