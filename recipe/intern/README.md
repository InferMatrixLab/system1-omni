# Intern-Decision-4B native text candidate

The draft implements Choice, Noul and Score for text and ordered JSON state.
On one NVIDIA L20 (sm_89), all 23 frozen text decisions agreed with the fixed
checkpoint HF oracle. Two fields exceeded the approved probability and
confidence error of `0.01`. Keep this worker experimental until the full
numerical gate passes.

## Run the candidate locally

Run from the repository root on Linux with Rust, CUDA 12.8 and a compatible
NVIDIA GPU. The checkpoint requires several GiB of storage and the worker
loads both the full 4B language weights and a separate BF16 vocabulary matrix.
The commands below assume a downloaded, immutable
[`internlm/Intern-Decision-4B@38c9dafbbef52dd04bae943c0c6ac918419adfd5`](https://huggingface.co/internlm/Intern-Decision-4B/tree/38c9dafbbef52dd04bae943c0c6ac918419adfd5)
snapshot. The worker reads the original safetensors and tokenizer directly.

```sh
hf download internlm/Intern-Decision-4B \
  --revision 38c9dafbbef52dd04bae943c0c6ac918419adfd5 \
  --local-dir "$PWD/checkpoints/intern-decision-4b"
cargo build --release --locked -p omni-intern-native -p omni-jev
bash src/backends/cuda/qwen3_5/build.sh "$PWD/target/intern-cuda" 89

INTERN_CHECKPOINT="$PWD/checkpoints/intern-decision-4b" \
INTERN_CUDA_LIB="$PWD/target/intern-cuda/libqwen3_5_cuda.so" \
INTERN_LISTEN=127.0.0.1:8000 \
  ./target/release/omni-intern-native
```

Start the [Rust frontend](../../src/frontend/README.md) in a second shell in
the repository root.

```sh
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev

curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "state": {"message": "I was billed twice. Please refund the duplicate charge."},
    "questions": {
      "route": {"type": "choice", "criteria": {"billing": "Billing", "technical": "Technical support"}},
      "refund": {"type": "noul", "instructions": "Is a refund requested?"},
      "urgency": {"type": "score", "criteria": {"0": "Low", "2": "Medium", "10": "High"}}
    }
  }'
```

Readiness follows a completed warmup. The worker accepts 1–16 ordered
questions and 1–62 options for Choice and Score. Score array criteria assign
levels `0..N-1`. The processor rejects nonempty media and inputs with more
than 8192 final tokens, and the worker limits JSON bodies to 2 MiB. Malformed
JSON returns 400, invalid decision or media inputs return 422, and length
overflow returns 413. The response carries field-count usage and a native
`rust-cuda` backend identity.

## Validation

```sh
cargo test --locked -p omni-intern-native
# With the immutable oracle text export, the pinned checkpoint and the library
INTERN_FROZEN_INPUTS=/path/to/frozen-text.json \
INTERN_PREFILL_OUT=/path/to/native-prefill.json \
INTERN_CHECKPOINT="$PWD/checkpoints/intern-decision-4b" \
INTERN_CUDA_LIB="$PWD/target/intern-cuda/libqwen3_5_cuda.so" \
  cargo test --locked --release -p omni-intern-native --test prefill frozen_prefill -- --ignored
```

The opt-in test expects a JSON array of records with `id`, `request`,
`prompt`, `ids`, `readout` and `fields` from the fixed HF capture. Each
field lists `field`, ordered `values` and `symbol_ids`. It checks all six
frozen text inputs exactly and writes the selected hidden rows plus the
62 symbol logits per row. Compare only the legal symbols per field against
the captured raw logits, and compare each typed result against the fixed
HF response. The uncalibrated reference is the official CUDA finishing
replay of the saved logits.

| L20 comparison, 6 requests / 23 fields | Default fixed prefill control | Eager framework-style candidate | Limit |
| --- | ---: | ---: | ---: |
| Maximum absolute selected-logit difference | 0.1875 | 0.125 | 0.125 |
| Maximum absolute calibrated probability difference | 0.013008 | 0.011571 | 0.01 |
| Maximum absolute confidence difference | 0.013008 | 0.011193 | 0.01 |
| Maximum absolute Noul difference | 0.004976 | 0.004976 | 0.01 |
| Exact categorical decisions | 23/23 | 23/23 | 23/23 |

Continuous Score met the per-field 1%-of-legend-range gate. The candidate
missed both probability and confidence gates for `field_10` and `field_13`
in the 16-field request. Neither run found a calibrated top-two margin of
0.02 or less. Real-weight near ties, extreme levels, image/video execution,
other checkpoint sizes, other hardware and performance remain outside this
qualification. See the [model contract](../../src/models/intern/README.md)
for execution and profile details.
