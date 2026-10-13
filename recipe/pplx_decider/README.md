# pplx-decider-v1.1-27b preparation

This recipe prepares [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/tree/14c9d90137962d300f3de95633f7b6535e80c5ce)
for a native worker: it regenerates the CPU fixtures that the contract and tokenization tests use, and records the
hosted model's answers for a later parity check. The worker itself is not available yet;
[the model README](../../src/models/pplx_decider/README.md) describes what is implemented. Run the commands from the
repository root.

## Reference environment

The fixtures run the checkpoint's own `source/src/autojev` code on the CPU, without the weights; no GPU is needed.
They were generated with Python 3.12, whose `sum` adds floats with Neumaier compensation, as the reference's
`answer` relies on.

```bash
python3.12 -m venv .venv-pplx-decider
.venv-pplx-decider/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 pillow==12.3.0 fastapi==0.141.1 huggingface-hub==1.31.0
```

## Download the pinned files

Only the code, the tokenizer and the configuration; the 54 GB of weights are not needed here.

```bash
hf download perplexity-ai/pplx-decider-v1.1-27b --revision 14c9d90137962d300f3de95633f7b6535e80c5ce \
  --include '*.json' '*.jinja' 'source/**' --local-dir pplx-decider
```

`tests/pplx_decider/make_fixtures.py` checks `model.py`, `server.py`, `tokenizer.json`, `chat_template.jinja` and
`decision_config.json` against their SHA-256 before it runs.

## Regenerate the fixtures

```bash
.venv-pplx-decider/bin/python tests/pplx_decider/make_fixtures.py pplx-decider tests/pplx_decider/data
```

It writes `tests/pplx_decider/data/contract.json`: 15 questions in 9 requests, one of them the recorded
System1-Agents Google Flights step in `data/flights_step.json`, and 11 requests the checkpoint's server rejects.

## Check

```bash
cargo test --locked -p omni-pplx-decider-native
PPLX_DECIDER_MODEL=pplx-decider cargo test --locked -p omni-pplx-decider-native --test tokenization -- --ignored
```

## Hosted answers (optional)

```bash
OPENROUTER_API_KEY=... python tests/pplx_decider/record_hosted.py tests/pplx_decider/data
```

This records `data/hosted.json`, what the hosted model answers to the same requests, for the executor's parity
check later. It costs a few thousandths of a cent.

## Limits

No worker, no inference and no images yet. A worker needs noncausal attention in the full-attention layers of the
shared Qwen3.5/3.8 prefill, the 255-row readout, and about 49 GiB of GPU memory for the BF16 weights.
