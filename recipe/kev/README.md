# Kev-4B: CPU export and pointer-head validation

This is a preparation recipe for [#27](https://github.com/InferMatrixLab/system1-omni/issues/27),
not a runnable Kev server. The [model contract](../../src/models/kev/README.md)
records the pinned artifacts, numerical boundaries and missing worker pieces.
All commands below run from the repository root. No GPU is needed.

## Environment and downloads

Use Python 3.12 and the stable Rust toolchain used by this repository. The CPU
export reads about 9.5 GB of inputs and writes about 8.5 GB; reserve at least 25 GB
of free disk and 24 GB of host RAM. It processes one source shard at a time.

```sh
python3.12 -m venv .venv-kev
.venv-kev/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv-kev/bin/pip install -r recipe/kev/requirements.txt huggingface-hub==1.33.0

.venv-kev/bin/hf download Qwen/Qwen3.5-4B-Base \
  --revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --include 'config.json' 'tokenizer.json' 'tokenizer_config.json' \
  'model.safetensors.index.json' '*.safetensors' --local-dir weights/kev-base
.venv-kev/bin/hf download jaredpalmer/kev-4b \
  --revision 139fdd94f1b6a6ad80cc15e08fcb99cac885a101 \
  --include 'adapter_config.json' 'adapter_model.safetensors' 'head.pt' \
  --local-dir weights/kev-adapter

.venv-kev/bin/python recipe/kev/export.py \
  --base weights/kev-base --checkpoint weights/kev-adapter --out weights/kev-native
```

Every required input must match its pinned SHA-256. The output directory must not
already exist. `kev_export.json` lists merged targets, output hashes, FP32 head
dimensions and calibration, plus the base tokenizer's five delimiter IDs.
Do not replace these artifacts with another Kev size or a bare Qwen checkpoint.

## CPU checks

```sh
.venv-kev/bin/python -m unittest discover -s tests/kev -p 'test_*.py' -v
cargo test --locked -p omni-kev-native -- --list
cargo test --locked -p omni-kev-native
KEV_HEAD="$PWD/weights/kev-native/head.safetensors" \
  cargo test --release --locked -p omni-kev-native --test head \
  -- --ignored --nocapture
```

The final test is opt-in because it needs the released head, but it is still
CPU-only. It compares the Rust implementation against the committed small oracle
for 1/2/7/31/255 options. Default CI executes the ordinary head tests without
downloading weights. Full backbone inference is not part of these checks.

To regenerate the oracle using the official class rather than this implementation:

```sh
git clone https://github.com/jaredpalmer/kev.git /tmp/kev-reference
git -C /tmp/kev-reference checkout --detach 2074cc44256200bb37540103fa4572901faa10f3
.venv-kev/bin/python tests/kev/generate_head_fixture.py \
  --reference-model /tmp/kev-reference/kev/model.py \
  --head weights/kev-adapter/head.pt --out /tmp/kev-head-reference.json
diff -u tests/kev/data/head.json /tmp/kev-head-reference.json
```

The generator verifies the complete source and head hashes, extracts only the
upstream `PointerHead` class, and uses CPU PyTorch. It does not need Transformers,
PEFT or the backbone. The fixture's Torch version includes the wheel's local
suffix, so a different wheel can change provenance even when numbers agree.

## Next integration slice

There is no worker binary, frontend launch command, health endpoint or example
HTTP request yet. The next slice adds exact prompt/row preparation and typed
answers, verified export loading, shared Qwen execution and serial admission.
It must qualify the BF16 merge against the full reference, then test real warmup
and first inference directly and through the frontend. Record hardware, full
dependency pins and numerical gates before those runs. No latency or accuracy
claim follows from export success or isolated head agreement.
