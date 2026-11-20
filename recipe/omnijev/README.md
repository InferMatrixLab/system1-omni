# OmniJev-4B v1.1 preparation

This recipe prepares OmniJev-4B v1.1 ([#114](https://github.com/ThinkFlowLab/system1-omni/issues/114))
for [the native worker](native.md): it exports the checkpoint and regenerates the CPU
fixtures that the contract, preparation and pixel tests use. [The model
README](../../src/models/omnijev/README.md) describes what is implemented. Run the
commands from the repository root.

## Reference environment

The export and the fixtures use the reference's own code and Python packages on the
CPU; no GPU is needed.

```bash
python3.10 -m venv .venv-omnijev
.venv-omnijev/bin/pip install torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 peft==0.19.1 accelerate pillow
git clone https://github.com/tinnel123666888/OmniJev
git -C OmniJev checkout 14dbec4f71e194852c8d7b88ab36ef639493f400
```

Leave `fla-core` uninstalled, so transformers runs its PyTorch Gated DeltaNet code.
Keep transformers at 5.7.0 or later: older releases do not continue the
linear-attention state across the reference's prefix. The fixtures were generated
with Python 3.10.12; Python 3.12 changed how `sum` adds floats, which the reference's
calibration uses.

## Download the pinned files

```bash
hf download tinnel123/OmniJev --revision ffe5f436eaf22e20e2f041f8e74e121fd057a6cb --local-dir omnijev-v1.1
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir qwen3.5-4b
```

## Export

```bash
.venv-omnijev/bin/python recipe/omnijev/export.py --base qwen3.5-4b --checkpoint omnijev-v1.1 --out omnijev-export
```

The export checks the checkpoint files it reads against the release's
`release_manifest.json` and the base model's shards, index and `config.json` against
their pinned SHA-256, then writes:

- the language model in BF16 with the adapter merged, its tied embedding resized to
  the tokenizer's 248,079 entries and the two option tokens' saved rows restored, and
  `config.json` with the vision configuration;
- `vision.safetensors`, the base vision tower, which the adapter does not change;
- `heads.safetensors`, the decision and ordinal heads in FP32;
- the tokenizer and chat template;
- `omnijev_export.json`, written last, with the revisions, calibration, image budget,
  and the SHA-256 of the checked inputs and of every output file.

It writes into `.<out>.partial` next to the output and renames it at the end, so a
failed export leaves no output directory; the next run removes the partial one. It
needs about 18 GB of host RAM and writes 9 GB; on a 48-core workstation it takes
under a minute.

## Fixtures and tests

### Separate unmerged reference artifact

For the eager one-shot Score **library API**, prepare a separate directory:

```bash
.venv-omnijev/bin/python recipe/omnijev/export.py --base qwen3.5-4b \
    --checkpoint omnijev-v1.1 --out omnijev-unmerged-reference \
    --language unmerged-reference
```

This writes `omnijev-unmerged-reference/1`: the original BF16 base language
weights, resized tied embedding and saved special rows, plus all 400 separate
FP32 rank-32 adapter tensors at alpha 64 / scale 2. Vision, heads, tokenizer and
calibration retain the same identity. It does not merge or overwrite an existing
artifact. Both loaders reject the other format; the merged worker does not serve
this artifact.

Use `Executor::load_one_shot_reference` and `score_one_shot(&prepared, index)`
after preparing the **complete original request**. Keep every question in the
processor input, since selecting a question before compilation changes its
longest common prefix. The returned selected hidden rows, ordinal probabilities
and calibrated answer support boundary validation. This API supports one Score
question at a time; whole-request execution, Choice/Noul, LM targets, prefix
reuse and Graphs are rejected. Set `CUA_S1_GRAPH=0`. There is no worker switch
for this incomplete all-type profile.

The root real-weight checks are opt-in and must run serially on a reserved GPU:

```bash
CUA_S1_GRAPH=0 CUA_S1_CUDA_LIB=/absolute/libqwen3_5_cuda.so \
QWEN3_5_CHECKPOINT=/absolute/omnijev-unmerged-reference \
QWEN3_5_ADAPTER=/absolute/omnijev-unmerged-reference/adapter_model.safetensors \
cargo test --release --locked -p omni-qwen3-5-native --test readout \
    reference_selected_rows_preserve_full_row_and_recover -- --ignored --exact --test-threads=1
```

`tests/omnijev/worker.rs::unmerged_reference_original_q2_gate` consumes the original
12-question PNG request and an official full-request oracle supplied through
`OMNIJEV_ORIGINAL_REQUEST` and `OMNIJEV_REFERENCE_ORACLE` (the observer JSON's
`official_full_answer.q2`). Also set `OMNIJEV_REFERENCE_EXPORT` and
`OMNIJEV_CUDA_LIB` to absolute paths. The test checks prefix 581, input usage
1214, mode rejection, repeated hidden/head outputs, the Score decision, and every
returned probability/confidence at the original 0.01 gate. Optional
`OMNIJEV_REFERENCE_RESULT` saves its selected hidden/head output outside the repo.
Passing this bounded check does not establish all-type/full-280, JPEG or HTTP
parity; those require separate complete execution evidence.

On one L20, the original full-context PNG `video.png/q2` check passes: selected
`zq` and three `u` rows exactly match the pinned official unmerged plain-row
control; native confidence is 0.5640 against the normative original branch's
0.5613, within the unchanged 0.01 gate. The branch oracle remains authoritative;
the plain-row control only verifies this library path. This coverage is limited
to that observed Score case and the selected-row state/recovery tests.

### Ordinary CPU fixtures

```bash
.venv-omnijev/bin/python tests/omnijev/generate_fixtures.py \
    --reference OmniJev --checkpoint omnijev-v1.1 --base qwen3.5-4b --out tests/omnijev/data
cargo test --locked -p omni-omnijev-native
OMNIJEV_CHECKPOINT=$PWD/omnijev-v1.1 OMNIJEV_EXPORT=$PWD/omnijev-export \
    cargo test --locked -p omni-omnijev-native --test processing --test heads -- --include-ignored
```

The generator runs the reference's image loading and processor, preparation, layout,
rotary positions, heads and finishing without loading the backbone. The ordinary tests need only the committed
fixtures; the opt-in ones compare token ids with the checkpoint's tokenizer and the
heads with the export. Cargo runs the tests from the crate's directory, so the two
paths must be absolute.
