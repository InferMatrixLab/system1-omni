# Kev-4B default-service acceptance protocol v2

Declared before the first v2 model run. This supplements the original FP32 eager
diagnostics; it does not replace, discard or retroactively qualify those runs.
The recorded outcome and the separately frozen supplemental experiment follow
the original protocol below. **Current status: failed numerical qualification.**

## Primary reference and controls

- Official Kev code: `2074cc44256200bb37540103fa4572901faa10f3`.
- Base: `1001bb4d826a52d1f399e183466143f4da7b741b`.
- Adapter/head: `139fdd94f1b6a6ad80cc15e08fcb99cac885a101`.
- Run the pinned `kev.serve.main` on CUDA with no `KEV_*` environment overrides:
  BF16 backbone, FP32 pointer head, merged LoRA, saved temperature
  `2.406050072164233`, SDPA, prefix cache and CUDA Graphs enabled by default.
- The fixed environment has Torch 2.8.0, Transformers 5.17.0, PEFT 0.21.0,
  FastAPI 0.141.1, Starlette 1.6.0 and Uvicorn 0.53.0. Optional FLA and
  causal-conv1d are absent. Consequently the official default does not enable
  fused_qwen35. This qualifies that environment, not every optional-kernel setup.
- The observer calls the original server response method unchanged, saves its
  already-computed unrounded probabilities, and counts original graph replays.
  It does not replace loading, forward, precision, response formatting or kernels.
- Freeze the full package list, source hashes, command, effective runtime,
  checkpoint hashes, native binary/library hashes, corpus and request order.
  A failure to start the official default is a failed/unverified reference run;
  do not silently disable Graphs or change its dtype to obtain a pass.
- Native remains the existing eager worker. Exact prompt/token/readout equality
  is checked against official CPU preprocessing for every retained request.

## Gates fixed before execution

- Every unrounded candidate probability: maximum absolute error <= **0.01**.
- Choice: identical first-max decision when official top-two margin > **0.02**.
- Noul: identical `p(true) >= 0.5` decision when official distance from 0.5 >
  **0.01**. Record all raw and response-rounded threshold disagreements.
- Score: maximum absolute difference <= **0.1 ordinal level**, for both the
  unrounded expectation and the returned scalar, on every question. Single-level
  scores must be zero. Report the raw error and error divided by `L-1` (zero for
  `L=1`). No unapproved alternative normalized gate is introduced.
- Precedent: merged [Decider #110](https://github.com/InferMatrixLab/system1-omni/pull/110)
  uses Score drift 0.1 on the same 0..L-1 ordinal scale. Its probability/winner
  gates 0.02/0.05 do not replace Kev's stricter existing 0.01/0.02 gates.
  [Open-Jev #102](https://github.com/InferMatrixLab/system1-omni/pull/102)'s 0.001
  checks a native optimization against native, not initial framework parity.
  [Cua-S1 #19](https://github.com/InferMatrixLab/system1-omni/pull/19) has no Score
  support. The earlier proposed 0.25-level gate is not adopted.
- All values must be finite, each distribution in [0,1] and sum within 1e-4 of
  one; counts, IDs and candidate order must match. Compare unrounded values;
  four-decimal wire rounding never hides a numerical failure.
- Boundary coverage: at least **10 actual model-scored Choice** questions with
  margin <=0.02 and **10 actual model-scored Noul** questions with distance <=0.01.
  Keep every qualifying item and every disagreement, including exact ties.
  Distinguish public labeled examples from explicitly synthetic inputs.
- Task-quality gates on the fixed labeled population: observed aggregate
  accuracy drop <=1 percentage point; per-type drop <=2 points; aggregate Brier
  increase <=0.01; normalized Score MAE increase <=0.005. Also report NLL, raw
  Score MAE, per-source results and record-clustered paired 95% bootstrap
  intervals (2,000 resamples, seed 27). These are finite-population qualification
  gates, not a general statistical noninferiority or model-quality claim.
- Direct worker and frontend responses must match exactly. Compare an explicit
  subset of contract and official-selected boundary cases through HTTP; record
  the subset. Input usage and response schema follow the official API. Timing
  fields and numerically dependent output billing tokens are not equated across
  implementations; each response's billing is checked against its own answers.
- Graph capture/replay and prefix cache hits must actually occur in the official
  server; report captures, failures, replay count and cache stats. Native repeated
  inputs must be identical. Official repetitions retain all observed variation.

## Frozen corpus and selection

The generator verifies development-file hashes before reading. It never opens
locked test partitions or changes labels, and makes no selection using model
correctness or native outputs.

1. Original 13 contract requests, including 1/2/many/255 candidates and Score
   1/3/255; three additional mixed repeats.
2. Every `clean` development record of official `evals/v4/transfer-v4`: 656
   questions (336 Choice, 280 Noul, 40 Score), including six held-out public
   sources and held-out synthetic policy structures.
3. Every Score question from `clean` development records of `evals/v4/decision-v4`: 240
   questions from SST5, Yelp and Amazon. These are in-distribution labeled
   sentiment tasks; do not describe them as unseen-source evaluation.
4. Boundary discovery: every `hard_ambiguous` and `hard_probability` development
   record of `hard-v1`, plus a declared 101-input lottery-probability sweep from
   45.0% to 55.0% with Choice and Noul questions. This last sweep is synthetic,
   has no accuracy labels and is excluded from task-quality metrics.

All rows are prepared and frozen before inference. All selected rows are run
through both implementations, whether or not they are close to a boundary.
Boundary membership is determined only from the official unrounded outputs,
then saved before native inference. No native disagreement can remove a row.
If the fixed pool produces too few boundary examples, report a coverage failure;
do not call distant or fabricated probability vectors near-boundary examples.

## Run budget and evidence

One official default-service pass and one native pass over the complete frozen
corpus, then the declared HTTP subset. Preserve all startup/harness/numerical
failures. A harness repair may resume only unexecuted work or repeat an affected
stage in a fresh directory with its reason recorded; never relax a threshold.
One additional diagnostic run is allowed for a concrete failure. All GPU child
processes remain inside `gpu run`, with a two-hour execution cap and at most the
previously authorized two-hour queue wait. Stop owned services in `finally`.

The original `run-01` FP32-unmerged / FP32-merged / BF16-export / native evidence
and its unchanged original gates remain diagnostic evidence. The v2 Score gate
applies to the primary BF16 service comparison, not retroactively to FP32 drift.
Do not claim throughput or latency improvement from this correctness campaign.

## Reproduction

Commands run from the repository root; use the pinned reference environment and
the original exported checkpoint. Install [service-requirements.txt](service-requirements.txt)
in a separate Python 3.12 environment. Build the native opt-in test before reserving
a GPU. Write every run to a fresh output directory.

```sh
python tests/kev/service_acceptance.py prepare --reference /path/to/kev \
  --base /path/to/base --out /path/to/corpus.json
gpu run --gpus 1 --wait 2h --timeout 2h --note 'Kev BF16 service acceptance' -- \
  python tests/kev/service_acceptance.py reference --reference /path/to/kev \
  --adapter /path/to/adapter --corpus /path/to/corpus.json --out /path/to/service-run
```

The reference may use an offline Hub snapshot cache of the pinned base. Point
`HF_HUB_CACHE` to it and set `HF_HUB_OFFLINE=1`; this changes file lookup only.
No `KEV_*` overrides may be present in the official server's environment.

```sh
# CPU-only full token/readout/billing check before device execution:
KEV_EXPORT=/path/to/export KEV_CORPUS=/path/to/corpus.json \
  cargo test --release --locked -p omni-kev-native --test processing \
  exact_tokens_readouts_and_billing_match_official -- --ignored --nocapture
gpu run --gpus 1 --wait 2h --timeout 2h --note 'Kev native service parity' -- \
  env KEV_EXPORT=/path/to/export KEV_CUDA_LIB=/path/to/libqwen3_5_cuda.so \
      KEV_CORPUS=/path/to/corpus.json KEV_NATIVE_OUT=/path/to/native.jsonl \
  cargo test --release --locked -p omni-kev-native --lib \
  native_service_capture -- --ignored --nocapture
python tests/kev/compare_service.py --corpus /path/to/corpus.json \
  --official /path/to/service-run/official-service.json --native /path/to/native.jsonl \
  --reference /path/to/kev --base /path/to/base --out /path/to/comparison.json
```

The comparator returns nonzero on a failed gate and retains the complete report.
For HTTP, use `recipe/kev/validate_http.py` as in the [worker recipe](native.md),
adding `--corpus /path/to/corpus.json` and
`--boundaries /path/to/service-run/boundary-membership.json`; `--native` accepts
the JSONL capture. It selects contract/repeat requests and the first ten official
boundary questions of each type, retaining the actual subset in `http.json`.

## Recorded v2 result and preserved failures

The original pool has 1,213 requests / 1,391 questions. Its 896 labeled questions
are 336 Choice, 280 Noul and 280 Score. Initial execution found a real tokenization
bug at request 233 (`tweet_offensive/test/356`, a development-partition record):
the pinned Transformers Qwen2Tokenizer rebuilds the split regex, differing from
the base JSON on Unicode marks such as emoji variation selectors. The processor
now installs that effective regex. A maintained Unicode oracle and CPU comparison
of every corpus row verify the repair. The partial failed capture is preserved;
the official run is reused and the repaired native run uses a fresh output file.
Regenerate the Unicode oracle with `tests/kev/generate_processing_fixture.py`
and the original reference/base arguments, adding `--unicode-only`; compare the
output with `tests/kev/data/unicode-processing.json`.

The first pool yielded 28 near-boundary Choice and 3 Noul, failing the ten-Noul
coverage gate. Before further inference, a supplemental diagnostic froze 2,002
Noul inputs: two random-event templates, tickets and balls, each at 0.0..100.0%
in steps of 0.1%. All were evaluated, with no native-dependent selection. No
supplemental Noul falls within 0.01 of the threshold (closest: 0.02489048).
The coverage gap remains; these synthetic inputs are excluded from task accuracy.
Generate this fixed supplement with `boundary-supplement` in place of `prepare`.
Run the same reference/native commands with fresh paths; add
`--supplement-corpus`, `--supplement-official` and `--supplement-native` to the
comparator to retain both populations in one report.

| Gate | Result |
| --- | --- |
| Exact tokens, positions, readouts, input/output usage | Pass for all 3,215 requests |
| Probability max absolute error <=0.01 | **Fail:** 0.06492195; 55 original + 84 supplemental questions fail |
| Score raw and wire absolute error <=0.1 level | **Fail:** 255-level raw error 0.16675543, wire error 0.1667 |
| Choice agreement above margin 0.02; Noul above distance 0.01 | Pass; all three decision flips are near-boundary |
| Ten actual near-boundary questions per type | Choice 28; **Noul 3/10, incomplete** |
| Fixed-population task-quality gates | Pass; figures below are not universal model-quality claims |
| Real warmup, direct worker/frontend/native equality | Pass, 32 paired requests and 8 HTTP error cases |

Official Score-255 is 96.48285554, native 96.64961097 (range 0..254). The difference
normalized by 254 is 0.00065652; the absolute gate still fails. The earlier 0.1149
was against the FP32 eager reference and is retained as precision diagnosis.

| Labeled metric | Official BF16 service | Native |
| --- | ---: | ---: |
| Accuracy, all 896 questions | 76.5625% | 76.6741% |
| Choice accuracy, 336 | 81.25% | 81.25% |
| Noul accuracy, 280 | 83.9286% | 84.2857% |
| Score argmax accuracy, 280 | 63.5714% | 63.5714% |
| Brier, all | 0.30749647 | 0.30729482 |
| NLL, all | 0.54907485 | 0.54867828 |
| Raw Score MAE | 0.45892220 | 0.45831086 |
| Normalized Score MAE | 0.12694017 | 0.12674938 |

The paired record-bootstrap accuracy difference interval is [-0.3223, +0.5470]
percentage points; normalized Score MAE difference interval is
[-0.00040033, +0.00001614]. These intervals do not establish general noninferiority.
Per-source results and all individual failures are retained in the local report.
The three raw/wire decision disagreements are:

| Request / question | Official | Native | Official distance |
| --- | --- | --- | ---: |
| `tweet_offensive/test/80` / offensive | Noul 0.49786508, false | 0.50078470, true | 0.00213492 |
| `boundary-probe-060` / outcome | lose | win | top-two 0.00460693 |
| `boundary-probe-095` / outcome | lose | win | top-two 0.01417685 |

Original/supplemental official passes captured 43/4 graphs and replayed 2,246/3,990
times, with no capture failures; the original pass had 30 prefix-cache hits.
Native repeated mixed inputs are identical. The actual execution is BF16 backbone,
FP32 head, SDPA, no remaining LoRA modules and no optional fused Qwen kernels.
Production Qwen/CUDA math is unchanged. Numerical attribution beyond the existing
FP32/merged/BF16 diagnostics requires a separate investigation; neither task
accuracy nor the small normalized Score error overrides a failed parity gate.
