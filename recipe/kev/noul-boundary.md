# Kev Noul boundary qualification (v4)

The fixed official BF16 default-service acceptance now passes all frozen gates:
**5,484 requests / 5,662 questions**, including **36 Noul** and **28 Choice**
near-boundary questions (minimum 10 each). Every observed decision agrees.
The [v3 numerical repair](numerical-diagnosis.md) and
[v2 protocol and failures](service-acceptance.md) remain available. This is author-run
qualification evidence; contributor/maintainer review remains pending.

## Evidence availability after environment recovery

On 2026-10-31 the previous `/tmp/system1-issue-27` directory was unavailable.
The v1-v5 raw captures, layer dumps and local handoff archive referenced below
have not been recovered. The committed reports and fixed fixtures preserve the
recorded results and gates, but are not a replacement for those raw artifacts.
The [review follow-up in #197](https://github.com/InferMatrixLab/system1-omni/pull/197)
distinguishes fresh checks after the main sync from the historical campaign.
Full-checkpoint BF16/HTTP and prefix-cache requalification remain unverified at
that updated head; the old measurements must not be labeled as fresh passes.

## Discovery frozen before inference

The additional pool reads only the pinned official repository's 29
`development.jsonl` files, in sorted path and original row/question order.
`tests/kev/data/boundary-sources.json` pins every source hash.
`tests/kev/boundary_discovery.py` selects clean Noul questions, one per request,
without consulting model outputs or native correctness. It removes prior and
repeated source-record/question identities and identical actual token rows.
Public dataset IDs containing `test` describe source provenance; no locked Kev
evaluation partition was read.

Before inference, the campaign limited state and branch to 1,024 tokens each.
The exclusion ledger retains 6,607 prior/repeated source identities, 309 length
exclusions and 120 duplicate token rows. The remaining **2,269** requests are
all retained and compared, including those far from the decision threshold.
The length budget limits this discovery claim; it does not qualify longer inputs.

Run the unchanged default service first: official `2074cc4`, base `1001bb4`,
adapter/head `139fdd9`, temperature `2.406050072164233`, Torch 2.8.0+cu128,
Transformers 5.17.0 and PEFT 0.21.0. The backbone is BF16, head FP32, LoRA merged,
SDPA/Graph/prefix cache defaults enabled, optional FLA/causal-conv1d absent.
Save actual unrounded probabilities and freeze membership before native capture.
The criterion remains **`abs(p(true) - 0.5) <= 0.01`**.

This yields **33 new** qualifying Noul questions, plus the original 3. All 36
have distinct source-record/question identities and token rows. None was selected
using native agreement, and none was dropped after comparison. The new discovery
population does not change the original 896-question labeled task population.

## Results under unchanged gates

| Check | Result | Gate |
| --- | ---: | ---: |
| Aggregate probability max absolute error | 0.0024709404 | <=0.01 |
| New 2,269-question probability max error | 0.00000059604645 | <=0.01 |
| All 36 near-boundary Noul probability max error | 0.00000047683716 | <=0.01 |
| Score raw maximum absolute error | 0.00000305218 levels | <=0.1 levels |
| Score returned-scalar maximum error | 0 | <=0.1 levels |
| Choice / Noul boundary coverage | 28 / 36 | >=10 each |
| Far-boundary decision disagreements | 0 | 0 |
| Near-boundary raw / wire disagreements | 0 / 0 | report every item |
| Original labeled task accuracy, official / native | 76.5625% / 76.5625% | aggregate drop <=1 pp, per type <=2 pp |

Brier increase remains `8.76751e-9`; normalized Score MAE change remains
`-2.57951e-9`. All original quality gates and the paired record bootstrap remain
unchanged. This is qualification on a fixed population, not a general model
quality or statistical noninferiority claim.

The closest Noul question is `flakeflagger/hector-client-hector/`
`me.prettyprint.hector.api.ApiV2SystemTest#testSuperCountQuery`, question `flaky`:
official `p(true)=0.5000787973403931`, distance `0.00007879734`.
The complete 36-question table is retained in the local handoff's
`evidence/noul-boundary-v4/boundary-cases.md`; raw, unrounded values remain in JSON.

## Independent replay and HTTP

A fresh official service and native process replay all 36 qualifying Noul cases
plus 16 contract/repeat requests: 52 requests / 75 questions. **36/36 still
qualify**, with zero probability drift within each implementation between the
first and repeat captures. All near-boundary decisions agree again. The replay
records 13 Graph captures, 37 replays, 6 prefix-cache hits and zero capture failures;
no official defaults were disabled.

Real worker warmup, the first mixed request after readiness, **55**
direct-worker/frontend/native response comparisons, and **8** error-response
comparisons pass. This HTTP set covers every qualified Noul case, not only ten.
All child services terminate, and all GPU execution uses `gpu run`.

## Reproduction and preserved evidence

The local handoff contains `reproduce-bf16.sh`, `protocol.md`, source/corpus hashes,
all requests and responses, official-only membership, native captures,
`comparison-all.json`, `comparison-replay.json`, the complete boundary table and
checksummed artifact manifest. Run the script from the handoff with a fresh output
directory under `gpu run --gpus 1 --wait 2h --timeout 2h --note 'Kev BF16 reproduction' --`.
The discovery generator and repeatable supplement comparator are in `tests/kev/`.

Production source, exported checkpoint, worker/capture binaries and CUDA library
are unchanged from v3. The 399 Rust tests, independent head checks and 83 CUDA
operator comparisons from v3 remain applicable; this round reruns 22 Python
checks, Ruff, strict documentation checks, new native captures and HTTP validation.
The comparator also reproduces the old v3 report exactly, including its failed
Noul coverage gate. The v2 probability/Score failures, v3 intermediate failures,
FP32 controls and unsuccessful earlier boundary sweep are preserved.

Other model sizes, optional FLA setups, arbitrary concurrent official batches,
other GPU architectures, cross-request native prefix reuse and performance remain
outside this evidence. The previous lab L20X capability-reporting caveat remains.

## Shared-backend precheck (v5)

A fresh ordinary CUDA build and a fresh Kev-profile build each pass 27 existing
Qwen kernel tests, 3 shared-prefix tests, 3 selected-readout tests and both Graph
switch settings. Full-checkpoint shared-engine tests use the same pinned merged
Kev weights with the default Qwen execution profile, including packed projection
options, prefix lifetime/reset, Graph replay and readout invariance. These are
shared-engine regressions, not Cua-S1/Open-Jev/JEMM task-accuracy results.

All 85 previously exported `cs1_*` symbols remain present. The legacy/default
entry points and the new Torch 2.8 profile pass 99 operator comparisons, including
5120-wide reference norms used by JEMM: zero differing elements under the
unchanged assertions. Ordinary builds need no PyTorch/CUTLASS headers. The
optional build emits redistribution notices alongside its library and has no
Python/libtorch runtime dependency.

The rebuilt Kev library reproduces the complete 52-request boundary/contract
capture byte for byte and passes all 55 HTTP comparisons and 8 error checks.
The 2,269 new discovery requests also pass exact CPU prompt/token/readout checks.
Fresh workspace validation passes 399 Rust tests (92 opt-in tests ignored by
that ordinary command), 22 Python tests, fmt, Clippy, release and strict docs.
Relevant GPU and released-head checks are separately executed. The independent
export/head slice passes its 7 Python tests, Rust/head checks and strict docs.
No functional model/kernel change was needed during this precheck, so the full
5,662-question result above remains the frozen v4 evidence, not a newly rerun
whole-corpus campaign. Raw v5 logs and source/binary hashes remain in the local
handoff under `evidence/precheck-v5/`.
