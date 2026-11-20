# Valen-Decision-1.5 reference contract

This directory contains the System1-Omni reference integration for
Valen-Decision-1.5. It is a reference-serving boundary only: the Python worker,
processor, executor, and postprocessor serve text and single-image `choice`,
`noul`, and `score` questions on a released checkpoint's own runtime. Native
CUDA execution is not part of it.

It is separate from the [Valen-Preview-0923 integration](../valen/README.md).
The two share no code: Preview loads a `checkpoint.pt` on a frozen Qwen base
from a pinned source checkout, while a Valen-Decision-1.5 release is a complete
Hugging Face checkpoint that carries its compiler and model code.

## Pinned artifacts

| Size | Served `model` | Checkpoint | Revision |
| --- | --- | --- | --- |
| 0.8B | `valen-decision-1.5-0.8b` | [Valen-Team/Valen-Decision-1.5-0.8B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-0.8B) | `e079395fe9b8e0763b384af403e636e91c1a1559` |
| 2B | `valen-decision-1.5-2b` | [Valen-Team/Valen-Decision-1.5-2B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-2B) | `1b890606578ba0f40bb70ed46891ab65f778e76d` |
| 4B | `valen-decision-1.5-4b` | [Valen-Team/Valen-Decision-1.5-4B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-4B) | `9b0cacc976d7fb72f252ee1adb1564f5f15e462c` |
| 9B | `valen-decision-1.5-9b` | [Valen-Team/Valen-Decision-1.5-9B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-9B) | `5a512117ebc0f2b69f311becfcf47cdef8abf568` |
| 27B | `valen-decision-1.5-27b` | [Valen-Team/Valen-Decision-1.5-27B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-27B) | `9f1b90bcb6841d9a4a44acfd84c6ecbd4c18fec6` |

Each checkpoint holds the weights, `config.json`, the tokenizer and processor
files, and twelve Python files that `trust_remote_code` executes (the compiler,
the model wrapper, block-structured attention and the heads). The configs set
`model_type="valen_qwen"`, `valen_decision_version="1.5"`,
`qwen_execution="shared_state"`, `valen_full_attention_mask="bbox_blocks"`,
`valen_max_length=16384` and BF16 weight storage.

[`pinned.py`](pinned.py) records the SHA-256 of every one of those files for
the revisions above. The worker identifies the size from the `config.json`
digest and hashes the Python, tokenizer, processor and chat-template files and
the weight shards before it imports anything from the directory; a directory
that matches no pinned release is refused. `--no-verify-weights` skips only the
weight shards. The served `model` follows from the verified size, so one
worker serves one size.

Only the 0.8B and 2B sizes have recorded GPU validation; see the
[recipe](../../../recipe/valen_decision_1_5/reference.md#reference-parity).

## Current serving scope

- `state` is either one inline PNG/JPEG image (`{"image": data URL}`) or text
  (a non-empty, non-whitespace string; objects and arrays are serialized to
  JSON text, matching the shared text-state wire convention);
- image limits: single-frame PNG/JPEG, at most 4 MiB, 2048 pixels per side,
  1,048,576 pixels, a 200:1 aspect ratio; text state, instructions and
  criteria are limited to 16,384 characters;
- `choice`, `noul` and `score` questions, at most 8 per request, freely mixed
  against one shared state;
- `choice`: up to 255 named candidates per question (a null criterion text
  defaults to its key);
- `noul`: fixed `true`/`false` candidates; `criteria` is optional and, when
  present, must give both a `true` and a `false` rubric;
- `score`: an ordered list of 2 to 10 level descriptions in `criteria`;
- no `bbox` questions, video, multi-image, or remote URL inputs;
- no token generation: the decision head returns candidate logits directly.

These request limits are the ones the Preview worker applies. The checkpoint's
runtime itself accepts more (for example 255 score levels and several images).

The public request keeps the Cua-S1-compatible shape:

~~~json
{
  "model": "valen-decision-1.5-0.8b",
  "state": {"image": "data:image/png;base64,..."},
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Choose the next move.",
      "criteria": {"up": "Move up", "down": "Move down"}
    },
    "blocked": {"type": "noul", "instructions": "The path ahead is blocked."}
  }
}
~~~

A text-only request replaces the image object with plain text.

### Unsupported: `bbox`

Valen-Decision-1.5 also answers `bbox` questions (a box, its centre point and
a confidence, from the grounding head). This worker rejects them with `422`.
The frontend's [`/v1/systemone` contract](../../frontend/README.md#http-interface)
names `choice`, `score` and `noul` decisions only, and adding a fourth answer
shape is a protocol decision outside this integration. The checkpoint's
grounding head is loaded but never called.

## Processing

Preprocessing validates and decodes the data URL, writes it to a request-scoped
directory, hashes the bytes, and builds the checkpoint compiler's SystemOne
record. Text state skips materialization. The worker owns that directory's
lifetime.

The compiler is the checkpoint's own `compiler.py` in `shared_state` mode. It
builds one sequence per request: the state, then every question's instructions
and candidates, then one decision readout per question. One backbone forward
answers all questions of a request:

- `choice` and `score` list their candidates in request order; `score`
  candidates carry no level index;
- `noul` uses the runtime's fixed true/false candidate wording, and a rubric
  is appended to the instructions as `True criterion: …` / `False criterion: …`;
- question names never enter the prompt; `choice` keys do.

In the full-attention layers the state attends only to itself, and the
decision questions share one bidirectional block that also sees the state
(`bbox_blocks` without `bbox` questions). The linear-attention layers are
causal. A question's logits can therefore depend on the other questions in the
same request, as in the released model.

## Answers

Postprocessing applies the formulas and field order of the checkpoint's own
`responses.answer` at temperature 1:

~~~text
p = softmax(logits)
choice confidence (n candidates):
  n = 1: 1
  n > 1: max(0, (max(p) - 1/n) / (1 - 1/n))
score confidence (K ordered levels, mode = argmax p):
  distance    = sum_i p_i * |i - mode|
  uniform_mad = (sum_i |i - (K-1)/2|) / K
  confidence  = max(0, 1 - distance / uniform_mad)
~~~

- `choice`: `{"type", "probabilities", "choice", "confidence"}`. `choice` is
  the candidate with the highest probability; ties go to the earliest
  candidate in request order.
- `noul`: `{"type": "noul", "noul": P(true)}`, with no `probabilities` or
  `confidence`. This differs from the Preview worker, which adds both.
- `score`: `{"type", "probabilities", "score", "legend", "confidence"}`.
  `probabilities` maps the level keys `"0"`… to their probability, `score` is
  the expected level `sum(i * p_i)`, and `legend` maps each level key to its
  description.

The response `model` is the served id from the table above; the checkpoint's
own `predict` reports `Valen`. `usage.input_tokens` and
`internal_usage.compute_tokens` are both the length of the one shared
sequence, image tokens included. `usage.output_tokens` is 0. The request has
no `temperature` field.

The confidence formulas are the model's own and, as for the Preview worker,
are not a statement about the shared `/v1/systemone` confidence semantics
discussed in [system1-omni#61](https://github.com/ThinkFlowLab/system1-omni/issues/61).

## Errors

An error rejects the whole request. Its body is `{"detail": "<message>"}`, and
the message names the problem. Transport statuses match the Preview worker:
`400` (unusable JSON body, invalid or mismatched `Content-Length`), `408`
(body read timeout), `411` (no single `Content-Length`; chunked requests are
unsupported), `413` (body or image over its limit), `415` (content type other
than `application/json`), `503` (another inference is running), `404`
(unknown route).

The status is `422` when a well-formed request cannot be evaluated:

- a `model` other than the served id;
- fields other than `model`, `state` and `questions`, or question fields other
  than `type`, `instructions` and `criteria`;
- a question whose `type` is not `choice`, `noul` or `score`;
- text over its limit, empty or whitespace-only text, or text containing a
  tokenizer control token (the loaded tokenizer's full special-token set);
  the guard covers text state, instructions, criterion texts, level
  descriptions and option keys;
- `noul` criteria that do not define exactly `true` and `false`, or `score`
  criteria that are not an ordered list of 2 to 10 level descriptions;
- a state that is neither an image data URL nor text, or an image that is not
  a single-frame PNG/JPEG matching its MIME type and limits;
- anything else the checkpoint's compiler rejects, for example a request whose
  shared sequence exceeds 16,384 tokens; the compiler's message is passed
  through with a `request rejected by the pinned compiler` prefix.

Any other failure is `500` with a generic detail and no request content in
the log (the exception type only).

## Ownership boundary

- `pinned.py` owns the release table and checkpoint verification.
- `protocol.py` owns wire validation and safe inline-image limits.
- `preprocess.py` owns media materialization, the SystemOne record, the
  tokenizer/processor and the checkpoint's compiler.
- `engine.py` owns the loaded model, the shared-state forward, the decision
  head, device state, and warmup. It returns FP32 logits.
- `postprocess.py` owns probability normalization, confidence, and response
  reconstruction.
- [`src/frontend/valen_decision_1_5.py`](../../frontend/valen_decision_1_5.py)
  owns HTTP lifecycle and worker orchestration.

The worker admits one request at a time and runs one unpadded sequence per
request; it does not batch across requests. Warmup runs one image and one text
request with all three question types before the port is bound.

The tests under [`tests/valen_decision_1_5/`](../../../tests/valen_decision_1_5/)
are CPU-only and need no weights, torch or GPU. Setup, launch, example
requests and the recorded GPU validation are in
[recipe/valen_decision_1_5/reference.md](../../../recipe/valen_decision_1_5/reference.md).
