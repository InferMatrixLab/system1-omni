# Intern-Decision-4B text candidate

This native Rust/CUDA worker implements the text decision contract of
[`internlm/Intern-Decision-4B`](https://huggingface.co/internlm/Intern-Decision-4B/tree/38c9dafbbef52dd04bae943c0c6ac918419adfd5).
The worker is experimental. Its frozen-corpus L20 qualification is blocked by
calibrated probability and confidence drift. See the [recipe and measured
limits](../../../recipe/intern/README.md).

- `processing.rs` renders ordered JSON state and the Choice, Noul and Score
  schema using the pinned checkpoint's text template. It assigns the 62
  one-token symbols `A–Z a–z 0–9` and takes the position immediately before
  each `<decision>` marker. It rejects requests beyond 8192 complete input
  tokens and rejects nonempty media before dispatch.
- `executor.rs` runs one causal prefill over the whole prompt. The shared
  Qwen executor returns selected final-norm BF16 rows widened losslessly
  to FP32. A model-owned full vocabulary CUDA projection reuses the
  checkpoint's tied BF16 `[248320, 2560]` embedding. Only the legal
  symbol logits for each field enter response finishing. The tied weight
  is presently uploaded a second time for the vocabulary head.
- `finishing.rs` applies restricted FP32 softmax, then the release's FP64
  log-probability temperature scaling at `1.99241824`. It keeps field and
  option order, uses lexical order for exact maxima ties and computes Noul
  as `P(yes)` and Score as a probability-weighted expectation. Both
  `usage.output_tokens` and `usage.decision_count` count scored fields.
  Each request generates zero tokens.
- The shared FIFO scheduler serializes access to one executor and all its
  device buffers. The executor completes device work before releasing
  admission, including when a client disconnects after dispatch. Hidden
  rows and responses are owned by the request. Each request has one batch-one
  execution and no retained prefix or generation state.

The candidate uses the shared merged 4B eager causal profile and existing
CUDA framework-style norms, attention and FLA-style GDN. The fixed reference
used Torch 2.9.1+cu128, Transformers 5.14.1, BF16/SDPA, FLA/fla-core
0.4.2, Triton 3.5.1 and Torch convolution. The candidate's GDN has not been
established to reproduce that FLA build on sm_89.

Prompt and calibration behavior are derived from the pinned checkpoint's
[`inference.py`](https://huggingface.co/internlm/Intern-Decision-4B/blob/38c9dafbbef52dd04bae943c0c6ac918419adfd5/inference.py)
and its bundled [Apache-2.0 license](https://huggingface.co/internlm/Intern-Decision-4B/blob/38c9dafbbef52dd04bae943c0c6ac918419adfd5/LICENSE).
The upstream [InternLM source](https://github.com/InternLM/Intern-Decision/tree/a7b01e276cd5e18bae640be925e685ba6f5159ad)
and [Qwen license](https://huggingface.co/internlm/Intern-Decision-4B/blob/38c9dafbbef52dd04bae943c0c6ac918419adfd5/LICENSE-QWEN)
apply to their respective components. Checkpoint files are loaded from the
user-supplied snapshot and are distributed by the upstream publisher.
