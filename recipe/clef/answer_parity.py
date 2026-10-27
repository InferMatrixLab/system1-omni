#!/usr/bin/env python3
"""Compare Clef FLA and FlashInfer answers. Exit 1 when the gate fails.

Run from the repository root, after two ``bench.py run`` directories and
the two ``/health`` documents exist. This does not flip the default
prefill and it does not start ``omni-clef``.

    PYTHONPATH=src python recipe/clef/answer_parity.py \
      --fla-run <fla-bench-dir> \
      --flashinfer-run <flashinfer-bench-dir> \
      --fla-health <fla-health.json> \
      --flashinfer-health <flashinfer-health.json> \
      --device "<nvidia-smi name>" \
      --compute-capability "<printed capability>" \
      --report <fresh-report.json>

The recipe for the two servers is in ``recipe/clef/reference.md``.
"""

from __future__ import annotations

from models.clef.answer_parity import main

if __name__ == "__main__":
    raise SystemExit(main())
