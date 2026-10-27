#!/usr/bin/env python3
"""Compare eager FLA Clef answers with the opt-in piecewise backbone.

Run from the repository root after two ``bench.py run`` directories and the
two ``/health`` documents exist. This does not flip the default and it does
not start ``omni-clef``. Exit 1 fails the tip. The tolerances live in
``models.clef.answer_parity`` and are not flags.

    PYTHONPATH=src python recipe/clef/answer_parity.py \
      --eager-run <eager-bench-dir> \
      --piecewise-run <piecewise-bench-dir> \
      --eager-health <eager-health.json> \
      --piecewise-health <piecewise-health.json> \
      --device "<nvidia-smi name>" \
      --compute-capability "<printed capability>" \
      --report <fresh-report.json>
"""

from __future__ import annotations

from models.clef.answer_parity import main

if __name__ == "__main__":
    raise SystemExit(main())
