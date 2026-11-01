#!/usr/bin/env python3
"""Recipe entrypoint for the Valen export regression battery.

Thin glue over the implementation in ``tests/valen/export_reference_checks.py``
(the contract battery plus the env-gated artifact validation); see that
module for what runs and how to enable the artifact-gated phases.

    python recipe/valen/test_export_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "valen"))

from export_reference_checks import main  # noqa: E402


if __name__ == "__main__":
    main()
