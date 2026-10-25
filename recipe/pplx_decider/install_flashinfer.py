"""Install the pinned flashinfer-python into this environment.

Head ``f968e576`` was ready on compute capability 9.0 with ``flashinfer``
null and ``linear_attention=fla-chunk``. That result is the primary.
This script installs the pin. Startup holds it on FLA chunk unless
``PPLX_GDN_PREFILL=short``. The worker is still ready on the FLA chunk
when the package is absent or its module does not import.
"""

from __future__ import annotations

import subprocess
import sys

from models.pplx_decider.gdn import FI_PIN


def main() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", FI_PIN]
    )
    from models.pplx_decider.gdn import prepare_flashinfer

    version, status = prepare_flashinfer()
    if status != "bound" or version is None:
        raise SystemExit(f"flashinfer install finished with status {status}")
    print(f"installed {FI_PIN} ({version}, {status})")


if __name__ == "__main__":
    main()
