"""Build causal-conv1d 1.7.0 with an sm_89 cubin and install it here.

The checkpoint requirements pin flash-linear-attention 0.5.2 and do not
install this package. Upstream 1.7.0 emits sm_90 when nvcc is 11.8 or
newer and skips sm_89, so a plain pip install has no L20X image. This
script inserts ``arch=compute_89,code=sm_89`` and keeps or inserts sm_90,
then installs with the torch already in the environment.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from models.pplx_decider.conv import CAUSAL_CONV1D_VERSION, ensure_prefill_gencode

PIN = f"causal-conv1d=={CAUSAL_CONV1D_VERSION}"


def main() -> None:
    import torch

    if torch.version.cuda is None:
        raise SystemExit("this torch build has no CUDA; refuse to build causal_conv1d")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "download", "--no-deps", PIN, "-d", str(root)]
        )
        archives = list(root.glob("causal_conv1d-*.tar.gz"))
        if len(archives) != 1:
            raise SystemExit(f"expected one sdist, found {archives}")
        with tarfile.open(archives[0]) as archive:
            archive.extractall(root)
        setup = next(root.glob("causal_conv1d-*/setup.py"))
        patched = ensure_prefill_gencode(setup.read_text(encoding="utf-8"))
        if "compute_89" not in patched:
            raise SystemExit("refusing to build causal_conv1d without an sm_89 cubin")
        setup.write_text(patched, encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                str(setup.parent),
            ]
        )
    import causal_conv1d

    print(f"installed causal_conv1d {causal_conv1d.__version__}")


if __name__ == "__main__":
    main()
