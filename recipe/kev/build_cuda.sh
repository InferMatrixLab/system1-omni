#!/usr/bin/env bash
# Build-only dependencies: PyTorch 2.8.0 headers and its exact CUTLASS submodule.
# The resulting worker/library does not link libtorch or load Python.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:?usage: build_cuda.sh OUTPUT [COMPUTE_CAPABILITY]}
arch=${2:-89}
python=${KEV_PYTHON:-python3}
cache=${KEV_BUILD_CACHE:-${TMPDIR:-/tmp}/kev-cuda-headers}
revision=ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e
digest=36429479df2ab1cd1c7981c2d34e786106326ccfefc8846f902f76151a65e112
export KEV_ATEN_INCLUDE_DIR
KEV_ATEN_INCLUDE_DIR=$("$python" - <<'PY'
from pathlib import Path
import torch
if torch.__version__.split('+')[0] != '2.8.0':
    raise RuntimeError('Kev requires PyTorch 2.8.0 build headers')
print(Path(torch.__file__).resolve().parent / 'include')
PY
)
mkdir -p "$cache"
archive="$cache/cutlass-$revision.tar.gz"
if [ ! -f "$archive" ]; then
    tmp=$(mktemp "$cache/download.XXXXXX")
    trap 'rm -f "$tmp"' EXIT
    curl --fail --location --retry 2 "https://codeload.github.com/NVIDIA/cutlass/tar.gz/$revision" -o "$tmp"
    printf '%s  %s\n' "$digest" "$tmp" | sha256sum --check --status
    mv "$tmp" "$archive"
fi
printf '%s  %s\n' "$digest" "$archive" | sha256sum --check --status
# Re-extract verified bytes into a private staging directory; do not trust a stale tree.
headers=$(mktemp -d "$cache/headers.XXXXXX")
trap 'rm -rf "$headers"' EXIT
tar -xzf "$archive" -C "$headers"
export KEV_CUTLASS_INCLUDE_DIR="$headers/cutlass-$revision/include"
bash "$root/src/backends/cuda/qwen3_5/build.sh" "$out" "$arch"
# Header instantiation embeds PyTorch/CUTLASS code in the library. Keep their
# redistribution notices next to the binary, even though neither runtime is linked.
cp "$root/src/models/qwen3_5/native/licenses/PYTORCH-2.8-LICENSE" "$out/PYTORCH-LICENSE"
cp "$root/src/models/qwen3_5/native/licenses/CUTLASS-LICENSE" "$out/CUTLASS-LICENSE"
cat > "$out/KEV_CUDA_NOTICES.txt" <<'NOTICE'
This library includes header-instantiated PyTorch 2.8.0 memory-efficient attention
(Copyright (c) Meta Platforms, Inc. and affiliates) and CUTLASS
ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e. Redistribute this notice and the adjacent
PYTORCH-LICENSE and CUTLASS-LICENSE files with the library.
NOTICE
