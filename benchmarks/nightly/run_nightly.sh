#!/bin/sh
# Local hk2 nightly entrypoint. Install from cron or systemd.
# This is not a GitHub Actions job and it does not start a GPU runner in CI.
set -eu

if [ -n "${OMNI_NIGHTLY_REPO:-}" ]; then
  REPO=$OMNI_NIGHTLY_REPO
else
  SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
  if [ -f "$SCRIPT_DIR/run_nightly.py" ]; then
    REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
  else
    REPO=""
    # Keep this order aligned with CONVENTIONAL_REPOS in run_nightly.py.
    for candidate in \
      "$HOME/repos/ThinkFlowLab/system1-omni" \
      "$HOME/repos/InferMatrixLab/system1-omni"
    do
      if [ -f "$candidate/benchmarks/nightly/run_nightly.py" ]; then
        REPO=$candidate
        break
      fi
    done
  fi
fi

if [ -z "${REPO:-}" ] || [ ! -f "$REPO/benchmarks/nightly/run_nightly.py" ]; then
  echo "error: nightly repo not found. Set OMNI_NIGHTLY_REPO or clone under ~/repos/ThinkFlowLab/system1-omni or ~/repos/InferMatrixLab/system1-omni." >&2
  echo "错误：找不到 nightly 仓库。请设置 OMNI_NIGHTLY_REPO，或把仓库放在 ~/repos/ThinkFlowLab/system1-omni 或 ~/repos/InferMatrixLab/system1-omni。" >&2
  exit 2
fi

LOCK=${OMNI_NIGHTLY_LOCK:-"$HOME/experiments/system1-omni-nightly/nightly.lock"}
mkdir -p "$(dirname "$LOCK")" "$HOME/experiments/system1-omni-nightly"
# Non-blocking: a second 02:00 start exits instead of waiting.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "error: another nightly is already running; this one exits." >&2
  echo "错误：已有 nightly 在运行，本次退出，不会等待。" >&2
  exit 2
fi

if [ -n "${OMNI_NIGHTLY_PYTHON:-}" ]; then
  PYTHON=$OMNI_NIGHTLY_PYTHON
elif [ -x "$REPO/.venv/bin/python" ]; then
  PYTHON=$REPO/.venv/bin/python
else
  PYTHON=python3
fi

cd "$REPO"
exec "$PYTHON" benchmarks/nightly/run_nightly.py "$@"
