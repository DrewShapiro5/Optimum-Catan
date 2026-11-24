#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE"
export PATH="$HOME/.cargo/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "==> Building native Rust worker (release)..."
agent_apis/neural/native_worker/build.sh

echo "==> Running CoreML 2,000-game cohort pipeline (CNN self-play + JSettlers vs CNN; W4 0.25x value loss for CNN & Transformer)..."
.conda/bin/python -B -m agent_apis.neural.coreml_cohort_pipeline "$@"
