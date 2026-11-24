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

echo "==> Running Rust CPU-only vs Batched GPU benchmark (latest XD-CNN checkpoint)..."
.conda/bin/python -B -m agent_apis.neural.native_gpu_benchmark "$@"
