#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .conda/bin/python ]]; then
  echo 'Run ./setup.sh first to create the Conda environment.' >&2
  exit 1
fi
exec .conda/bin/python play_catan.py "$@"
