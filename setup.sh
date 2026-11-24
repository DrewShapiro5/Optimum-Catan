#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export CONDA_PKGS_DIRS="$PWD/.cache/conda-pkgs"
if [[ ! -x .conda/bin/python ]]; then
  conda env create --prefix "$PWD/.conda" --file environment.yml --yes
fi
export PATH="$PWD/.conda/bin:$PATH"
export PIP_CACHE_DIR="$PWD/.cache/pip"
python -m pip install -e './framework[web]' -e .
if [[ -f integrations/jsettlers/lib/gson-2.8.6.jar ]]; then
  integrations/jsettlers/build.sh
fi
cd framework/ui
npm ci --cache ../../.cache/npm
CTRON_API_URL='' npm run build
