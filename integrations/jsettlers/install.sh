#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
revision=7a756baac6ad675f0930605bfb3a8e1bd7a796a5
if [[ ! -d integrations/jsettlers/upstream/.git ]]; then
  git clone https://github.com/jdmonin/JSettlers2.git integrations/jsettlers/upstream
  git -C integrations/jsettlers/upstream checkout "$revision"
fi
if [[ "$(git -C integrations/jsettlers/upstream rev-parse HEAD)" != "$revision" ]]; then
  echo 'JSettlers source differs from the pinned revision; preserve your edits and use the pinned revision before building.' >&2
  exit 1
fi
mkdir -p integrations/jsettlers/lib
if [[ ! -f integrations/jsettlers/lib/gson-2.8.6.jar ]]; then
  curl --fail --location --output integrations/jsettlers/lib/gson-2.8.6.jar https://repo.maven.apache.org/maven2/com/google/code/gson/gson/2.8.6/gson-2.8.6.jar
fi
python3 - <<'PY'
from pathlib import Path
import hashlib
expected='c8fb4839054d280b3033f800d1f5a97de2f028eb8ba2eb458ad287e536f3f25f'
if hashlib.sha256(Path('integrations/jsettlers/lib/gson-2.8.6.jar').read_bytes()).hexdigest()!=expected:
    raise SystemExit('Gson checksum mismatch')
PY
integrations/jsettlers/build.sh
