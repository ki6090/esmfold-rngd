#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec .venv/bin/hf download facebook/esmfold_v1 \
  --revision 75a3841ee059df2bf4d56688166c8fb459ddd97a \
  --local-dir models/esmfold_v1 --exclude '*.msgpack' '*.h5'
