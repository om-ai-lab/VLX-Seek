#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

VLX_SEEK_BACKEND=${VLX_SEEK_BACKEND:-apple} "${PYTHON_BIN:-python3}" inference.py \
  --task detection \
  --text "orange; apple"
