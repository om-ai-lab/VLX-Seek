#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

VLX_SEEK_BACKEND=${VLX_SEEK_BACKEND:-ascend} \
VLX_SEEK_NPU_FLA=${VLX_SEEK_NPU_FLA:-full} \
  "${PYTHON_BIN:-python}" inference.py \
    --task detection \
    --text "orange; apple"
