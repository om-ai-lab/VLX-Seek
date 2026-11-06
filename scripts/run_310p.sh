#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

launcher=("${PYTHON_BIN:-python}")
tp_args=()
if [[ "${TP_SIZE:-1}" == "2" ]]; then
  launcher=(torchrun --standalone --nproc-per-node=2)
  tp_args=(--tp-size 2)
elif [[ "${TP_SIZE:-1}" != "1" ]]; then
  echo "TP_SIZE must be 1 or 2 (310P only)." >&2
  exit 2
fi

VLX_SEEK_BACKEND=${VLX_SEEK_BACKEND:-ascend} VLX_SEEK_NPU_FLA=${VLX_SEEK_NPU_FLA:-full} "${launcher[@]}" inference.py \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh "${tp_args[@]}"
