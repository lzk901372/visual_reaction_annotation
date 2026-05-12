#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/detect_multi_gpu.py" \
  --input-root "/data/zikai/Data/Seamless" \
  --output-root "${SCRIPT_DIR}/output_seamless" \
  --gpus "0,1" \
  --workers-per-gpu 1 \
  --batch-size 512 \
  --size-multiplier 1 \
  --mtcnn \