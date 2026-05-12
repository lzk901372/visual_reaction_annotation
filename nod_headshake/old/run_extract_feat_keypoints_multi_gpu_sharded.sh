#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/extract_feat_keypoints_multi_gpu_sharded.py" \
  --input-root "/data/zikai/Data/RealTalkListening/Original" \
  --output-root "${SCRIPT_DIR}/output" \
  --gpus "0,1" \
  --workers-per-gpu 1 \
  --batch-size 64 \
  --num-workers 2 \
  --output-size 512 \
  --max-videos 12
