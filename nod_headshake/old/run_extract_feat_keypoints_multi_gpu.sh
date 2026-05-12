#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/extract_feat_keypoints_multi_gpu.py" \
  --input-root "/data/zikai/Data/RealTalkListening/Original" \
  --output-root "${SCRIPT_DIR}/output" \
  --gpus "0,1,2,3" \
  --workers-per-gpu 1 \
  --batch-size 1 \
  --num-workers 2 \
  --no-identity \
  --no-emotion \
  --output-size 512 \
  --max-videos 2
