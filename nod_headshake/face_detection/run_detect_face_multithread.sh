#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/detect_face_multithread.py" \
  --input_dir "/data/zikai/Data/Seamless" \
  --output_dir "${SCRIPT_DIR}/output" \
  --model_path "${SCRIPT_DIR}/face_landmarker_v2_with_blendshapes.task" \
  --num_workers 8 \
  --skip_existing
