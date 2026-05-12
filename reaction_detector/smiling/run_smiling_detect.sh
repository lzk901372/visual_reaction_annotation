#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/run_smiling_intensity.py" \
  --usable_txt "/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt" \
  --output_dir "${SCRIPT_DIR}/output" \
  --original_root "/data/zikai/Data/RealTalkListening/Original" \
  --validate \
  --validate_n 20 \
#   --render_overlay_video \
#   --render_overlay_video_output_dir "${SCRIPT_DIR}/output/videos"