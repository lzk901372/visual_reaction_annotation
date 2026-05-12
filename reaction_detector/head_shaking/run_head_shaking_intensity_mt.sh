#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# python "${SCRIPT_DIR}/run_head_shaking_intensity_mt.py" "$@"

# Moderate
# python "${SCRIPT_DIR}/run_head_shaking_intensity_mt.py" \
#   --usable_txt "/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt" \
#   --output_dir "${SCRIPT_DIR}/output" \
#   --original_root "/data/zikai/Data/RealTalkListening/Original" \
#   --validate \
#   --validate_n 20 \
#   --render_overlay_video \
#   --render_overlay_video_output_dir "${SCRIPT_DIR}/output/videos"


# Loose
# python "${SCRIPT_DIR}/run_head_shaking_intensity_mt.py" \
#   --num_workers 8 \
#   --smooth_window 7 \
#   --baseline_window 35 \
#   --th_on 0.45 \
#   --th_off 0.30 \
#   --min_event_len 6 \
#   --min_turns_per_event 1 \
#   --min_yaw_peak_to_valley_deg 1.6 \
#   --min_peak_yaw_vel_deg 0.08 \
#   --min_cycles_per_event 0 \
#   --pitch_penalty_weight 0.30 \
#   --validate \
#   --validate_n 200




python "${SCRIPT_DIR}/run_head_shaking_intensity_mt.py" \
  --usable_txt "/data/zikai/Data/RealTalkListening/reaction_detect/usable_seamless_2.txt" \
  --output_dir "${SCRIPT_DIR}/output_seamless" \
  --original_root "/data/zikai/Data/Seamless" \
  # --validate \
  # --validate_n 100 \
  # --render_overlay_video \
  # --render_overlay_video_output_dir "${SCRIPT_DIR}/output/videos"