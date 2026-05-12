#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# python "${SCRIPT_DIR}/run_nodding_intensity_mt_v2.py" "$@"

# V2 is stricter than V1
# python "${SCRIPT_DIR}/run_nodding_intensity_mt_v2.py" \
#   --usable_txt "/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt" \
#   --output_dir "${SCRIPT_DIR}/output" \
#   --original_root "/data/zikai/Data/RealTalkListening/Original" \
#   --validate \
#   --validate_n 20 \
#   --render_overlay_video \
#   --render_overlay_video_output_dir "${SCRIPT_DIR}/output/videos"


# A moderate version of V2
python "${SCRIPT_DIR}/run_nodding_intensity_mt_v2.py" \
#   --validate --validate_n 20 
  --num_workers 8 \
  --th_on 0.50 --th_off 0.35 \
  --min_event_len 7 \
  --min_turns_per_event 1 \
  --min_pitch_peak_to_valley_deg 2.0 \
  --min_peak_pitch_vel_deg 0.12 \
  --min_cycles_per_event 0 \
  --yaw_penalty_weight 0.40 \
  --smooth_window 7 --baseline_window 35


# Include more "slow" and "gentle" nodding events. Might have to fune many parameters.
# python "${SCRIPT_DIR}/run_nodding_intensity_mt_v2.py" \
#   --validate --validate_n 20 --num_workers 8 \
#   --th_on 0.35 --th_off 0.35 \
#   --min_event_len 7 \
#   --min_turns_per_event 1 \
#   --min_pitch_peak_to_valley_deg 2.0 \
#   --min_peak_pitch_vel_deg 0.12 \
#   --min_cycles_per_event 0 \
#   --yaw_penalty_weight 0.40 \
#   --smooth_window 7 --baseline_window 35