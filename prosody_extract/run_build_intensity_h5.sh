#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/build_qwen_intensity_h5.py" \
  --dataset_name "seamless" \
  --txt_root "/data/zikai/Data/Qwen2-Audio_extract/seamless" \
  --converted_root "/data/zikai/Data/Qwen2-Audio_extract/seamless_converted" \
  --input_h5 "/data/zikai/Data/H5_files/hdf5_reaction/Seamless_v3_listening_preprocessed_reaction_pruned.h5" \
  --output_h5 "/data/zikai/Data/H5_files/hdf5_reaction/Seamless_v3_listening_preprocessed_reaction_pruned_intensity.h5" \
  --overwrite_output_h5 \