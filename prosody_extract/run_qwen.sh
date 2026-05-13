#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/batch_emotion_fluctuation.py" \
  --input-root "/data/zikai/Data/RealTalkListening/Original_wav" \
  --output-root "/data/zikai/Data/Qwen2-Audio_extract/realtalk" \
  --model-path "/data/zikai/Codes/base_checkpoints/qwen2-audio-7b-instruct" \
  --prompt-file "${SCRIPT_DIR}/input_prompt.txt" \
  --sample-rate 16000 \
  --chunk-seconds 2.4 \
  --max-new-tokens 64 \
  --batch-size 48 \
  --chunk-index-width 3 \
  --skip-existing \
  --scan-non-null \
#   --max-audios 100 \


# --max-new-tokens 256 is the original max length of the output text