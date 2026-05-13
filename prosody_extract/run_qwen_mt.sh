#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/batch_emotion_fluctuation_multi_gpu.py" \
  --dataset-name "seamless" \
  --input-root "/data/zikai/Data/Seamless_audio" \
  --output-root "/data/zikai/Data/Qwen2-Audio_extract/seamless" \
  --list-file "/data/zikai/Data/RealTalkListening/reaction_detect/usable_seamless_2.txt" \
  --model-path "/data/zikai/Codes/base_checkpoints/qwen2-audio-7b-instruct" \
  --prompt-file "${SCRIPT_DIR}/input_prompt.txt" \
  --sample-rate 16000 \
  --chunk-seconds 2.4 \
  --max-new-tokens 64 \
  --batch-size 16 \
  --chunk-index-width 3 \
  --gpu-ids "0,1,2" \
  --num-workers 3 \
  --skip-existing \
  --scan-non-null \
  --normalize-loudness \
#   --max-audios 100 \




# --normalize-loudness：开启响度归一化（默认关闭）
# --target-dbfs：目标 RMS dBFS（默认 -20.0）
# --peak-limit：归一化后峰值限制（默认 0.98）
# --enable-min-rms-gate：开启最小能量门限（默认关闭）
# --min-chunk-rms：门限值（默认 0.003）
# 实现细节：

# 归一化在整段音频切块前进行（统一增益，保留 chunk 间相对动态）
# 最小门限作用在每个 chunk：
# 若开启且 chunk RMS 低于阈值，则直接写 NULL，不送模型推理
# 统计信息新增：
# chunks_low_rms_filtered
# 同时打印 normalize_loudness / enable_min_rms_gate 开关状态
# 多卡进度条里也加入了 chunks_low_rms