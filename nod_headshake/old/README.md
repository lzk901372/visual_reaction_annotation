# Legacy Head Pose and Keypoint Extraction

This folder contains older scripts for extracting head-pose, frown, and facial keypoint features from RealTalkListening video clips. The code uses `py-feat` to run face detection, action-unit extraction, head-pose estimation, and 68-point facial landmark extraction.

This repository is discontinued because of some inaccuracy. One can use these codes if desired, but proceed at your own risk.

## Contents

- `extract_feat_keypoints.py`: original single-GPU prototype script.
- `extract_feat_keypoints_multi_gpu.py`: multi-process, multi-GPU batch extraction script using a shared task queue.
- `extract_feat_keypoints_multi_gpu_sharded.py`: multi-process, multi-GPU batch extraction script that assigns videos to workers by round-robin sharding.
- `run_extract_feat_keypoints_multi_gpu.sh`: example launcher for the queue-based multi-GPU script.
- `run_extract_feat_keypoints_multi_gpu_sharded.sh`: example launcher for the sharded multi-GPU script.
- `output_old/`: legacy extracted CSV outputs.

## Inputs and Outputs

The scripts expect input videos under an `input-root` directory, with `.mp4` files discovered recursively. By default, the scripts point to the RealTaklListening directory.

For each input video, the output directory is organized as:

```text
<output-root>/<video-parent-name>/<video-stem>/
```

Each processed clip writes two CSV files:

- `head_and_frown.csv`: frame-level `Pitch`, `Roll`, `Yaw`, and `AU04` values.
- `keypoints.csv`: frame-level 68-point facial landmarks, stored as `x_0`, `y_0`, ..., `x_67`, `y_67`.

## Example Usage

Run the queue-based multi-GPU extractor:

```bash
bash run_extract_feat_keypoints_multi_gpu.sh
```

Run the sharded multi-GPU extractor:

```bash
bash run_extract_feat_keypoints_multi_gpu_sharded.sh
```

The launcher scripts set `LD_LIBRARY_PATH` from the active Conda environment and pass example values for `--input-root`, `--output-root`, `--gpus`, `--workers-per-gpu`, `--batch-size`, and `--num-workers`.

## Notes

- These files are kept as legacy extraction utilities and may not reflect the latest pipeline.
- The scripts require a working CUDA environment and the `feat` Python package.
- Use `--force` if you need to re-process videos whose CSV outputs already exist.
- Use `--max-videos` for quick smoke tests before launching a full extraction run.
