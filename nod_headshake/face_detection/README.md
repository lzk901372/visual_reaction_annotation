# Face Detection Feature Extraction

This folder contains Python utilities for extracting per-frame face information from videos with MediaPipe Face Landmarker.

The generated `.pkl` files contains rich information such as AU04 (estimated, not real), Face bbox (normalized and pixelized), blendshapes (Mediapipe analysis), detected (face detected nor not), landmarks (478-point landmarks, including regular 68-point information), pitch/roll/yaw (for head pose analysis) [Note that 'identity' information is discarded in `.pkl` files because it's not helpful for downstream task, but one can restore it by modifying some codes]. All these `.pkl` files are intended for downstream reaction analysis, such as nod/headshake or facial-expression features.

## Contents

- `detect_face.py`: Processes videos under a directory and saves one pickle file per video. This script can request the MediaPipe GPU delegate.
- `detect_face_multithread.py`: Processes one video or a directory of videos with multiple CPU worker threads. Note that GPU-based Mediapipe is difficult to configure, and since Mediapipe is highly optimized, using multithread CPU is not a bad idea.
- `inspect_pkl.py`: Prints a quick summary of one generated pickle file. You can easily get an idea of what is in the `.pkl` file.
- `examine_output.py`: Compares source videos with generated pickle outputs and writes an alignment report.

## Requirements

Install the required Python packages in your environment:

```bash
pip install opencv-python mediapipe numpy tqdm
```

You also need the MediaPipe Face Landmarker model file:

```text
face_landmarker_v2_with_blendshapes.task
```

Place the model file in this folder, or pass its location with `--model_path`.

## Extract Face Features

Process a directory of videos with the multi-threaded CPU script:

```bash
python detect_face_multithread.py \
  --input_dir /path/to/videos \
  --output_dir ./output \
  --model_path ./face_landmarker_v2_with_blendshapes.task \
  --num_workers 4 \
  --skip_existing
```

Process a single video:

```bash
python detect_face_multithread.py \
  --video_path /path/to/video.mp4 \
  --output_dir ./output \
  --model_path ./face_landmarker_v2_with_blendshapes.task
```

The original single-thread script can also process videos recursively from `--video_dir`:

```bash
python detect_face.py \
  --video_dir /path/to/videos \
  --output_dir ./output \
  --model_path ./face_landmarker_v2_with_blendshapes.task
```

## Output Format

Each output pickle file stores a list with one dictionary per video frame. Each frame dictionary contains:

- `detected`: whether a face was detected.
- `landmarks`: 478-point MediaPipe face landmarks as `[x, y, z]` coordinates.
- `bbox_norm`: normalized face bounding box `[x_min, y_min, x_max, y_max]`.
- `bbox_px`: pixel-space face bounding box.
- `blendshapes`: MediaPipe blendshape scores.
- `au04_proxy`: average of `browDownLeft` and `browDownRight`, used as a proxy for brow lowering. If one wants real AU04 scores, they can use PyFeat.
- `pitch`, `roll`, `yaw`: head-pose angles in degrees.

## Inspect Outputs

Preview one generated pickle file:

```bash
python inspect_pkl.py \
  --pkl_path ./output/example.pkl \
  --sample_indices 0,1,2,-1
```

Check whether generated pickle frame counts match the original videos:

```bash
python examine_output.py \
  --video_root /path/to/videos \
  --pkl_root ./output \
  --output_txt ./alignment_report.txt \
  --num_workers 8
```

The alignment report includes the video path, pickle frame count, source video frame count, detection rate, and whether the frame counts match. We can use the detection rate to examine whether a video is good or bad, and whether to use it or not.
