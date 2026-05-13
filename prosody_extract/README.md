# Prosody Extract

This folder contains tools to detect **localized prosodic fluctuation** in speech using **Qwen2-Audio-7B-Instruct**. Audio is split into short chunks; the model returns time-bounded events with intensity scores. Results can be written as per-chunk text files, converted to frame-level CSV, and injected into an HDF5 file under `/intensity`.

## Contents

| File | Purpose |
|------|---------|
| `batch_emotion_fluctuation.py` | Single-process batch inference (GPU batching over chunks). |
| `batch_emotion_fluctuation_multi_gpu.py` | Multi-GPU worker processes with a task queue. |
| `build_qwen_intensity_h5.py` | Convert chunk `.txt` outputs to frame CSV, copy an input HDF5, inject datasets under `/intensity`. |
| `input_prompt.txt` | Prompt template for the model (strict output format: lines or `NULL`). |
| `requirements.txt` | Python dependencies for inference. |
| `run_qwen.sh` | Example launcher for single-process inference. |
| `run_qwen_mt.sh` | Example launcher for multi-GPU inference (e.g. Seamless). |
| `run_build_intensity_h5.sh` | Example launcher for HDF5 build. |

## Environment

Install dependencies (add `h5py` for `build_qwen_intensity_h5.py`):

```bash
pip install -r requirements.txt h5py
```

You need a local **Qwen2-Audio-7B-Instruct** checkpoint (see `--model-path` in the scripts) and a CUDA-capable setup for `torch` / `transformers`. The model has to be downloaded from huggingface.

## Input List and Audio Paths

Both batch scripts read a **usable list** text file (`--list-file`). Each line should have three comma-separated fields; **only the first column** is used. That field is normally a video path; the script rewrites it to a **WAV** path:

- **`realtalk`**: `Original` → `Original_wav`, `.mp4` → `.wav`.
- **`seamless`**: `Seamless` → `Seamless_audio`; in the multi-GPU script, `.mp4` is replaced with `_speak.wav` (see `batch_emotion_fluctuation_multi_gpu.py`).

`--input-root` is used when the resolved path is not absolute.

## Inference Pipeline

1. Load mono waveform at `--sample-rate` (default 16000 Hz) with **librosa**.
2. Split into fixed-length chunks (`--chunk-seconds`, default **2.4** seconds).
3. Run **Qwen2AudioForConditionalGeneration** with the prompt from `--prompt-file`.
4. For each chunk, write a `.txt` file under `--output-root`, mirroring a folder layout derived from the audio path.

Optional preprocessing:

- **`--normalize-loudness`**: RMS-based loudness normalization before chunking (whole clip, shared gain).
- **`--enable-min-rms-gate`**: Skip model inference for very quiet chunks and write `NULL` instead.

Other useful flags: `--skip-existing`, `--max-audios`, `--scan-non-null` (slow full-tree scan of outputs).

### Model Output Format

The prompt in `input_prompt.txt` asks the model for lines:

```text
start_time, end_time, intensity_score
```

Times are in **seconds** within the chunk; intensity is in **[0, 1]** (the prompt asks for two decimal places). If nothing is detected, the file should contain exactly:

```text
NULL
```

The batch scripts parse valid lines with a strict numeric regex.

## Running Inference

Single-process example (see `run_qwen.sh` for paths):

```bash
bash run_qwen.sh
```

Multi-GPU example:

```bash
bash run_qwen_mt.sh
```

Or call Python directly, for example:

```bash
python batch_emotion_fluctuation_multi_gpu.py \
  --dataset-name realtalk \
  --list-file /path/to/usable_realtalk.txt \
  --input-root /path/to/Original_wav \
  --output-root /path/to/Qwen2-Audio_extract/realtalk \
  --model-path /path/to/qwen2-audio-7b-instruct \
  --gpu-ids "0,1" \
  --num-workers 2 \
  --skip-existing
```

## HDF5 Injection (`build_qwen_intensity_h5.py`)

This script:

1. Walks `--txt_root` for `*.txt`, converts each chunk file to a **frame-level** CSV with columns `frame_index`, `intensity_score` (using `--fps` and `--chunk_size` to map time intervals to frames; segments fill contiguous frame ranges).
2. Copies `--input_h5` to `--output_h5`.
3. Creates `/intensity` (or `--intensity_group_name`) and writes one **1-D** float16 dataset per chunk, aligned by default with existing **`--anchor_group_name`** (default `audio`) clip and chunk names.

Example:

```bash
bash run_build_intensity_h5.sh
```

Important flags: `--overwrite_output_h5`, `--keep_existing_intensity`, `--disable_anchor_match`, `--clean_converted_root`, `--compression` / `--gzip_level`.

`--dataset_name` must match how clip keys are derived from relative CSV paths (`realtalk` vs `seamless`).

## Notes

- Default paths in the shell scripts point to machine-specific data roots; update them for your environment.
- Chunk text outputs can be large; use `--skip-existing` for resumable runs.
- For HDF5 injection, ensure chunk naming and directory layout under `txt_root` matches what `clip_key_and_chunk_from_csv_rel` expects for your `dataset_name`.
