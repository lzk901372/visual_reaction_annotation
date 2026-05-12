# Reaction Detection Utilities

This directory contains helper scripts for preparing file lists used by the reaction detection pipeline. The two top-level scripts, `generate_usable.py` and `txt_process.py`, are used to create or inspect text files that connect each source video with its corresponding emotion and head-motion feature outputs.

## `generate_usable.py`

`generate_usable.py` builds a usable sample list by combining an alignment report (from `nod_headshake/face_detection` with Mediapipe) with a list of incomplete emotion CSV files (from `fer_emotion` with FER).

By default, it reads:

- `nod_headshake/face_detection/output_seamless_alignment_report_seamless.txt`
- `fer_emotion/incomplete_csv_seamless.txt`

It writes:

- `usable_seamless.txt`

The script keeps only videos that satisfy both conditions:

- The video is not listed in the incomplete CSV list (Means that it contains complete detection from FER).
- The `detection_rate` value in the alignment report is exactly `1.0` (Means that all-frame results are valid from MediaPipe).

For each kept video, the output line has three comma-separated paths:

```text
<video_path>,<fer_csv_path>,<nod_pkl_path>
```

The generated paths point to:

- the original input video,
- the corresponding FER emotion CSV under `reaction_detect/fer_emotion/output` (or the specified folder that contains csv files of the corresponding dataset),
- the corresponding nod/headshake feature pickle under `reaction_detect/nod_headshake/face_detection/output` (or the specified folder that contains pkl files of the corresponding dataset).

Example usage:

```bash
python generate_usable.py
```

Custom paths can be passed with:

```bash
python generate_usable.py \
  --alignment_report <alignment_report.txt> \
  --incomplete_list <incomplete_csv_list.txt> \
  --output_txt <output_usable_list.txt>
```

After running, the script prints summary counts for the total rows, removed rows, kept rows, and output path.

## `txt_process.py`

`txt_process.py` is a small temporary utility for inspecting or rewriting usable-list text files. It is specifically used for seamless dataset, because it has a different directory structure to RealTalk, and one may want to keep consistency between these two.

In its current active form, it:

- opens `usable_seamless_2.txt`,
- reads all lines,
- prints the number of lines,
- exits immediately.

The commented section below the `exit()` call shows an older path-rewriting workflow. That workflow reads `usable_seamless.txt`, splits each comma-separated line into:

```text
video_path,fer_csv_path,pkl_path
```

and rewrites the FER CSV and PKL paths from `Seamless`-based paths to project-local output paths:

- `RealTalkListening/reaction_detect/fer_emotion/output_seamless`
- `RealTalkListening/reaction_detect/nod_headshake/face_detection/output_seamless`

If one wants to change paths in txt files, use the commented parts of `txt_process.py`. The uncommented part is only used to examine number of lines.

## `build_reaction_pruned_h5.py`

`build_reaction_pruned_h5.py` creates a reaction-aware HDF5 file from an existing preprocessed HDF5 file. It first copies the source HDF5 to a new output path, then injects frame-level reaction scores from the combined reaction CSV files, and finally prunes all non-reaction groups so that they contain only the clips and chunks that also exist in the injected `/reaction` group (aligning all groups).

By default, it reads combined reaction CSV files from:

```text
reaction_detector/combined_(realtalk or seamless)
```

Each combined CSV is expected to contain:

```text
frame_index,smiling,laughing,frowning,surprised,nodding,head_shaking
```

The six reaction columns are loaded as a `(num_frames, 6)` matrix, split into fixed-length chunks, converted to `float16`, and written into the output HDF5 under:

```text
/reaction/<clip_key>/chunk_000
/reaction/<clip_key>/chunk_001
...
```

The default chunk size is `60` frames. When anchor matching is enabled, the script uses an existing HDF5 group such as `/audio` to limit the number of reaction chunks so that reaction data stays aligned with the rest of the modalities.

Example usage:

```bash
python build_reaction_pruned_h5.py --overwrite_output
```

Important options include:

- `--dataset_name`: controls how clip keys are derived from CSV paths; supported values in the script are `realtalk` and `seamless`.
- `--input_h5`: source HDF5 file to copy.
- `--output_h5`: destination HDF5 file with injected reaction data.
- `--combined_root`: folder containing combined reaction CSV files.
- `--reaction_group_name`: top-level HDF5 group for reaction data, defaulting to `reaction`.
- `--anchor_group_name`: reference group used to align chunk counts, defaulting to `audio`.
- `--disable_anchor_match`: use CSV-derived chunk counts without matching an anchor group.
- `--chunk_size`: number of frames per reaction chunk.
- `--compression` and `--gzip_level`: compression settings for reaction datasets.
- `--keep_existing_reaction`: keep an existing reaction group and skip clips that are already present.

After injection, the pruning step removes clip groups and chunk datasets from other HDF5 groups if they do not have a matching entry in `/reaction`. This keeps the final HDF5 consistent with the subset of clips that have valid reaction labels.
