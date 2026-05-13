#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import h5py. Please use a Python environment with a compatible "
        f"numpy/h5py build. Original error: {e}"
    ) from e


LINE_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$"
)


@dataclass
class ConvertStats:
    total_txt: int = 0
    converted_csv: int = 0
    null_txt: int = 0
    parse_error_lines: int = 0
    valid_segments: int = 0


@dataclass
class InjectStats:
    discovered_csv: int = 0
    written_chunks: int = 0
    skipped_chunks: int = 0
    failed_chunks: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Qwen2-Audio chunk txt outputs to frame-level csv and inject "
            "them into a copied HDF5 under /intensity."
        )
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="realtalk",
        help="Dataset name.",
    )
    parser.add_argument(
        "--txt_root",
        type=Path,
        default=Path("/data/zikai/Data/Qwen2-Audio_extract/realtalk"),
        help="Root directory containing Qwen2-Audio txt chunk outputs.",
    )
    parser.add_argument(
        "--converted_root",
        type=Path,
        default=Path("/data/zikai/Data/Qwen2-Audio_extract/realtalk_converted"),
        help="Output root for converted csv files (same folder structure as txt_root).",
    )
    parser.add_argument(
        "--input_h5",
        type=Path,
        required=True,
        help="Input HDF5 path.",
    )
    parser.add_argument(
        "--output_h5",
        type=Path,
        required=True,
        help="Output HDF5 path (input_h5 will be copied first).",
    )
    parser.add_argument(
        "--intensity_group_name",
        type=str,
        default="intensity",
        help="Top-level group name for injected intensity chunks.",
    )
    parser.add_argument(
        "--anchor_group_name",
        type=str,
        default="audio",
        help="Reference group used to validate clip/chunk path existence.",
    )
    parser.add_argument(
        "--disable_anchor_match",
        action="store_true",
        help="If set, skip anchor group validation.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=60,
        help="Frame count in each chunk.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate used to map start/end time (seconds) to frame numbers.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "none"],
        help="Compression used for injected intensity datasets.",
    )
    parser.add_argument(
        "--gzip_level",
        type=int,
        default=6,
        help="Gzip level [0-9], used when --compression=gzip.",
    )
    parser.add_argument(
        "--overwrite_output_h5",
        action="store_true",
        help="Overwrite output_h5 if it already exists.",
    )
    parser.add_argument(
        "--keep_existing_intensity",
        action="store_true",
        help="Keep existing /intensity group and skip existing chunk datasets.",
    )
    parser.add_argument(
        "--clean_converted_root",
        action="store_true",
        help="Delete converted_root before conversion.",
    )
    return parser.parse_args()


def time_to_start_frame_1_based(start_time: float, fps: float) -> int:
    return int(np.floor(start_time * fps)) + 1


def time_to_end_frame_1_based(end_time: float, fps: float) -> int:
    return int(np.ceil(end_time * fps))


def fill_frames_from_segments(
    lines: list[str],
    chunk_size: int,
    fps: float,
) -> tuple[np.ndarray, bool, int, int]:
    scores = np.zeros((chunk_size,), dtype=np.float32)
    saw_null = False
    parse_error_lines = 0
    valid_segments = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper() == "NULL":
            saw_null = True
            continue

        match = LINE_PATTERN.match(line)
        if not match:
            parse_error_lines += 1
            continue

        start_time = float(match.group(1))
        end_time = float(match.group(2))
        intensity = float(match.group(3))
        if end_time <= start_time:
            parse_error_lines += 1
            continue

        start_frame = time_to_start_frame_1_based(start_time, fps)
        end_frame = time_to_end_frame_1_based(end_time, fps)
        start_idx = max(0, start_frame - 1)
        end_idx = min(chunk_size - 1, end_frame - 1)
        if end_idx < start_idx:
            continue

        scores[start_idx : end_idx + 1] = intensity
        valid_segments += 1

    if valid_segments == 0 and saw_null:
        return scores, True, parse_error_lines, valid_segments
    return scores, False, parse_error_lines, valid_segments


def write_csv(csv_path: Path, scores: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "intensity_score"])
        for frame_index, score in enumerate(scores):
            writer.writerow([frame_index, f"{float(score):.6f}"])


def convert_txt_to_csv(
    txt_root: Path,
    converted_root: Path,
    chunk_size: int,
    fps: float,
) -> ConvertStats:
    txt_paths = sorted(txt_root.rglob("*.txt"))
    stats = ConvertStats(total_txt=len(txt_paths))
    if not txt_paths:
        print(f"No txt files found under: {txt_root}")
        return stats

    for idx, txt_path in enumerate(txt_paths, start=1):
        rel = txt_path.relative_to(txt_root)
        csv_rel = rel.with_suffix(".csv")
        csv_path = converted_root / csv_rel

        lines = txt_path.read_text(encoding="utf-8").splitlines()
        scores, is_null, parse_error_lines, valid_segments = fill_frames_from_segments(
            lines=lines,
            chunk_size=chunk_size,
            fps=fps,
        )

        write_csv(csv_path, scores)
        stats.converted_csv += 1
        stats.parse_error_lines += parse_error_lines
        stats.valid_segments += valid_segments
        if is_null:
            stats.null_txt += 1

        if idx % 2000 == 0 or idx == len(txt_paths):
            print(
                "Convert progress "
                f"{idx}/{len(txt_paths)} | converted={stats.converted_csv} "
                f"null={stats.null_txt} parse_errors={stats.parse_error_lines}",
                flush=True,
            )
    return stats


def count_files_by_suffix(root: Path, suffix: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(f"*{suffix}"))


def clip_key_and_chunk_from_csv_rel(rel_csv_path: Path, dataset_name: str) -> tuple[str, str]:
    if dataset_name == "realtalk":
        if len(rel_csv_path.parts) < 3:
            raise ValueError(f"Unexpected realtalk csv relative path: {rel_csv_path}")
        video_id = rel_csv_path.parts[-3]
        clip_id = rel_csv_path.parts[-2]
        chunk_name = rel_csv_path.stem
        clip_key = f"{video_id}_{clip_id}"
        return clip_key, chunk_name
    elif dataset_name == "seamless":
        if len(rel_csv_path.parts) < 2:
            raise ValueError(f"Unexpected seamless csv relative path: {rel_csv_path}")
        video_id = rel_csv_path.parts[-2].replace("_speak", "")
        clip_id = rel_csv_path.parts[-1]
        chunk_name = rel_csv_path.stem
        # clip_key = f"{video_id}_{clip_id}"
        clip_key = video_id
        return clip_key, chunk_name
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_csv_intensity(csv_path: Path, chunk_size: int) -> np.ndarray:
    scores = np.zeros((chunk_size,), dtype=np.float32)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        required = {"frame_index", "intensity_score"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required columns: {required}")
        for row in reader:
            frame_index = int(float(row["frame_index"]))
            if 0 <= frame_index < chunk_size:
                scores[frame_index] = float(row["intensity_score"])
    return scores


def inject_intensity_from_csv(
    h5_file: h5py.File,
    converted_root: Path,
    intensity_group_name: str,
    anchor_group_name: str,
    chunk_size: int,
    disable_anchor_match: bool,
    keep_existing_intensity: bool,
    compression: str,
    gzip_level: int,
    dataset_name: str,
) -> InjectStats:
    csv_paths = sorted(converted_root.rglob("*.csv"))
    stats = InjectStats(discovered_csv=len(csv_paths))
    if not csv_paths:
        print(f"No csv files found under: {converted_root}")
        return stats

    if intensity_group_name in h5_file and not keep_existing_intensity:
        del h5_file[intensity_group_name]
    intensity_group = h5_file.require_group(intensity_group_name)

    anchor_group = h5_file.get(anchor_group_name)
    match_anchor = (not disable_anchor_match) and isinstance(anchor_group, h5py.Group)
    if match_anchor:
        print(f"Anchor alignment: enabled via /{anchor_group_name}")
    else:
        print("Anchor alignment: disabled or anchor group missing.")

    for idx, csv_path in enumerate(csv_paths, start=1):
        clip_key = ""
        chunk_name = ""
        try:
            rel = csv_path.relative_to(converted_root)
            clip_key, chunk_name = clip_key_and_chunk_from_csv_rel(rel, dataset_name)

            if match_anchor:
                if clip_key not in anchor_group:
                    stats.skipped_chunks += 1
                    continue
                if chunk_name not in anchor_group[clip_key]:
                    stats.skipped_chunks += 1
                    continue

            clip_group = intensity_group.require_group(clip_key)
            if chunk_name in clip_group and keep_existing_intensity:
                stats.skipped_chunks += 1
                continue
            if chunk_name in clip_group:
                del clip_group[chunk_name]

            chunk_scores = load_csv_intensity(csv_path, chunk_size).astype(np.float16, copy=False)
            dataset_kwargs = {
                "data": chunk_scores,
                "dtype": np.float16,
                "chunks": (chunk_size,),
            }
            if compression == "gzip":
                dataset_kwargs["compression"] = "gzip"
                dataset_kwargs["compression_opts"] = gzip_level
            clip_group.create_dataset(chunk_name, **dataset_kwargs)
            stats.written_chunks += 1
        except Exception:
            print(f"Failed chunk injection: clip={clip_key} chunk={chunk_name} csv={csv_path}")
            stats.failed_chunks += 1

        if idx % 2000 == 0 or idx == len(csv_paths):
            print(
                "Inject progress "
                f"{idx}/{len(csv_paths)} | written={stats.written_chunks} "
                f"skipped={stats.skipped_chunks} failed={stats.failed_chunks}",
                flush=True,
            )
    return stats


def main() -> None:
    args = parse_args()
    if not args.txt_root.exists():
        raise FileNotFoundError(f"txt_root not found: {args.txt_root}")
    if not args.input_h5.exists():
        raise FileNotFoundError(f"input_h5 not found: {args.input_h5}")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if args.fps <= 0:
        raise ValueError("fps must be > 0")
    if not (0 <= args.gzip_level <= 9):
        raise ValueError("gzip_level must be in [0, 9]")

    if args.clean_converted_root and args.converted_root.exists():
        shutil.rmtree(args.converted_root)
    args.converted_root.mkdir(parents=True, exist_ok=True)

    if args.output_h5.exists():
        if not args.overwrite_output_h5:
            raise FileExistsError(
                f"output_h5 already exists: {args.output_h5}. "
                "Use --overwrite_output_h5 to replace it."
            )
        args.output_h5.unlink()
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)

    txt_count = count_files_by_suffix(args.txt_root, ".txt")
    csv_count = count_files_by_suffix(args.converted_root, ".csv")
    if txt_count > 0 and txt_count == csv_count:
        print(
            "Step 1/2: Skip conversion (existing converted csv count matches txt count) "
            f"[txt={txt_count}, csv={csv_count}]"
        )
        convert_stats = ConvertStats(
            total_txt=txt_count,
            converted_csv=csv_count,
        )
    else:
        print(
            "Step 1/2: Convert txt -> frame csv "
            f"[txt={txt_count}, existing_csv={csv_count}]"
        )
        convert_stats = convert_txt_to_csv(
            txt_root=args.txt_root,
            converted_root=args.converted_root,
            chunk_size=args.chunk_size,
            fps=args.fps,
        )

    print(f"Copying HDF5:\n  from: {args.input_h5}\n  to:   {args.output_h5}")
    shutil.copy2(args.input_h5, args.output_h5)

    print("Step 2/2: Inject intensity into copied HDF5")
    with h5py.File(args.output_h5, "r+") as h5f:
        inject_stats = inject_intensity_from_csv(
            h5_file=h5f,
            converted_root=args.converted_root,
            intensity_group_name=args.intensity_group_name,
            anchor_group_name=args.anchor_group_name,
            chunk_size=args.chunk_size,
            disable_anchor_match=args.disable_anchor_match,
            keep_existing_intensity=args.keep_existing_intensity,
            compression=args.compression,
            gzip_level=args.gzip_level,
            dataset_name=args.dataset_name,
        )

    print("Completed.")
    print(
        "Convert stats: "
        f"total_txt={convert_stats.total_txt}, "
        f"converted_csv={convert_stats.converted_csv}, "
        f"null_txt={convert_stats.null_txt}, "
        f"valid_segments={convert_stats.valid_segments}, "
        f"parse_error_lines={convert_stats.parse_error_lines}"
    )
    print(
        "Inject stats: "
        f"discovered_csv={inject_stats.discovered_csv}, "
        f"written_chunks={inject_stats.written_chunks}, "
        f"skipped_chunks={inject_stats.skipped_chunks}, "
        f"failed_chunks={inject_stats.failed_chunks}"
    )


if __name__ == "__main__":
    main()
