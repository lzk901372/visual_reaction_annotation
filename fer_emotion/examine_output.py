from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("OpenCV (cv2) is required to read video frame count.") from exc


# PROJECT_ROOT = Path("/data/zikai/Data/RealTalkListening")
# ORIGINAL_DIR = PROJECT_ROOT / "Original"
# OUTPUT_DIR = PROJECT_ROOT / "reaction_detect" / "fer_emotion" / "output"

PROJECT_ROOT = Path("/data/zikai/Data/RealTalkListening")
ORIGINAL_DIR = Path("/data/zikai/Data/Seamless")
OUTPUT_DIR = PROJECT_ROOT / "reaction_detect" / "fer_emotion" / "output_seamless"
RESULT_FILE = PROJECT_ROOT / "reaction_detect" / "fer_emotion" / "incomplete_csv_seamless.txt"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def normalize_csv_parent(parent: Path) -> Path:
    """
    Remove the optional 'emo_scores' segment from CSV parent path.
    Example: O98D2Tlgs-E/emo_scores -> O98D2Tlgs-E
    """
    parts = tuple(p for p in parent.parts if p != "emo_scores")
    return Path(*parts) if parts else Path(".")


def build_video_map(video_root: Path) -> Dict[Tuple[Path, str], Path]:
    mapping: Dict[Tuple[Path, str], Path] = {}
    for video_path in tqdm(sorted(video_root.rglob("*")), desc="Building video map"):
        if not video_path.is_file():
            continue
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative = video_path.relative_to(video_root)
        key = (relative.parent, video_path.stem)
        mapping[key] = video_path
    return mapping


def build_csv_map(csv_root: Path) -> Dict[Tuple[Path, str], Path]:
    mapping: Dict[Tuple[Path, str], Path] = {}
    for csv_path in tqdm(sorted(csv_root.rglob("*.csv")), desc="Building CSV map"):
        if not csv_path.is_file():
            continue
        relative = csv_path.relative_to(csv_root)
        normalized_parent = normalize_csv_parent(relative.parent)
        key = (normalized_parent, csv_path.stem)
        mapping[key] = csv_path
    return mapping


def get_video_frame_count(video_path: Path) -> Optional[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def get_csv_data_row_count(csv_path: Path) -> int:
    """
    Count CSV data rows (excluding header).
    Empty rows are ignored.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for row in reader if any(cell.strip() for cell in row))


def check_mismatched_csvs(
    video_map: Dict[Tuple[Path, str], Path],
    csv_map: Dict[Tuple[Path, str], Path],
) -> List[Tuple[Path, int, int]]:
    common_keys = sorted(set(video_map) & set(csv_map))

    unmatched_videos = sorted(set(video_map) - set(csv_map))
    unmatched_csvs = sorted(set(csv_map) - set(video_map))
    if unmatched_videos:
        print(f"Warning: {len(unmatched_videos)} videos have no matching CSV.")
    if unmatched_csvs:
        print(f"Warning: {len(unmatched_csvs)} CSV files have no matching video.")

    mismatched_csv_entries: List[Tuple[Path, int, int]] = []
    for key in tqdm(common_keys, desc="Checking mismatched CSVs"):
        video_path = video_map[key]
        csv_path = csv_map[key]

        frame_count = get_video_frame_count(video_path)
        if frame_count is None:
            print(f"Warning: cannot open video: {video_path}")
            continue

        csv_rows = get_csv_data_row_count(csv_path)
        if frame_count != csv_rows:
            mismatched_csv_entries.append((csv_path, csv_rows, frame_count))

    return mismatched_csv_entries


def write_result(entries: Iterable[Tuple[Path, int, int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(entries, key=lambda item: str(item[0]))
    with output_file.open("w", encoding="utf-8") as f:
        for csv_path, csv_rows, video_frames in sorted_entries:
            f.write(f"{csv_path},{csv_rows},{video_frames}\n")


def main() -> None:
    video_map = build_video_map(ORIGINAL_DIR)
    csv_map = build_csv_map(OUTPUT_DIR)

    print("Done building video and CSV maps.")

    mismatched_csvs = check_mismatched_csvs(video_map, csv_map)
    write_result(mismatched_csvs, RESULT_FILE)

    print(f"Total videos found: {len(video_map)}")
    print(f"Total CSVs found: {len(csv_map)}")
    print(f"Mismatched CSV count: {len(mismatched_csvs)}")
    print(f"Result written to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
