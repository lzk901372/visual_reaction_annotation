import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".MOV", ".AVI", ".MKV", ".WEBM"}


def count_video_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def load_pkl_info(pkl_path: Path) -> tuple[int, int, float]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"PKL top-level object is not list: {pkl_path}")

    total = len(data)
    detected = sum(
        1
        for x in data
        if isinstance(x, dict) and (x.get("detected") is True or x.get("landmarks") is not None)
    )
    rate = (detected / total) if total > 0 else 0.0
    return total, detected, rate


def discover_pairs(video_root: Path, pkl_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for video_path in video_root.rglob("*"):
        if not video_path.is_file():
            continue
        if video_path.suffix not in VIDEO_EXTS:
            continue

        rel = video_path.relative_to(video_root)
        pkl_path = pkl_root / rel.with_suffix(".pkl")
        if pkl_path.exists():
            pairs.append((video_path, pkl_path))

    return sorted(pairs, key=lambda x: str(x[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check frame-count alignment between videos and paired pkl outputs."
    )
    parser.add_argument(
        "--video_root",
        # default="/data/zikai/Data/RealTalkListening/Original",
        default="/data/zikai/Data/Seamless",
        help="Root directory of source videos.",
    )
    parser.add_argument(
        "--pkl_root",
        # default="/data/zikai/Data/RealTalkListening/reaction_detect/nod_headshake/face_detection/output_realtalk",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/nod_headshake/face_detection/output_seamless",
        help="Root directory of pkl outputs mirroring video structure.",
    )
    parser.add_argument(
        "--output_txt",
        # default="/data/zikai/Data/RealTalkListening/reaction_detect/nod_headshake/face_detection/output_realtalk_alignment_report.txt",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/nod_headshake/face_detection/output_seamless_alignment_report_seamless.txt",
        help="Path to output txt report (comma-separated).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for pair analysis.",
    )
    return parser.parse_args()


def analyze_pair(video_path: Path, pkl_path: Path) -> dict:
    pkl_total, _, detection_rate = load_pkl_info(pkl_path)
    video_total = count_video_frames(video_path)
    return {
        "video_path": str(video_path),
        "pkl_total_frames": pkl_total,
        "video_total_frames": video_total,
        "detection_rate": detection_rate,
        "frame_count_match": (pkl_total == video_total),
    }


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    pkl_root = Path(args.pkl_root)
    output_txt = Path(args.output_txt)

    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")
    if not pkl_root.exists():
        raise FileNotFoundError(f"PKL root not found: {pkl_root}")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be > 0")

    pairs = discover_pairs(video_root, pkl_root)
    if not pairs:
        raise ValueError("No paired video/pkl files found.")

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = ["video_path,pkl_total_frames,video_total_frames,detection_rate,frame_count_match"]
    results_by_video: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_video = {
            executor.submit(analyze_pair, video_path, pkl_path): str(video_path)
            for video_path, pkl_path in pairs
        }

        for future in tqdm(as_completed(future_to_video), total=len(future_to_video), desc="Analyzing pairs"):
            video_key = future_to_video[future]
            try:
                item = future.result()
                results_by_video[video_key] = (
                    f"{item['video_path']},"
                    f"{item['pkl_total_frames']},"
                    f"{item['video_total_frames']},"
                    f"{item['detection_rate']:.6f},"
                    f"{item['frame_count_match']}"
                )
            except Exception as exc:
                results_by_video[video_key] = f"{video_key},ERROR,ERROR,ERROR,{exc}"

    for video_path, _ in pairs:
        item = results_by_video[str(video_path)]
        detection_rate = item.split(",")[3].strip()
        if detection_rate == "ERROR" or float(detection_rate) < 1.0:
            print(f"Detection rate is error orless than 1.0 for {video_path}")
            continue
        rows.append(item)

    with output_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(rows))

    print(f"Paired files analyzed: {len(pairs)}")
    print(f"Workers used: {args.num_workers}")
    print(f"Saved report: {output_txt}")


if __name__ == "__main__":
    main()
