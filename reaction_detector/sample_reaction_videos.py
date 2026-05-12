import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Candidate:
    video_path: Path
    csv_path: Path
    peak_score: float
    max_consecutive: int
    active_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample videos with clear reaction evidence from reaction csv outputs, "
            "then render overlay videos into reaction_sample folder."
        )
    )
    parser.add_argument(
        "--reaction", 
        required=True, 
        help="Reaction folder name (e.g., frowning)."
    )
    parser.add_argument(
        "--number",
         required=True,
          type=int, 
          help="Number of videos to randomly sample."
    )
    parser.add_argument(
        "--usable_txt",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt",
        help="Path to usable_realtalk.txt.",
    )
    parser.add_argument(
        "--reaction_detector_root",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector",
        help="Root folder of reaction_detector.",
    )
    parser.add_argument(
        "--original_root",
        default="/data/zikai/Data/RealTalkListening/Original",
        help="Root folder of original videos.",
    )
    parser.add_argument(
        "--output_subdirs",
        default="output_realtalk,output",
        help="Comma-separated subdir search order inside reaction folder for csvs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--max_scan_videos",
        type=int,
        default=0,
        help="If >0, only scan first N usable videos (for quick debugging).",
    )
    parser.add_argument(
        "--active_threshold",
        type=float,
        default=0.40,
        help="Frame-level threshold to count active reaction frames.",
    )
    parser.add_argument(
        "--min_peak_score",
        type=float,
        default=0.60,
        help="Minimum peak score required for a csv to be considered clear.",
    )
    parser.add_argument(
        "--min_consecutive_frames",
        type=int,
        default=5,
        help="Minimum longest consecutive active frames required.",
    )
    parser.add_argument(
        "--min_active_ratio",
        type=float,
        default=0.03,
        help="Minimum fraction of active frames required.",
    )
    return parser.parse_args()


def parse_usable_videos(usable_txt: Path) -> list[Path]:
    videos: list[Path] = []
    with usable_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 3:
                continue
            video_path = Path(parts[0])
            videos.append(video_path)
    return videos


def get_reaction_csv_path(
    video_path: Path,
    original_root: Path,
    reaction_root: Path,
    output_subdirs: list[str],
) -> Path | None:
    try:
        rel = video_path.relative_to(original_root).with_suffix(".csv")
    except ValueError:
        return None

    for subdir in output_subdirs:
        csv_path = reaction_root / subdir / rel
        if csv_path.exists():
            return csv_path
    return None


def longest_consecutive(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = 0
    cur = 0
    for x in mask:
        if x:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def load_scores(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "frame_index" not in fieldnames:
            raise ValueError(f"Missing frame_index in {csv_path}")

        score_col = None
        for name in fieldnames:
            if name.endswith("_intensity"):
                score_col = name
                break
        if score_col is None:
            # fallback: second column
            non_frame_cols = [x for x in fieldnames if x != "frame_index"]
            if not non_frame_cols:
                raise ValueError(f"No score column found in {csv_path}")
            score_col = non_frame_cols[0]

        rows: list[tuple[int, float]] = []
        for row in reader:
            try:
                i = int((row.get("frame_index") or "").strip())
                s = float((row.get(score_col) or "0").strip())
            except ValueError:
                continue
            rows.append((i, float(np.clip(s, 0.0, 1.0))))

    if not rows:
        return np.zeros(0, dtype=np.float32)

    max_idx = max(i for i, _ in rows)
    scores = np.zeros(max_idx + 1, dtype=np.float32)
    for i, s in rows:
        if 0 <= i < len(scores):
            scores[i] = s
    return scores


def evaluate_candidate(scores: np.ndarray, active_threshold: float) -> tuple[float, int, float]:
    if scores.size == 0:
        return 0.0, 0, 0.0
    active_mask = scores >= active_threshold
    peak = float(np.max(scores))
    max_consec = longest_consecutive(active_mask)
    ratio = float(np.mean(active_mask.astype(np.float32)))
    return peak, max_consec, ratio


def render_overlay(video_path: Path, out_video_path: Path, scores: np.ndarray, reaction: str) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer: {out_video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = float(scores[idx]) if idx < len(scores) else 0.0
        cv2.putText(
            frame,
            f"{reaction}: {score:.3f}",
            (20, max(height - 20, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()


def main() -> None:
    args = parse_args()
    if args.number <= 0:
        raise ValueError("--number must be > 0")
    if args.min_consecutive_frames <= 0:
        raise ValueError("--min_consecutive_frames must be > 0")
    if not (0 <= args.active_threshold <= 1):
        raise ValueError("--active_threshold must be in [0,1]")

    usable_txt = Path(args.usable_txt)
    reaction_detector_root = Path(args.reaction_detector_root)
    reaction_dir = reaction_detector_root / args.reaction
    reaction_sample_dir = reaction_dir / "reaction_sample"
    original_root = Path(args.original_root)
    output_subdirs = [x.strip() for x in args.output_subdirs.split(",") if x.strip()]

    if not usable_txt.exists():
        raise FileNotFoundError(f"usable txt not found: {usable_txt}")
    if not reaction_dir.exists():
        raise FileNotFoundError(f"reaction folder not found: {reaction_dir}")
    if not output_subdirs:
        raise ValueError("No output_subdirs provided.")

    videos = parse_usable_videos(usable_txt)
    if args.max_scan_videos > 0:
        videos = videos[: args.max_scan_videos]
    candidates: list[Candidate] = []

    for video_path in videos:
        csv_path = get_reaction_csv_path(
            video_path=video_path,
            original_root=original_root,
            reaction_root=reaction_dir,
            output_subdirs=output_subdirs,
        )
        if csv_path is None:
            continue

        scores = load_scores(csv_path)
        peak, max_consec, ratio = evaluate_candidate(scores, args.active_threshold)

        if peak < args.min_peak_score:
            continue
        if max_consec < args.min_consecutive_frames:
            continue
        if ratio < args.min_active_ratio:
            continue

        candidates.append(
            Candidate(
                video_path=video_path,
                csv_path=csv_path,
                peak_score=peak,
                max_consecutive=max_consec,
                active_ratio=ratio,
            )
        )

    if not candidates:
        raise RuntimeError("No eligible videos found. Try lowering thresholds.")

    rng = random.Random(args.seed)
    k = min(args.number, len(candidates))
    selected = rng.sample(candidates, k)
    selected = sorted(selected, key=lambda x: str(x.video_path))

    reaction_sample_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = reaction_sample_dir / "selected_samples.csv"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_path",
                "score_csv_path",
                "peak_score",
                "max_consecutive_active_frames",
                "active_ratio",
                "overlay_video_path",
            ]
        )

        for item in selected:
            rel = item.video_path.relative_to(original_root).with_suffix(".mp4")
            out_video_path = reaction_sample_dir / rel
            scores = load_scores(item.csv_path)
            render_overlay(item.video_path, out_video_path, scores, args.reaction)
            writer.writerow(
                [
                    str(item.video_path),
                    str(item.csv_path),
                    f"{item.peak_score:.6f}",
                    str(item.max_consecutive),
                    f"{item.active_ratio:.6f}",
                    str(out_video_path),
                ]
            )

    print(f"Reaction: {args.reaction}")
    print(f"Total usable videos checked: {len(videos)}")
    print(f"Eligible videos found: {len(candidates)}")
    print(f"Selected videos: {len(selected)}")
    print(f"Saved overlay videos to: {reaction_sample_dir}")
    print(f"Selection manifest: {summary_csv}")


if __name__ == "__main__":
    main()
