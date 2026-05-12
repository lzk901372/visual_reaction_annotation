import argparse
import csv
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SampleItem:
    video_path: Path
    fer_csv_path: Path
    pkl_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-frame smiling intensity from usable_realtalk list."
    )
    parser.add_argument(
        "--usable_txt",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt",
        help="Path to usable_realtalk.txt.",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector/smiling/output",
        help="Output directory for smiling csv files.",
    )
    parser.add_argument(
        "--original_root",
        default="/data/zikai/Data/RealTalkListening/Original",
        help="Root path of source videos; used to keep relative output structure.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="If set, process random samples for quick validation.",
    )
    parser.add_argument(
        "--validate_n",
        type=int,
        default=20,
        help="Number of randomly sampled items when --validate is enabled.",
    )
    parser.add_argument(
        "--validate_seed",
        type=int,
        default=42,
        help="Random seed used by validate sampling.",
    )
    parser.add_argument(
        "--render_overlay_video",
        action="store_true",
        help="If set, render smiling intensity text on each frame and save video.",
    )
    parser.add_argument(
        "--render_overlay_video_output_dir",
        type=str,
        default="./output/videos",
        help="Output directory for rendered overlay videos.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=7,
        help="Moving-average window for smoothing scores.",
    )
    parser.add_argument(
        "--th_on",
        type=float,
        default=0.45,
        help="Start threshold for smiling event gating.",
    )
    parser.add_argument(
        "--th_off",
        type=float,
        default=0.30,
        help="End threshold for smiling event gating.",
    )
    parser.add_argument(
        "--min_event_len",
        type=int,
        default=6,
        help="Minimum event length (frames) for smiling segments.",
    )
    return parser.parse_args()


def resolve_fer_csv_path(path_from_txt: Path) -> Path:
    if path_from_txt.exists():
        return path_from_txt

    # Fallback for current usable_realtalk format:
    # .../fer_emotion/output/<video_id>/<clip>.csv
    # Actual FER output is usually:
    # .../fer_emotion/output/<video_id>/emo_scores/<clip>.csv
    candidate = path_from_txt.parent / "emo_scores" / path_from_txt.name
    if candidate.exists():
        return candidate

    return path_from_txt


def parse_usable_list(usable_txt: Path) -> list[SampleItem]:
    items: list[SampleItem] = []
    with usable_txt.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 3:
                continue
            video_path = Path(parts[0])
            fer_csv = resolve_fer_csv_path(Path(parts[1]))
            pkl_path = Path(parts[2])
            items.append(SampleItem(video_path=video_path, fer_csv_path=fer_csv, pkl_path=pkl_path))
    return items


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def gate_events(
    scores: np.ndarray,
    th_on: float,
    th_off: float,
    min_event_len: int,
) -> np.ndarray:
    mask = np.zeros_like(scores, dtype=np.float32)
    n = scores.shape[0]
    i = 0
    while i < n:
        if scores[i] < th_on:
            i += 1
            continue
        start = i
        i += 1
        while i < n and scores[i] >= th_off:
            i += 1
        end = i
        if (end - start) >= min_event_len:
            mask[start:end] = 1.0
    return mask


def load_fer_happy_scores(fer_csv_path: Path) -> dict[int, float]:
    happy_by_frame: dict[int, float] = {}
    with fer_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int((row.get("frame_index") or "").strip())
                happy_score = float((row.get("happy") or "0").strip())
            except ValueError:
                continue
            happy_by_frame[frame_idx] = float(np.clip(happy_score, 0.0, 1.0))
    return happy_by_frame


def load_smile_blendshape_scores(pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"PKL top-level is not list: {pkl_path}")

    n = len(data)
    detected = np.zeros(n, dtype=np.float32)
    smile = np.zeros(n, dtype=np.float32)

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if item.get("detected") is True:
            detected[i] = 1.0
        bs = item.get("blendshapes")
        if isinstance(bs, dict):
            left = float(bs.get("mouthSmileLeft", 0.0) or 0.0)
            right = float(bs.get("mouthSmileRight", 0.0) or 0.0)
            smile[i] = float(np.clip((left + right) * 0.5, 0.0, 1.0))

    return detected, smile


def compute_smiling_intensity(
    fer_happy_by_frame: dict[int, float],
    detected_mask: np.ndarray,
    smile_bs: np.ndarray,
    smooth_window: int,
    th_on: float,
    th_off: float,
    min_event_len: int,
) -> np.ndarray:
    n = smile_bs.shape[0]
    happy = np.zeros(n, dtype=np.float32)
    for idx, score in fer_happy_by_frame.items():
        if 0 <= idx < n:
            happy[idx] = score

    # Raw score: FER happiness + MediaPipe mouth smile blendshape.
    raw = (0.65 * happy + 0.35 * smile_bs) * detected_mask
    raw = np.clip(raw, 0.0, 1.0)

    smooth = moving_average(raw, smooth_window)
    event_mask = gate_events(smooth, th_on=th_on, th_off=th_off, min_event_len=min_event_len)
    final_score = smooth * event_mask
    return np.clip(final_score, 0.0, 1.0)


def output_base_for_video(video_path: Path, original_root: Path, output_dir: Path) -> Path:
    try:
        rel = video_path.relative_to(original_root).with_suffix("")
        return output_dir / rel
    except ValueError:
        # Fallback when video is not under original_root.
        safe_name = f"{video_path.parent.name}_{video_path.stem}"
        return output_dir / safe_name


def write_intensity_csv(csv_path: Path, scores: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "smiling_intensity"])
        for i, s in enumerate(scores):
            writer.writerow([i, f"{float(s):.6f}"])


def render_overlay_video(video_path: Path, out_video_path: Path, scores: np.ndarray) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {out_video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = float(scores[idx]) if idx < len(scores) else 0.0
        text = f"smiling: {score:.3f}"
        cv2.putText(
            frame,
            text,
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
    usable_txt = Path(args.usable_txt)
    output_dir = Path(args.output_dir)
    render_overlay_video_output_dir = Path(args.render_overlay_video_output_dir)
    original_root = Path(args.original_root)

    if not usable_txt.exists():
        raise FileNotFoundError(f"usable txt not found: {usable_txt}")
    if args.validate and args.validate_n <= 0:
        raise ValueError("--validate_n must be > 0")
    if args.smooth_window <= 0:
        raise ValueError("--smooth_window must be > 0")
    if args.min_event_len <= 0:
        raise ValueError("--min_event_len must be > 0")
    if args.th_off > args.th_on:
        raise ValueError("--th_off should be <= --th_on")

    items = parse_usable_list(usable_txt)
    if args.validate:
        sample_n = min(args.validate_n, len(items))
        rng = random.Random(args.validate_seed)
        items = rng.sample(items, sample_n)
        # Keep output stable/readable in folder listing.
        items = sorted(items, key=lambda x: str(x.video_path))

    # Validation mode is intended for visual inspection, so always export videos.
    should_render_video = args.render_overlay_video or args.validate

    output_dir.mkdir(parents=True, exist_ok=True)
    render_overlay_video_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log_smiling.txt"

    ok_count = 0
    fail_count = 0
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"Total items scheduled: {len(items)}\n")
        for idx, item in enumerate(items, start=1):
            try:
                if not item.video_path.exists():
                    raise FileNotFoundError(f"video not found: {item.video_path}")
                if not item.fer_csv_path.exists():
                    raise FileNotFoundError(f"fer csv not found: {item.fer_csv_path}")
                if not item.pkl_path.exists():
                    raise FileNotFoundError(f"pkl not found: {item.pkl_path}")

                fer_happy = load_fer_happy_scores(item.fer_csv_path)
                detected_mask, smile_bs = load_smile_blendshape_scores(item.pkl_path)
                smiling_scores = compute_smiling_intensity(
                    fer_happy_by_frame=fer_happy,
                    detected_mask=detected_mask,
                    smile_bs=smile_bs,
                    smooth_window=args.smooth_window,
                    th_on=args.th_on,
                    th_off=args.th_off,
                    min_event_len=args.min_event_len,
                )

                out_base_csv = output_base_for_video(item.video_path, original_root, output_dir)
                out_csv = out_base_csv.with_suffix(".csv")
                write_intensity_csv(out_csv, smiling_scores)

                if should_render_video:
                    out_base_video = output_base_for_video(
                        item.video_path, original_root, render_overlay_video_output_dir
                    )
                    out_video = out_base_video.with_suffix(".mp4")
                    render_overlay_video(item.video_path, out_video, smiling_scores)

                ok_count += 1
                log_f.write(f"[OK] {idx}/{len(items)} {item.video_path}\n")
            except Exception as exc:
                fail_count += 1
                log_f.write(f"[FAIL] {idx}/{len(items)} {item.video_path} :: {exc}\n")

    print(f"Scheduled: {len(items)}")
    print(f"Succeeded: {ok_count}")
    print(f"Failed: {fail_count}")
    print(f"Output dir: {output_dir}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
