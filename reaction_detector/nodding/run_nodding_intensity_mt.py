import argparse
import csv
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class SampleItem:
    video_path: Path
    fer_csv_path: Path
    pkl_path: Path


@dataclass
class ProcessResult:
    ok: bool
    idx: int
    total: int
    video_path: Path
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-thread nodding intensity extraction from usable_realtalk list."
    )
    parser.add_argument(
        "--usable_txt",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt",
        help="Path to usable_realtalk.txt.",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector/nodding/output",
        help="Output directory for nodding csv files.",
    )
    parser.add_argument(
        "--render_overlay_video_output_dir",
        type=str,
        default="/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector/nodding/output/videos",
        help="Output directory for rendered overlay videos.",
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
        help="If set, render nodding intensity text on each frame and save video.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of worker threads for parallel video processing.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=7,
        help="Moving-average window for final score smoothing.",
    )
    parser.add_argument(
        "--baseline_window",
        type=int,
        default=31,
        help="Window for removing slow head pose drift from pitch/yaw.",
    )
    parser.add_argument(
        "--th_on",
        type=float,
        default=0.42,
        help="Start threshold for nodding event gating.",
    )
    parser.add_argument(
        "--th_off",
        type=float,
        default=0.30,
        help="End threshold for nodding event gating.",
    )
    parser.add_argument(
        "--min_event_len",
        type=int,
        default=6,
        help="Minimum event length (frames) for nodding segments.",
    )
    parser.add_argument(
        "--min_turns_per_event",
        type=int,
        default=1,
        help="Minimum pitch turning points required in an event.",
    )
    parser.add_argument(
        "--yaw_penalty_weight",
        type=float,
        default=0.35,
        help="Penalty weight for yaw motion (head shake suppression).",
    )
    return parser.parse_args()


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
            items.append(
                SampleItem(
                    video_path=Path(parts[0]),
                    fer_csv_path=Path(parts[1]),
                    pkl_path=Path(parts[2]),
                )
            )
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


def robust_norm(x: np.ndarray, q: float = 95.0) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    scale = float(np.percentile(np.abs(x), q))
    if scale <= 1e-6:
        scale = 1e-6
    return np.clip(np.abs(x) / scale, 0.0, 1.0)


def fill_pose_gaps(values: np.ndarray, detected_mask: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32).copy()
    valid = detected_mask > 0.5
    if np.sum(valid) == 0:
        return np.zeros_like(out, dtype=np.float32)
    idx = np.arange(out.shape[0], dtype=np.float32)
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


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


def count_turns(signal: np.ndarray, segment_mask: np.ndarray) -> int:
    idx = np.where(segment_mask > 0.5)[0]
    if idx.size < 3:
        return 0
    seg = signal[idx]
    d = np.diff(seg)
    sign = np.sign(d)
    # remove zeros to avoid fake flips
    nz = sign[sign != 0]
    if nz.size < 2:
        return 0
    flips = np.sum(nz[1:] * nz[:-1] < 0)
    return int(flips)


def enforce_turn_constraint(
    score: np.ndarray,
    event_mask: np.ndarray,
    pitch_residual: np.ndarray,
    min_turns_per_event: int,
) -> np.ndarray:
    if min_turns_per_event <= 0:
        return score * event_mask

    final_mask = np.zeros_like(event_mask, dtype=np.float32)
    n = event_mask.shape[0]
    i = 0
    while i < n:
        if event_mask[i] < 0.5:
            i += 1
            continue
        start = i
        while i < n and event_mask[i] > 0.5:
            i += 1
        end = i
        seg_mask = np.zeros(n, dtype=np.float32)
        seg_mask[start:end] = 1.0
        if count_turns(pitch_residual, seg_mask) >= min_turns_per_event:
            final_mask[start:end] = 1.0
    return score * final_mask


def load_pose_from_pkl(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"PKL top-level is not list: {pkl_path}")

    n = len(data)
    detected = np.zeros(n, dtype=np.float32)
    pitch = np.zeros(n, dtype=np.float32)
    yaw = np.zeros(n, dtype=np.float32)

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if item.get("detected") is True:
            detected[i] = 1.0
        pitch[i] = float(item.get("pitch", 0.0) or 0.0)
        yaw[i] = float(item.get("yaw", 0.0) or 0.0)

    return detected, pitch, yaw


def compute_nodding_intensity(
    detected_mask: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    smooth_window: int,
    baseline_window: int,
    th_on: float,
    th_off: float,
    min_event_len: int,
    min_turns_per_event: int,
    yaw_penalty_weight: float,
) -> np.ndarray:
    pitch_filled = fill_pose_gaps(pitch, detected_mask)
    yaw_filled = fill_pose_gaps(yaw, detected_mask)

    pitch_trend = moving_average(pitch_filled, baseline_window)
    yaw_trend = moving_average(yaw_filled, baseline_window)
    pitch_res = pitch_filled - pitch_trend
    yaw_res = yaw_filled - yaw_trend

    pitch_vel = np.diff(pitch_res, prepend=pitch_res[0])
    yaw_vel = np.diff(yaw_res, prepend=yaw_res[0])
    pitch_acc = np.diff(pitch_vel, prepend=pitch_vel[0])

    pitch_amp_n = robust_norm(pitch_res, q=95.0)
    pitch_vel_n = robust_norm(pitch_vel, q=95.0)
    pitch_acc_n = robust_norm(pitch_acc, q=95.0)
    yaw_vel_n = robust_norm(yaw_vel, q=95.0)

    # Main nodding cue: pitch oscillation (amplitude + speed + turning).
    # raw = 0.40 * pitch_amp_n + 0.35 * pitch_vel_n + 0.25 * pitch_acc_n
    raw = 0.7 * pitch_amp_n + 0.5 * pitch_vel_n + 0.4 * pitch_acc_n
    raw = raw - yaw_penalty_weight * yaw_vel_n
    raw = np.clip(raw, 0.0, 1.0)
    raw = raw * detected_mask

    smooth = moving_average(raw, smooth_window)
    event_mask = gate_events(smooth, th_on=th_on, th_off=th_off, min_event_len=min_event_len)
    final_score = enforce_turn_constraint(
        score=smooth,
        event_mask=event_mask,
        pitch_residual=pitch_res,
        min_turns_per_event=min_turns_per_event,
    )
    return np.clip(final_score, 0.0, 1.0)


def output_base_for_video(video_path: Path, original_root: Path, output_dir: Path) -> Path:
    try:
        rel = video_path.relative_to(original_root).with_suffix("")
        return output_dir / rel
    except ValueError:
        safe_name = f"{video_path.parent.name}_{video_path.stem}"
        return output_dir / safe_name


def write_intensity_csv(csv_path: Path, scores: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "nodding_intensity"])
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
        text = f"nodding: {score:.3f}"
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


def process_one(
    idx: int,
    total: int,
    item: SampleItem,
    args: argparse.Namespace,
    output_dir: Path,
    video_output_dir: Path,
    original_root: Path,
    should_render_video: bool,
) -> ProcessResult:
    try:
        if not item.video_path.exists():
            raise FileNotFoundError(f"video not found: {item.video_path}")
        if not item.pkl_path.exists():
            raise FileNotFoundError(f"pkl not found: {item.pkl_path}")

        detected_mask, pitch, yaw = load_pose_from_pkl(item.pkl_path)
        nodding_scores = compute_nodding_intensity(
            detected_mask=detected_mask,
            pitch=pitch,
            yaw=yaw,
            smooth_window=args.smooth_window,
            baseline_window=args.baseline_window,
            th_on=args.th_on,
            th_off=args.th_off,
            min_event_len=args.min_event_len,
            min_turns_per_event=args.min_turns_per_event,
            yaw_penalty_weight=args.yaw_penalty_weight,
        )

        out_base_csv = output_base_for_video(item.video_path, original_root, output_dir)
        write_intensity_csv(out_base_csv.with_suffix(".csv"), nodding_scores)

        if should_render_video:
            out_base_video = output_base_for_video(item.video_path, original_root, video_output_dir)
            render_overlay_video(item.video_path, out_base_video.with_suffix(".mp4"), nodding_scores)

        return ProcessResult(
            ok=True,
            idx=idx,
            total=total,
            video_path=item.video_path,
            message=f"[OK] {idx}/{total} {item.video_path}",
        )
    except Exception as exc:
        return ProcessResult(
            ok=False,
            idx=idx,
            total=total,
            video_path=item.video_path,
            message=f"[FAIL] {idx}/{total} {item.video_path} :: {exc}",
        )


def main() -> None:
    args = parse_args()
    usable_txt = Path(args.usable_txt)
    output_dir = Path(args.output_dir)
    video_output_dir = Path(args.render_overlay_video_output_dir)
    original_root = Path(args.original_root)

    if not usable_txt.exists():
        raise FileNotFoundError(f"usable txt not found: {usable_txt}")
    if args.validate and args.validate_n <= 0:
        raise ValueError("--validate_n must be > 0")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be > 0")
    if args.smooth_window <= 0:
        raise ValueError("--smooth_window must be > 0")
    if args.baseline_window <= 0:
        raise ValueError("--baseline_window must be > 0")
    if args.min_event_len <= 0:
        raise ValueError("--min_event_len must be > 0")
    if args.min_turns_per_event < 0:
        raise ValueError("--min_turns_per_event must be >= 0")
    if args.th_off > args.th_on:
        raise ValueError("--th_off should be <= --th_on")

    items = parse_usable_list(usable_txt)
    if args.validate:
        sample_n = min(args.validate_n, len(items))
        rng = random.Random(args.validate_seed)
        items = rng.sample(items, sample_n)
        items = sorted(items, key=lambda x: str(x.video_path))

    should_render_video = args.render_overlay_video or args.validate
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log_nodding_mt.txt"

    ok_count = 0
    fail_count = 0
    total = len(items)

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"Total items scheduled: {total}\n")
        log_f.write(f"num_workers: {args.num_workers}\n")
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    process_one,
                    idx,
                    total,
                    item,
                    args,
                    output_dir,
                    video_output_dir,
                    original_root,
                    should_render_video,
                )
                for idx, item in enumerate(items, start=1)
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing videos",
                unit="video",
            ):
                result = future.result()
                if result.ok:
                    ok_count += 1
                else:
                    fail_count += 1
                log_f.write(result.message + "\n")

    print(f"Scheduled: {total}")
    print(f"Succeeded: {ok_count}")
    print(f"Failed: {fail_count}")
    print(f"Workers used: {args.num_workers}")
    print(f"CSV output dir: {output_dir}")
    print(f"Video output dir: {video_output_dir}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
