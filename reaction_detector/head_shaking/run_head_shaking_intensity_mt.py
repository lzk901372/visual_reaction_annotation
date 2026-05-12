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
        description="Multi-thread head-shaking intensity extraction (moderate defaults)."
    )
    parser.add_argument(
        "--usable_txt",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/usable_realtalk.txt",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector/head_shaking/output",
    )
    parser.add_argument(
        "--render_overlay_video_output_dir",
        type=str,
        default="/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector/head_shaking/output/videos",
    )
    parser.add_argument(
        "--original_root",
        default="/data/zikai/Data/RealTalkListening/Original",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate_n", type=int, default=20)
    parser.add_argument("--validate_seed", type=int, default=42)
    parser.add_argument("--render_overlay_video", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
    )
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--baseline_window", type=int, default=40)
    parser.add_argument("--th_on", type=float, default=0.50)
    parser.add_argument("--th_off", type=float, default=0.35)
    parser.add_argument("--min_event_len", type=int, default=7)
    parser.add_argument("--min_turns_per_event", type=int, default=1)
    parser.add_argument(
        "--pitch_penalty_weight",
        type=float,
        default=0.40,
        help="Penalty weight for pitch motion (nodding suppression).",
    )

    # Hard constraints (moderate defaults).
    parser.add_argument(
        "--min_yaw_peak_to_valley_deg",
        type=float,
        default=2.0,
        help="Minimum yaw range (max-min) within an event.",
    )
    parser.add_argument(
        "--min_peak_yaw_vel_deg",
        type=float,
        default=0.12,
        help="Minimum peak absolute yaw velocity within an event.",
    )
    parser.add_argument(
        "--min_cycles_per_event",
        type=int,
        default=0,
        help="Minimum oscillation cycles per event.",
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
            items.append(SampleItem(Path(parts[0]), Path(parts[1]), Path(parts[2])))
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
    scale = max(scale, 1e-6)
    return np.clip(np.abs(x) / scale, 0.0, 1.0)


def fill_pose_gaps(values: np.ndarray, detected_mask: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32).copy()
    valid = detected_mask > 0.5
    if np.sum(valid) == 0:
        return np.zeros_like(out, dtype=np.float32)
    idx = np.arange(out.shape[0], dtype=np.float32)
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def gate_events(scores: np.ndarray, th_on: float, th_off: float, min_event_len: int) -> np.ndarray:
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


def signal_turns_count(signal: np.ndarray) -> int:
    if signal.size < 3:
        return 0
    d = np.diff(signal)
    sign = np.sign(d)
    nz = sign[sign != 0]
    if nz.size < 2:
        return 0
    return int(np.sum(nz[1:] * nz[:-1] < 0))


def signal_cycles_count(signal: np.ndarray) -> int:
    return signal_turns_count(signal) // 2


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


def enforce_event_constraints(
    smooth_score: np.ndarray,
    event_mask: np.ndarray,
    yaw_res: np.ndarray,
    yaw_vel: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    n = event_mask.shape[0]
    out_mask = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        if event_mask[i] < 0.5:
            i += 1
            continue
        start = i
        while i < n and event_mask[i] > 0.5:
            i += 1
        end = i

        seg_yaw = yaw_res[start:end]
        seg_yaw_vel = yaw_vel[start:end]
        if seg_yaw.size == 0:
            continue

        turns = signal_turns_count(seg_yaw)
        cycles = signal_cycles_count(seg_yaw)
        peak_to_valley = float(np.max(seg_yaw) - np.min(seg_yaw))
        peak_abs_vel = float(np.max(np.abs(seg_yaw_vel))) if seg_yaw_vel.size > 0 else 0.0

        if turns < args.min_turns_per_event:
            continue
        if cycles < args.min_cycles_per_event:
            continue
        if peak_to_valley < args.min_yaw_peak_to_valley_deg:
            continue
        if peak_abs_vel < args.min_peak_yaw_vel_deg:
            continue

        out_mask[start:end] = 1.0

    return smooth_score * out_mask


def compute_head_shaking_intensity(
    detected_mask: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    pitch_filled = fill_pose_gaps(pitch, detected_mask)
    yaw_filled = fill_pose_gaps(yaw, detected_mask)

    pitch_trend = moving_average(pitch_filled, args.baseline_window)
    yaw_trend = moving_average(yaw_filled, args.baseline_window)
    pitch_res = pitch_filled - pitch_trend
    yaw_res = yaw_filled - yaw_trend

    pitch_vel = np.diff(pitch_res, prepend=pitch_res[0])
    yaw_vel = np.diff(yaw_res, prepend=yaw_res[0])
    yaw_acc = np.diff(yaw_vel, prepend=yaw_vel[0])

    yaw_amp_n = robust_norm(yaw_res, q=95.0)
    yaw_vel_n = robust_norm(yaw_vel, q=95.0)
    yaw_acc_n = robust_norm(yaw_acc, q=95.0)
    pitch_vel_n = robust_norm(pitch_vel, q=95.0)

    # raw = 0.40 * yaw_amp_n + 0.35 * yaw_vel_n + 0.25 * yaw_acc_n
    raw = 0.7 * yaw_amp_n + 0.5 * yaw_vel_n + 0.4 * yaw_acc_n
    raw = raw - args.pitch_penalty_weight * pitch_vel_n
    raw = np.clip(raw, 0.0, 1.0) * detected_mask

    smooth = moving_average(raw, args.smooth_window)
    event_mask = gate_events(smooth, args.th_on, args.th_off, args.min_event_len)
    final_score = enforce_event_constraints(smooth, event_mask, yaw_res, yaw_vel, args)
    return np.clip(final_score, 0.0, 1.0)


def output_base_for_video(video_path: Path, original_root: Path, output_dir: Path) -> Path:
    try:
        rel = video_path.relative_to(original_root).with_suffix("")
        return output_dir / rel
    except ValueError:
        return output_dir / f"{video_path.parent.name}_{video_path.stem}"


def write_intensity_csv(csv_path: Path, scores: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "head_shaking_intensity"])
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
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {out_video_path}")

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = float(scores[i]) if i < len(scores) else 0.0
        cv2.putText(
            frame,
            f"head_shaking: {score:.3f}",
            (20, max(height - 20, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        i += 1
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

        detected, pitch, yaw = load_pose_from_pkl(item.pkl_path)
        scores = compute_head_shaking_intensity(detected, pitch, yaw, args)

        out_base_csv = output_base_for_video(item.video_path, original_root, output_dir)
        write_intensity_csv(out_base_csv.with_suffix(".csv"), scores)

        if should_render_video:
            out_base_video = output_base_for_video(item.video_path, original_root, video_output_dir)
            render_overlay_video(item.video_path, out_base_video.with_suffix(".mp4"), scores)

        return ProcessResult(True, idx, total, item.video_path, f"[OK] {idx}/{total} {item.video_path}")
    except Exception as exc:
        return ProcessResult(False, idx, total, item.video_path, f"[FAIL] {idx}/{total} {item.video_path} :: {exc}")


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
    if args.smooth_window <= 0 or args.baseline_window <= 0:
        raise ValueError("window params must be > 0")
    if args.min_event_len <= 0:
        raise ValueError("--min_event_len must be > 0")
    if args.min_turns_per_event < 0 or args.min_cycles_per_event < 0:
        raise ValueError("turn/cycle params must be >= 0")
    if args.th_off > args.th_on:
        raise ValueError("--th_off should be <= --th_on")

    items = parse_usable_list(usable_txt)
    if args.validate:
        sample_n = min(args.validate_n, len(items))
        rng = random.Random(args.validate_seed)
        items = sorted(rng.sample(items, sample_n), key=lambda x: str(x.video_path))

    should_render_video = args.render_overlay_video or args.validate
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log_head_shaking_mt.txt"

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
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos", unit="video"):
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
