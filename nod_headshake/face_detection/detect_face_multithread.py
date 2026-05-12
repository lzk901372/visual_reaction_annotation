import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# MediaPipe Tasks aliases
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def build_options(model_path: str) -> FaceLandmarkerOptions:
    # CPU only: do not set GPU delegate.
    base_options = BaseOptions(model_asset_path=model_path)
    return FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )


def matrix_to_euler_degrees(matrix_like) -> tuple[float, float, float]:
    mat = np.array(matrix_like, dtype=np.float64)
    if mat.size == 16:
        mat = mat.reshape(4, 4)
    if mat.shape != (4, 4):
        raise ValueError(f"Unexpected matrix shape: {mat.shape}")

    r = mat[:3, :3]
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(r[2, 1], r[2, 2])      # pitch
        y = np.arctan2(-r[2, 0], sy)          # yaw
        z = np.arctan2(r[1, 0], r[0, 0])      # roll
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])     # pitch
        y = np.arctan2(-r[2, 0], sy)          # yaw
        z = 0.0                               # roll

    return np.degrees(x).item(), np.degrees(z).item(), np.degrees(y).item()


def landmarks_to_bbox(landmarks, frame_w: int, frame_h: int):
    if not landmarks:
        return None, None

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min = float(np.clip(min(xs), 0.0, 1.0))
    y_min = float(np.clip(min(ys), 0.0, 1.0))
    x_max = float(np.clip(max(xs), 0.0, 1.0))
    y_max = float(np.clip(max(ys), 0.0, 1.0))

    bbox_norm = [x_min, y_min, x_max, y_max]
    bbox_px = [
        int(round(x_min * frame_w)),
        int(round(y_min * frame_h)),
        int(round(x_max * frame_w)),
        int(round(y_max * frame_h)),
    ]
    return bbox_norm, bbox_px


def process_video(video_path: Path, output_path: Path, model_path: str) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = build_options(model_path=model_path)
    all_frames_data = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_idx * 1000.0) / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_data = {
                "detected": False,
                "landmarks": None,
                "bbox_norm": None,
                "bbox_px": None,
                "blendshapes": {},
                "au04_proxy": None,
                "pitch": None,
                "roll": None,
                "yaw": None,
            }

            if result.face_landmarks:
                frame_h, frame_w = frame_bgr.shape[:2]
                frame_data["detected"] = True
                landmarks_obj = result.face_landmarks[0]
                frame_data["landmarks"] = [[lm.x, lm.y, lm.z] for lm in landmarks_obj]
                bbox_norm, bbox_px = landmarks_to_bbox(landmarks_obj, frame_w, frame_h)
                frame_data["bbox_norm"] = bbox_norm
                frame_data["bbox_px"] = bbox_px

                if result.face_blendshapes:
                    blend = {b.category_name: b.score for b in result.face_blendshapes[0]}
                    frame_data["blendshapes"] = blend
                    brow_l = blend.get("browDownLeft", 0.0)
                    brow_r = blend.get("browDownRight", 0.0)
                    frame_data["au04_proxy"] = (brow_l + brow_r) / 2.0

                if result.facial_transformation_matrixes:
                    pitch, roll, yaw = matrix_to_euler_degrees(
                        result.facial_transformation_matrixes[0]
                    )
                    frame_data["pitch"] = pitch
                    frame_data["roll"] = roll
                    frame_data["yaw"] = yaw

            all_frames_data.append(frame_data)
            frame_idx += 1

    cap.release()

    with output_path.open("wb") as f:
        pickle.dump(all_frames_data, f)

    detected_frames = sum(1 for x in all_frames_data if x["detected"])
    return {
        "video_path": str(video_path),
        "output_path": str(output_path),
        "frames": len(all_frames_data),
        "detected_frames": detected_frames,
    }


def discover_videos(video_path: str | None, input_dir: str | None) -> list[Path]:
    videos: list[Path] = []
    if video_path:
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {p}")
        videos.append(p)
    if input_dir:
        root = Path(input_dir)
        if not root.exists():
            raise FileNotFoundError(f"Input directory not found: {root}")
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV"):
            videos.extend(root.rglob(ext))

    unique_sorted = sorted({v.resolve() for v in videos})
    return [Path(x) for x in unique_sorted]


def output_path_for_video(video: Path, output_dir: Path, input_dir: str | None) -> Path:
    if input_dir:
        root = Path(input_dir).resolve()
        rel = video.resolve().relative_to(root)
        return output_dir / rel.with_suffix(".pkl")
    return output_dir / f"{video.stem}.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-thread face landmark extraction with MediaPipe (CPU only)."
    )
    parser.add_argument(
        "--video_path",
        default=None,
        help="Single input video path. Optional when --input_dir is provided.",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory to recursively discover videos.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        help="Output directory for pkl files.",
    )
    parser.add_argument(
        "--model_path",
        default="./face_landmarker_v2_with_blendshapes.task",
        help="Path to MediaPipe face landmarker model (.task).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos whose output pkl already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not args.video_path and not args.input_dir:
        raise ValueError("You must provide at least one of --video_path or --input_dir")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be > 0")

    videos = discover_videos(video_path=args.video_path, input_dir=args.input_dir)
    if not videos:
        raise ValueError("No videos found to process.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_out_pairs = []
    skipped = 0
    for video in videos:
        out_path = output_path_for_video(video, output_dir, args.input_dir)
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        video_out_pairs.append((video, out_path))

    if skipped > 0:
        print(f"[SKIP] Existing outputs: {skipped}")
    if not video_out_pairs:
        print("No videos to process after applying --skip_existing.")
        return

    tasks = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for video, out_path in video_out_pairs:
            tasks.append(executor.submit(process_video, video, out_path, args.model_path))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing videos"):
            try:
                stats = future.result()
                print(
                    f"[OK] {stats['video_path']} -> {stats['output_path']} | "
                    f"frames={stats['frames']}, detected={stats['detected_frames']}"
                )
            except Exception as exc:
                print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
