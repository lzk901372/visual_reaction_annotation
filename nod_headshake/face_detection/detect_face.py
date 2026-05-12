import argparse
import pickle
from pathlib import Path
import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from glob import glob

# MediaPipe Tasks aliases
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def build_options(model_path: str, use_gpu: bool) -> FaceLandmarkerOptions:
    base_kwargs = {"model_asset_path": model_path}
    if use_gpu:
        base_kwargs["delegate"] = BaseOptions.Delegate.GPU
    base_options = BaseOptions(**base_kwargs)

    return FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )


def process_video(video_path: str, output_path: str, model_path: str, use_gpu: bool) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback to a common frame rate if metadata is unavailable.
        fps = 30.0

    options = build_options(model_path=model_path, use_gpu=use_gpu)

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
                    # MediaPipe does not output FACS AU04 directly. Brow-down scores are a common proxy.
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

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_path_obj.open("wb") as f:
        pickle.dump(all_frames_data, f)

    detected_frames = sum(1 for x in all_frames_data if x["detected"])
    print(f"Saved: {output_path_obj}")
    print(f"Frames: {len(all_frames_data)} | Detected face frames: {detected_frames}")
    print("Each frame contains landmarks, blendshapes, au04_proxy and pitch/roll/yaw.")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Face Landmarker on one video and save per-frame outputs."
    )
    parser.add_argument(
        "--video_dir", 
        default="/data/zikai/Data/RealTalkListening/Original/", 
        help="Path to input video."
    )
    parser.add_argument(
        "--output_dir",
        default="./output/",
        help="Path to output pickle file.",
    )
    parser.add_argument(
        "--model_path",
        default="./face_landmarker_v2_with_blendshapes.task",
        help="Path to MediaPipe face landmarker model (.task).",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU delegate (if the local mediapipe build supports it).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}. "
            "Download face_landmarker.task from MediaPipe official release first."
        )
    os.makedirs(args.output_dir, exist_ok=True)

    all_videos = glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True)
    for video_path in tqdm(all_videos):
        video_subdir = Path(video_path).parent.stem
        video_name = Path(video_path).stem
        os.makedirs(os.path.join(args.output_dir, video_subdir), exist_ok=True)
        output_path = os.path.join(args.output_dir, video_subdir, f"{video_name}.pkl")
        process_video(video_path, output_path, args.model_path, args.use_gpu)