import argparse
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from queue import Empty

import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


KEYPOINT_LIST = [
    "frame",
    "approx_time",
    "x_0",
    "y_0",
    "x_1",
    "y_1",
    "x_2",
    "y_2",
    "x_3",
    "y_3",
    "x_4",
    "y_4",
    "x_5",
    "y_5",
    "x_6",
    "y_6",
    "x_7",
    "y_7",
    "x_8",
    "y_8",
    "x_9",
    "y_9",
    "x_10",
    "y_10",
    "x_11",
    "y_11",
    "x_12",
    "y_12",
    "x_13",
    "y_13",
    "x_14",
    "y_14",
    "x_15",
    "y_15",
    "x_16",
    "y_16",
    "x_17",
    "y_17",
    "x_18",
    "y_18",
    "x_19",
    "y_19",
    "x_20",
    "y_20",
    "x_21",
    "y_21",
    "x_22",
    "y_22",
    "x_23",
    "y_23",
    "x_24",
    "y_24",
    "x_25",
    "y_25",
    "x_26",
    "y_26",
    "x_27",
    "y_27",
    "x_28",
    "y_28",
    "x_29",
    "y_29",
    "x_30",
    "y_30",
    "x_31",
    "y_31",
    "x_32",
    "y_32",
    "x_33",
    "y_33",
    "x_34",
    "y_34",
    "x_35",
    "y_35",
    "x_36",
    "y_36",
    "x_37",
    "y_37",
    "x_38",
    "y_38",
    "x_39",
    "y_39",
    "x_40",
    "y_40",
    "x_41",
    "y_41",
    "x_42",
    "y_42",
    "x_43",
    "y_43",
    "x_44",
    "y_44",
    "x_45",
    "y_45",
    "x_46",
    "y_46",
    "x_47",
    "y_47",
    "x_48",
    "y_48",
    "x_49",
    "y_49",
    "x_50",
    "y_50",
    "x_51",
    "y_51",
    "x_52",
    "y_52",
    "x_53",
    "y_53",
    "x_54",
    "y_54",
    "x_55",
    "y_55",
    "x_56",
    "y_56",
    "x_57",
    "y_57",
    "x_58",
    "y_58",
    "x_59",
    "y_59",
    "x_60",
    "y_60",
    "x_61",
    "y_61",
    "x_62",
    "y_62",
    "x_63",
    "y_63",
    "x_64",
    "y_64",
    "x_65",
    "y_65",
    "x_66",
    "y_66",
    "x_67",
    "y_67",
]

LIST_TO_SAVE = [
    "frame",
    "approx_time",
    "Pitch",
    "Roll",
    "Yaw",
    "AU04",
    # "anger",
    # "disgust",
    # "fear",
    # "happiness",
    # "sadness",
    # "surprise",
    # "neutral",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-process multi-GPU batch extraction for py-feat head/frown and keypoints."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="/data/zikai/Data/RealTalkListening/Original",
        help="Root directory containing input .mp4 files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "output"),
        help="Root directory where csv outputs are saved.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help='Comma-separated GPU ids, e.g. "0,1,2,3".',
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Worker processes per GPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size passed to detector.detect_video().",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="num_workers passed to detector.detect_video().",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="skip_frames passed to detector.detect_video().",
    )
    parser.add_argument(
        "--dropna-how",
        type=str,
        default="any",
        choices=["any", "all"],
        help="Row removal rule for LIST_TO_SAVE columns.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if both output csv already exist.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Only process first N videos after sorting.",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Don't detect face pose.",
    )
    parser.add_argument(
        "--no-identity",
        action="store_true",
        help="Don't detect face identity.",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="Don't detect emotion.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=512,
        help="Output size passed to detector.detect_video().",
    )
    return parser.parse_args()


def get_video_paths(input_root: Path):
    video_paths = sorted([p for p in input_root.rglob("*.mp4") if p.is_file()])
    return video_paths


def get_output_paths(video_path: Path, output_root: Path):
    video_output_subdir = output_root / video_path.parent.stem / video_path.stem
    return {
        "csv_dir": video_output_subdir,
        "general_csv": video_output_subdir / "head_and_frown.csv",
        "keypoint_csv": video_output_subdir / "keypoints.csv",
    }


def ensure_columns(df: pd.DataFrame, columns):
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in prediction: {missing_cols}")


def process_one_video(video_path: Path, detector, args):
    out = get_output_paths(video_path, Path(args["output_root"]))
    out["csv_dir"].mkdir(parents=True, exist_ok=True)

    video_prediction = detector.detect_video(
        str(video_path),
        skip_frames=args["skip_frames"],
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        no_pose=args["no_pose"],
        no_identity=args["no_identity"],
        no_emotion=args["no_emotion"],
        output_size=args["output_size"],
    )

    print(f"Done video prediction.")

    video_prediction = video_prediction.replace(r"^\s*$", np.nan, regex=True)
    ensure_columns(video_prediction, LIST_TO_SAVE)
    ensure_columns(video_prediction, KEYPOINT_LIST)
    video_prediction = video_prediction.dropna(
        subset=LIST_TO_SAVE, how=args["dropna_how"]
    )

    general_results = video_prediction.loc[:, LIST_TO_SAVE].copy()
    keypoint_results = video_prediction.loc[:, KEYPOINT_LIST].copy()
    general_results.to_csv(out["general_csv"], index=False)
    keypoint_results.to_csv(out["keypoint_csv"], index=False)

    print(f"Done saving csv.")


def worker_loop(gpu_id, task_queue, result_queue, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        from feat import Detector

        detector = Detector(
            face_model="retinaface",
            au_model="xgb",
            emotion_model="resmasknet",
            facepose_model="img2pose",
            device="cuda",
            n_jobs=2,
            no_pose=args["no_pose"],
            no_identity=args["no_identity"],
            no_emotion=args["no_emotion"],
        )
    except Exception:
        result_queue.put(
            {
                "status": "worker_init_error",
                "gpu": gpu_id,
                "error": traceback.format_exc(),
            }
        )
        return

    while True:
        try:
            video_path = task_queue.get(timeout=1)
        except Empty:
            continue

        if video_path is None:
            break

        try:
            process_one_video(Path(video_path), detector, args)
            result_queue.put(
                {
                    "status": "ok",
                    "gpu": gpu_id,
                    "video_path": video_path,
                }
            )
        except Exception:
            result_queue.put(
                {
                    "status": "error",
                    "gpu": gpu_id,
                    "video_path": video_path,
                    "error": traceback.format_exc(),
                }
            )


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus is empty. Example: --gpus 0,1")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be >= 1")

    all_videos = get_video_paths(input_root)
    if not all_videos:
        print(f"[INFO] No .mp4 found under: {input_root}")
        return

    if args.force:
        pending_videos = all_videos
    else:
        pending_videos = []
        for p in all_videos:
            out = get_output_paths(p, output_root)
            if not (out["general_csv"].exists() and out["keypoint_csv"].exists()):
                pending_videos.append(p)

    if args.max_videos is not None:
        pending_videos = pending_videos[:args.max_videos]

    print(f"[INFO] input_root: {input_root}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] total videos: {len(all_videos)}")
    print(f"[INFO] pending videos: {len(pending_videos)}")
    print(f"[INFO] gpus: {gpu_ids}, workers_per_gpu: {args.workers_per_gpu}")
    print(f"[INFO] batch_size: {args.batch_size}, num_workers: {args.num_workers}, skip_frames: {args.skip_frames}, dropna_how: {args.dropna_how}")
    print(f"[INFO] no_pose: {args.no_pose}, no_identity: {args.no_identity}, no_emotion: {args.no_emotion}, output_size: {args.output_size}")

    if not pending_videos:
        print("[INFO] Nothing to do.")
        return

    queue_size = max(32, len(gpu_ids) * args.workers_per_gpu * 4)
    task_queue = mp.Queue(maxsize=queue_size)
    result_queue = mp.Queue()

    worker_args = {
        "output_root": str(output_root),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "skip_frames": args.skip_frames,
        "dropna_how": args.dropna_how,
        "no_pose": args.no_pose,
        "no_identity": args.no_identity,
        "no_emotion": args.no_emotion,
        "output_size": args.output_size,
    }

    workers = []
    for gpu_id in gpu_ids:
        for _ in range(args.workers_per_gpu):
            p = mp.Process(
                target=worker_loop,
                args=(gpu_id, task_queue, result_queue, worker_args),
                daemon=False,
            )
            p.start()
            workers.append(p)

    for p in tqdm(pending_videos, desc="Enqueue videos"):
        task_queue.put(str(p))

    for _ in range(len(workers)):
        task_queue.put(None)

    ok_count = 0
    err_count = 0
    init_err_count = 0
    total_results_expected = len(pending_videos)

    for _ in range(total_results_expected):
        result = result_queue.get()
        status = result.get("status")
        if status == "ok":
            ok_count += 1
            print(f"[OK][GPU {result['gpu']}] {result['video_path']}")
        elif status == "error":
            err_count += 1
            print(f"[ERR][GPU {result['gpu']}] {result['video_path']}")
            print(result["error"])
        elif status == "worker_init_error":
            init_err_count += 1
            print(f"[INIT_ERR][GPU {result['gpu']}]")
            print(result["error"])

    for p in workers:
        p.join()

    print(
        f"[DONE] ok={ok_count}, error={err_count}, worker_init_error={init_err_count}, "
        f"total={len(pending_videos)}"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
