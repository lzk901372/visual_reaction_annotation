import argparse
import multiprocessing as mp
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


KEYPOINT_LIST = ["frame", "approx_time"] + [
    coord
    for i in range(68)
    for coord in (f"x_{i}", f"y_{i}")
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
        description="Sharded multi-process multi-GPU py-feat extraction without task queue."
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
        default=64,
        help="Batch size for detector.detect_video().",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count for detector.detect_video().",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="skip_frames for detector.detect_video().",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=512,
        help="output_size for detector.detect_video().",
    )
    parser.add_argument(
        "--dropna-how",
        type=str,
        default="any",
        choices=["any", "all"],
        help="dropna rule on LIST_TO_SAVE.",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose output in detector pipeline if supported by backend.",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="Disable emotion output in detector pipeline if supported by backend.",
    )
    parser.add_argument(
        "--no-identity",
        action="store_true",
        help="Disable identity output in detector pipeline if supported by backend.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if output csv files already exist.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Only process first N videos after sorting.",
    )
    return parser.parse_args()


def get_video_paths(input_root: Path):
    video_paths = sorted([p for p in input_root.rglob("*.mp4") if p.is_file()])
    return video_paths


def get_output_paths(video_path: Path, output_root: Path):
    csv_dir = output_root / video_path.parent.stem / video_path.stem
    return {
        "csv_dir": csv_dir,
        "general_csv": csv_dir / "head_and_frown.csv",
        "keypoint_csv": csv_dir / "keypoints.csv",
    }


def ensure_columns(df: pd.DataFrame, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in prediction: {missing}")


def process_one_video(video_path: Path, detector, worker_args):
    out = get_output_paths(video_path, Path(worker_args["output_root"]))
    # print(str(out['csv_dir']))
    os.makedirs(str(out['csv_dir']), exist_ok=True)

    # print("start video prediction")

    video_prediction = detector.detect_video(
        str(video_path),
        skip_frames=worker_args["skip_frames"],
        output_size=worker_args["output_size"],
        batch_size=worker_args["batch_size"],
        num_workers=worker_args["num_workers"],
        no_pose=worker_args["no_pose"],
        no_emotion=worker_args["no_emotion"],
        no_identity=worker_args["no_identity"],
    )

    video_prediction = video_prediction.replace(r"^\s*$", np.nan, regex=True)
    ensure_columns(video_prediction, LIST_TO_SAVE)
    ensure_columns(video_prediction, KEYPOINT_LIST)
    video_prediction = video_prediction.dropna(
        subset=LIST_TO_SAVE, how=worker_args["dropna_how"]
    )

    general_results = video_prediction.loc[:, LIST_TO_SAVE].copy()
    keypoint_results = video_prediction.loc[:, KEYPOINT_LIST].copy()
    general_results.to_csv(out["general_csv"], index=False)
    keypoint_results.to_csv(out["keypoint_csv"], index=False)


def shard_videos_round_robin(videos, n_shards):
    shards = [[] for _ in range(n_shards)]
    for idx, video in enumerate(videos):
        shards[idx % n_shards].append(video)
    return shards


def worker_entry(slot_name, gpu_id, assigned_videos, worker_args, summary_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    ok_count = 0
    err_count = 0
    init_error = None
    errors = []

    try:
        from feat import Detector

        detector = Detector(
            face_model="retinaface",
            au_model="xgb",
            emotion_model="resmasknet",
            facepose_model="img2pose",
            device="cuda",
            n_jobs=worker_args["n_jobs"],
            no_pose=worker_args["no_pose"],
            no_emotion=worker_args["no_emotion"],
            no_identity=worker_args["no_identity"],
        )
    except Exception:
        init_error = traceback.format_exc()
        summary_queue.put(
            {
                "slot": slot_name,
                "gpu": gpu_id,
                "assigned": len(assigned_videos),
                "ok": ok_count,
                "err": err_count,
                "init_error": init_error,
                "errors": errors,
            }
        )
        return

    for video_path in assigned_videos:
        try:
            process_one_video(video_path, detector, worker_args)
            ok_count += 1
            print(f"[OK][{slot_name}] {video_path}")
        except Exception:
            err_count += 1
            tb = traceback.format_exc()
            errors.append({"video": str(video_path), "error": tb})
            print(f"[ERR][{slot_name}] {video_path}")
            print(tb)

    summary_queue.put(
        {
            "slot": slot_name,
            "gpu": gpu_id,
            "assigned": len(assigned_videos),
            "ok": ok_count,
            "err": err_count,
            "init_error": init_error,
            "errors": errors[:5],  # avoid oversized queue payload
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

    slots = []
    for gpu in gpu_ids:
        for worker_idx in range(args.workers_per_gpu):
            slots.append((f"gpu{gpu}-w{worker_idx}", gpu))

    print(f"[INFO] input_root: {input_root}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] total videos: {len(all_videos)}")
    print(f"[INFO] pending videos: {len(pending_videos)}")
    print(f"[INFO] slots: {len(slots)} ({slots})")
    print(
        f"[INFO] batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"skip_frames={args.skip_frames}, output_size={args.output_size}, dropna_how={args.dropna_how}"
    )
    print(
        f"[INFO] no_pose={args.no_pose}, no_emotion={args.no_emotion}, "
        f"no_identity={args.no_identity}"
    )

    if not pending_videos:
        print("[INFO] Nothing to do.")
        return

    shards = shard_videos_round_robin(pending_videos, len(slots))
    for i, (slot_name, gpu) in enumerate(slots):
        print(f"[INFO] {slot_name} -> GPU {gpu}, assigned videos: {len(shards[i])}")

    summary_queue = mp.Queue()
    worker_args = {
        "output_root": str(output_root),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "skip_frames": args.skip_frames,
        "output_size": args.output_size,
        "dropna_how": args.dropna_how,
        "no_pose": args.no_pose,
        "no_emotion": args.no_emotion,
        "no_identity": args.no_identity,
        "n_jobs": 2,
    }

    print(f"[INFO] Starting workers.")

    workers = []
    for i, (slot_name, gpu) in enumerate(slots):
        p = mp.Process(
            target=worker_entry,
            args=(slot_name, gpu, shards[i], worker_args, summary_queue),
            daemon=False,
        )
        p.start()
        workers.append(p)

    summaries = []
    for _ in range(len(workers)):
        summaries.append(summary_queue.get())

    for p in workers:
        p.join()

    total_assigned = sum(s["assigned"] for s in summaries)
    total_ok = sum(s["ok"] for s in summaries)
    total_err = sum(s["err"] for s in summaries)
    total_init_err = sum(1 for s in summaries if s["init_error"] is not None)

    print(
        f"[DONE] assigned={total_assigned}, ok={total_ok}, err={total_err}, "
        f"worker_init_error={total_init_err}"
    )
    for s in summaries:
        if s["init_error"] is not None:
            print(f"[INIT_ERR][{s['slot']}]")
            print(s["init_error"])
        elif s["err"] > 0:
            print(f"[ERR_SUMMARY][{s['slot']}] err={s['err']}")
            for e in s["errors"]:
                print(f"- {e['video']}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
