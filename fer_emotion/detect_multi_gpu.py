import argparse
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from queue import Empty
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU batch FER inference for large video sets."
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
        default="output",
        help="Root directory to save outputs (csv and optional annotated videos).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help='Comma-separated gpu ids, e.g. "0,1,2,3".',
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
        help="Batch size passed to video.analyze().",
    )
    parser.add_argument(
        "--size-multiplier",
        type=int,
        default=1,
        help="size_multiplier passed to video.analyze().",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="RealTalkListening",
        help="dataset_name passed to fer.Video.",
    )
    parser.add_argument(
        "--mtcnn",
        action="store_true",
        default=True,
        help="Use MTCNN face detector for higher precision.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Whether to save annotated output videos.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Whether to save per-frame images.",
    )
    parser.add_argument(
        "--draw-annotation",
        action="store_true",
        default=True,
        help="Whether to draw annotation on the video.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process videos even when CSV already exists.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process.",
    )
    return parser.parse_args()


def get_video_paths(input_root: Path, max_videos: int = None):
    video_paths = sorted([p for p in input_root.rglob("*.mp4") if p.is_file()])
    if max_videos is not None:
        video_paths = video_paths[:max_videos]
    return video_paths


def output_csv_path(video_path: Path, output_root: Path):
    subdir = video_path.parent.stem
    video_name = video_path.stem
    return output_root / subdir / "emo_scores" / f"{video_name}.csv"


def process_one_video(video_path: Path, detector, args):
    from fer.classes import Video

    subdir = video_path.parent.stem
    video_name = video_path.stem

    # outdir = str(Path(args["output_root"]) / subdir)
    video = Video(
        str(video_path),
        dataset_name=args["dataset_name"],
        outdir=str(Path(args["output_root"])),
    )

    raw_data = video.analyze(
        detector,
        display=False,
        output=None,
        save_frames=args["save_frames"],
        save_video=args["save_video"],
        annotate_frames=args["draw_annotation"],
        zip_images=False,
        detection_box=None,
        lang="en",
        include_audio=False,
        size_multiplier=args["size_multiplier"],
        batch_size=args["batch_size"],
        use_async_io=True,
    )

    df = video.to_pandas(raw_data)
    csv_dir = Path(args["output_root"]) / subdir / "emo_scores"
    csv_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_dir / f"{video_name}.csv", index=False)


def worker_loop(gpu_id, task_queue, result_queue, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        from fer.fer import FER

        detector = FER(mtcnn=args["mtcnn"])
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

    all_videos = get_video_paths(input_root, args.max_videos)
    if not all_videos:
        print(f"[INFO] No .mp4 found under: {input_root}")
        return

    if args.force:
        pending_videos = all_videos
    else:
        pending_videos = [
            p for p in all_videos if not output_csv_path(p, output_root).exists()
        ]

    print(f"[INFO] input_root: {input_root}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] total videos: {len(all_videos)}")
    print(f"[INFO] pending videos: {len(pending_videos)}")
    print(f"[INFO] gpus: {gpu_ids}, workers_per_gpu: {args.workers_per_gpu}")

    if not pending_videos:
        print("[INFO] Nothing to do.")
        return

    task_queue = mp.Queue(maxsize=max(32, len(gpu_ids) * args.workers_per_gpu * 4))
    result_queue = mp.Queue()

    print(f"[INFO] task_queue created.")

    worker_args = {
        "output_root": str(output_root),
        "dataset_name": args.dataset_name,
        "mtcnn": args.mtcnn,
        "save_video": args.save_video,
        "save_frames": args.save_frames,
        "size_multiplier": args.size_multiplier,
        "batch_size": args.batch_size,
        "draw_annotation": args.draw_annotation,
    }

    print(f"[INFO] Starting workers.")

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

    print(f"[INFO] Started {len(workers)} workers. Begin enqueue tasks.")

    for p in tqdm(pending_videos, desc="Putting videos into task_queue"):
        task_queue.put(str(p))

    print(f"[INFO] task_queue put all videos.")

    for _ in range(len(workers)):
        task_queue.put(None)

    print(f"[INFO] task_queue put None for each worker.")

    ok_count = 0
    err_count = 0
    init_err_count = 0

    for _ in range(len(pending_videos)):
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
