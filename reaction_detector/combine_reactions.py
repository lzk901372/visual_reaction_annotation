import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REACTION_COLUMNS: list[tuple[str, str]] = [
    ("smiling", "smiling_intensity"),
    ("laughing", "laughing_intensity"),
    ("frowning", "frowning_intensity"),
    ("surprised", "surprised_intensity"),
    ("nodding", "nodding_intensity"),
    ("head_shaking", "head_shaking_intensity"),
]


@dataclass
class ProcessResult:
    rel_csv_path: Path
    ok: bool
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine 6 reaction frame-wise intensity csv files for <dataset_name> and "
            "enforce one active reaction (max intensity > 0) per frame."
        )
    )
    parser.add_argument(
        "--usable_txt",
        type=Path,
        default=Path("/data/zikai/Data/RealTalkListening/reaction_detect/usable_seamless_2.txt"),
        help="Path to usable_<dataset_name>.txt.",
    )
    parser.add_argument(
        "--reaction_root",
        type=Path,
        default=Path("/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector"),
        help="Root path containing six reaction folders.",
    )
    parser.add_argument(
        "--original_root",
        type=Path,
        default=Path("/data/zikai/Data/Seamless"),
        help="Root path of source videos for building relative output structure.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector/combined_seamless"
        ),
        help="Output root directory for combined csv files.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="seamless",
        help="Dataset name.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Worker threads for parallel csv processing.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip writing files that already exist in output_dir.",
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help="Raise error if any reaction csv is missing instead of skipping.",
    )
    return parser.parse_args()


def parse_usable_list(usable_txt: Path, original_root: Path) -> list[Path]:
    rel_csv_paths: list[Path] = []
    seen: set[Path] = set()

    with usable_txt.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if not parts:
                continue

            video_path = Path(parts[0])
            if video_path.suffix.lower() != ".mp4":
                continue

            # Keep the same structure as Original/.../*.mp4 -> .../*.csv
            try:
                rel_video_path = video_path.relative_to(original_root)
                rel_csv_path = rel_video_path.with_suffix(".csv")
            except ValueError:
                if len(video_path.parts) < 2:
                    continue
                rel_csv_path = Path(video_path.parts[-2]) / f"{video_path.stem}.csv"
            if rel_csv_path in seen:
                continue
            seen.add(rel_csv_path)
            rel_csv_paths.append(rel_csv_path)
    return rel_csv_paths


def read_reaction_csv(csv_path: Path, expected_col: str) -> dict[int, float]:
    frame_to_score: dict[int, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return frame_to_score

        score_col = expected_col
        if score_col not in reader.fieldnames:
            candidates = [c for c in reader.fieldnames if c != "frame_index"]
            if len(candidates) == 1:
                score_col = candidates[0]
            else:
                raise ValueError(
                    f"Cannot determine score column in {csv_path}. fieldnames={reader.fieldnames}"
                )

        for row in reader:
            if "frame_index" not in row or score_col not in row:
                continue
            try:
                frame_idx = int(float(row["frame_index"]))
                score = float(row[score_col])
            except (TypeError, ValueError):
                continue
            frame_to_score[frame_idx] = score

    return frame_to_score


def combine_one_file(
    rel_csv_path: Path,
    reaction_root: Path,
    output_dir: Path,
    skip_existing: bool,
    strict_missing: bool,
    dataset_name: str = "realtalk"
) -> ProcessResult:
    out_path = output_dir / rel_csv_path
    if skip_existing and out_path.exists():
        return ProcessResult(rel_csv_path=rel_csv_path, ok=True, message="skipped existing")

    per_reaction_scores: dict[str, dict[int, float]] = {}
    missing: list[str] = []

    for reaction_name, score_col in REACTION_COLUMNS:
        src_csv = reaction_root / reaction_name / f"output_{dataset_name}" / rel_csv_path
        # print(src_csv)
        # exit()
        if not src_csv.exists():
            missing.append(str(src_csv))
            continue
        try:
            per_reaction_scores[reaction_name] = read_reaction_csv(src_csv, score_col)
        except Exception as e:  # pragma: no cover - defensive
            return ProcessResult(
                rel_csv_path=rel_csv_path,
                ok=False,
                message=f"failed reading {src_csv}: {e}",
            )

    if missing:
        msg = f"missing {len(missing)} reaction csv(s)"
        if strict_missing:
            return ProcessResult(rel_csv_path=rel_csv_path, ok=False, message=f"{msg}: {missing}")
        return ProcessResult(rel_csv_path=rel_csv_path, ok=False, message=msg)

    frame_indices: set[int] = set()
    for reaction_name, _ in REACTION_COLUMNS:
        frame_indices.update(per_reaction_scores[reaction_name].keys())
    ordered_frames = sorted(frame_indices)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index"] + [x[0] for x in REACTION_COLUMNS])

        for frame_idx in ordered_frames:
            scores = [
                per_reaction_scores[reaction_name].get(frame_idx, 0.0)
                for reaction_name, _ in REACTION_COLUMNS
            ]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_score = scores[best_idx]

            merged_scores = [0.0] * len(scores)
            if best_score > 0:
                merged_scores[best_idx] = best_score

            writer.writerow([frame_idx] + [f"{v:.6f}" for v in merged_scores])

    return ProcessResult(rel_csv_path=rel_csv_path, ok=True, message="ok")


def main() -> None:
    args = parse_args()

    if not args.usable_txt.exists():
        raise FileNotFoundError(f"usable_txt not found: {args.usable_txt}")
    if not args.reaction_root.exists():
        raise FileNotFoundError(f"reaction_root not found: {args.reaction_root}")

    rel_csv_paths = parse_usable_list(args.usable_txt, args.original_root)
    total = len(rel_csv_paths)
    if total == 0:
        print(f"No usable {args.dataset_name} entries found.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Total usable clips: {total}")
    print(f"Output dir: {args.output_dir}")

    ok_count = 0
    fail_count = 0
    fail_examples: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = [
            executor.submit(
                combine_one_file,
                rel_csv_path,
                args.reaction_root,
                args.output_dir,
                args.skip_existing,
                args.strict_missing,
                args.dataset_name,
            )
            for rel_csv_path in rel_csv_paths
        ]
        for idx, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            if result.ok:
                ok_count += 1
            else:
                fail_count += 1
                if len(fail_examples) < 10:
                    fail_examples.append(f"{result.rel_csv_path}: {result.message}")

            if idx % 500 == 0 or idx == total:
                print(
                    f"Progress {idx}/{total} | ok={ok_count} fail={fail_count}",
                    flush=True,
                )

    print("Done.")
    print(f"ok={ok_count}, fail={fail_count}")
    if fail_examples:
        print("Failure examples:")
        for item in fail_examples:
            print(f"- {item}")


if __name__ == "__main__":
    main()
