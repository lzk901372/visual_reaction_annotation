import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build usable_<dataset_name>.txt by filtering report A with list B."
    )
    parser.add_argument(
        "--alignment_report",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/nod_headshake/face_detection/output_seamless_alignment_report_seamless.txt",
        help="Path to file A (alignment report).",
    )
    parser.add_argument(
        "--incomplete_list",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/fer_emotion/incomplete_csv_seamless.txt",
        help="Path to file B (incomplete csv list).",
    )
    parser.add_argument(
        "--output_txt",
        default="/data/zikai/Data/RealTalkListening/reaction_detect/usable_seamless.txt",
        help="Output file path.",
    )
    return parser.parse_args()


def load_excluded_videos_from_b(path_b: Path) -> set[str]:
    excluded: set[str] = set()
    with path_b.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            video_path = row[0].strip()
            if video_path:
                excluded.add(video_path)
    return excluded


def build_output_paths(video_path: str) -> tuple[str, str, str]:
    video = Path(video_path)

    fer_csv = str(video).replace(
        "Original", "reaction_detect/fer_emotion/output", 1
    )
    fer_csv = str(Path(fer_csv).with_suffix(".csv"))

    nod_pkl = str(video).replace(
        "Original", "reaction_detect/nod_headshake/face_detection/output_realtalk", 1
    )
    nod_pkl = str(Path(nod_pkl).with_suffix(".pkl"))

    return str(video), fer_csv, nod_pkl


def main() -> None:
    args = parse_args()
    path_a = Path(args.alignment_report)
    path_b = Path(args.incomplete_list)
    out_path = Path(args.output_txt)

    if not path_a.exists():
        raise FileNotFoundError(f"File A not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"File B not found: {path_b}")

    excluded_by_b = load_excluded_videos_from_b(path_b)

    kept_rows: list[str] = []
    total_a = 0
    removed_by_b = 0
    removed_by_rate = 0

    with path_a.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_path", "detection_rate"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "File A must contain columns: video_path,detection_rate"
            )

        for row in reader:
            total_a += 1
            video_path = (row.get("video_path") or "").strip()
            if not video_path:
                continue

            if video_path in excluded_by_b:
                removed_by_b += 1
                continue

            detection_rate_str = (row.get("detection_rate") or "").strip()
            try:
                detection_rate = float(detection_rate_str)
            except ValueError:
                removed_by_rate += 1
                continue

            if detection_rate != 1.0:
                removed_by_rate += 1
                continue

            video, fer_csv, nod_pkl = build_output_paths(video_path)
            kept_rows.append(f"{video},{fer_csv},{nod_pkl}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(kept_rows))

    print(f"Total rows in A: {total_a}")
    print(f"Removed by B: {removed_by_b}")
    print(f"Removed by detection_rate != 1.0: {removed_by_rate}")
    print(f"Kept rows: {len(kept_rows)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
