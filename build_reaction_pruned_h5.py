import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import h5py. Please use a Python environment with a compatible "
        f"numpy/h5py build. Original error: {e}"
    ) from e


REACTION_COLUMNS = [
    "smiling",
    "laughing",
    "frowning",
    "surprised",
    "nodding",
    "head_shaking",
]
CHUNK_NAME_PATTERN = re.compile(r"^chunk_\d+$")


@dataclass
class InjectStats:
    written_clips: int = 0
    skipped_clips: int = 0
    failed_clips: int = 0
    written_chunks: int = 0


@dataclass
class PruneStats:
    visited_groups: int = 0
    visited_clip_groups: int = 0
    deleted_clip_groups: int = 0
    deleted_chunk_datasets: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new HDF5 by copying input, inject reaction scores from combined_<dataset_name>, "
            "then prune all non-reaction groups to keep only clips/chunks existing in /reaction."
        )
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="seamless",
        help="Dataset name.",
    )
    parser.add_argument(
        "--input_h5",
        type=Path,
        default=Path("/data/zikai/Data/H5_files/hdf5_LIA/Seamless_v3_listening_preprocessed.h5"),
        help="Source HDF5 file.",
    )
    parser.add_argument(
        "--output_h5",
        type=Path,
        default=Path(
            "/data/zikai/Data/H5_files/hdf5_reaction/Seamless_v3_listening_preprocessed_reaction_pruned.h5"
        ),
        help="Output HDF5 file. Script copies input_h5 to this path first.",
    )
    parser.add_argument(
        "--combined_root",
        type=Path,
        default=Path(
            "/data/zikai/Data/RealTalkListening/reaction_detect/reaction_detector/combined_seamless"
        ),
        help="Root directory containing combined reaction csv files.",
    )
    parser.add_argument(
        "--reaction_group_name",
        type=str,
        default="reaction",
        help="Top-level group name for injected reaction chunks.",
    )
    parser.add_argument(
        "--anchor_group_name",
        type=str,
        default="audio",
        help="Reference group used to align chunk count (default: audio).",
    )
    parser.add_argument(
        "--disable_anchor_match",
        action="store_true",
        help="If set, do not align chunk count to anchor group; use csv-only chunk count.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=60,
        help="Semantic frame chunk length for reaction data. Default: 60.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "none"],
        help="Compression used for reaction datasets.",
    )
    parser.add_argument(
        "--gzip_level",
        type=int,
        default=6,
        help="Gzip level [0-9], only used when --compression=gzip.",
    )
    parser.add_argument(
        "--h5_chunk_layout",
        type=str,
        default="full",
        choices=["full", "frame"],
        help=(
            "HDF5 internal chunking for each reaction dataset of shape (chunk_size, 6): "
            "full -> chunks=(chunk_size,6); frame -> chunks=(1,6)."
        ),
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Overwrite output_h5 if it already exists.",
    )
    parser.add_argument(
        "--keep_existing_reaction",
        action="store_true",
        help="Keep existing /reaction group in copied file and skip already existing clips.",
    )
    return parser.parse_args()


def clip_key_from_csv_path(csv_path: Path, combined_root: Path, dataset_name: str) -> str:
    rel = csv_path.relative_to(combined_root)
    # print(rel)
    if dataset_name == "realtalk":
        if len(rel.parts) < 2:
            raise ValueError(f"Unexpected csv path under combined_root: {csv_path}")
        video_id = rel.parts[-2]
        clip_id = Path(rel.parts[-1]).stem
        return f"{video_id}_{clip_id}"
    elif dataset_name == "seamless":
        if len(rel.parts) < 1:
            raise ValueError(f"Unexpected csv path under combined_root: {csv_path}")
        video_id = Path(rel.parts[-1]).stem
        # clip_id = Path(rel.parts[-1]).stem
        return video_id
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def load_csv_matrix(csv_path: Path) -> np.ndarray:
    rows: list[tuple[int, list[float]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        missing_cols = [c for c in ["frame_index"] + REACTION_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"CSV missing columns: {missing_cols}")
        for row in reader:
            frame_idx = int(float(row["frame_index"]))
            values = [float(row[col]) for col in REACTION_COLUMNS]
            rows.append((frame_idx, values))

    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    rows.sort(key=lambda x: x[0])
    return np.asarray([x[1] for x in rows], dtype=np.float32)


def inject_reaction(
    dataset_name: str,
    h5_file: h5py.File,
    combined_root: Path,
    reaction_group_name: str,
    anchor_group_name: str,
    chunk_size: int,
    disable_anchor_match: bool,
    keep_existing_reaction: bool,
    compression: str,
    gzip_level: int,
    h5_chunk_layout: str,
) -> InjectStats:
    stats = InjectStats()
    csv_files = sorted(combined_root.rglob("*.csv"))
    if not csv_files:
        print("No csv files found under combined_root.")
        return stats

    if reaction_group_name in h5_file and not keep_existing_reaction:
        del h5_file[reaction_group_name]
    reaction_group = h5_file.require_group(reaction_group_name)
    anchor_group = h5_file.get(anchor_group_name)
    match_anchor = (not disable_anchor_match) and isinstance(anchor_group, h5py.Group)

    print(f"CSV files found: {len(csv_files)}")
    print("Reaction dtype: float16")
    if h5_chunk_layout == "full":
        print(f"Reaction HDF5 chunks: ({chunk_size}, 6)")
    else:
        print("Reaction HDF5 chunks: (1, 6)")
    if compression == "gzip":
        print(f"Reaction compression: gzip(level={gzip_level})")
    else:
        print("Reaction compression: none")
    if not match_anchor:
        print("Anchor alignment: disabled or anchor group missing.")
    else:
        print(f"Anchor alignment: enabled via /{anchor_group_name}")

    for idx, csv_path in enumerate(csv_files, start=1):
        try:
            clip_key = clip_key_from_csv_path(csv_path, combined_root, dataset_name)

            if keep_existing_reaction and clip_key in reaction_group:
                print(f"Skipping clip: {clip_key} (already exists)")
                stats.skipped_clips += 1
                continue

            if match_anchor and clip_key not in anchor_group:
                print(f"Skipping clip: {clip_key} (not in anchor group {anchor_group_name})")
                stats.skipped_clips += 1
                continue

            matrix = load_csv_matrix(csv_path)
            csv_chunk_count = matrix.shape[0] // chunk_size
            if csv_chunk_count <= 0:
                print(f"Skipping clip: {clip_key} (csv chunk count <= 0)")
                stats.skipped_clips += 1
                continue

            target_chunk_count = csv_chunk_count
            if match_anchor:
                anchor_chunk_count = len(anchor_group[clip_key])
                target_chunk_count = min(csv_chunk_count, anchor_chunk_count)
            if target_chunk_count <= 0:
                print(f"Skipping clip: {clip_key} (target chunk count <= 0)")
                stats.skipped_clips += 1
                continue

            if clip_key in reaction_group and not keep_existing_reaction:
                del reaction_group[clip_key]
            clip_group = reaction_group.create_group(clip_key)

            h5_chunks = (chunk_size, 6) if h5_chunk_layout == "full" else (1, 6)
            for i in range(target_chunk_count):
                start = i * chunk_size
                end = start + chunk_size
                chunk_data = matrix[start:end, :].astype(np.float16, copy=False)
                dataset_kwargs = {
                    "data": chunk_data,
                    "dtype": np.float16,
                    "chunks": h5_chunks,
                }
                if compression == "gzip":
                    dataset_kwargs["compression"] = "gzip"
                    dataset_kwargs["compression_opts"] = gzip_level
                clip_group.create_dataset(f"chunk_{i:03d}", **dataset_kwargs)
                stats.written_chunks += 1

            stats.written_clips += 1
        except Exception:
            print(f"Failed to inject clip: {clip_key}")
            stats.failed_clips += 1

        if idx % 500 == 0 or idx == len(csv_files):
            print(
                "Inject progress "
                f"{idx}/{len(csv_files)} | written={stats.written_clips} "
                f"skipped={stats.skipped_clips} failed={stats.failed_clips}",
                flush=True,
            )
    return stats


def build_reaction_index(reaction_group: h5py.Group) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for clip_key in reaction_group.keys():
        clip_obj = reaction_group[clip_key]
        if not isinstance(clip_obj, h5py.Group):
            continue
        keep = {
            name
            for name in clip_obj.keys()
            if CHUNK_NAME_PATTERN.match(name) and isinstance(clip_obj[name], h5py.Dataset)
        }
        index[clip_key] = keep
    return index


def prune_group_by_reaction(
    group: h5py.Group,
    reaction_index: dict[str, set[str]],
    stats: PruneStats,
) -> None:
    stats.visited_groups += 1
    for clip_key in list(group.keys()):
        obj = group[clip_key]
        if not isinstance(obj, h5py.Group):
            continue

        stats.visited_clip_groups += 1
        if clip_key not in reaction_index:
            del group[clip_key]
            stats.deleted_clip_groups += 1
            continue

        keep_chunks = reaction_index[clip_key]
        for dataset_name in list(obj.keys()):
            if not CHUNK_NAME_PATTERN.match(dataset_name):
                continue
            child = obj[dataset_name]
            if not isinstance(child, h5py.Dataset):
                continue
            if dataset_name in keep_chunks:
                continue
            del obj[dataset_name]
            stats.deleted_chunk_datasets += 1

        remaining_chunk = 0
        for name in obj.keys():
            if CHUNK_NAME_PATTERN.match(name) and isinstance(obj[name], h5py.Dataset):
                remaining_chunk += 1
        if remaining_chunk == 0:
            del group[clip_key]
            stats.deleted_clip_groups += 1


def prune_by_reaction(h5_file: h5py.File, reaction_group_name: str) -> PruneStats:
    if reaction_group_name not in h5_file:
        raise KeyError(f"Missing '/{reaction_group_name}' for pruning reference.")
    reaction_group = h5_file[reaction_group_name]
    if not isinstance(reaction_group, h5py.Group):
        raise TypeError(f"'/{reaction_group_name}' is not a group.")

    reaction_index = build_reaction_index(reaction_group)
    print(f"Reference clips in /{reaction_group_name}: {len(reaction_index)}")

    stats = PruneStats()
    for group_name in list(h5_file.keys()):
        if group_name == reaction_group_name:
            continue
        obj = h5_file[group_name]
        if not isinstance(obj, h5py.Group):
            continue
        print(f"Pruning group: /{group_name}")
        prune_group_by_reaction(obj, reaction_index, stats)
    return stats


def main() -> None:
    args = parse_args()

    if not args.input_h5.exists():
        raise FileNotFoundError(f"input_h5 not found: {args.input_h5}")
    if not args.combined_root.exists():
        raise FileNotFoundError(f"combined_root not found: {args.combined_root}")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0 <= args.gzip_level <= 9):
        raise ValueError("gzip_level must be in [0, 9]")

    if args.output_h5.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"output_h5 already exists: {args.output_h5}. Use --overwrite_output to replace it."
        )
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    if args.output_h5.exists() and args.overwrite_output:
        args.output_h5.unlink()

    print(f"Copying HDF5:\n  from: {args.input_h5}\n  to:   {args.output_h5}")
    shutil.copy2(args.input_h5, args.output_h5)

    with h5py.File(args.output_h5, "r+") as h5f:
        inject_stats = inject_reaction(
            dataset_name=args.dataset_name,
            h5_file=h5f,
            combined_root=args.combined_root,
            reaction_group_name=args.reaction_group_name,
            anchor_group_name=args.anchor_group_name,
            chunk_size=args.chunk_size,
            disable_anchor_match=args.disable_anchor_match,
            keep_existing_reaction=args.keep_existing_reaction,
            compression=args.compression,
            gzip_level=args.gzip_level,
            h5_chunk_layout=args.h5_chunk_layout,
        )
        prune_stats = prune_by_reaction(h5f, args.reaction_group_name)

    print("Completed.")
    print(
        "Inject stats: "
        f"written_clips={inject_stats.written_clips}, "
        f"skipped_clips={inject_stats.skipped_clips}, "
        f"failed_clips={inject_stats.failed_clips}, "
        f"written_chunks={inject_stats.written_chunks}"
    )
    print(
        "Prune stats: "
        f"visited_groups={prune_stats.visited_groups}, "
        f"visited_clip_groups={prune_stats.visited_clip_groups}, "
        f"deleted_clip_groups={prune_stats.deleted_clip_groups}, "
        f"deleted_chunk_datasets={prune_stats.deleted_chunk_datasets}"
    )


if __name__ == "__main__":
    main()
