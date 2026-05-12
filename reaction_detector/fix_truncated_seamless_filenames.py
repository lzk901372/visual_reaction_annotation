import argparse
from collections import defaultdict
from pathlib import Path


REACTION_NAMES = [
    "smiling",
    "laughing",
    "frowning",
    "surprised",
    "nodding",
    "head_shaking",
]


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    reaction_root_default = script_path.parent
    usable_txt_default = script_path.parent.parent / "usable_seamless_2.txt"

    parser = argparse.ArgumentParser(
        description=(
            "Fix truncated csv filenames in reaction output_seamless folders by "
            "matching against names from usable_seamless_2.txt."
        )
    )
    parser.add_argument(
        "--usable_txt",
        type=Path,
        default=usable_txt_default,
        help="Path to usable_seamless_2.txt.",
    )
    parser.add_argument(
        "--reaction_root",
        type=Path,
        default=reaction_root_default,
        help="Root directory containing six reaction folders.",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="output_seamless",
        help="Sub-directory name under each reaction folder.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview rename operations without changing files.",
    )
    parser.add_argument(
        "--print_each_rename",
        action="store_true",
        help="Print every rename operation (can generate very large logs).",
    )
    return parser.parse_args()


def load_expected_stems(usable_txt: Path) -> set[str]:
    expected_stems: set[str] = set()
    with usable_txt.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            video_path_str = line.split(",", 1)[0].strip()
            if not video_path_str:
                continue
            video_path = Path(video_path_str)
            if video_path.suffix.lower() != ".mp4":
                continue
            expected_stems.add(video_path.stem)
    return expected_stems


def split_prefix_end(stem: str) -> tuple[str, str] | None:
    if "_" not in stem:
        return None
    prefix, end_token = stem.rsplit("_", 1)
    if not prefix or not end_token:
        return None
    return prefix + "_", end_token


def pick_target_stem(
    existing_stem: str,
    expected_by_prefix: dict[str, set[str]],
) -> tuple[str | None, str]:
    split = split_prefix_end(existing_stem)
    if split is None:
        return None, "invalid_filename_pattern"

    prefix, existing_end = split
    candidates = expected_by_prefix.get(prefix)
    if not candidates:
        return None, "prefix_not_in_expected"

    if existing_end in candidates:
        return None, "already_complete"

    matched = [
        expected_end
        for expected_end in candidates
        if len(existing_end) < len(expected_end) and expected_end.startswith(existing_end)
    ]
    if not matched:
        return None, "not_truncated_or_no_match"
    if len(matched) > 1:
        return None, "ambiguous_match"

    return prefix + matched[0], "truncated"


def fix_one_folder(
    folder: Path,
    expected_stems: set[str],
    dry_run: bool,
    print_each_rename: bool,
) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)

    expected_by_prefix: dict[str, set[str]] = defaultdict(set)
    for stem in expected_stems:
        split = split_prefix_end(stem)
        if split is None:
            continue
        prefix, end_token = split
        expected_by_prefix[prefix].add(end_token)

    for file_path in folder.glob("*.csv"):
        stats["total_csv"] += 1
        stem = file_path.stem

        if stem in expected_stems:
            stats["already_complete"] += 1
            continue

        target_stem, reason = pick_target_stem(stem, expected_by_prefix)
        if target_stem is None:
            stats[reason] += 1
            continue

        target_path = file_path.with_name(f"{target_stem}.csv")
        if target_path.exists():
            stats["target_exists"] += 1
            continue

        stats["rename_candidates"] += 1
        if print_each_rename:
            print(f"[RENAME] {file_path.name} -> {target_path.name}")
        if not dry_run:
            file_path.rename(target_path)
            stats["renamed"] += 1
        else:
            stats["dry_run_renamed"] += 1

    return stats


def main() -> None:
    args = parse_args()
    if not args.usable_txt.exists():
        raise FileNotFoundError(f"usable_txt not found: {args.usable_txt}")
    if not args.reaction_root.exists():
        raise FileNotFoundError(f"reaction_root not found: {args.reaction_root}")

    expected_stems = load_expected_stems(args.usable_txt)
    if not expected_stems:
        raise RuntimeError("No expected stems loaded from usable list.")

    print(f"Loaded expected clip names: {len(expected_stems)}")
    print(f"usable list: {args.usable_txt}")
    print(f"reaction root: {args.reaction_root}")
    print(f"mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"print_each_rename: {args.print_each_rename}")

    grand_totals: dict[str, int] = defaultdict(int)

    for reaction in REACTION_NAMES:
        folder = args.reaction_root / reaction / args.output_dir_name
        if not folder.exists():
            print(f"[SKIP] missing folder: {folder}")
            grand_totals["missing_folder"] += 1
            continue

        print(f"\n=== {reaction} | {folder} ===")
        stats = fix_one_folder(
            folder=folder,
            expected_stems=expected_stems,
            dry_run=args.dry_run,
            print_each_rename=args.print_each_rename,
        )
        for k, v in stats.items():
            grand_totals[k] += v
        print(
            "stats: "
            + ", ".join(
                f"{k}={stats[k]}"
                for k in [
                    "total_csv",
                    "already_complete",
                    "rename_candidates",
                    "renamed",
                    "dry_run_renamed",
                    "ambiguous_match",
                    "target_exists",
                    "prefix_not_in_expected",
                    "not_truncated_or_no_match",
                    "invalid_filename_pattern",
                ]
                if k in stats
            )
        )

    print("\n=== GRAND TOTAL ===")
    for key in sorted(grand_totals):
        print(f"{key}: {grand_totals[key]}")


if __name__ == "__main__":
    main()
