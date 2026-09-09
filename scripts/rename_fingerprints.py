#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import zlib
from pathlib import Path


FILE_PATTERN = re.compile(
    r"^(?P<prefix>(?P<person>0\d{2})_(?P<finger>[^_]+))_(?P<impression>[^_]+)\.(?P<ext>[a-zA-Z0-9]+)$"
)


def natural_sort_key(s: str) -> list[int | str]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def generate_prefix_mapping(
    prefixes: list[str],
    strategy: str = "sequential",
    start_index: int = 1,
) -> dict[str, int]:
    sorted_prefixes = sorted(set(prefixes), key=natural_sort_key)
    mapping: dict[str, int] = {}

    if strategy == "sequential":
        for i, prefix in enumerate(sorted_prefixes, start=start_index):
            mapping[prefix] = i
    elif strategy == "hash":
        assigned_ids: set[int] = set()
        for prefix in sorted_prefixes:
            candidate_id = zlib.crc32(prefix.encode("utf-8")) % 100000
            while candidate_id in assigned_ids:
                candidate_id += 1
            assigned_ids.add(candidate_id)
            mapping[prefix] = candidate_id
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return mapping


def find_renames(
    target_directory: Path,
    recursive: bool = False,
    extensions: set[str] | None = None,
    strategy: str = "sequential",
    start_index: int = 1,
) -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    file_iterator = target_directory.rglob("*") if recursive else target_directory.glob("*")
    matched_files: list[tuple[Path, re.Match]] = []

    for path in file_iterator:
        if not path.is_file():
            continue

        if extensions and path.suffix.lower().lstrip(".") not in extensions:
            continue

        match = FILE_PATTERN.match(path.name)
        if match:
            matched_files.append((path, match))

    if not matched_files:
        return [], {}

    all_prefixes = [m.group("prefix") for _, m in matched_files]
    prefix_mapping = generate_prefix_mapping(
        all_prefixes, strategy=strategy, start_index=start_index
    )

    planned_renames: list[tuple[Path, Path]] = []
    for path, match in matched_files:
        prefix = match.group("prefix")
        unique_finger_id = prefix_mapping[prefix]
        impression = match.group("impression")
        ext = match.group("ext")

        new_filename = f"{unique_finger_id}_{impression}.{ext}"
        destination = path.parent / new_filename
        planned_renames.append((path, destination))

    return planned_renames, prefix_mapping


def validate_renames(planned_renames: list[tuple[Path, Path]]) -> tuple[list[str], list[str]]:
    destination_counts: dict[Path, list[Path]] = {}
    collisions: list[str] = []
    existing_conflicts: list[str] = []

    for source, destination in planned_renames:
        destination_counts.setdefault(destination, []).append(source)

    for destination, sources in destination_counts.items():
        if len(sources) > 1:
            sources_str = ", ".join(s.name for s in sources)
            collisions.append(f"Multiple files map to '{destination.name}': {sources_str}")

        source_set = set(sources)
        if destination.exists() and destination not in source_set:
            existing_conflicts.append(
                f"Destination file already exists on disk: '{destination}'"
            )

    return collisions, existing_conflicts


def save_mapping_file(mapping: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["original_prefix", "assigned_id"])
            for prefix, assigned_id in mapping.items():
                writer.writerow([prefix, assigned_id])
    print(f"Saved prefix mapping to: '{output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename fingerprint images from <person_id>_<finger>_<impression> to <unique_id>_<impression>."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to directory containing images to rename.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, script runs in dry-run mode.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively rename matching files in subdirectories.",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        help="Optional extensions filter (e.g. -e tif png bmp).",
    )
    parser.add_argument(
        "--strategy",
        choices=["sequential", "hash"],
        default="sequential",
        help="Strategy to assign IDs to unique '<person_id>_<finger>' prefixes (default: sequential).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting number for sequential ID assignment (default: 1).",
    )
    parser.add_argument(
        "--save-mapping",
        type=Path,
        help="Path to save mapping file (.json or .csv).",
    )

    args = parser.parse_args()
    directory = args.directory.resolve()

    if not directory.is_dir():
        print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
        sys.exit(1)

    extensions = {ext.lower().lstrip(".") for ext in args.extensions} if args.extensions else None
    planned_renames, prefix_mapping = find_renames(
        directory,
        recursive=args.recursive,
        extensions=extensions,
        strategy=args.strategy,
        start_index=args.start_index,
    )

    if not planned_renames:
        print(f"No files matching pattern '0XX_<finger>_<impression>.<ext>' found in '{directory}'.")
        return

    collisions, conflicts = validate_renames(planned_renames)

    if collisions or conflicts:
        print("Safety check failed! Collisions or existing target files detected:", file=sys.stderr)
        for issue in collisions:
            print(f"  - Collision: {issue}", file=sys.stderr)
        for issue in conflicts:
            print(f"  - Conflict: {issue}", file=sys.stderr)
        print("\nAborting rename operation to avoid data loss.", file=sys.stderr)
        sys.exit(1)

    mode_label = "[APPLY]" if args.apply else "[DRY-RUN]"
    print(f"{mode_label} Found {len(planned_renames)} files ({len(prefix_mapping)} unique fingers) in '{directory}':\n")

    print("Sample prefix mappings:")
    for prefix, assigned_id in list(prefix_mapping.items())[:10]:
        print(f"  Prefix '{prefix}' -> {assigned_id}")
    if len(prefix_mapping) > 10:
        print(f"  ... and {len(prefix_mapping) - 10} more.")
    print()

    # Show first 15 renames
    for source, destination in planned_renames[:15]:
        print(f"  {source.name}  ->  {destination.name}")
    if len(planned_renames) > 15:
        print(f"  ... and {len(planned_renames) - 15} more files.")

    if args.apply:
        for source, destination in planned_renames:
            os.rename(source, destination)
        print(f"\nSuccessfully renamed {len(planned_renames)} files.")
        if args.save_mapping:
            save_mapping_file(prefix_mapping, args.save_mapping)
    else:
        print(f"\nDry-run complete. Run with '--apply' to perform the {len(planned_renames)} renames.")
        if args.save_mapping:
            save_mapping_file(prefix_mapping, args.save_mapping)


if __name__ == "__main__":
    main()
