#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import sys
from PIL import Image


def organize_and_convert_bmp(
    target_directory: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    if not target_directory.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_directory}")
    if not target_directory.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_directory}")

    bmp_directory = target_directory / "bmp"
    tif_directory = target_directory / "tif"

    candidate_files = [
        item
        for item in sorted(target_directory.iterdir())
        if item.is_file() and item.suffix.lower() == ".bmp" and not item.name.startswith(".")
    ]

    if not candidate_files:
        return 0, 0

    if not dry_run:
        bmp_directory.mkdir(parents=True, exist_ok=True)
        tif_directory.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    moved_count = 0

    for file_path in candidate_files:
        tif_target_path = tif_directory / f"{file_path.stem}.tif"
        bmp_target_path = bmp_directory / file_path.name

        if dry_run:
            print(f"[DRY-RUN] Convert: {file_path.name} -> tif/{tif_target_path.name}")
            print(f"[DRY-RUN] Move:    {file_path.name} -> bmp/{bmp_target_path.name}")
            converted_count += 1
            moved_count += 1
            continue

        with Image.open(file_path) as image:
            dpi = image.info.get("dpi")
            if dpi:
                image.save(tif_target_path, format="TIFF", dpi=dpi)
            else:
                image.save(tif_target_path, format="TIFF")
        converted_count += 1

        shutil.move(str(file_path), str(bmp_target_path))
        moved_count += 1

    return converted_count, moved_count


import os


def find_directories_with_bmp(root_directory: Path) -> list[Path]:
    directories_with_bmp: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root_directory):
        directory_names[:] = [
            name
            for name in directory_names
            if name.lower() not in {"bmp", "tif"} and not name.startswith(".")
        ]
        has_bmp = any(
            name.lower().endswith(".bmp") and not name.startswith(".")
            for name in file_names
        )
        if has_bmp:
            directories_with_bmp.append(Path(current_root))
    return sorted(directories_with_bmp)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize BMP files into a 'bmp' subfolder and convert them to TIFF in a 'tif' subfolder."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        type=Path,
        help="Path to one or more directories containing BMP files to process.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search recursively for all directories containing BMP files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview conversion and move operations without modifying the filesystem.",
    )
    return parser.parse_args(arguments)


def main() -> int:
    args = parse_arguments(sys.argv[1:])

    target_directories: list[Path] = []
    for folder in args.folders:
        if args.recursive:
            discovered = find_directories_with_bmp(folder)
            target_directories.extend(discovered)
        else:
            target_directories.append(folder)

    total_converted = 0
    total_moved = 0

    for directory in target_directories:
        try:
            converted, moved = organize_and_convert_bmp(directory, dry_run=args.dry_run)
            action_prefix = "[DRY-RUN] " if args.dry_run else ""
            print(
                f"{action_prefix}{directory}: {converted} BMP file(s) converted to TIF, {moved} moved to bmp/"
            )
            total_converted += converted
            total_moved += moved
        except Exception as error:
            print(f"Error processing '{directory}': {error}", file=sys.stderr)
            return 1

    action_summary = "would be" if args.dry_run else "were"
    print(f"Total: {total_converted} file(s) converted, {total_moved} file(s) {action_summary} moved across {len(target_directories)} directory(ies).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
