#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import re
import shutil
import sys


class TransferMode(StrEnum):
    COPY = "copy"
    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    MOVE = "move"


class UnmatchedAction(StrEnum):
    SKIP = "skip"
    COPY = "copy"
    ERROR = "error"


@dataclass(frozen=True)
class SplitRule:
    subfolder_name: str
    allowed_numbers: set[int]


def parse_range_specification(range_text: str) -> set[int]:
    matched_numbers: set[int] = set()
    cleaned_range_text = range_text.strip()
    if not cleaned_range_text:
        raise ValueError("Range specification cannot be empty.")

    tokens = [token.strip() for token in cleaned_range_text.split(",") if token.strip()]
    for token in tokens:
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"Invalid range segment '{token}'. Expected format: 'start-end'.")
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(f"Start index {start} cannot exceed end index {end} in range '{token}'.")
            matched_numbers.update(range(start, end + 1))
        elif ".." in token:
            parts = token.split("..")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"Invalid range segment '{token}'. Expected format: 'start..end'.")
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(f"Start index {start} cannot exceed end index {end} in range '{token}'.")
            matched_numbers.update(range(start, end + 1))
        elif token.isdigit():
            matched_numbers.add(int(token))
        else:
            raise ValueError(f"Invalid token '{token}' in range specification. Expected integer or range.")

    return matched_numbers


def parse_split_rule(rule_text: str) -> SplitRule:
    delimiter = "=" if "=" in rule_text else (":" if ":" in rule_text else None)
    if not delimiter:
        raise ValueError(f"Invalid split rule '{rule_text}'. Expected '<subfolder>=<range>' (e.g. 'gallery=1-4').")

    subfolder_name, range_specification = rule_text.split(delimiter, 1)
    subfolder_name = subfolder_name.strip()
    if not subfolder_name:
        raise ValueError(f"Missing subfolder name in rule '{rule_text}'.")

    if "/" in subfolder_name or "\\" in subfolder_name or ".." in subfolder_name:
        raise ValueError(f"Subfolder name '{subfolder_name}' cannot contain path separators or parent directory references.")

    allowed_numbers = parse_range_specification(range_specification)
    return SplitRule(subfolder_name=subfolder_name, allowed_numbers=allowed_numbers)


def validate_split_rules(rules: list[SplitRule]) -> None:
    seen_subfolders: set[str] = set()
    number_assignment: dict[int, str] = {}

    for rule in rules:
        if rule.subfolder_name in seen_subfolders:
            raise ValueError(f"Duplicate rule definition for subfolder '{rule.subfolder_name}'.")
        seen_subfolders.add(rule.subfolder_name)

        for number in rule.allowed_numbers:
            if number in number_assignment:
                conflicting_subfolder = number_assignment[number]
                raise ValueError(
                    f"Conflicting rules: number {number} is assigned to both '{conflicting_subfolder}' and '{rule.subfolder_name}'."
                )
            number_assignment[number] = rule.subfolder_name


def extract_identifier(file_path: Path, pattern: re.Pattern[str] | None = None) -> int | None:
    if pattern:
        match = pattern.search(file_path.name)
        if not match:
            return None
        if match.lastindex and match.lastindex >= 1:
            raw_value = match.group(1)
        else:
            raw_value = match.group(0)
        return int(raw_value) if raw_value.isdigit() else None

    trailing_digits_match = re.search(r"(\d+)$", file_path.stem)
    if trailing_digits_match:
        return int(trailing_digits_match.group(1))

    return None


def find_category_for_file(
    file_path: Path,
    rules: list[SplitRule],
    pattern: re.Pattern[str] | None = None,
) -> str | None:
    number = extract_identifier(file_path, pattern)
    if number is None:
        return None

    for rule in rules:
        if number in rule.allowed_numbers:
            return rule.subfolder_name

    return None


@dataclass(frozen=True)
class PlannedTransfer:
    source_path: Path
    destination_path: Path
    subfolder_name: str


def natural_sort_key(text: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def plan_transfers(
    source_directory: Path,
    destination_directory: Path,
    rules: list[SplitRule],
    recursive: bool = False,
    extensions: set[str] | None = None,
    pattern: re.Pattern[str] | None = None,
    unmatched_action: UnmatchedAction = UnmatchedAction.SKIP,
    preserve_structure: bool = False,
) -> tuple[list[PlannedTransfer], list[Path]]:
    file_iterator = source_directory.rglob("*") if recursive else source_directory.glob("*")
    planned_transfers: list[PlannedTransfer] = []
    unmatched_files: list[Path] = []

    resolved_source = source_directory.resolve()
    resolved_destination = destination_directory.resolve()

    for path in file_iterator:
        if not path.is_file() or path.name.startswith("."):
            continue

        resolved_file = path.resolve()
        # Prevent picking up files from inside destination if destination is placed inside source
        if resolved_destination in resolved_file.parents or resolved_file == resolved_destination:
            continue

        if extensions and path.suffix.lower().lstrip(".") not in extensions:
            continue

        category = find_category_for_file(path, rules, pattern)

        if category is not None:
            if preserve_structure:
                relative_parent = path.relative_to(source_directory).parent
                target_directory = destination_directory / category / relative_parent
            else:
                target_directory = destination_directory / category

            planned_transfers.append(
                PlannedTransfer(
                    source_path=path,
                    destination_path=target_directory / path.name,
                    subfolder_name=category,
                )
            )
        else:
            unmatched_files.append(path)
            if unmatched_action == UnmatchedAction.COPY:
                if preserve_structure:
                    relative_parent = path.relative_to(source_directory).parent
                    target_directory = destination_directory / "unmatched" / relative_parent
                else:
                    target_directory = destination_directory / "unmatched"

                planned_transfers.append(
                    PlannedTransfer(
                        source_path=path,
                        destination_path=target_directory / path.name,
                        subfolder_name="unmatched",
                    )
                )

    planned_transfers.sort(key=lambda item: natural_sort_key(str(item.destination_path)))
    unmatched_files.sort(key=lambda path: natural_sort_key(str(path)))
    return planned_transfers, unmatched_files


def validate_transfers(
    transfers: list[PlannedTransfer],
    force: bool = False,
) -> tuple[list[str], list[str]]:
    destination_counts: dict[Path, list[Path]] = {}
    collisions: list[str] = []
    existing_conflicts: list[str] = []

    for transfer in transfers:
        destination_counts.setdefault(transfer.destination_path, []).append(transfer.source_path)

    for destination_path, sources in destination_counts.items():
        if len(sources) > 1:
            sources_summary = ", ".join(source.name for source in sources)
            collisions.append(f"Multiple sources map to '{destination_path.name}': {sources_summary}")

        if not force and destination_path.exists():
            existing_conflicts.append(f"Target already exists on disk: '{destination_path}'")

    return collisions, existing_conflicts


def execute_transfer(
    source_path: Path,
    destination_path: Path,
    mode: TransferMode,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == TransferMode.COPY:
        shutil.copy2(source_path, destination_path)
    elif mode == TransferMode.HARDLINK:
        try:
            os.link(source_path, destination_path)
        except OSError as error:
            # Caveat: hardlinks fail when source and target reside on different filesystem devices.
            raise OSError(
                f"Cannot hardlink across different filesystems ({source_path} -> {destination_path}). "
                "Use 'copy' or 'symlink' mode instead."
            ) from error
    elif mode == TransferMode.SYMLINK:
        destination_path.symlink_to(source_path.resolve())
    elif mode == TransferMode.MOVE:
        shutil.move(source_path, destination_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of a directory with files partitioned into subfolders based on numeric identifiers in filenames "
            "(e.g. 1-4 into 'gallery', 5-8 into 'query')."
        )
    )
    parser.add_argument(
        "source_directory",
        type=Path,
        help="Source directory containing files to partition.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Destination directory. Defaults to '<source_directory>_split'.",
    )
    parser.add_argument(
        "--rule",
        action="append",
        dest="rules",
        metavar="SUBFOLDER=RANGE",
        help=(
            "Partition rule in '<subfolder>=<range>' format (e.g. 'gallery=1-4', 'query=5-8'). "
            "Can be specified multiple times. Defaults to 'gallery=1-4' and 'query=5-8'."
        ),
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default=None,
        help=(
            "Custom regex pattern to match and extract the numeric identifier from the filename. "
            "If a capture group is included, group 1 is parsed as integer; otherwise group 0 is used. "
            "Defaults to trailing digits in filename stem (e.g. '101_1.tif' -> 1)."
        ),
    )
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        help="Filter source files by extension without leading dots (e.g. -e tif png bmp).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan source directory subfolders.",
    )
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve relative directory hierarchy within each partitioned subfolder when scanning recursively.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=[mode.value for mode in TransferMode],
        default=TransferMode.COPY.value,
        help="Transfer mechanism: 'copy' (default), 'hardlink', 'symlink', or 'move'.",
    )
    parser.add_argument(
        "--unmatched",
        choices=[action.value for action in UnmatchedAction],
        default=UnmatchedAction.SKIP.value,
        help="Action for files not matching any rule: 'skip' (default), 'copy' (to 'unmatched/'), or 'error'.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned operations without creating or modifying files on disk.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    source_directory = arguments.source_directory.resolve()
    if not source_directory.is_dir():
        print(f"Error: Source directory '{source_directory}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    destination_directory = (
        arguments.output.resolve()
        if arguments.output
        else source_directory.parent / f"{source_directory.name}_split"
    )

    raw_rules = arguments.rules if arguments.rules else ["gallery=1-4", "query=5-8"]
    try:
        rules = [parse_split_rule(rule_text) for rule_text in raw_rules]
        validate_split_rules(rules)
    except ValueError as validation_error:
        print(f"Rule configuration error: {validation_error}", file=sys.stderr)
        sys.exit(1)

    compiled_pattern = None
    if arguments.pattern:
        try:
            compiled_pattern = re.compile(arguments.pattern)
        except re.error as regex_error:
            print(f"Invalid regex pattern '{arguments.pattern}': {regex_error}", file=sys.stderr)
            sys.exit(1)

    extensions = (
        {token.lower().lstrip(".").strip() for ext in arguments.extensions for token in ext.split(",") if token.strip()}
        if arguments.extensions
        else None
    )
    unmatched_action = UnmatchedAction(arguments.unmatched)
    transfer_mode = TransferMode(arguments.mode)

    planned_transfers, unmatched_files = plan_transfers(
        source_directory=source_directory,
        destination_directory=destination_directory,
        rules=rules,
        recursive=arguments.recursive,
        extensions=extensions,
        pattern=compiled_pattern,
        unmatched_action=unmatched_action,
        preserve_structure=arguments.preserve_structure,
    )

    if unmatched_files and unmatched_action == UnmatchedAction.ERROR:
        print(
            f"Error: {len(unmatched_files)} files did not match any rule and --unmatched is set to 'error':",
            file=sys.stderr,
        )
        for unmatched_path in unmatched_files[:10]:
            print(f"  - {unmatched_path.name}", file=sys.stderr)
        if len(unmatched_files) > 10:
            print(f"  ... and {len(unmatched_files) - 10} more.", file=sys.stderr)
        sys.exit(1)

    if not planned_transfers and not unmatched_files:
        print(f"No matching files found in source directory '{source_directory}'.")
        return

    collisions, conflicts = validate_transfers(planned_transfers, force=arguments.force)
    if collisions or conflicts:
        print("Safety check failed! Conflicts detected:", file=sys.stderr)
        for collision in collisions:
            print(f"  - Collision: {collision}", file=sys.stderr)
        for conflict in conflicts:
            print(f"  - Existing destination: {conflict}", file=sys.stderr)
        print("\nAborting operation to prevent unintentional data overwrite.", file=sys.stderr)
        sys.exit(1)

    subfolder_counts: dict[str, int] = {}
    for transfer in planned_transfers:
        subfolder_counts[transfer.subfolder_name] = subfolder_counts.get(transfer.subfolder_name, 0) + 1

    mode_label = "[DRY-RUN]" if arguments.dry_run else "[APPLY]"
    print(f"{mode_label} Partitioning '{source_directory}' -> '{destination_directory}'")
    print(f"Mode: {transfer_mode.value}")
    print("Files partitioned per subfolder:")
    for rule in rules:
        count = subfolder_counts.get(rule.subfolder_name, 0)
        sorted_numbers = sorted(rule.allowed_numbers)
        number_summary = f"{sorted_numbers[0]}-{sorted_numbers[-1]}" if len(sorted_numbers) > 1 else str(sorted_numbers[0])
        print(f"  - {rule.subfolder_name} (ids: {number_summary}): {count} files")

    if unmatched_action == UnmatchedAction.COPY:
        unmatched_copied_count = subfolder_counts.get("unmatched", 0)
        print(f"  - unmatched: {unmatched_copied_count} files")
    else:
        print(f"  - unmatched (skipped): {len(unmatched_files)} files")

    print(f"Total files to transfer: {len(planned_transfers)}")

    if arguments.dry_run:
        if planned_transfers:
            print("\nSample planned transfers:")
            for transfer in planned_transfers[:10]:
                print(f"  {transfer.source_path.name} -> {transfer.destination_path.relative_to(destination_directory)}")
            if len(planned_transfers) > 10:
                print(f"  ... and {len(planned_transfers) - 10} more.")
        print("\nDry-run completed successfully. No files were modified.")
        return

    for transfer in planned_transfers:
        execute_transfer(transfer.source_path, transfer.destination_path, transfer_mode)

    print(f"\nSuccessfully transferred {len(planned_transfers)} files to '{destination_directory}'.")


if __name__ == "__main__":
    main()
