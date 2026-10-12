#!/usr/bin/env python3
"""Replace 'Walinet' with 'Walrus' in data-directory names."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


OLD_TEXT = "Walinet"
NEW_TEXT = "Walrus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renames. Without this flag, only show a dry run.",
    )
    return parser.parse_args()


def find_directories(branch_roots: list[Path]) -> list[Path]:
    matches: list[Path] = []
    for root in branch_roots:
        for current_root, directory_names, _ in os.walk(root):
            current_path = Path(current_root)
            matches.extend(
                current_path / name
                for name in directory_names
                if OLD_TEXT in name
            )

    # Rename children before parents if a matching directory is nested.
    return sorted(set(matches), key=lambda path: (-len(path.parts), str(path)))


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    branch_roots = [
        project_root / "data/3T/B0corrected_wo_LipidMask",
        project_root / "data/3T/NoB0Correction",
        project_root / "data/7T/B0corrected_wo_LipidMask",
        project_root / "data/7T/NoB0Correction",
    ]

    missing_roots = [root for root in branch_roots if not root.is_dir()]
    if missing_roots:
        formatted = "\n".join(f"  {path}" for path in missing_roots)
        raise FileNotFoundError(f"Missing data branch(es):\n{formatted}")

    sources = find_directories(branch_roots)
    rename_pairs = [
        (source, source.with_name(source.name.replace(OLD_TEXT, NEW_TEXT)))
        for source in sources
    ]

    conflicts = [destination for _, destination in rename_pairs if destination.exists()]
    if conflicts:
        formatted = "\n".join(f"  {path}" for path in conflicts)
        raise FileExistsError(
            "Refusing to merge with or overwrite existing target directories:\n"
            f"{formatted}"
        )

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Directories found: {len(rename_pairs)}")

    for source, destination in rename_pairs:
        print(f"{source} -> {destination}")
        if args.apply:
            source.rename(destination)

    if not rename_pairs:
        print("Nothing to rename.")
    elif not args.apply:
        print("\nNo directories changed. Run again with --apply to perform the renames.")
    else:
        print(f"\nRenamed {len(rename_pairs)} directories successfully.")


if __name__ == "__main__":
    main()
