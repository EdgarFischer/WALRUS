#!/usr/bin/env python3
"""Rename data_after_walinet.npy to data_after_walrus.npy in all data branches."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_NAME = "data_after_walinet.npy"
NEW_NAME = "data_after_walrus.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renames. Without this flag, only show a dry run.",
    )
    return parser.parse_args()


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

    sources = sorted(
        source
        for root in branch_roots
        for source in root.rglob(OLD_NAME)
        if source.parent.name == "OriginalData"
    )

    conflicts = [source.with_name(NEW_NAME) for source in sources if source.with_name(NEW_NAME).exists()]
    if conflicts:
        formatted = "\n".join(f"  {path}" for path in conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing data_after_walrus.npy file(s):\n"
            f"{formatted}"
        )

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Files found: {len(sources)}")

    for source in sources:
        destination = source.with_name(NEW_NAME)
        print(f"{source} -> {destination}")
        if args.apply:
            source.rename(destination)

    if not sources:
        print("Nothing to rename.")
    elif not args.apply:
        print("\nNo files changed. Run again with --apply to perform the renames.")
    else:
        print(f"\nRenamed {len(sources)} files successfully.")


if __name__ == "__main__":
    main()
