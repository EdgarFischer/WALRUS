#!/usr/bin/env python3
"""Run the 7T PaperFig1 simulation evaluation for the WALINET-style model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _project_root() -> Path:
    for parent in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (parent / "src" / "walinet").is_dir():
            return parent
    raise FileNotFoundError("Could not locate WALRUS project root")


def main() -> None:
    root = _project_root()
    sys.path.insert(0, str(root / "src"))

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate WALINET-style on the same 7T synthetic test set used by "
            "the quantitative paper figure."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/PaperFig1/model_comparison_walinet_style.yaml",
        help="PaperFig1 evaluation YAML.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional CUDA device override, e.g. cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing WALINET-style HDF5 result file.",
    )
    args = parser.parse_args()

    if args.device is not None:
        import yaml

        raw = yaml.safe_load(args.config.read_text())
        raw["device"] = args.device
        temporary_config = (
            args.config.resolve().parent
            / ".model_comparison_walinet_style_runtime.yaml"
        )
        temporary_config.write_text(yaml.safe_dump(raw, sort_keys=False))
        config_path = temporary_config
    else:
        config_path = args.config

    from walinet.paper_figures.paper_fig1 import run_evaluation

    output = run_evaluation(config_path, overwrite=args.overwrite)
    print(f"Wrote: {output.resolve()}")


if __name__ == "__main__":
    main()
