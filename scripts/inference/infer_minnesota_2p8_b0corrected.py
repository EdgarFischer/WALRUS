#!/usr/bin/env python3
"""Run WALRUS on the B0-corrected Minnesota 2p8 FIDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_fid  # noqa: E402


SUBJECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "7T"
    / "B0corrected_wo_LipidMask"
    / "Minnesota"
    / "2p8"
)
INPUT_PATH = SUBJECT_DIR / "OriginalData" / "data.npy"
MASK_PATH = SUBJECT_DIR / "masks" / "brain_mask.npy"
OUTPUT_PATH = SUBJECT_DIR / "OriginalData" / "data_after_walrus.npy"
MODEL_DIR = PROJECT_ROOT / "models" / "7T_Final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the 7T WALRUS model to B0-corrected Minnesota 2p8."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing data_after_walrus.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            "The finalized B0-corrected input does not exist yet: "
            f"{INPUT_PATH}"
        )
    if not MASK_PATH.is_file():
        raise FileNotFoundError(f"Brain mask does not exist: {MASK_PATH}")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")
    if not (MODEL_DIR / args.checkpoint).is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {MODEL_DIR / args.checkpoint}"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if OUTPUT_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}. Use --overwrite to replace it."
        )

    print(f"Input:      {INPUT_PATH}")
    print(f"Brain mask: {MASK_PATH}")
    print(f"Model:      {MODEL_DIR}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {args.device}")
    print(f"Output:     {OUTPUT_PATH}")

    infer_fid(
        fid=INPUT_PATH,
        headmask=MASK_PATH,
        model_dir=MODEL_DIR,
        output_path=OUTPUT_PATH,
        # Minnesota 2p8 data has shape (80, 80, 43, 840).
        fid_axis=3,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
