#!/usr/bin/env python3
"""Run 7T_Final WALRUS inference on the B0-corrected tumor dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_fid  # noqa: E402


SUBJECT_DIR = Path(
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/"
    "Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/Tumor/"
    "Tumor_Patient_160_B_2026-03-17"
)
INPUT_PATH = SUBJECT_DIR / "OriginalData/data.npy"
MASK_PATH = SUBJECT_DIR / "masks/brain_mask.npy"
OUTPUT_PATH = SUBJECT_DIR / "OriginalData/data_after_walrus.npy"
MODEL_DIR = PROJECT_ROOT / "models/7T_Final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
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
    checkpoint_path = MODEL_DIR / args.checkpoint

    for description, path in (
        ("B0-corrected input", INPUT_PATH),
        ("brain mask", MASK_PATH),
        ("model checkpoint", checkpoint_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if OUTPUT_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Use --overwrite only if it should be replaced."
        )

    source = np.load(INPUT_PATH, mmap_mode="r", allow_pickle=False)
    mask = np.load(MASK_PATH, mmap_mode="r", allow_pickle=False)
    expected_shape = source.shape
    if source.ndim != 4 or expected_shape[-1] != 840:
        raise ValueError(f"Expected 4-D input with 840 FID points, got {expected_shape}")
    if mask.shape != expected_shape[:3]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data shape {expected_shape[:3]}"
        )
    del source, mask

    temporary_path = OUTPUT_PATH.with_name(
        f".data_after_walrus.{os.getpid()}.tmp.npy"
    )
    temporary_path.unlink(missing_ok=True)

    print(f"Input:      {INPUT_PATH}", flush=True)
    print(f"Brain mask: {MASK_PATH}", flush=True)
    print(f"Model:      {checkpoint_path}", flush=True)
    print(f"Device:     {args.device}", flush=True)
    print(f"Output:     {OUTPUT_PATH}", flush=True)

    try:
        infer_fid(
            fid=INPUT_PATH,
            headmask=MASK_PATH,
            model_dir=MODEL_DIR,
            output_path=temporary_path,
            fid_axis=-1,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
        )

        candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if candidate.shape != expected_shape:
            raise RuntimeError(
                f"Output shape {candidate.shape} does not match {expected_shape}"
            )
        if not np.isfinite(candidate).all():
            raise RuntimeError("WALRUS output contains NaN or Inf")
        print(
            f"Validated output: shape={candidate.shape}, dtype={candidate.dtype}",
            flush=True,
        )
        del candidate
        os.replace(temporary_path, OUTPUT_PATH)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"DONE: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
