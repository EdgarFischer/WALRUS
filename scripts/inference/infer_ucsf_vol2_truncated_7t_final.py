#!/usr/bin/env python3
"""Run 7T_Final WALRUS inference on truncated UCSF Vol2 B0-corrected FIDs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_fid  # noqa: E402


SUBJECT_DIR = (
    PROJECT_ROOT
    / "data/7T/B0corrected_wo_LipidMask/UCSF/Vol2"
)
ORIGINAL_DATA = SUBJECT_DIR / "OriginalData"
MASK_PATH = SUBJECT_DIR / "masks/brain_mask.npy"
MODEL_DIR = PROJECT_ROOT / "models/7T_Final"

JOBS = (
    ("data_420.npy", "data_after_walrus_420.npy"),
    ("data_630.npy", "data_after_walrus_630.npy"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:1.")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(MODEL_DIR)
    if not MASK_PATH.is_file():
        raise FileNotFoundError(MASK_PATH)

    for input_name, output_name in JOBS:
        input_path = ORIGINAL_DATA / input_name
        output_path = ORIGINAL_DATA / output_name
        temporary_path = output_path.with_name(
            f".{output_path.stem}.{os.getpid()}.tmp.npy"
        )
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if output_path.exists() and not args.overwrite:
            print(f"SKIP existing output: {output_path}", flush=True)
            continue

        source = np.load(input_path, mmap_mode="r", allow_pickle=False)
        input_shape = tuple(source.shape)
        input_dtype = source.dtype
        del source

        print(f"Input:      {input_path}", flush=True)
        print(f"Input data: {input_shape}, {input_dtype}", flush=True)
        print(f"Mask:       {MASK_PATH}", flush=True)
        print(f"Model:      {MODEL_DIR}", flush=True)
        print(f"Device:     {args.device}", flush=True)
        print(f"Output:     {output_path}", flush=True)

        temporary_path.unlink(missing_ok=True)
        try:
            infer_fid(
                fid=input_path,
                headmask=MASK_PATH,
                model_dir=MODEL_DIR,
                output_path=temporary_path,
                fid_axis=-1,
                checkpoint=args.checkpoint,
                batch_size=args.batch_size,
                device=args.device,
            )
            candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
            if candidate.shape[:3] != input_shape[:3]:
                raise RuntimeError(
                    f"{output_name}: spatial shape {candidate.shape[:3]} "
                    f"!= {input_shape[:3]}"
                )
            if candidate.dtype != np.complex64:
                raise RuntimeError(
                    f"{output_name}: dtype {candidate.dtype} != complex64"
                )
            if not np.isfinite(candidate).all():
                raise RuntimeError(f"{output_name}: output contains NaN or Inf")
            print(
                f"Validated output shape: {candidate.shape}, dtype={candidate.dtype}",
                flush=True,
            )
            del candidate
            temporary_path.replace(output_path)
            print(f"Saved: {output_path}", flush=True)
        finally:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
