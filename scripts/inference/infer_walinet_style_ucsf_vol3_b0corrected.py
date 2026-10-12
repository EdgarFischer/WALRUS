#!/usr/bin/env python3
"""Infer the B0-corrected UCSF Vol3 data with the WALINET-Style model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_fid  # noqa: E402


SUBJECT_DIR = (
    PROJECT_ROOT.parent
    / "Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/UCSF/Vol3"
)
INPUT_PATH = SUBJECT_DIR / "OriginalData/data.npy"
MASK_PATH = SUBJECT_DIR / "masks/brain_mask.npy"
OUTPUT_PATH = SUBJECT_DIR / "OriginalData/data_WALINET_Style.npy"
MODEL_DIR = PROJECT_ROOT / "models/WALINET-Style"
RESOURCE_PATH = SUBJECT_DIR / "TrainData/SimulationResources_water_lipid_v1.h5"
OPERATOR_KEY = "lipid_projection/operator_840"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> tuple[tuple[int, ...], np.dtype]:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for path in (INPUT_PATH, MASK_PATH, RESOURCE_PATH, MODEL_DIR / args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}. Use --overwrite to replace it."
        )

    data = np.load(INPUT_PATH, mmap_mode="r", allow_pickle=False)
    mask = np.load(MASK_PATH, mmap_mode="r", allow_pickle=False)
    if data.shape != (64, 64, 35, 840):
        raise ValueError(f"Unexpected input shape: {data.shape}")
    if mask.shape != data.shape[:-1]:
        raise ValueError(f"Mask shape {mask.shape} != data space {data.shape[:-1]}")
    if not np.issubdtype(data.dtype, np.complexfloating):
        raise TypeError(f"Input is not complex: {data.dtype}")

    with h5py.File(RESOURCE_PATH, "r") as resources:
        if OPERATOR_KEY not in resources:
            raise KeyError(f"Missing {OPERATOR_KEY} in {RESOURCE_PATH}")
        operator = resources[OPERATOR_KEY]
        if operator.shape != (data.shape[-1], data.shape[-1]):
            raise ValueError(
                f"Operator shape {operator.shape} != "
                f"({data.shape[-1]}, {data.shape[-1]})"
            )
        target = float(resources["lipid_projection"].attrs.get("target", np.nan))
        if not np.isclose(target, 0.99, rtol=0.0, atol=1e-8):
            raise ValueError(f"Expected projection target 0.99, got {target}")
    return data.shape, data.dtype


def main() -> None:
    args = parse_args()
    expected_shape, input_dtype = preflight(args)
    print(f"Input:      {INPUT_PATH}")
    print(f"Input data: {expected_shape}, {input_dtype}")
    print(f"Mask:       {MASK_PATH}")
    print(f"Operator:   {RESOURCE_PATH}::{OPERATOR_KEY} (target=0.99)")
    print(f"Model:      {MODEL_DIR}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {args.device}")
    print(f"Output:     {OUTPUT_PATH}")
    if args.dry_run:
        print("Dry run successful; no output written.")
        return

    temporary_path = OUTPUT_PATH.with_name(
        f".{OUTPUT_PATH.stem}.{os.getpid()}.tmp.npy"
    )
    temporary_path.unlink(missing_ok=True)
    try:
        with h5py.File(RESOURCE_PATH, "r") as resources:
            operator = np.asarray(resources[OPERATOR_KEY], dtype=np.complex64)
        infer_fid(
            fid=INPUT_PATH,
            headmask=MASK_PATH,
            model_dir=MODEL_DIR,
            lipid_projection_operator=operator,
            output_path=temporary_path,
            fid_axis=-1,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
        )
        output = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if output.shape != expected_shape:
            raise RuntimeError(f"Output shape {output.shape} != {expected_shape}")
        if output.dtype != np.complex64:
            raise RuntimeError(f"Output dtype {output.dtype} != complex64")
        if not np.isfinite(output).all():
            raise RuntimeError("Output contains NaN or Inf")
        del output
        temporary_path.replace(OUTPUT_PATH)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"DONE: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
