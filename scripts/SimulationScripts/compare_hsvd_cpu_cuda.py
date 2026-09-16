#!/usr/bin/env python3
"""Compare established CPU HSVD with batched CUDA HSVD on real 3T FIDs.

The script is read-only: it samples brain voxels from one validation subject,
prints numerical differences and does not save arrays or modify data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.config.build_water_lipid import (  # noqa: E402
    build_water_lipid_extraction_config,
)
from walinet.training_data.hsvd_water import estimate_water_hsvd  # noqa: E402


WATER_CONFIG = PROJECT_ROOT / "configs/Simulation/3T_water_lipid_v1.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "configs/Training/train_3T.yaml"
N_TIMEPOINTS = 288


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        help="Validation-subject path; default is the first entry in train_3T.yaml.",
    )
    parser.add_argument("--n-fids", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--cuda-dtype",
        choices=("complex64", "complex128"),
        default="complex64",
        help="Use complex128 for a stricter numerical comparison.",
    )
    parser.add_argument(
        "--show-worst",
        type=int,
        default=5,
        help="Number of worst individual FID errors to print.",
    )
    parser.add_argument(
        "--fail-above",
        type=float,
        default=None,
        help="Optional maximum permitted relative water error.",
    )
    return parser.parse_args()


def load_setup(subject_override: str | None):
    water_raw = yaml.safe_load(WATER_CONFIG.read_text(encoding="utf-8"))
    water_cfg = build_water_lipid_extraction_config(
        water_raw, config_dir=WATER_CONFIG.parent
    )
    training_raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    validation_subjects = list(training_raw["data"]["val_subjects"])
    subject = subject_override or validation_subjects[0]
    if subject not in validation_subjects:
        raise ValueError(
            f"{subject!r} is not one of the train_3T.yaml validation subjects"
        )
    subject_dir = Path(water_cfg.data.base_dir) / subject
    data_path = subject_dir / water_cfg.data.paths.input_data
    mask_path = subject_dir / water_cfg.data.paths.brain_mask
    return water_cfg, subject, data_path, mask_path


def sample_fids(
    data_path: Path,
    mask_path: Path,
    *,
    n_fids: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    mask = np.asarray(np.load(mask_path, allow_pickle=False)).squeeze() > 0
    if data.ndim != 4 or data.shape[-1] < N_TIMEPOINTS:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    if mask.shape != data.shape[:-1]:
        raise ValueError(f"Mask shape {mask.shape} != data grid {data.shape[:-1]}")
    brain_indices = np.flatnonzero(mask.reshape(-1))
    if n_fids > len(brain_indices):
        raise ValueError(
            f"Requested {n_fids} FIDs, but mask contains {len(brain_indices)} voxels"
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(brain_indices, size=n_fids, replace=False)
    flat_data = data.reshape(-1, data.shape[-1])
    fids = np.asarray(flat_data[selected, :N_TIMEPOINTS], dtype=np.complex64)
    coordinates = np.column_stack(np.unravel_index(selected, mask.shape))
    return np.ascontiguousarray(fids), coordinates


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(candidate - reference, axis=-1)
    denominator = np.linalg.norm(reference, axis=-1)
    return numerator / np.maximum(denominator, np.finfo(np.float64).eps)


def print_summary(name: str, values: np.ndarray) -> None:
    print(
        f"{name}: mean={values.mean():.6g}, median={np.median(values):.6g}, "
        f"p95={np.percentile(values, 95):.6g}, max={values.max():.6g}"
    )


def main() -> None:
    args = parse_args()
    if args.n_fids <= 0 or args.batch_size <= 0:
        raise ValueError("--n-fids and --batch-size must be positive")
    if args.show_worst < 0:
        raise ValueError("--show-worst must not be negative")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device, e.g. cuda:1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process")

    cfg, subject, data_path, mask_path = load_setup(args.subject)
    fids, coordinates = sample_fids(
        data_path, mask_path, n_fids=args.n_fids, seed=args.seed
    )
    hsvd_kwargs = dict(
        sampling_frequency_hz=cfg.water_extraction.bandwidth,
        n_components=cfg.water_extraction.hsvd_components,
        min_frequency_hz=cfg.water_extraction.min_freq,
        max_frequency_hz=cfg.water_extraction.max_freq,
    )
    cuda_dtype = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }[args.cuda_dtype]

    print(f"Subject: {subject}")
    print(f"Data: {data_path}")
    print(f"Sampled FIDs: {len(fids)} x {fids.shape[-1]}")
    print(
        "HSVD: "
        f"components={cfg.water_extraction.hsvd_components}, "
        f"bandwidth={cfg.water_extraction.bandwidth:.6g} Hz, "
        f"range=[{cfg.water_extraction.min_freq:.6g}, "
        f"{cfg.water_extraction.max_freq:.6g}] Hz"
    )
    print(
        f"CUDA: device={device}, dtype={args.cuda_dtype}, "
        f"batch_size={args.batch_size}"
    )

    start = time.perf_counter()
    cpu_water = estimate_water_hsvd(fids, device="cpu", **hsvd_kwargs)
    cpu_seconds = time.perf_counter() - start

    # Synchronize around the CUDA timing and include one complete production
    # call. This deliberately reports transfer plus computation time.
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    cuda_water = estimate_water_hsvd(
        fids,
        device=device,
        batch_size=args.batch_size,
        cuda_dtype=cuda_dtype,
        **hsvd_kwargs,
    )
    torch.cuda.synchronize(device)
    cuda_seconds = time.perf_counter() - start

    water_error = relative_l2(cuda_water, cpu_water)
    cpu_residual = fids - cpu_water
    cuda_residual = fids - cuda_water
    residual_error = relative_l2(cuda_residual, cpu_residual)

    print("\nNumerical comparison (CPU is reference)")
    print_summary("Water relative L2", water_error)
    print_summary("Water-removed residual relative L2", residual_error)
    print(f"\nRuntime: CPU={cpu_seconds:.3f}s, CUDA={cuda_seconds:.3f}s")
    if cuda_seconds > 0:
        print(f"Observed speedup: {cpu_seconds / cuda_seconds:.2f}x")

    n_worst = min(args.show_worst, len(water_error))
    if n_worst:
        print(f"\nWorst {n_worst} water estimates")
        for rank, index in enumerate(np.argsort(water_error)[-n_worst:][::-1], 1):
            xyz = tuple(int(value) for value in coordinates[index])
            print(
                f"  {rank}: sample={index}, voxel={xyz}, "
                f"water_rel_l2={water_error[index]:.6g}, "
                f"residual_rel_l2={residual_error[index]:.6g}"
            )

    if args.fail_above is not None:
        if not np.isfinite(water_error).all() or water_error.max() > args.fail_above:
            raise SystemExit(
                f"FAIL: maximum water relative L2 {water_error.max():.6g} "
                f"> {args.fail_above:.6g}"
            )
        print(f"PASS: maximum water relative L2 <= {args.fail_above:.6g}")


if __name__ == "__main__":
    main()

