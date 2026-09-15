#!/usr/bin/env python3
"""Generate data_HSVD_L2.npy for the 20 non-MS 7T B0-corrected subjects."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/7T/B0corrected_wo_LipidMask"
RESOURCE_NAME = "TrainData/SimulationResources_water_lipid_v1.h5"
EXPECTED_SUBJECTS = 20
N_TIMEPOINTS = 840
EXPECTED_OPERATOR_TARGET = 0.99
EXPECTED_OPERATOR_TOLERANCE = 0.0001
DEFAULT_LOG = PROJECT_ROOT / "logs/generate_hsvd_l2_7t_b0corrected.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--subject", help="process only this Vol* subject")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> logging.Logger:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=(logging.StreamHandler(), logging.FileHandler(path)),
        force=True,
    )
    return logging.getLogger("generate_hsvd_l2_7t")


def discover_subjects() -> list[Path]:
    subjects = sorted(
        (path for path in DATA_ROOT.glob("Vol*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if len(subjects) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} Vol* subjects, found {len(subjects)}."
        )
    return subjects


def paths_for(subject_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        subject_dir / "OriginalData/data.npy",
        subject_dir / "masks/brain_mask.npy",
        subject_dir / RESOURCE_NAME,
        subject_dir / "OriginalData/data_HSVD_L2.npy",
    )


def preflight(subjects: list[Path], overwrite: bool) -> None:
    problems: list[str] = []
    expected_data_shape = (64, 64, 35, N_TIMEPOINTS)
    for subject_dir in subjects:
        data_path, mask_path, resource_path, output_path = paths_for(subject_dir)
        for path in (data_path, mask_path, resource_path):
            if not path.is_file():
                problems.append(f"{subject_dir.name}: missing {path}")
        if problems and any(item.startswith(subject_dir.name) for item in problems):
            continue
        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if data.shape != expected_data_shape:
            problems.append(f"{subject_dir.name}: data shape {data.shape}")
        if mask.shape != expected_data_shape[:-1]:
            problems.append(f"{subject_dir.name}: mask shape {mask.shape}")
        if not np.issubdtype(data.dtype, np.complexfloating):
            problems.append(f"{subject_dir.name}: data dtype {data.dtype}")
        if output_path.exists() and not overwrite:
            output = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if output.shape != data.shape or output.dtype != np.complex64:
                problems.append(
                    f"{subject_dir.name}: existing output has shape/dtype "
                    f"{output.shape}/{output.dtype}, expected {data.shape}/complex64"
                )
        with h5py.File(resource_path, "r") as h5:
            if "water_fids" not in h5:
                problems.append(f"{subject_dir.name}: missing water_fids")
            elif h5["water_fids"].shape != expected_data_shape:
                problems.append(
                    f"{subject_dir.name}: water shape {h5['water_fids'].shape}"
                )
            name = "lipid_projection/operator_840"
            if name not in h5:
                problems.append(f"{subject_dir.name}: missing {name}")
            elif h5[name].shape != (N_TIMEPOINTS, N_TIMEPOINTS):
                problems.append(f"{subject_dir.name}: operator shape {h5[name].shape}")
            target = float(h5["lipid_projection"].attrs.get("target", np.nan))
            tolerance = float(h5["lipid_projection"].attrs.get("tol", np.nan))
            if (
                target != EXPECTED_OPERATOR_TARGET
                or tolerance != EXPECTED_OPERATOR_TOLERANCE
            ):
                problems.append(
                    f"{subject_dir.name}: operator target/tol "
                    f"{target}/{tolerance}, expected "
                    f"{EXPECTED_OPERATOR_TARGET}/{EXPECTED_OPERATOR_TOLERANCE}"
                )
    if problems:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  {p}" for p in problems))


def process_subject(
    subject_dir: Path,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
    logger: logging.Logger,
) -> None:
    data_path, mask_path, resource_path, output_path = paths_for(subject_dir)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.npy"
    )
    for stale_path in output_path.parent.glob(
        f".{output_path.stem}.*.tmp.npy"
    ):
        stale_path.unlink()
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False) > 0

    with h5py.File(resource_path, "r") as h5:
        operator_np = np.asarray(
            h5["lipid_projection/operator_840"][:], dtype=np.complex64
        )
        water = h5["water_fids"]
        operator = torch.as_tensor(operator_np, device=device)
        output = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.complex64, shape=data.shape
        )
        completed = 0
        total = int(mask.sum())
        try:
            with torch.no_grad():
                for z in range(data.shape[2]):
                    output[:, :, z, :] = 0
                    slice_mask = mask[:, :, z].reshape(-1)
                    indices = np.flatnonzero(slice_mask)
                    if not len(indices):
                        continue
                    data_slice = np.asarray(data[:, :, z, :], dtype=np.complex64).reshape(
                        -1, N_TIMEPOINTS
                    )
                    water_slice = np.asarray(
                        water[:, :, z, :], dtype=np.complex64
                    ).reshape(-1, N_TIMEPOINTS)
                    output_slice = output[:, :, z, :].reshape(-1, N_TIMEPOINTS)
                    for start in range(0, len(indices), batch_size):
                        chosen = indices[start : start + batch_size]
                        x_fid = torch.as_tensor(data_slice[chosen], device=device)
                        w_fid = torch.as_tensor(water_slice[chosen], device=device)
                        x = torch.fft.fftshift(torch.fft.fft(x_fid, dim=-1), dim=-1)
                        w = torch.fft.fftshift(torch.fft.fft(w_fid, dim=-1), dim=-1)
                        clean = x - w - x @ operator
                        clean_fid = torch.fft.ifft(
                            torch.fft.ifftshift(clean, dim=-1), dim=-1
                        )
                        result = clean_fid.cpu().numpy().astype(np.complex64, copy=False)
                        if not np.isfinite(result).all():
                            raise RuntimeError(
                                f"{subject_dir.name}: non-finite result at slice {z}"
                            )
                        output_slice[chosen] = result
                        completed += len(chosen)
                    logger.info(
                        "%s: slice %d/%d, brain voxels %d/%d",
                        subject_dir.name,
                        z + 1,
                        data.shape[2],
                        completed,
                        total,
                    )
            output.flush()
            del output
            candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
            if candidate.shape != data.shape or candidate.dtype != np.complex64:
                raise RuntimeError(f"{subject_dir.name}: invalid output shape or dtype")
            for z in range(candidate.shape[2]):
                values = np.asarray(candidate[:, :, z, :])[mask[:, :, z]]
                if not np.isfinite(values).all():
                    raise RuntimeError(f"{subject_dir.name}: output contains NaN or Inf")
            del candidate
            if output_path.exists() and not overwrite:
                raise FileExistsError(output_path)
            temporary_path.replace(output_path)
        except BaseException:
            try:
                del output
            except UnboundLocalError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise
    logger.info("%s DONE: %s", subject_dir.name, output_path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    logger = configure_logging(args.log_file)
    subjects = discover_subjects()
    if args.subject:
        subjects = [subject for subject in subjects if subject.name == args.subject]
        if not subjects:
            raise ValueError(f"Unknown 7T Vol* subject: {args.subject}")
    preflight(subjects, args.overwrite)
    completed = sum(paths_for(subject)[3].is_file() for subject in subjects)
    pending = len(subjects) if args.overwrite else len(subjects) - completed
    logger.info(
        "Validated %d subjects; completed=%d, pending=%d; MS and Maps are excluded.",
        len(subjects),
        completed,
        pending,
    )
    for subject in subjects:
        state = (
            "RECOMPUTE"
            if args.overwrite
            else "SKIP" if paths_for(subject)[3].is_file() else "PENDING"
        )
        logger.info("SUBJECT %s: %s", subject.name, state)
    if args.dry_run:
        logger.info("Dry run complete; no output generated.")
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(device)
    for index, subject in enumerate(subjects, start=1):
        output_path = paths_for(subject)[3]
        if output_path.is_file() and not args.overwrite:
            logger.info("[%d/%d] SKIP completed %s", index, len(subjects), subject.name)
            continue
        logger.info("[%d/%d] processing %s", index, len(subjects), subject.name)
        process_subject(subject, device, args.batch_size, args.overwrite, logger)
    logger.info("All %d data_HSVD_L2.npy files generated successfully.", len(subjects))


if __name__ == "__main__":
    main()
