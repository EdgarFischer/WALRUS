#!/usr/bin/env python3
"""Generate HSVD+L2 data for the 3T validation subjects in train_3T.yaml."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG = PROJECT_ROOT / "configs/Training/train_3T.yaml"
EXPECTED_SUBJECTS = 12
N_TIMEPOINTS = 288
EXPECTED_TARGET = 0.97
EXPECTED_TOLERANCE = 0.0001
DEFAULT_LOG = PROJECT_ROOT / "logs/generate_hsvd_l2_3t_validation.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
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
    return logging.getLogger("generate_hsvd_l2_3t_validation")


def load_subjects() -> list[tuple[str, Path, str]]:
    raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    data_cfg = raw["data"]
    base_dir = (TRAINING_CONFIG.parent / data_cfg["base_dir"]).resolve()
    filename = data_cfg["resources"]["filename"].format(
        version=data_cfg["resources"]["version"]
    )
    subjects = list(data_cfg["val_subjects"])
    if len(subjects) != EXPECTED_SUBJECTS or len(set(subjects)) != len(subjects):
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} unique val_subjects, found "
            f"{len(subjects)} entries and {len(set(subjects))} unique entries."
        )
    return [(subject, base_dir / subject, filename) for subject in subjects]


def paths_for(subject_dir: Path, resource_name: str) -> tuple[Path, Path, Path, Path]:
    return (
        subject_dir / "OriginalData/data.npy",
        subject_dir / "masks/brain_mask.npy",
        subject_dir / resource_name,
        subject_dir / "OriginalData/data_HSVD_L2.npy",
    )


def preflight(
    subjects: list[tuple[str, Path, str]], overwrite: bool
) -> None:
    problems: list[str] = []
    for subject, subject_dir, resource_name in subjects:
        data_path, mask_path, resource_path, output_path = paths_for(
            subject_dir, resource_name
        )
        missing = [
            str(path)
            for path in (data_path, mask_path, resource_path)
            if not path.is_file()
        ]
        if missing:
            problems.append(f"{subject}: missing: {', '.join(missing)}")
            continue
        if output_path.exists() and not overwrite:
            problems.append(f"{subject}: output exists; use --overwrite: {output_path}")
            continue

        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if data.ndim != 4 or data.shape[-1] != N_TIMEPOINTS:
            problems.append(f"{subject}: data shape {data.shape}, expected (..., 288)")
            continue
        if mask.shape != data.shape[:-1]:
            problems.append(
                f"{subject}: brain mask shape {mask.shape} != {data.shape[:-1]}"
            )
        if not np.issubdtype(data.dtype, np.complexfloating):
            problems.append(f"{subject}: data dtype is {data.dtype}, expected complex")

        with h5py.File(resource_path, "r") as h5:
            if "water_fids" not in h5:
                problems.append(f"{subject}: missing water_fids")
            elif h5["water_fids"].shape != data.shape:
                problems.append(
                    f"{subject}: water shape {h5['water_fids'].shape} != {data.shape}"
                )
            operator_name = "lipid_projection/operator_288"
            if operator_name not in h5:
                problems.append(f"{subject}: missing {operator_name}")
                continue
            if h5[operator_name].shape != (N_TIMEPOINTS, N_TIMEPOINTS):
                problems.append(f"{subject}: invalid operator shape {h5[operator_name].shape}")
            group = h5["lipid_projection"]
            target = float(group.attrs.get("target", np.nan))
            tolerance = float(group.attrs.get("tol", np.nan))
            if target != EXPECTED_TARGET or tolerance != EXPECTED_TOLERANCE:
                problems.append(
                    f"{subject}: operator target/tol is {target}/{tolerance}, "
                    f"expected {EXPECTED_TARGET}/{EXPECTED_TOLERANCE}"
                )
    if problems:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  {p}" for p in problems))


def process_subject(
    subject: str,
    subject_dir: Path,
    resource_name: str,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
    logger: logging.Logger,
) -> None:
    data_path, mask_path, resource_path, output_path = paths_for(
        subject_dir, resource_name
    )
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.npy"
    )
    temporary_path.unlink(missing_ok=True)
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False) > 0

    try:
        with h5py.File(resource_path, "r") as h5:
            operator = torch.as_tensor(
                np.asarray(
                    h5["lipid_projection/operator_288"][:], dtype=np.complex64
                ),
                device=device,
            )
            water = h5["water_fids"]
            output = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.complex64,
                shape=data.shape,
            )
            completed = 0
            total = int(mask.sum())
            with torch.no_grad():
                for z in range(data.shape[2]):
                    output[:, :, z, :] = 0
                    indices = np.flatnonzero(mask[:, :, z].reshape(-1))
                    if not len(indices):
                        continue
                    data_slice = np.asarray(
                        data[:, :, z, :], dtype=np.complex64
                    ).reshape(-1, N_TIMEPOINTS)
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
                        result = clean_fid.cpu().numpy().astype(
                            np.complex64, copy=False
                        )
                        if not np.isfinite(result).all():
                            raise RuntimeError(
                                f"{subject}: non-finite result at slice {z}"
                            )
                        output_slice[chosen] = result
                        completed += len(chosen)
                    logger.info(
                        "%s: slice %d/%d, brain voxels %d/%d",
                        subject,
                        z + 1,
                        data.shape[2],
                        completed,
                        total,
                    )
            output.flush()
            del output

        candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if candidate.shape != data.shape or candidate.dtype != np.complex64:
            raise RuntimeError(f"{subject}: invalid output shape or dtype")
        for z in range(candidate.shape[2]):
            values = np.asarray(candidate[:, :, z, :])[mask[:, :, z]]
            if not np.isfinite(values).all():
                raise RuntimeError(f"{subject}: output contains NaN or Inf")
        del candidate
        if output_path.exists() and not overwrite:
            raise FileExistsError(output_path)
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    logger.info("%s DONE: %s", subject, output_path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    logger = configure_logging(args.log_file)
    subjects = load_subjects()
    preflight(subjects, args.overwrite)
    logger.info("Validated %d 3T validation datasets.", len(subjects))
    for subject, subject_dir, _ in subjects:
        logger.info("SUBJECT %s: %s", subject, subject_dir)
    if args.dry_run:
        logger.info("Dry run complete; no output generated.")
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(device)
    for index, (subject, subject_dir, resource_name) in enumerate(subjects, start=1):
        logger.info("[%d/%d] processing %s", index, len(subjects), subject)
        process_subject(
            subject,
            subject_dir,
            resource_name,
            device,
            args.batch_size,
            args.overwrite,
            logger,
        )
    logger.info("All %d 3T validation outputs generated successfully.", len(subjects))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        logging.getLogger("generate_hsvd_l2_3t_validation").exception(
            "HSVD+L2 generation failed."
        )
        raise
