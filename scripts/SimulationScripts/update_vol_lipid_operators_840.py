#!/usr/bin/env python3
"""Recompute operator_840 from stored water-suppressed lipid FIDs for Vol* subjects."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.training_data.lipid_removal import (  # noqa: E402
    compute_lipid_projection_operator,
)


DATA_ROOT = PROJECT_ROOT / "data/7T/B0corrected_wo_LipidMask"
RESOURCE_RELATIVE_PATH = Path(
    "TrainData/SimulationResources_water_lipid_v1.h5"
)
DEFAULT_TARGET = 0.99
DEFAULT_TOLERANCE = 0.0001
N_TIMEPOINTS = 840
EXPECTED_SUBJECTS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("auto", "gpu", "eigenvalue", "legacy"),
        default="auto",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_resources() -> list[tuple[str, Path]]:
    resources = [
        (directory.name, directory / RESOURCE_RELATIVE_PATH)
        for directory in sorted(DATA_ROOT.glob("Vol*"), key=lambda path: path.name)
        if directory.is_dir()
    ]
    if len(resources) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} Vol* directories, found {len(resources)}."
        )
    return resources


def preflight(resources: list[tuple[str, Path]]) -> None:
    problems = []
    for subject, path in resources:
        if not path.is_file():
            problems.append(f"{subject}: missing {path}")
            continue
        try:
            with h5py.File(path, "r") as h5:
                if "lipid_fids" not in h5:
                    problems.append(f"{subject}: missing lipid_fids")
                    continue
                if "lipid_projection/operator_840" not in h5:
                    problems.append(f"{subject}: missing lipid_projection/operator_840")
                    continue
                lipid_shape = h5["lipid_fids"].shape
                operator_shape = h5["lipid_projection/operator_840"].shape
                if len(lipid_shape) != 2 or lipid_shape[0] == 0 or lipid_shape[1] != N_TIMEPOINTS:
                    problems.append(f"{subject}: invalid lipid_fids shape {lipid_shape}")
                if operator_shape != (N_TIMEPOINTS, N_TIMEPOINTS):
                    problems.append(f"{subject}: invalid operator shape {operator_shape}")
        except OSError as error:
            problems.append(f"{subject}: cannot read HDF5: {error}")
    if problems:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  {p}" for p in problems))


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def save_backup(
    *,
    resource_path: Path,
    operator: np.ndarray,
    old_target: float,
    old_tolerance: float,
) -> Path:
    backup_dir = resource_path.parent / "operator_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"operator_840_target_{_number_label(old_target)}_"
        f"tol_{_number_label(old_tolerance)}"
    )
    backup_path = backup_dir / f"{stem}.npy"
    metadata_path = backup_dir / f"{stem}.json"
    if not backup_path.exists():
        temporary_path = backup_path.with_name(f".{backup_path.name}.tmp.npy")
        np.save(temporary_path, np.asarray(operator, dtype=np.complex64))
        temporary_path.replace(backup_path)
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(
                {
                    "source": str(resource_path),
                    "dataset": "lipid_projection/operator_840",
                    "target": old_target,
                    "tol": old_tolerance,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return backup_path


def recompute_subject(
    *,
    subject: str,
    resource_path: Path,
    method: str,
    gpu_index: int,
    target: float,
    tolerance: float,
) -> None:
    started = time.time()
    with h5py.File(resource_path, "r") as h5:
        lipid_fids = np.asarray(h5["lipid_fids"][:], dtype=np.complex64)
        old_operator = np.asarray(
            h5["lipid_projection/operator_840"][:],
            dtype=np.complex64,
        )
        group = h5["lipid_projection"]
        old_target = float(group.attrs.get("target", np.nan))
        old_tolerance = float(group.attrs.get("tol", np.nan))

    backup_path = save_backup(
        resource_path=resource_path,
        operator=old_operator,
        old_target=old_target,
        old_tolerance=old_tolerance,
    )
    del old_operator

    lipid_spectra = np.fft.fftshift(
        np.fft.fft(lipid_fids, axis=-1),
        axes=-1,
    ).astype(np.complex64, copy=False)
    del lipid_fids

    operator = compute_lipid_projection_operator(
        spectra=lipid_spectra[:, None, None, :],
        lipid_mask=np.ones((lipid_spectra.shape[0], 1, 1), dtype=bool),
        target=target,
        tol=tolerance,
        method=method,
        gpu_index=gpu_index,
    )
    operator = np.asarray(operator, dtype=np.complex64)
    if operator.shape != (N_TIMEPOINTS, N_TIMEPOINTS):
        raise RuntimeError(f"{subject}: unexpected new shape {operator.shape}")
    if not np.isfinite(operator).all():
        raise RuntimeError(f"{subject}: new operator contains NaN or Inf")

    achieved = float(np.mean(np.abs(np.diag(np.eye(N_TIMEPOINTS) - operator))))
    if abs(achieved - target) > tolerance:
        raise RuntimeError(
            f"{subject}: achieved {achieved:.8f}, outside "
            f"{target} ± {tolerance}"
        )

    with h5py.File(resource_path, "r+") as h5:
        dataset = h5["lipid_projection/operator_840"]
        dataset[...] = operator
        group = h5["lipid_projection"]
        group.attrs["target"] = target
        group.attrs["tol"] = tolerance
        group.attrs["source"] = "water_suppressed_lipid_fids"
        group.attrs["updated_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        )
        group.attrs["operator_840_achieved_diag_mean"] = achieved
        h5.flush()

    print(
        f"[{subject}] updated: achieved={achieved:.8f}, "
        f"backup={backup_path}, elapsed={time.time() - started:.1f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target <= 1.0:
        raise ValueError("--target must be in (0, 1].")
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive.")
    resources = discover_resources()
    preflight(resources)
    print(f"Validated {len(resources)} Vol* resources.")
    for subject, path in resources:
        print(f"  {subject}: {path}")
    if args.dry_run:
        print("Dry run complete; no operators changed.")
        return
    for index, (subject, path) in enumerate(resources, start=1):
        print(f"[{index}/{len(resources)}] recomputing {subject}", flush=True)
        recompute_subject(
            subject=subject,
            resource_path=path,
            method=args.method,
            gpu_index=args.gpu_index,
            target=args.target,
            tolerance=args.tolerance,
        )
    print("All 20 operator_840 datasets updated successfully.")


if __name__ == "__main__":
    main()
