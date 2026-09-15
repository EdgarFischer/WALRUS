#!/usr/bin/env python3
"""Replace only operator_288 in all 3T B0-corrected resources."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.training_data.lipid_removal import (  # noqa: E402
    compute_lipid_projection_operator,
)


DATA_ROOT = PROJECT_ROOT / "data/3T/B0corrected_wo_LipidMask"
RESOURCE_NAME = "SimulationResources_water_lipid_v1.h5"
DEFAULT_LOG = PROJECT_ROOT / "logs/update_all_3t_operator_288_target_0p97.log"
EXPECTED_RESOURCES = 61
N_TIMEPOINTS = 288
DEFAULT_TARGET = 0.97
DEFAULT_TOLERANCE = 0.0001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--method",
        choices=("gpu", "eigenvalue", "legacy"),
        default="gpu",
        help="GPU is the default; unlike auto, it cannot silently fall back to CPU.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def discover_resources() -> list[Path]:
    resources = sorted(DATA_ROOT.rglob(f"TrainData/{RESOURCE_NAME}"))
    if len(resources) != EXPECTED_RESOURCES:
        raise RuntimeError(
            f"Expected {EXPECTED_RESOURCES} resources below {DATA_ROOT}, "
            f"found {len(resources)}."
        )
    return resources


def preflight(resources: list[Path]) -> None:
    problems: list[str] = []
    for path in resources:
        try:
            with h5py.File(path, "r") as h5:
                if "lipid_fids" not in h5:
                    problems.append(f"{path}: missing lipid_fids")
                    continue
                lipid_shape = h5["lipid_fids"].shape
                if len(lipid_shape) != 2 or lipid_shape[0] == 0 or lipid_shape[1] < N_TIMEPOINTS:
                    problems.append(f"{path}: invalid lipid_fids shape {lipid_shape}")
                name = f"lipid_projection/operator_{N_TIMEPOINTS}"
                if name not in h5:
                    problems.append(f"{path}: missing {name}")
                elif h5[name].shape != (N_TIMEPOINTS, N_TIMEPOINTS):
                    problems.append(f"{path}: invalid {name} shape {h5[name].shape}")
        except OSError as error:
            problems.append(f"{path}: cannot read HDF5: {error}")
    if problems:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  {p}" for p in problems))


def number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def save_backup(
    resource_path: Path,
    operator: np.ndarray,
    old_target: float,
    old_tolerance: float,
) -> Path:
    backup_dir = resource_path.parent / "operator_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"operator_288_target_{number_label(old_target)}_"
        f"tol_{number_label(old_tolerance)}"
    )
    backup_path = backup_dir / f"{stem}.npy"
    metadata_path = backup_dir / f"{stem}.json"
    if not backup_path.exists():
        temporary = backup_dir / f".{stem}.tmp.npy"
        np.save(temporary, np.asarray(operator, dtype=np.complex64))
        temporary.replace(backup_path)
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(
                {
                    "source": str(resource_path),
                    "dataset": "lipid_projection/operator_288",
                    "target": old_target,
                    "tol": old_tolerance,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return backup_path


def recompute(
    path: Path,
    target: float,
    tolerance: float,
    method: str,
    gpu_index: int,
) -> None:
    started = time.monotonic()
    with h5py.File(path, "r") as h5:
        lipid_fids = np.asarray(h5["lipid_fids"][:, :N_TIMEPOINTS], dtype=np.complex64)
        group = h5["lipid_projection"]
        dataset = group["operator_288"]
        old_operator = np.asarray(dataset[:], dtype=np.complex64)
        old_target = float(dataset.attrs.get("target", group.attrs.get("target", np.nan)))
        old_tolerance = float(dataset.attrs.get("tol", group.attrs.get("tol", np.nan)))

    backup = save_backup(path, old_operator, old_target, old_tolerance)
    del old_operator
    spectra = np.fft.fftshift(np.fft.fft(lipid_fids, axis=-1), axes=-1).astype(
        np.complex64, copy=False
    )
    del lipid_fids
    operator = compute_lipid_projection_operator(
        spectra=spectra[:, None, None, :],
        lipid_mask=np.ones((len(spectra), 1, 1), dtype=bool),
        target=target,
        tol=tolerance,
        method=method,
        gpu_index=gpu_index,
    )
    operator = np.asarray(operator, dtype=np.complex64)
    if operator.shape != (N_TIMEPOINTS, N_TIMEPOINTS) or not np.isfinite(operator).all():
        raise RuntimeError(f"{path}: invalid newly computed operator_288")
    achieved = float(
        np.mean(np.abs(np.diag(np.eye(N_TIMEPOINTS, dtype=np.complex64) - operator)))
    )
    if abs(achieved - target) > tolerance:
        raise RuntimeError(
            f"{path}: achieved {achieved:.8f}, outside {target} +/- {tolerance}"
        )

    with h5py.File(path, "r+") as h5:
        group = h5["lipid_projection"]
        dataset = group["operator_288"]
        dataset[...] = operator
        dataset.attrs["target"] = target
        dataset.attrs["tol"] = tolerance
        dataset.attrs["source"] = "water_suppressed_lipid_fids"
        dataset.attrs["achieved_diag_mean"] = achieved
        group.attrs["operator_288_target"] = target
        group.attrs["operator_288_tol"] = tolerance
        group.attrs["operator_288_achieved_diag_mean"] = achieved
        # Retain these legacy attributes for existing operator_288 consumers.
        group.attrs["target"] = target
        group.attrs["tol"] = tolerance
        group.attrs["source"] = "water_suppressed_lipid_fids"
        group.attrs["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h5.flush()

    print(
        f"UPDATED {path.relative_to(DATA_ROOT)}: achieved={achieved:.8f}, "
        f"backup={backup}, elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )


def launch_background(args: argparse.Namespace) -> int:
    log_file = args.log_file.expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--target",
        str(args.target),
        "--tolerance",
        str(args.tolerance),
        "--method",
        args.method,
        "--gpu-index",
        str(args.gpu_index),
        "--log-file",
        str(log_file),
    ]
    with log_file.open("a", buffering=1) as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            start_new_session=True,
        )
    pid_file = log_file.with_suffix(log_file.suffix + ".pid")
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"3T operator_288 update started (PID {process.pid}).")
    print(f"Log: {log_file}")
    print(f"Follow: tail -f {log_file}")
    return 0


def run(args: argparse.Namespace) -> int:
    if not 0.0 < args.target <= 1.0:
        raise ValueError("--target must be in (0, 1].")
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive.")
    resources = discover_resources()
    preflight(resources)
    print(
        f"Validated {len(resources)} resources; only operator_288 will be changed.",
        flush=True,
    )
    if args.dry_run:
        for path in resources:
            print(path.relative_to(DATA_ROOT))
        print("Dry run complete; no files changed.")
        return 0
    for index, path in enumerate(resources, start=1):
        print(f"[{index}/{len(resources)}] recomputing {path.relative_to(DATA_ROOT)}", flush=True)
        recompute(path, args.target, args.tolerance, args.method, args.gpu_index)
    print(f"All {len(resources)} operator_288 datasets updated successfully.", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if not args.foreground and not args.child and not args.dry_run:
        return launch_background(args)
    try:
        return run(args)
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
