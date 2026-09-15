#!/usr/bin/env python3
"""Update 3T training or validation resources with target-trace 0.991 operators."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.training_data.lipid_removal import (  # noqa: E402
    compute_lipid_projection_operator,
)


TRAINING_CONFIG = PROJECT_ROOT / "configs/Training/train_3T.yaml"
TARGET = 0.991
TOLERANCE = 0.001
N_TIMEPOINTS = (142, 200, 288)
EXPECTED_RESOURCES = {"train": 49, "validation": 12}
DEFAULT_LOG = PROJECT_ROOT / "logs/update_train_3t_lipid_operators.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("auto", "gpu", "eigenvalue", "legacy"),
        default="auto",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="train",
        help="Select train_subjects or val_subjects from train_3T.yaml.",
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list resources without changing anything.",
    )
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
    return logging.getLogger("update_train_3t_lipid_operators")


def load_resources(split: str) -> list[tuple[str, Path]]:
    raw = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    data_cfg = raw["data"]
    base_dir = (TRAINING_CONFIG.parent / data_cfg["base_dir"]).resolve()
    filename = data_cfg["resources"]["filename"].format(
        version=data_cfg["resources"]["version"]
    )
    subject_key = "train_subjects" if split == "train" else "val_subjects"
    subjects = list(data_cfg[subject_key])
    expected = EXPECTED_RESOURCES[split]
    if len(subjects) != expected or len(set(subjects)) != len(subjects):
        raise RuntimeError(
            f"Expected {expected} unique {subject_key}, found "
            f"{len(subjects)} entries and {len(set(subjects))} unique entries."
        )
    return [(subject, base_dir / subject / filename) for subject in subjects]


def preflight(resources: list[tuple[str, Path]]) -> None:
    problems: list[str] = []
    for subject, path in resources:
        if not path.is_file():
            problems.append(f"{subject}: missing resource file: {path}")
            continue
        try:
            with h5py.File(path, "r") as h5:
                if "lipid_fids" not in h5:
                    problems.append(f"{subject}: missing lipid_fids")
                    continue
                shape = h5["lipid_fids"].shape
                if len(shape) != 2 or shape[0] == 0 or shape[1] < max(N_TIMEPOINTS):
                    problems.append(f"{subject}: invalid lipid_fids shape {shape}")
                if "lipid_projection" not in h5:
                    problems.append(f"{subject}: missing lipid_projection group")
                    continue
                for n in N_TIMEPOINTS[:-1]:
                    name = f"operator_{n}"
                    if name not in h5["lipid_projection"]:
                        problems.append(f"{subject}: missing lipid_projection/{name}")
                    elif h5["lipid_projection"][name].shape != (n, n):
                        problems.append(
                            f"{subject}: invalid {name} shape "
                            f"{h5['lipid_projection'][name].shape}"
                        )
        except OSError as error:
            problems.append(f"{subject}: cannot read HDF5: {error}")
    if problems:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  {p}" for p in problems))


def number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def save_backup(
    resource_path: Path,
    dataset_name: str,
    operator: np.ndarray,
    old_target: float,
    old_tolerance: float,
) -> Path:
    backup_dir = resource_path.parent / "operator_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{dataset_name}_target_{number_label(old_target)}_"
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
                    "dataset": f"lipid_projection/{dataset_name}",
                    "target": old_target,
                    "tol": old_tolerance,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return backup_path


def prepare_fids(fids: np.ndarray, n_timepoints: int) -> np.ndarray:
    if fids.shape[-1] == n_timepoints:
        return fids
    if fids.shape[-1] > n_timepoints:
        return np.ascontiguousarray(fids[:, :n_timepoints])
    prepared = np.zeros((len(fids), n_timepoints), dtype=np.complex64)
    prepared[:, : fids.shape[-1]] = fids
    return prepared


def achieved_trace(operator: np.ndarray) -> float:
    n = operator.shape[0]
    return float(np.mean(np.abs(np.diag(np.eye(n) - operator))))


def recompute_resource(
    subject: str,
    resource_path: Path,
    method: str,
    gpu_index: int,
    logger: logging.Logger,
) -> None:
    started = time.monotonic()
    with h5py.File(resource_path, "r") as h5:
        lipid_fids = np.asarray(h5["lipid_fids"][:], dtype=np.complex64)
        group = h5["lipid_projection"]
        old_target = float(group.attrs.get("target", np.nan))
        old_tolerance = float(group.attrs.get("tol", np.nan))
        old_operators = {
            name: np.asarray(group[name][:], dtype=np.complex64)
            for name in ("operator_142", "operator_200")
        }

    for name, operator in old_operators.items():
        backup = save_backup(
            resource_path,
            name,
            operator,
            old_target,
            old_tolerance,
        )
        logger.info("%s: backup saved/present: %s", subject, backup)
    del old_operators

    new_operators: dict[int, np.ndarray] = {}
    achieved: dict[int, float] = {}
    for n in N_TIMEPOINTS:
        prepared = prepare_fids(lipid_fids, n)
        spectra = np.fft.fftshift(
            np.fft.fft(prepared, axis=-1), axes=-1
        ).astype(np.complex64, copy=False)
        operator = compute_lipid_projection_operator(
            spectra=spectra[:, None, None, :],
            lipid_mask=np.ones((len(spectra), 1, 1), dtype=bool),
            target=TARGET,
            tol=TOLERANCE,
            method=method,
            gpu_index=gpu_index,
        )
        operator = np.asarray(operator, dtype=np.complex64)
        if operator.shape != (n, n) or not np.isfinite(operator).all():
            raise RuntimeError(f"{subject}: invalid newly computed operator_{n}")
        value = achieved_trace(operator)
        if abs(value - TARGET) > TOLERANCE:
            raise RuntimeError(
                f"{subject}: operator_{n} achieved {value:.8f}, outside "
                f"{TARGET} +/- {TOLERANCE}"
            )
        new_operators[n] = operator
        achieved[n] = value

    with h5py.File(resource_path, "r+") as h5:
        group = h5["lipid_projection"]
        for n, operator in new_operators.items():
            name = f"operator_{n}"
            if name in group:
                group[name][...] = operator
            else:
                group.create_dataset(name, data=operator, dtype=np.complex64)
            group.attrs[f"operator_{n}_achieved_diag_mean"] = achieved[n]
        group.attrs["configured_n_timepoints"] = np.asarray(
            N_TIMEPOINTS, dtype=np.int64
        )
        group.attrs["target"] = TARGET
        group.attrs["tol"] = TOLERANCE
        group.attrs["source"] = "water_suppressed_lipid_fids"
        group.attrs["updated_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        h5.flush()

    values = ", ".join(f"{n}={achieved[n]:.8f}" for n in N_TIMEPOINTS)
    logger.info(
        "%s updated (%s; %.1fs)", subject, values, time.monotonic() - started
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file)
    resources = load_resources(args.split)
    preflight(resources)
    logger.info("Validated %d 3T %s resources.", len(resources), args.split)
    for subject, path in resources:
        logger.info("RESOURCE %s: %s", subject, path)
    if args.dry_run:
        logger.info("Dry run complete; no files changed.")
        return
    for index, (subject, path) in enumerate(resources, start=1):
        logger.info("[%d/%d] recomputing %s", index, len(resources), subject)
        recompute_resource(
            subject, path, args.method, args.gpu_index, logger
        )
    logger.info(
        "All %d 3T %s resources updated successfully.",
        len(resources),
        args.split,
    )


if __name__ == "__main__":
    main()
