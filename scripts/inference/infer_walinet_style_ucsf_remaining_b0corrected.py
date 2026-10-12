#!/usr/bin/env python3
"""Run WALINET-Style inference on B0-corrected UCSF Vol1, Vol2, Vol4 and Vol5."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_fid  # noqa: E402


DATA_ROOT = (
    PROJECT_ROOT.parent
    / "Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/UCSF"
)
SUBJECTS = ("Vol1", "Vol2", "Vol4", "Vol5")
MODEL_DIR = PROJECT_ROOT / "models/WALINET-Style"
RESOURCE_NAME = "SimulationResources_water_lipid_v1.h5"
OPERATOR_KEY = "lipid_projection/operator_840"
DEFAULT_LOG = PROJECT_ROOT / "logs/infer_walinet_style_ucsf_remaining_b0corrected.log"


def paths_for(subject: str) -> tuple[Path, Path, Path, Path]:
    subject_dir = DATA_ROOT / subject
    return (
        subject_dir / "OriginalData/data.npy",
        subject_dir / "masks/brain_mask.npy",
        subject_dir / "TrainData" / RESOURCE_NAME,
        subject_dir / "OriginalData/data_WALINET_Style.npy",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    checkpoint = MODEL_DIR / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    problems: list[str] = []
    for subject in SUBJECTS:
        data_path, mask_path, resource_path, output_path = paths_for(subject)
        for path in (data_path, mask_path, resource_path):
            if not path.is_file():
                problems.append(f"{subject}: missing {path}")
        if problems and any(str(path) in problems[-1] for path in (data_path, mask_path, resource_path)):
            continue

        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if data.shape != (64, 64, 35, 840):
            problems.append(f"{subject}: unexpected data shape {data.shape}")
        if mask.shape != data.shape[:-1]:
            problems.append(f"{subject}: mask {mask.shape} != data space {data.shape[:-1]}")
        if not np.issubdtype(data.dtype, np.complexfloating):
            problems.append(f"{subject}: data dtype {data.dtype} is not complex")

        with h5py.File(resource_path, "r") as resources:
            if OPERATOR_KEY not in resources:
                problems.append(f"{subject}: missing HDF5 key {OPERATOR_KEY}")
            elif resources[OPERATOR_KEY].shape != (840, 840):
                problems.append(
                    f"{subject}: operator shape {resources[OPERATOR_KEY].shape} != (840, 840)"
                )
            target = float(resources["lipid_projection"].attrs.get("target", np.nan))
            if not np.isclose(target, 0.99, rtol=0.0, atol=1e-8):
                problems.append(f"{subject}: projection target {target} != 0.99")

        if output_path.exists() and not args.overwrite:
            output = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if output.shape != data.shape or output.dtype != np.complex64:
                problems.append(
                    f"{subject}: existing output is {output.shape}/{output.dtype}, "
                    f"expected {data.shape}/complex64"
                )

    if problems:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  {p}" for p in problems))


def process_one(subject: str, args: argparse.Namespace) -> None:
    data_path, mask_path, resource_path, output_path = paths_for(subject)
    if output_path.exists() and not args.overwrite:
        print(f"SKIP {subject}: output already exists", flush=True)
        return

    source = np.load(data_path, mmap_mode="r", allow_pickle=False)
    expected_shape = source.shape
    del source
    with h5py.File(resource_path, "r") as resources:
        operator = np.asarray(resources[OPERATOR_KEY], dtype=np.complex64)

    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.npy"
    )
    temporary_path.unlink(missing_ok=True)
    try:
        infer_fid(
            fid=data_path,
            headmask=mask_path,
            model_dir=MODEL_DIR,
            lipid_projection_operator=operator,
            output_path=temporary_path,
            fid_axis=-1,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
        )
        output = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if output.shape != expected_shape or output.dtype != np.complex64:
            raise RuntimeError(
                f"Invalid output {output.shape}/{output.dtype}; "
                f"expected {expected_shape}/complex64"
            )
        if not np.isfinite(output).all():
            raise RuntimeError("Output contains NaN or Inf")
        del output
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"DONE {subject}: {output_path}", flush=True)


def run(args: argparse.Namespace) -> int:
    preflight(args)
    print(f"Model:      {MODEL_DIR}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Device:     {args.device}", flush=True)
    for subject in SUBJECTS:
        data_path, mask_path, resource_path, output_path = paths_for(subject)
        state = "OVERWRITE" if output_path.exists() and args.overwrite else (
            "SKIP" if output_path.exists() else "PENDING"
        )
        print(f"{state}: {subject}", flush=True)
        print(f"  Input:    {data_path}", flush=True)
        print(f"  Mask:     {mask_path}", flush=True)
        print(f"  Operator: {resource_path}::{OPERATOR_KEY}", flush=True)
        print(f"  Output:   {output_path}", flush=True)
    if args.dry_run:
        print("Dry run successful; no output written.", flush=True)
        return 0

    for index, subject in enumerate(SUBJECTS, 1):
        print(f"[{index}/{len(SUBJECTS)}] START {subject}", flush=True)
        process_one(subject, args)
    print("All remaining UCSF WALINET-Style inferences completed.", flush=True)
    return 0


def launch_background(args: argparse.Namespace) -> int:
    log_file = args.log_file.expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()), "--child",
        "--device", args.device, "--batch-size", str(args.batch_size),
        "--checkpoint", args.checkpoint, "--log-file", str(log_file),
    ]
    if args.overwrite:
        command.append("--overwrite")
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
    print(f"WALINET-Style batch started in background (PID {process.pid}).")
    print(f"Log: {log_file}")
    print(f"Follow: tail -f {log_file}")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return run(args)
    if not args.foreground and not args.child:
        preflight(args)
        return launch_background(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
