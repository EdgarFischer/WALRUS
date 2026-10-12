#!/usr/bin/env python3
"""Run the three WALINET-calibration UNets on all five B0-corrected UCSF subjects."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_fid  # noqa: E402


DATA_ROOT = (
    PROJECT_ROOT.parent
    / "Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/UCSF"
)
SUBJECTS = tuple(f"Vol{index}" for index in range(1, 6))
MODELS = tuple(
    (
        f"U20_WALINET_Calibration_S{seed}",
        f"UNet_ablation_WALINET_Calibration_S{seed}",
    )
    for seed in range(1, 4)
)
DEFAULT_LOG = PROJECT_ROOT / "logs/infer_walinet_calibration_ucsf_7t_b0corrected.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="For example cuda:0 or cuda:1")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files after validating the new inference result.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in this terminal instead of launching a background process.",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def paths_for(subject: str, output_name: str) -> tuple[Path, Path, Path]:
    subject_dir = DATA_ROOT / subject
    original_data = subject_dir / "OriginalData"
    return (
        original_data / "data.npy",
        subject_dir / "masks/brain_mask.npy",
        original_data / f"data_{output_name}.npy",
    )


def preflight(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    problems: list[str] = []
    for output_name, model_name in MODELS:
        checkpoint = PROJECT_ROOT / "models" / model_name / args.checkpoint
        if not checkpoint.is_file():
            problems.append(f"{output_name}: missing checkpoint {checkpoint}")

    for subject in SUBJECTS:
        data_path, mask_path, _ = paths_for(subject, MODELS[0][0])
        if not data_path.is_file():
            problems.append(f"{subject}: missing input {data_path}")
            continue
        if not mask_path.is_file():
            problems.append(f"{subject}: missing mask {mask_path}")
            continue

        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if data.ndim != 4 or data.shape[-1] != 840:
            problems.append(f"{subject}: expected (..., 840) data, got {data.shape}")
        if mask.shape != data.shape[:-1]:
            problems.append(f"{subject}: mask {mask.shape} != data space {data.shape[:-1]}")
        if not np.issubdtype(data.dtype, np.complexfloating):
            problems.append(f"{subject}: input dtype {data.dtype} is not complex")

        for output_name, _ in MODELS:
            output_path = paths_for(subject, output_name)[2]
            if output_path.exists() and not args.overwrite:
                existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
                if existing.shape != data.shape or existing.dtype != np.complex64:
                    problems.append(
                        f"{subject}/{output_name}: existing output has "
                        f"{existing.shape}/{existing.dtype}, expected {data.shape}/complex64"
                    )

    if problems:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  {p}" for p in problems))


def process_one(
    subject: str,
    output_name: str,
    model_name: str,
    args: argparse.Namespace,
) -> None:
    data_path, mask_path, output_path = paths_for(subject, output_name)
    if output_path.exists() and not args.overwrite:
        print(f"SKIP {subject}/{output_name}: output already exists", flush=True)
        return

    source = np.load(data_path, mmap_mode="r", allow_pickle=False)
    expected_shape = source.shape
    del source

    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.npy"
    )
    temporary_path.unlink(missing_ok=True)
    try:
        infer_fid(
            fid=data_path,
            headmask=mask_path,
            model_dir=PROJECT_ROOT / "models" / model_name,
            output_path=temporary_path,
            fid_axis=-1,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
        )
        candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if candidate.shape != expected_shape or candidate.dtype != np.complex64:
            raise RuntimeError(
                f"Invalid output {candidate.shape}/{candidate.dtype}; "
                f"expected {expected_shape}/complex64"
            )
        if not np.isfinite(candidate).all():
            raise RuntimeError("Output contains NaN or Inf")
        del candidate
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"DONE {subject}/{output_name}: {output_path}", flush=True)


def run(args: argparse.Namespace) -> int:
    preflight(args)
    print(f"Data root:  {DATA_ROOT.resolve()}", flush=True)
    print(f"Device:     {args.device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Subjects:   {', '.join(SUBJECTS)}", flush=True)
    print("Models:", flush=True)
    for output_name, model_name in MODELS:
        print(f"  {output_name}: {model_name}", flush=True)

    jobs = [
        (subject, output_name, model_name)
        for output_name, model_name in MODELS
        for subject in SUBJECTS
    ]
    pending = 0
    for subject, output_name, _ in jobs:
        output_path = paths_for(subject, output_name)[2]
        state = "OVERWRITE" if output_path.exists() and args.overwrite else (
            "SKIP" if output_path.exists() else "PENDING"
        )
        pending += state != "SKIP"
        print(f"{state}: {subject}/{output_name} -> {output_path}", flush=True)
    print(f"Total: {len(jobs)} outputs; {pending} to calculate.", flush=True)

    if args.dry_run:
        print("Dry run complete; no inference performed.", flush=True)
        return 0

    for index, (subject, output_name, model_name) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] START {subject}/{output_name}", flush=True)
        process_one(subject, output_name, model_name, args)

    print("All three models on all five UCSF subjects completed.", flush=True)
    return 0


def launch_background(args: argparse.Namespace) -> int:
    log_file = args.log_file.expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--checkpoint",
        args.checkpoint,
        "--log-file",
        str(log_file),
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
    print(f"WALINET-calibration UCSF inference started in background (PID {process.pid}).")
    print(f"Log: {log_file}")
    print(f"Follow: tail -f {log_file}")
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
