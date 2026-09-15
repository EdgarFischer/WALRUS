#!/usr/bin/env python3
"""Run 7T_Final inference for the five B0-corrected UCSF subjects."""

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


DATA_ROOT = PROJECT_ROOT / "data/7T/B0corrected_wo_LipidMask"
MODEL_DIR = PROJECT_ROOT / "models/7T_Final"
SUBJECTS = tuple(f"Vol{index}_UCSF" for index in range(1, 6))
DEFAULT_LOG = PROJECT_ROOT / "logs/infer_ucsf_7t_b0corrected.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def paths_for(subject: str) -> tuple[Path, Path, Path]:
    subject_dir = DATA_ROOT / subject
    return (
        subject_dir / "OriginalData/data.npy",
        subject_dir / "masks/brain_mask.npy",
        subject_dir / "OriginalData/data_after_walinet.npy",
    )


def preflight(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    checkpoint = MODEL_DIR / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    problems: list[str] = []
    for subject in SUBJECTS:
        data_path, mask_path, output_path = paths_for(subject)
        if not data_path.is_file():
            problems.append(f"{subject}: missing {data_path}")
            continue
        if not mask_path.is_file():
            problems.append(f"{subject}: missing {mask_path}")
            continue
        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if data.ndim != 4 or data.shape[-1] != 840:
            problems.append(f"{subject}: unexpected data shape {data.shape}")
        if mask.shape != data.shape[:-1]:
            problems.append(f"{subject}: mask shape {mask.shape} != {data.shape[:-1]}")
        if not np.issubdtype(data.dtype, np.complexfloating):
            problems.append(f"{subject}: data dtype {data.dtype} is not complex")
        if output_path.exists() and not args.overwrite:
            output = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if output.shape != data.shape or output.dtype != np.complex64:
                problems.append(
                    f"{subject}: existing output has shape/dtype "
                    f"{output.shape}/{output.dtype}"
                )
    if problems:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  {p}" for p in problems))


def process_subject(subject: str, args: argparse.Namespace) -> None:
    data_path, mask_path, output_path = paths_for(subject)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.npy"
    )
    if output_path.exists() and not args.overwrite:
        print(f"SKIP {subject}: output already exists", flush=True)
        return

    source = np.load(data_path, mmap_mode="r", allow_pickle=False)
    expected_shape = source.shape
    del source
    temporary_path.unlink(missing_ok=True)
    try:
        infer_fid(
            fid=data_path,
            headmask=mask_path,
            model_dir=MODEL_DIR,
            output_path=temporary_path,
            fid_axis=-1,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
        )
        candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if candidate.shape != expected_shape:
            raise RuntimeError(
                f"{subject}: output shape {candidate.shape} != {expected_shape}"
            )
        if candidate.dtype != np.complex64:
            raise RuntimeError(f"{subject}: output dtype {candidate.dtype} != complex64")
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"{subject}: output contains NaN or Inf")
        del candidate
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"DONE {subject}: {output_path}", flush=True)


def run(args: argparse.Namespace) -> int:
    preflight(args)
    print(f"Data root:  {DATA_ROOT.resolve()}", flush=True)
    print(f"Model:      {MODEL_DIR.resolve()}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Device:     {args.device}", flush=True)
    print(f"Subjects:   {len(SUBJECTS)} ({', '.join(SUBJECTS)})", flush=True)
    for subject in SUBJECTS:
        data_path, mask_path, output_path = paths_for(subject)
        state = "OVERWRITE" if output_path.exists() and args.overwrite else (
            "SKIP" if output_path.exists() else "PENDING"
        )
        print(f"{subject}: {state}\n  data={data_path}\n  mask={mask_path}\n  output={output_path}", flush=True)
    if args.dry_run:
        print("Dry run complete; no inference performed.", flush=True)
        return 0
    for index, subject in enumerate(SUBJECTS, start=1):
        print(f"[{index}/5] processing {subject}", flush=True)
        process_subject(subject, args)
    print("All five UCSF outputs completed successfully.", flush=True)
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
    print(f"UCSF 7T inference started in background (PID {process.pid}).")
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
