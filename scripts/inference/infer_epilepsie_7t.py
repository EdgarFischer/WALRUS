#!/usr/bin/env python3
"""Run 7T_Final WALRUS inference on the two prepared epilepsy datasets."""

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


DATA_ROOT = PROJECT_ROOT / "data/7T/NoB0Correction/Epilepsie"
MODEL_DIR = PROJECT_ROOT / "models/7T_Final"
SUBJECTS = ("EPI_0001", "EPI_0002")
DEFAULT_LOG = PROJECT_ROOT / "logs/infer_epilepsie_7t.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def paths_for(subject: str) -> tuple[Path, Path, Path, Path]:
    subject_dir = DATA_ROOT / subject
    return (
        subject_dir / "OriginalData/data.npy",
        subject_dir / "masks/brain_mask.npy",
        subject_dir / "masks/lipid_mask.npy",
        subject_dir / "OriginalData/data_after_walrus.npy",
    )


def preflight(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    checkpoint_path = MODEL_DIR / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    problems: list[str] = []
    for subject in SUBJECTS:
        data_path, brain_path, lipid_path, output_path = paths_for(subject)
        missing = [
            path
            for path in (data_path, brain_path, lipid_path)
            if not path.is_file()
        ]
        if missing:
            problems.extend(f"{subject}: missing {path}" for path in missing)
            continue

        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        brain = np.load(brain_path, mmap_mode="r", allow_pickle=False)
        lipid = np.load(lipid_path, mmap_mode="r", allow_pickle=False)

        if data.ndim != 4 or data.shape[-1] != 840:
            problems.append(
                f"{subject}: expected data shape (X,Y,Z,840), found {data.shape}"
            )
        if brain.shape != data.shape[:-1]:
            problems.append(
                f"{subject}: brain-mask shape {brain.shape} != {data.shape[:-1]}"
            )
        if lipid.shape != data.shape[:-1]:
            problems.append(
                f"{subject}: lipid-mask shape {lipid.shape} != {data.shape[:-1]}"
            )
        if not np.issubdtype(data.dtype, np.complexfloating):
            problems.append(f"{subject}: data dtype {data.dtype} is not complex")

        if output_path.exists() and not args.overwrite:
            output = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if output.shape != data.shape or output.dtype != np.complex64:
                problems.append(
                    f"{subject}: existing output has unexpected shape/dtype "
                    f"{output.shape}/{output.dtype}; use --overwrite to replace it"
                )

    if problems:
        raise RuntimeError(
            "Preflight failed:\n" + "\n".join(f"  {item}" for item in problems)
        )


def process_subject(subject: str, args: argparse.Namespace) -> None:
    data_path, brain_path, lipid_path, output_path = paths_for(subject)

    if output_path.exists() and not args.overwrite:
        print(f"SKIP {subject}: output already exists", flush=True)
        return

    source = np.load(data_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(source.shape)
    input_dtype = source.dtype
    del source

    brain = np.load(brain_path, allow_pickle=False) > 0
    lipid = np.load(lipid_path, allow_pickle=False) > 0
    headmask = brain | lipid
    del brain, lipid

    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.npy"
    )
    temporary_path.unlink(missing_ok=True)

    print(f"Input:      {data_path}", flush=True)
    print(f"Input data: {expected_shape}, {input_dtype}", flush=True)
    print(
        f"Head mask:  brain OR lipid ({np.count_nonzero(headmask)} voxels)",
        flush=True,
    )
    print(f"Output:     {output_path}", flush=True)

    try:
        infer_fid(
            fid=data_path,
            headmask=headmask,
            model_dir=MODEL_DIR,
            output_path=temporary_path,
            fid_axis=-1,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            lipid_projection_operator=None,
        )

        candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if candidate.shape != expected_shape:
            raise RuntimeError(
                f"{subject}: output shape {candidate.shape} != {expected_shape}"
            )
        if candidate.dtype != np.complex64:
            raise RuntimeError(
                f"{subject}: output dtype {candidate.dtype} != complex64"
            )
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"{subject}: output contains NaN or Inf")
        del candidate

        os.replace(temporary_path, output_path)
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
    print(f"Batch size: {args.batch_size}", flush=True)
    print("Head mask: brain_mask OR lipid_mask", flush=True)
    print("Lipid projection: disabled (U-Net)", flush=True)

    for subject in SUBJECTS:
        data_path, brain_path, lipid_path, output_path = paths_for(subject)
        if output_path.exists():
            state = "OVERWRITE" if args.overwrite else "SKIP"
        else:
            state = "PENDING"
        print(
            f"{subject}: {state}\n"
            f"  input={data_path}\n"
            f"  brain_mask={brain_path}\n"
            f"  lipid_mask={lipid_path}\n"
            f"  output={output_path}",
            flush=True,
        )

    if args.dry_run:
        print("Dry run complete; no inference performed.", flush=True)
        return 0

    for index, subject in enumerate(SUBJECTS, start=1):
        print(f"[{index}/{len(SUBJECTS)}] PROCESS {subject}", flush=True)
        process_subject(subject, args)

    print("Both epilepsy WALRUS outputs completed successfully.", flush=True)
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
    print(f"Epilepsy inference started in background (PID {process.pid}).")
    print(f"Log: {log_file}")
    print(f"Follow: tail -f {log_file}")
    print(f"PID file: {pid_file}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        preflight(args)
        if not args.foreground and not args.child and not args.dry_run:
            return launch_background(args)
        return run(args)
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
