#!/usr/bin/env python3
"""Save the two Vol5 notebook predictions as complex FIDs in OriginalData."""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_fid


def main():
    subject = PROJECT_ROOT / "data/7T/B0corrected_wo_LipidMask/Vol5"
    original = subject / "OriginalData"
    models = (
        ("U20_S1", "UNet_ablation_parameter_matched", False),
        ("Y16_S1", "YNet_ablation", True),
    )
    for label, model, _ in models:
        if (original / f"data_{label}.npy").exists():
            raise FileExistsError(original / f"data_{label}.npy")
        if not (PROJECT_ROOT / "models" / model / "model_last.pt").is_file():
            raise FileNotFoundError(model)

    fid = np.load(original / "data.npy", mmap_mode="r")
    headmask = (
        np.load(subject / "masks/brain_mask.npy")
        + np.load(subject / "masks/lipid_mask.npy")
    )
    with h5py.File(subject / "TrainData/SimulationResources_water_lipid_v1.h5", "r") as f:
        projection = f["lipid_projection"]
        target = float(projection.attrs["target"])
        if not np.isclose(target, 0.99, rtol=0, atol=1e-8):
            raise ValueError(f"Expected projection target 0.99, got {target}")
        operator = np.asarray(projection[f"operator_{fid.shape[-1]}"][:], dtype=np.complex64)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Vol5: {fid.shape}; device={device}; model_last.pt; headmask=brain+lipid; target={target}", flush=True)
    for label, model, needs_operator in models:
        output = original / f"data_{label}.npy"
        temporary = original / f".{output.stem}.{os.getpid()}.tmp.npy"
        print(f"START {label}: {model}", flush=True)
        try:
            clean = infer_fid(
                fid=fid,
                model_dir=PROJECT_ROOT / "models" / model,
                fid_axis=-1,
                headmask=headmask,
                lipid_projection_operator=operator if needs_operator else None,
                checkpoint="model_last.pt",
                batch_size=200,
                device=device,
            )
            if clean.shape != fid.shape or clean.dtype != np.complex64:
                raise ValueError(f"Unexpected output: {clean.shape}, {clean.dtype}")
            if not np.isfinite(clean).all():
                raise ValueError("Output contains NaN or Inf")
            np.save(temporary, clean)
            temporary.replace(output)
            del clean
        finally:
            temporary.unlink(missing_ok=True)
        print(f"SAVED {output}", flush=True)


if __name__ == "__main__":
    main()
