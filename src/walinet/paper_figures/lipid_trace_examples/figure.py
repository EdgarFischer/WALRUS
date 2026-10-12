"""Measured full spectra with trace-dependent L2 lipid estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from walinet.paper_figures.spectrum_selection import load_selected_voxels
from walinet.training_data.lipid_removal import (
    DEFAULT_TRACE_TOLERANCE,
    compute_lipid_projection_operator,
)


def _ppm_axis(n_timepoints: int, bandwidth_hz: float, nmr_frequency_hz: float):
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(n_timepoints, d=1.0 / bandwidth_hz)
    )
    return 4.7 - frequencies / (nmr_frequency_hz / 1e6)


def _prepare_lipid_spectra(resource_path: Path, n_timepoints: int) -> np.ndarray:
    with h5py.File(resource_path, "r") as resources:
        if "lipid_fids" not in resources:
            raise KeyError(f"{resource_path}: missing lipid_fids")
        lipid_fids = np.asarray(resources["lipid_fids"], dtype=np.complex64)
    if lipid_fids.shape[1] < n_timepoints:
        prepared = np.zeros((lipid_fids.shape[0], n_timepoints), dtype=np.complex64)
        prepared[:, :lipid_fids.shape[1]] = lipid_fids
        lipid_fids = prepared
    lipid_fids = lipid_fids[:, :n_timepoints]
    return np.fft.fftshift(
        np.fft.fft(lipid_fids, axis=-1), axes=-1
    ).astype(np.complex64, copy=False)


def compute_trace_examples(
    *,
    subject_dir: str | Path,
    selection_path: str | Path,
    targets: Sequence[float] = (0.95, 0.97, 0.99),
    bandwidth_hz: float,
    nmr_frequency_hz: float,
    resource_relative_path: str = "TrainData/SimulationResources_water_lipid_v1.h5",
    gpu_index: int = 0,
) -> dict:
    """Compute L2 estimates for the first saved voxel and requested targets."""
    subject = Path(subject_dir).expanduser().resolve()
    selected = load_selected_voxels(selection_path)
    if not selected:
        raise ValueError("The selection file contains no voxels")
    label, voxel = next(iter(selected.items()))
    data_path = subject / "OriginalData/data.npy"
    data = np.load(data_path, mmap_mode="r")
    full_fid = np.asarray(data[voxel], dtype=np.complex64)
    full_spectrum = np.fft.fftshift(np.fft.fft(full_fid)).astype(
        np.complex64, copy=False
    )
    # Match the max-absolute normalization used for the network input. The
    # same scale is applied to the full spectrum and every L2 estimate.
    normalization_scale = float(np.max(np.abs(full_spectrum)))
    if not np.isfinite(normalization_scale) or normalization_scale <= 0:
        raise ValueError(f"Invalid full-spectrum normalization: {normalization_scale}")
    normalized_full = full_spectrum / normalization_scale
    n_timepoints = full_spectrum.size
    lipid_spectra = _prepare_lipid_spectra(
        subject / resource_relative_path, n_timepoints
    )
    compact = lipid_spectra[:, None, None, :]
    mask = np.ones(compact.shape[:-1], dtype=bool)
    estimates = {}
    for target in targets:
        target = float(target)
        print(f"Computing trace target {target:.2f} ...", flush=True)
        operator = compute_lipid_projection_operator(
            compact,
            mask,
            target=target,
            tol=DEFAULT_TRACE_TOLERANCE,
            method="gpu",
            gpu_index=int(gpu_index),
        ).astype(np.complex64, copy=False)
        estimates[target] = normalized_full @ operator
    return {
        "voxel_label": label,
        "voxel_index": tuple(voxel),
        "ppm": _ppm_axis(n_timepoints, bandwidth_hz, nmr_frequency_hz),
        "full": normalized_full,
        "estimates": estimates,
        "operator_tolerance": DEFAULT_TRACE_TOLERANCE,
        "normalization": "full_spectrum_max_abs",
        "normalization_scale": normalization_scale,
        "resource_path": str((subject / resource_relative_path).resolve()),
    }


def create_trace_examples_figure(
    datasets: dict[str, dict],
    *,
    targets: Sequence[float] = (0.95, 0.97, 0.99),
    ppm_range: tuple[float, float] = (0.0, 4.0),
    figsize: tuple[float, float] = (10.5, 5.2),
):
    """Return a 2×3 full-spectrum/L2-estimate comparison figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fields = ("3T", "7T")
    figure, axes = plt.subplots(
        2, len(targets), figsize=figsize, sharex=True,
        constrained_layout=True,
    )
    low, high = sorted(ppm_range)
    for row, field in enumerate(fields):
        data = datasets[field]
        ppm = np.asarray(data["ppm"])
        visible = (ppm >= low) & (ppm <= high)
        for column, target in enumerate(targets):
            axis = axes[row, column]
            axis.plot(
                ppm[visible], np.asarray(data["full"])[visible].real,
                color="0.35", linewidth=1.0, alpha=0.72, zorder=1,
            )
            axis.plot(
                ppm[visible], np.asarray(data["estimates"][float(target)])[visible].real,
                color="#D55E00", linewidth=1.35, alpha=0.95, zorder=2,
            )
            axis.set_xlim(high, low)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=9)
            if row == 0:
                axis.set_title(
                    rf"Trace target = {target:.2f}",
                    fontsize=11, fontweight="bold",
                )
            if column == 0:
                axis.set_ylabel(
                    f"{field}\nAmplitude [a.u.]", fontsize=10,
                    fontweight="normal",
                )
            if row == 1:
                axis.set_xlabel("ppm", fontsize=10)
    figure.legend(
        handles=[
            Line2D([0], [0], color="0.35", linewidth=1.0, alpha=0.72,
                   label="Full spectrum"),
            Line2D([0], [0], color="#D55E00", linewidth=1.35,
                   label="L2 lipid estimate"),
        ],
        loc="outside lower center", ncol=2, frameon=False, fontsize=10,
    )
    return figure
