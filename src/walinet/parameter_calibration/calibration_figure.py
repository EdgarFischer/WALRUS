from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm

from walinet.parameter_calibration.pipeline_FWHM_SNR_shifts import (
    ParameterCalibrationResult,
)


def _zero_truncated_normal_pdf(
    x: np.ndarray,
    *,
    mean: float,
    std: float,
) -> np.ndarray:
    probability_above_zero = norm.sf(-mean / std)
    density = norm.pdf(x, loc=mean, scale=std) / probability_above_zero
    return np.where(x >= 0, density, 0.0)


def plot_snr_phase_calibration_pair(
    snr_result: ParameterCalibrationResult,
    phase_result: ParameterCalibrationResult,
    *,
    snr_xmax: float = 100.0,
    bins: int = 30,
    figsize: tuple[float, float] = (12.0, 4.6),
    save_path: str | Path | None = None,
    dpi: int = 300,
    font_size: float = 12.0,
    left_title: str = "SNR",
    left_xlabel: str = "SNR",
    right_title: str = "Zero-order phase",
    right_xlabel: str = "Zero-order phase [degrees]",
    left_heuristic_truncated_normal: tuple[float, float] | None = None,
    right_heuristic_uniform: tuple[float, float] | None = None,
    heuristic_label: str = "Original WALINET distribution",
    heuristic_color: str = "#7B3294",
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot positive- and symmetric-mixture calibrations side by side."""
    if getattr(snr_result, "distribution", "positive_mixture") != "positive_mixture":
        raise ValueError("snr_result must use distribution='positive_mixture'.")
    if phase_result.distribution != "symmetric_mixture":
        raise ValueError("phase_result must use distribution='symmetric_mixture'.")
    if not np.isfinite(snr_xmax) or snr_xmax <= 0:
        raise ValueError("snr_xmax must be finite and greater than zero.")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = np.asarray(axes, dtype=object).reshape(-1)
        if len(axes) != 2:
            raise ValueError("axes must contain exactly two matplotlib axes.")
        fig = axes[0].figure
        if axes[1].figure is not fig:
            raise ValueError("Both axes must belong to the same figure.")

    # SNR: 50% zero-truncated normal + 50% lognormal.
    snr_x = np.linspace(max(snr_xmax * 1e-8, np.finfo(float).tiny), snr_xmax, 1200)
    snr_normal = _zero_truncated_normal_pdf(
        snr_x,
        mean=snr_result.normal_mu,
        std=snr_result.normal_sigma,
    )
    snr_lognormal = lognorm.pdf(
        snr_x,
        s=snr_result.log_sigma,
        scale=np.exp(snr_result.log_mu),
    )
    snr_mixture = (
        snr_result.normal_weight * snr_normal
        + snr_result.lognormal_weight * snr_lognormal
    )

    ax = axes[0]
    ax.hist(
        snr_result.pooled_values,
        bins=bins,
        range=(0.0, snr_xmax),
        density=True,
        alpha=0.6,
        label=r"$\mathit{In\,vivo}$ values",
    )
    ax.plot(
        snr_x,
        snr_normal,
        "--",
        linewidth=1.5,
        label="Zero-truncated normal component",
    )
    ax.plot(
        snr_x,
        snr_lognormal,
        ":",
        linewidth=1.8,
        label="Lognormal component",
    )
    ax.plot(snr_x, snr_mixture, linewidth=3, label="Final 50/50 mixture")
    if left_heuristic_truncated_normal is not None:
        heuristic_mean, heuristic_std = left_heuristic_truncated_normal
        heuristic_density = _zero_truncated_normal_pdf(
            snr_x,
            mean=heuristic_mean,
            std=heuristic_std,
        )
        ax.plot(
            snr_x,
            heuristic_density,
            color=heuristic_color,
            linestyle="-.",
            linewidth=2.5,
            label=heuristic_label,
            zorder=6,
        )
    ax.axvline(
        snr_result.median,
        linestyle="--",
        linewidth=1,
        label="Median",
    )
    ax.set(xlim=(0.0, snr_xmax), xlabel=left_xlabel, ylabel="Probability density")
    ax.set_title(left_title, fontsize=font_size + 1)
    ax.set_xlabel(left_xlabel, fontsize=font_size)
    ax.set_ylabel("Probability density", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size - 1)
    ax.legend(fontsize=font_size - 1)

    # Phase: 50% central normal + 25% tail on either side.
    phase_center = phase_result.center
    if phase_center is None:
        raise ValueError("phase_result does not contain a symmetric center.")
    phase_xmin = phase_result.plot_xmin
    phase_xmax = phase_result.plot_xmax
    if right_heuristic_uniform is not None:
        uniform_min, uniform_max = right_heuristic_uniform
        if not uniform_min < uniform_max:
            raise ValueError("right_heuristic_uniform must satisfy minimum < maximum.")
        phase_xmin = min(phase_xmin, uniform_min)
        phase_xmax = max(phase_xmax, uniform_max)
    phase_x = np.linspace(phase_xmin, phase_xmax, 1200)
    phase_normal = norm.pdf(
        phase_x,
        loc=phase_center,
        scale=phase_result.normal_sigma,
    )
    phase_tail = 0.5 * lognorm.pdf(
        np.abs(phase_x - phase_center),
        s=phase_result.log_sigma,
        scale=np.exp(phase_result.log_mu),
    )
    phase_mixture = (
        phase_result.normal_weight * phase_normal
        + phase_result.lognormal_weight * phase_tail
    )

    ax = axes[1]
    ax.hist(
        phase_result.pooled_values,
        bins=bins,
        range=(phase_result.plot_xmin, phase_result.plot_xmax),
        density=True,
        alpha=0.6,
        label=r"$\mathit{In\,vivo}$ values",
    )
    ax.plot(
        phase_x,
        phase_normal,
        "--",
        linewidth=1.5,
        label="Central normal component",
    )
    ax.plot(
        phase_x,
        phase_tail,
        ":",
        linewidth=1.8,
        label="Symmetric lognormal-tail component",
    )
    ax.plot(phase_x, phase_mixture, linewidth=3, label="Final 50/25/25 mixture")
    if right_heuristic_uniform is not None:
        uniform_density = 1.0 / (uniform_max - uniform_min)
        ax.plot(
            [phase_xmin, uniform_min, uniform_min, uniform_max, uniform_max, phase_xmax],
            [0.0, 0.0, uniform_density, uniform_density, 0.0, 0.0],
            color=heuristic_color,
            linestyle="-.",
            linewidth=2.5,
            label=heuristic_label,
            zorder=6,
        )
    ax.axvline(
        phase_center,
        linestyle="--",
        linewidth=1,
        label="Median",
    )
    ax.set(
        xlim=(phase_xmin, phase_xmax),
        xlabel=right_xlabel,
        ylabel="Probability density",
    )
    ax.set_title(right_title, fontsize=font_size + 1)
    ax.set_xlabel(right_xlabel, fontsize=font_size)
    ax.set_ylabel("Probability density", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size - 1)
    ax.legend(fontsize=font_size - 1)

    fig.tight_layout()
    if save_path is not None:
        destination = Path(save_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=dpi, bbox_inches="tight")

    return fig, axes
