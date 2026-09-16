"""Three-panel 3T simulation-error and LCModel fit-rate figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from walinet.paper_figures.hlsvd_l2_comparison.results import results_dataframe
from walinet.paper_figures.lcmodel_quality import plot_fit_rates


METHODS = ("HLSVD + L2", "WALRUS")
# Match the first two metabolite colors used in panel c (tNAA and tCr).
METHOD_COLORS = {"HLSVD + L2": "#0072b2", "WALRUS": "#e69f00"}
SIMULATION_PANELS = (
    (
        "residual_l2_over_nuisance_l2",
        "a) Nuisance-normalized residual",
        r"$\|\hat n-n\|_2 / \|n\|_2$",
    ),
    (
        "residual_l2_over_clean_metabolite_l2",
        "b) Metabolite-normalized residual",
        r"$\|\hat n-n\|_2 / \|m\|_2$",
    ),
)


def _plot_simulation_metric(axis, frame, metric, title, ylabel, *, logarithmic):
    subset = frame[frame.metric == metric]
    subjects = list(dict.fromkeys(subset.subject))
    x = np.arange(len(METHODS), dtype=float)
    for method_index, method in enumerate(METHODS):
        indexed = subset[subset.method == method].set_index("subject")
        missing = set(subjects) - set(indexed.index)
        if missing:
            raise ValueError(f"{metric}/{method}: missing subjects {missing}")
        values = indexed.loc[subjects, "mean"].to_numpy(dtype=float)
        axis.errorbar(
            method_index,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=1.8,
            color=METHOD_COLORS[method],
        )
    axis.set_xticks(x, METHODS)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.set_title(title, loc="left", fontweight="bold", fontsize=11)
    axis.tick_params(axis="both", labelsize=11)
    if logarithmic:
        axis.set_yscale("log")
    else:
        axis.margins(y=0.15)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2, which="major")


def create_quantitative_figure(
    *,
    simulation_results: str | Path,
    subject_fit_rates,
    simulation_log_scale: bool = True,
    figsize=(12.5, 4.0),
):
    """Return the combined figure and the subject-level simulation table."""
    import matplotlib.pyplot as plt

    simulation_frame = results_dataframe(simulation_results)
    if set(simulation_frame.method) != set(METHODS):
        raise ValueError(
            f"Expected simulation methods {METHODS}, got "
            f"{tuple(dict.fromkeys(simulation_frame.method))}"
        )
    if set(subject_fit_rates.method) != set(METHODS):
        raise ValueError(
            f"Expected fit-rate methods {METHODS}, got "
            f"{tuple(dict.fromkeys(subject_fit_rates.method))}"
        )
    simulation_subjects = set(simulation_frame.subject)
    fit_subjects = set(subject_fit_rates.subject)
    if simulation_subjects != fit_subjects:
        raise ValueError(
            "Simulation and fit-rate subject sets differ: "
            f"simulation-only={sorted(simulation_subjects - fit_subjects)}, "
            f"fit-only={sorted(fit_subjects - simulation_subjects)}"
        )

    figure, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1, 1.65]},
        constrained_layout=True,
    )
    for axis, panel in zip(axes[:2], SIMULATION_PANELS):
        _plot_simulation_metric(
            axis, simulation_frame, *panel, logarithmic=simulation_log_scale
        )
    plot_fit_rates(subject_fit_rates, axis=axes[2])
    axes[2].set_ylabel(axes[2].get_ylabel(), fontsize=12)
    axes[2].tick_params(axis="both", labelsize=11)
    threshold = float(subject_fit_rates.crlb_threshold.iloc[0])
    axes[2].set_title("", loc="center")
    axes[2].set_title(
        f"c) LCModel fit rate (CRLB ≤ {threshold:g}%)",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    legend = axes[2].get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(9)
        legend.get_title().set_fontsize(9)
    return figure, simulation_frame
