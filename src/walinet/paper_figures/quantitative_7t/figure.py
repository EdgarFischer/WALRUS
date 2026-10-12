"""Two-panel 7T model-family comparison with seed-aware aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import warnings

import numpy as np

from walinet.paper_figures.paper_fig1.metrics import (
    METABOLITE_RESIDUAL_METRIC,
    NUISANCE_RESIDUAL_METRIC,
    REFERENCE_RATIO_METRIC,
)
from walinet.paper_figures.paper_fig1.results import results_dataframe
from walinet.paper_figures.lcmodel_quality import plot_fit_rates_by_method


DEFAULT_MODEL_GROUPS = {
    "U20*": (
        "UNet_WALINET_calibrated_seed1",
        "UNet_WALINET_calibrated_seed2",
        "UNet_WALINET_calibrated_seed3",
    ),
    "U20": (
        "UNet_parameter_matched_seed1",
        "UNet_parameter_matched_seed2",
        "UNet_parameter_matched_seed3",
    ),
    "Y16": ("YNet_seed1", "YNet_seed2", "YNet_seed3"),
    "WALRUS": ("WALRUS",),
}

MODEL_COLORS = {
    "U20": "#e69f00",
    "U20*": "#d55e00",
    "Y16": "#009e73",
    "WALINET-style": "#8e44ad",
    "WALRUS": "#cc79a7",
    "Full WALRUS": "#cc79a7",
}

DISPLAY_LABELS = {
    "U20*": "U-Net (heuristic training)",
    "U20": "U-Net (calibrated training)",
    "Y16": "Y-Net + L2 prior (calibrated training)",
    "WALINET-style": "WALINET-style",
    "WALRUS": "Final WALRUS",
}

TICK_LABELS = {
    "U20*": "shallow U-Net\n(heuristic training)",
    "U20": "shallow U-Net\n(calibrated training)",
    "Y16": "shallow Y-Net\n+ L2 prior (calibrated training)",
    "WALINET-style": "WALINET\nstyle",
    "WALRUS": "deep U-Net\n(calibrated training)",
    "Full WALRUS": "deep U-Net\n(calibrated training)",
}

PANELS = (
    (
        NUISANCE_RESIDUAL_METRIC,
        "a) Nuisance-normalized residual",
        r"$\|\hat n-n\|_2 / \|n\|_2$",
    ),
    (
        METABOLITE_RESIDUAL_METRIC,
        "b) Metabolite-normalized residual",
        r"$\|\hat n-n\|_2 / \|m\|_2$",
    ),
)


def grouped_subject_results(
    path: str | Path,
    *,
    model_groups: Mapping[str, Sequence[str]] = DEFAULT_MODEL_GROUPS,
):
    """Average model seeds within each subject, never across subjects first."""
    frame = results_dataframe(path)
    available = set(frame["model"])
    rows = []
    for label, models in model_groups.items():
        missing = set(models) - available
        if missing:
            raise ValueError(f"{label}: missing configured models {sorted(missing)}")
        selected = frame[frame["model"].isin(models)]
        grouped = (
            selected.groupby(["subject", "metric"], as_index=False, sort=False)["mean"]
            .mean()
        )
        grouped["model_group"] = label
        grouped["n_seeds"] = len(models)
        rows.append(grouped)
    import pandas as pd

    return pd.concat(rows, ignore_index=True)


def create_quantitative_figure(
    results_path: str | Path,
    *,
    model_groups: Mapping[str, Sequence[str]] = DEFAULT_MODEL_GROUPS,
    logarithmic: bool = True,
    figsize=(8.4, 4.0),
):
    """Return the two-panel figure and subject-level, seed-averaged table."""
    import matplotlib.pyplot as plt

    frame = grouped_subject_results(results_path, model_groups=model_groups)
    labels = list(model_groups)
    figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes, PANELS):
        subset = frame[frame["metric"] == metric]
        subjects = list(dict.fromkeys(subset["subject"]))
        for index, label in enumerate(labels):
            indexed = subset[subset["model_group"] == label].set_index("subject")
            missing = set(subjects) - set(indexed.index)
            if missing:
                raise ValueError(f"{metric}/{label}: missing subjects {sorted(missing)}")
            values = indexed.loc[subjects, "mean"].to_numpy(dtype=float)
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                markersize=8,
                capsize=5,
                linewidth=1.8,
                color=MODEL_COLORS.get(label, "0.35"),
            )
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.set_title(title, loc="left", fontweight="bold", fontsize=11)
        axis.tick_params(axis="y", labelsize=11)
        axis.tick_params(axis="x", labelsize=12.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2, which="major")
        if logarithmic:
            axis.set_yscale("log")
        else:
            axis.margins(y=0.15)
    return figure, frame


def reference_subject_results(grouped_frame):
    """Return one method-independent ||n||/||m|| mean per subject."""
    reference = grouped_frame[
        grouped_frame["metric"] == REFERENCE_RATIO_METRIC
    ]
    return reference.groupby("subject", sort=True)["mean"].mean()


def create_combined_quantitative_figure(
    results_path: str | Path,
    *,
    subject_fit_rates,
    model_groups: Mapping[str, Sequence[str]] = DEFAULT_MODEL_GROUPS,
    logarithmic: bool = True,
    figsize=(16.5, 4.8),
):
    """Create the 7T simulation-metric and LCModel fit-rate figure."""
    import matplotlib.pyplot as plt

    simulation_frame = grouped_subject_results(results_path, model_groups=model_groups)
    def subject_identity(value):
        match = re.search(r"Vol\d+", str(value), flags=re.IGNORECASE)
        return match.group(0).lower() if match else str(value)

    simulation_subjects = {
        subject_identity(value) for value in simulation_frame["subject"]
    }
    fit_subjects = {
        subject_identity(value) for value in subject_fit_rates["subject"]
    }
    if fit_subjects - simulation_subjects:
        raise ValueError(
            "Simulation and fit-rate subject sets differ: "
            f"simulation-only={sorted(simulation_subjects - fit_subjects)}, "
            f"fit-only={sorted(fit_subjects - simulation_subjects)}"
        )
    if simulation_subjects - fit_subjects:
        warnings.warn(
            "LCModel fitting is not complete for all simulation subjects; "
            "the CRLB panel uses the subjects currently available. Missing: "
            f"{sorted(simulation_subjects - fit_subjects)}",
            RuntimeWarning,
            stacklevel=2,
        )

    labels = list(model_groups)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1, 1.55]},
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.13,
        top=0.90,
        wspace=0.34,
    )

    for axis, (metric, title, ylabel) in zip(axes[:2], PANELS):
        subset = simulation_frame[simulation_frame["metric"] == metric]
        subjects = list(dict.fromkeys(subset["subject"]))
        for index, label in enumerate(labels):
            indexed = subset[subset["model_group"] == label].set_index("subject")
            missing = set(subjects) - set(indexed.index)
            if missing:
                raise ValueError(f"{metric}/{label}: missing subjects {sorted(missing)}")
            values = indexed.loc[subjects, "mean"].to_numpy(dtype=float)
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                markersize=8,
                capsize=5,
                linewidth=1.8,
                color=MODEL_COLORS.get(label, "0.35"),
                label=TICK_LABELS.get(label, label).replace("\n", " "),
            )
        axis.set_xticks(
            np.arange(len(labels)),
            [""] * len(labels),
        )
        axis.set_ylabel(ylabel, fontsize=14.5)
        axis.set_title(title, loc="left", fontweight="bold", fontsize=13.5)
        axis.tick_params(axis="y", labelsize=12.5)
        axis.tick_params(axis="x", labelsize=12.5, pad=4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2, which="major")
        if logarithmic:
            axis.set_yscale("log")
        else:
            axis.margins(y=0.15)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles[: len(labels)],
            legend_labels[: len(labels)],
            loc="lower left",
            bbox_to_anchor=(0.055, 0.015),
            frameon=False,
            fontsize=14.5,
            ncols=4,
            handlelength=1.2,
            borderpad=0.2,
            labelspacing=0.25,
            columnspacing=1.2,
        )

    fit_method_colors = {
        "U20": MODEL_COLORS["U20"],
        "U20*": MODEL_COLORS["U20*"],
        "Y16": MODEL_COLORS["Y16"],
        "WALINET-style": MODEL_COLORS["WALINET-style"],
        "Full WALRUS": MODEL_COLORS["WALRUS"],
    }
    fit_tick_labels = {
        "U20": "",
        "U20*": "",
        "Y16": "",
        "WALINET-style": "",
        "Full WALRUS": "",
    }
    plot_fit_rates_by_method(
        subject_fit_rates,
        axis=axes[2],
        method_colors=fit_method_colors,
        display_labels=fit_tick_labels,
    )
    axes[2].set_title("")
    threshold = float(subject_fit_rates["crlb_threshold"].iloc[0])
    axes[2].set_title(
        f"c) LCModel fit rate (CRLB ≤ {threshold:g}%)",
        loc="left",
        fontweight="bold",
        fontsize=13.5,
    )
    axes[2].set_ylabel("Fit rate [%]", fontsize=14.5)
    axes[2].tick_params(axis="y", labelsize=12.5)
    axes[2].tick_params(axis="x", labelsize=12.5, pad=4)
    legend = axes[2].get_legend()
    if legend is not None:
        legend.set_title("Metabolite")
        for text in legend.get_texts():
            text.set_fontsize(12)
        legend.get_title().set_fontsize(12)

    return figure, simulation_frame
