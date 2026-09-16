"""Two-panel 7T model-family comparison with seed-aware aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from walinet.paper_figures.paper_fig1.metrics import (
    METABOLITE_RESIDUAL_METRIC,
    NUISANCE_RESIDUAL_METRIC,
    REFERENCE_RATIO_METRIC,
)
from walinet.paper_figures.paper_fig1.results import results_dataframe


DEFAULT_MODEL_GROUPS = {
    "U16": ("UNet_seed1", "UNet_seed2", "UNet_seed3"),
    "U20": (
        "UNet_parameter_matched_seed1",
        "UNet_parameter_matched_seed2",
        "UNet_parameter_matched_seed3",
    ),
    "Y16": ("YNet_seed1", "YNet_seed2", "YNet_seed3"),
    "WALRUS": ("WALRUS",),
}

MODEL_COLORS = {
    "U16": "#0072b2",
    "U20": "#e69f00",
    "Y16": "#009e73",
    "WALRUS": "#cc79a7",
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
        axis.tick_params(axis="both", labelsize=11)
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
