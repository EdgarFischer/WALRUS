"""Summaries and plotting for the 3T paired simulation comparison."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from walinet.paper_figures.paper_fig1.metrics import PREDICTION_METRICS

from .pipeline import METHODS, METRICS


def results_dataframe(path: str | Path):
    import pandas as pd

    rows = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        for subject_group in h5["subjects"].values():
            subject = str(subject_group.attrs["name"])
            for method in METHODS:
                for metric in METRICS:
                    values = subject_group[f"methods/{method}/{metric}"][:]
                    rows.append(
                        {
                            "subject": subject,
                            "method": method,
                            "metric": metric,
                            "mean": float(np.mean(values, dtype=np.float64)),
                            "std_over_spectra": float(np.std(values, ddof=1)),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def plot_comparison(path: str | Path, *, figsize=(8.2, 4.0)):
    import matplotlib.pyplot as plt

    frame = results_dataframe(path)
    labels = {
        "residual_l2_over_nuisance_l2":
            r"$\|\hat n-n_{true}\|_2 / \|n_{true}\|_2$",
        "residual_l2_over_clean_metabolite_l2":
            r"$\|\hat n-n_{true}\|_2 / \|m_{clean}\|_2$",
    }
    colors = {"HLSVD + L2": "#7f7f7f", "WALRUS": "#377eb8"}
    figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for axis, metric in zip(axes, PREDICTION_METRICS):
        subset = frame[frame.metric == metric]
        subject_order = list(dict.fromkeys(subset.subject))
        x = np.arange(len(METHODS), dtype=float)
        method_values = []
        for method in METHODS:
            indexed = subset[subset.method == method].set_index("subject")
            method_values.append(indexed.loc[subject_order, "mean"].to_numpy())
        matrix = np.stack(method_values, axis=1)
        for row in matrix:
            axis.plot(x, row, color="0.75", linewidth=0.8, zorder=1)
        for index, method in enumerate(METHODS):
            values = matrix[:, index]
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                markersize=8,
                capsize=5,
                linewidth=1.8,
                color=colors[method],
                zorder=3,
            )
            axis.scatter(
                np.full(len(values), index), values, s=18,
                facecolors="white", edgecolors=colors[method], zorder=2,
            )
        axis.set_xticks(x, METHODS)
        axis.set_ylabel(labels[metric])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    return figure, frame
