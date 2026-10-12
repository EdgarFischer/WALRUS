"""Combined 7T/3T quantitative evaluation figure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from walinet.paper_figures.hlsvd_l2_comparison.results import (
    results_dataframe as results_dataframe_3t,
)
from walinet.paper_figures.lcmodel_quality import plot_fit_rates_by_method
from walinet.paper_figures.quantitative_3t.figure import (
    METHOD_COLORS as METHOD_COLORS_3T,
    METHOD_LABELS as METHOD_LABELS_3T,
    METHODS as METHODS_3T,
)
from walinet.paper_figures.quantitative_7t.figure import (
    MODEL_COLORS as MODEL_COLORS_7T,
    TICK_LABELS as MODEL_LABELS_7T,
    grouped_subject_results,
)
from walinet.paper_figures.paper_fig1.metrics import (
    METABOLITE_RESIDUAL_METRIC,
    NUISANCE_RESIDUAL_METRIC,
)


PANELS_7T = (
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

PANELS_3T = (
    (
        "residual_l2_over_nuisance_l2",
        "d) Nuisance-normalized residual",
        r"$\|\hat n-n\|_2 / \|n\|_2$",
    ),
    (
        "residual_l2_over_clean_metabolite_l2",
        "e) Metabolite-normalized residual",
        r"$\|\hat n-n\|_2 / \|m\|_2$",
    ),
)


def _label_without_breaks(labels: Mapping[str, str], key: str) -> str:
    return labels.get(key, key).replace("\n", " ")


def _plot_grouped_metric(
    axis,
    frame,
    *,
    group_column: str,
    metric: str,
    methods: Sequence[str],
    colors: Mapping[str, str],
    title: str,
    ylabel: str,
    logarithmic: bool,
):
    subset = frame[frame["metric"] == metric]
    subjects = list(dict.fromkeys(subset["subject"]))
    x = np.arange(len(methods), dtype=float)
    for method_index, method in enumerate(methods):
        indexed = subset[subset[group_column] == method].set_index("subject")
        missing = set(subjects) - set(indexed.index)
        if missing:
            raise ValueError(f"{metric}/{method}: missing subjects {sorted(missing)}")
        values = indexed.loc[subjects, "mean"].to_numpy(dtype=float)
        axis.errorbar(
            method_index,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=1.8,
            color=colors[method],
            markerfacecolor=colors[method],
            markeredgecolor=colors[method],
            zorder=2,
        )
    axis.set_xticks(x, [""] * len(methods))
    axis.set_ylabel(ylabel, fontsize=13.5)
    axis.set_title(title, loc="left", fontweight="bold", fontsize=13.5)
    axis.tick_params(axis="y", labelsize=12)
    axis.tick_params(axis="x", length=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2, which="major")
    if logarithmic:
        axis.set_yscale("log")
    else:
        axis.margins(y=0.15)


def _method_legend(figure, *, methods, colors, labels, anchor):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[method],
            markeredgecolor=colors[method],
            markersize=8,
            label=_label_without_breaks(labels, method),
        )
        for method in methods
    ]
    return figure.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=anchor,
        frameon=False,
        fontsize=12.5,
        ncols=len(methods),
        handlelength=1.1,
        borderpad=0.2,
        labelspacing=0.25,
        columnspacing=1.15,
    )


def _configure_fit_panel(axis, title: str, threshold: float):
    axis.set_title("", loc="center")
    axis.set_title("", loc="right")
    axis.set_title(
        f"{title} LCModel fit rate (CRLB ≤ {threshold:g}%)",
        loc="left",
        fontweight="bold",
        fontsize=13.5,
    )
    axis.set_ylabel("Fit rate [%]", fontsize=13.5)
    axis.tick_params(axis="y", labelsize=12)
    axis.tick_params(axis="x", length=0)
    legend = axis.get_legend()
    if legend is not None:
        legend.set_title("Metabolite")
        for text in legend.get_texts():
            text.set_fontsize(11)
        legend.get_title().set_fontsize(11)


def create_combined_3t_7t_quantitative_figure(
    *,
    results_7t: str | Path | Sequence[str | Path],
    subject_fit_rates_7t,
    model_groups_7t: Mapping[str, Sequence[str]],
    simulation_results_3t: str | Path,
    subject_fit_rates_3t,
    logarithmic_7t: bool = True,
    logarithmic_3t: bool = True,
    figsize=(18.0, 9.0),
):
    """Create a 2x3 combined 7T/3T quantitative figure."""
    import matplotlib.pyplot as plt

    frame_7t = grouped_subject_results(results_7t, model_groups=model_groups_7t)
    methods_7t = list(model_groups_7t)
    colors_7t = {method: MODEL_COLORS_7T[method] for method in methods_7t}

    frame_3t = results_dataframe_3t(simulation_results_3t)
    methods_3t = list(METHODS_3T)
    if set(frame_3t.method) != set(methods_3t):
        raise ValueError(
            f"Expected 3T simulation methods {methods_3t}, got "
            f"{tuple(dict.fromkeys(frame_3t.method))}"
        )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1, 1.55]},
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.09,
        top=0.88,
        hspace=0.62,
        wspace=0.34,
    )

    for axis, (metric, title, ylabel) in zip(axes[0, :2], PANELS_7T):
        _plot_grouped_metric(
            axis,
            frame_7t,
            group_column="model_group",
            metric=metric,
            methods=methods_7t,
            colors=colors_7t,
            title=title,
            ylabel=ylabel,
            logarithmic=logarithmic_7t,
        )

    plot_fit_rates_by_method(
        subject_fit_rates_7t,
        axis=axes[0, 2],
        method_colors={
            "U20*": MODEL_COLORS_7T["U20*"],
            "U20": MODEL_COLORS_7T["U20"],
            "Y16": MODEL_COLORS_7T["Y16"],
            "WALINET-style": MODEL_COLORS_7T["WALINET-style"],
            "Full WALRUS": MODEL_COLORS_7T["WALRUS"],
        },
        display_labels={method: "" for method in subject_fit_rates_7t["method"].unique()},
    )
    _configure_fit_panel(
        axes[0, 2],
        "c)",
        float(subject_fit_rates_7t["crlb_threshold"].iloc[0]),
    )

    for axis, (metric, title, ylabel) in zip(axes[1, :2], PANELS_3T):
        _plot_grouped_metric(
            axis,
            frame_3t,
            group_column="method",
            metric=metric,
            methods=methods_3t,
            colors=METHOD_COLORS_3T,
            title=title,
            ylabel=ylabel,
            logarithmic=logarithmic_3t,
        )

    plot_fit_rates_by_method(
        subject_fit_rates_3t,
        axis=axes[1, 2],
        method_colors=METHOD_COLORS_3T,
        display_labels={method: "" for method in METHODS_3T},
    )
    _configure_fit_panel(
        axes[1, 2],
        "f)",
        float(subject_fit_rates_3t["crlb_threshold"].iloc[0]),
    )

    figure.text(
        0.055,
        0.975,
        "7 T: Framework optimization",
        fontsize=15.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    figure.text(
        0.055,
        0.505,
        "3 T: Transfer and benchmarking",
        fontsize=15.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    _method_legend(
        figure,
        methods=methods_7t,
        colors=colors_7t,
        labels=MODEL_LABELS_7T,
        anchor=(0.055, 0.520),
    )
    _method_legend(
        figure,
        methods=methods_3t,
        colors=METHOD_COLORS_3T,
        labels=METHOD_LABELS_3T,
        anchor=(0.055, 0.010),
    )
    return figure, {"7T": frame_7t, "3T": frame_3t}
