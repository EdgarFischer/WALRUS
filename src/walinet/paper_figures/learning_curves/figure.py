"""Seed-averaged learning curves and fixed/on-the-fly comparison."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


LOSS_PATTERN = re.compile(
    r"Epoch:\s*(\d+),\s*Loss:\s*([0-9.eE+-]+),\s*"
    r"Val Loss:\s*([0-9.eE+-]+)"
)

DEFAULT_MODEL_GROUPS = {
    "Y16": (
        "YNet_ablation",
        "YNet_ablation_Seed_2",
        "YNet_ablation_Seed_3",
    ),
    "U20": (
        "UNet_ablation_parameter_matched",
        "UNet_ablation_parameter_matched_Seed_2",
        "UNet_ablation_parameter_matched_Seed_3",
    ),
}

DEFAULT_PANEL_A_CALIBRATED_MODEL = "UNet_ablation_WALINET_Calibration_S1"

DEFAULT_FULL_WALRUS_RUNS = {
    # 3T_Final/run_summary.txt records this archive model as its warm start.
    "3T": ("Archiv/3T_Final_32_Channels", "3T_Final"),
    # 7T_Final/run_summary.txt records 7T_New_WALI_2.1 as its warm start.
    "7T": ("Archiv/7T_New_WALI_2.1", "7T_Final"),
}

DEFAULT_FULL_WALRUS_TRAINING_SET_COMPARISON = {
    "On-the-fly training": DEFAULT_FULL_WALRUS_RUNS["7T"],
    "Fixed training set": (
        "UNet_32_6_fixed_trainingset",
        "UNet_32_6_fixed_trainingset_100ep",
    ),
}


def load_loss_history(path: str | Path) -> dict[str, np.ndarray]:
    """Parse one model ``loss.txt`` into epoch, training, and validation arrays."""
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "loss.txt"
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = [LOSS_PATTERN.search(line) for line in source.read_text().splitlines()]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise ValueError(f"No loss entries found in {source}")
    epochs = np.asarray([int(row.group(1)) for row in rows], dtype=int)
    if len(np.unique(epochs)) != len(epochs):
        raise ValueError(f"Duplicate epochs in {source}")
    return {
        "epoch": epochs,
        "training": np.asarray([float(row.group(2)) for row in rows]),
        "validation": np.asarray([float(row.group(3)) for row in rows]),
    }


def _aligned_histories(model_root: Path, models: Sequence[str]):
    histories = [load_loss_history(model_root / model) for model in models]
    common = set(histories[0]["epoch"])
    for history in histories[1:]:
        common &= set(history["epoch"])
    epochs = np.asarray(sorted(common), dtype=int)
    if epochs.size == 0:
        raise ValueError(f"Models have no common epochs: {models}")
    aligned = {}
    for metric in ("training", "validation"):
        rows = []
        for history in histories:
            lookup = dict(zip(history["epoch"], history[metric]))
            rows.append([lookup[epoch] for epoch in epochs])
        aligned[metric] = np.asarray(rows, dtype=float)
    return epochs, aligned


def _plot_seed_average(axis, model_root, models, *, show_legend):
    epochs, values = _aligned_histories(model_root, models)
    styles = {
        "training": ("Training", "#0072B2", "-"),
        "validation": ("Validation", "#E69F00", "--"),
    }
    for metric, (label, color, linestyle) in styles.items():
        matrix = values[metric]
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        axis.plot(
            epochs, mean, color=color, linestyle=linestyle,
            linewidth=1.5, label=label,
        )
        axis.fill_between(
            epochs, mean - std, mean + std,
            color=color, alpha=0.14, linewidth=0,
        )
    if show_legend:
        axis.legend(frameon=False, fontsize=9, loc="best")


def _plot_u_net_heuristic_calibrated(axis, model_root, heuristic_models, calibrated_model):
    epochs, values = _aligned_histories(model_root, heuristic_models)
    calibrated = load_loss_history(model_root / calibrated_model)
    styles = {
        "training": (
            "Training (heuristic)",
            "Training (calibrated)",
            "#0072B2",
            "#009E73",
            "-",
        ),
        "validation": (
            "Validation (heuristic)",
            "Validation (calibrated)",
            "#E69F00",
            "#D55E00",
            "--",
        ),
    }
    for metric, (
        heuristic_label,
        calibrated_label,
        heuristic_color,
        calibrated_color,
        linestyle,
    ) in styles.items():
        matrix = values[metric]
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        axis.plot(
            epochs,
            mean,
            color=heuristic_color,
            linestyle=linestyle,
            linewidth=1.5,
            label=heuristic_label,
        )
        axis.fill_between(
            epochs, mean - std, mean + std,
            color=heuristic_color, alpha=0.14, linewidth=0,
        )
        axis.plot(
            calibrated["epoch"],
            calibrated[metric],
            color=calibrated_color,
            linestyle=linestyle,
            linewidth=1.5,
            label=calibrated_label,
        )
    axis.legend(frameon=False, fontsize=8.0, loc="best")


def _plot_full_walrus_training_set_comparison(axis, model_root, comparison):
    colors = {"On-the-fly training": "#0072B2", "Fixed training set": "#E69F00"}
    restart_epochs = {}
    for regimen, runs in comparison.items():
        history, regimen_restart_epochs = _concatenated_history(model_root, runs)
        restart_epochs[regimen] = regimen_restart_epochs
        color = colors.get(regimen, "0.35")
        axis.plot(
            history["epoch"], history["training"],
            color=color, linestyle="-", linewidth=1.4,
            label=f"{regimen} - Training",
        )
        axis.plot(
            history["epoch"], history["validation"],
            color=color, linestyle="--", linewidth=1.4,
            label=f"{regimen} - Validation",
        )
    axis.legend(frameon=False, fontsize=8.5, loc="best")
    return restart_epochs


def _concatenated_history(model_root: Path, runs: Sequence[str]):
    """Concatenate restart runs while retaining their chronological epochs."""
    if not runs:
        raise ValueError("At least one run is required")
    combined = {"epoch": [], "training": [], "validation": []}
    restart_epochs: list[int] = []
    offset = 0
    for run_index, run in enumerate(runs):
        history = load_loss_history(model_root / run)
        epochs = history["epoch"] + offset
        if run_index:
            restart_epochs.append(int(epochs[0]))
        combined["epoch"].append(epochs)
        combined["training"].append(history["training"])
        combined["validation"].append(history["validation"])
        offset = int(epochs[-1])
    return (
        {key: np.concatenate(parts) for key, parts in combined.items()},
        tuple(restart_epochs),
    )


def _plot_full_walrus(axis, model_root, runs):
    history, restart_epochs = _concatenated_history(model_root, runs)
    styles = {
        "training": ("Training", "#0072B2", "-"),
        "validation": ("Validation", "#E69F00", "--"),
    }
    for metric, (label, color, linestyle) in styles.items():
        axis.plot(
            history["epoch"],
            history[metric],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )
    return restart_epochs


def create_learning_curve_figure(
    *,
    model_root: str | Path,
    model_groups: Mapping[str, Sequence[str]] = DEFAULT_MODEL_GROUPS,
    panel_a_calibrated_model: str | None = DEFAULT_PANEL_A_CALIBRATED_MODEL,
    full_walrus_runs: Mapping[str, Sequence[str]] = DEFAULT_FULL_WALRUS_RUNS,
    full_walrus_training_set_comparison: Mapping[
        str, Sequence[str]
    ] = DEFAULT_FULL_WALRUS_TRAINING_SET_COMPARISON,
    logarithmic: bool = True,
    figsize: tuple[float, float] = (9.0, 7.0),
):
    """Return the paper figure and the curves' source configuration."""
    import matplotlib.pyplot as plt

    root = Path(model_root).expanduser().resolve()
    expected = ("U20", "Y16")
    missing = set(expected) - set(model_groups)
    if missing:
        raise ValueError(f"Missing model groups: {sorted(missing)}")
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axes = {
        "u_net": figure.add_subplot(grid[0, 0]),
        "y_net": figure.add_subplot(grid[0, 1]),
        "3T": figure.add_subplot(grid[1, 0]),
        "comparison": figure.add_subplot(grid[1, 1]),
    }
    panel_specs = (
        (axes["u_net"], "U20", "a) shallow U-Net"),
        (axes["y_net"], "Y16", "b) shallow Y-Net + L2 prior (calibrated training)"),
    )
    for panel_index, (axis, group, title) in enumerate(panel_specs):
        if panel_index == 0 and panel_a_calibrated_model is not None:
            _plot_u_net_heuristic_calibrated(
                axis, root, model_groups[group], panel_a_calibrated_model
            )
        else:
            _plot_seed_average(
                axis, root, model_groups[group], show_legend=panel_index == 0
            )
        axis.set_title(title, loc="left", fontweight="bold", fontsize=11)
    restart_epochs = {}
    if "3T" not in full_walrus_runs:
        raise ValueError("Missing Full WALRUS runs for 3T")
    restart_epochs["3T"] = _plot_full_walrus(
        axes["3T"], root, full_walrus_runs["3T"]
    )
    axes["3T"].set_title(
        "c) deep U-Net (calibrated training, 3T)",
        loc="left", fontweight="bold", fontsize=11,
    )
    comparison_restart_epochs = _plot_full_walrus_training_set_comparison(
        axes["comparison"], root, full_walrus_training_set_comparison
    )
    axes["comparison"].set_title(
        "d) deep U-Net (calibrated training, 7T)",
        loc="left", fontweight="bold", fontsize=11,
    )
    for axis in axes.values():
        if logarithmic:
            axis.set_yscale("log")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18, which="major")
        axis.tick_params(axis="both", labelsize=10)
    for axis in (axes["3T"], axes["comparison"]):
        axis.set_xlabel("Epoch", fontsize=11)
    for axis in (axes["u_net"], axes["3T"]):
        axis.set_ylabel("Loss", fontsize=11)
    metadata = {
        "model_groups": {key: tuple(value) for key, value in model_groups.items()},
        "panel_a_calibrated_model": panel_a_calibrated_model,
        "full_walrus_training_set_comparison": {
            key: tuple(value)
            for key, value in full_walrus_training_set_comparison.items()
        },
        "full_walrus_training_set_restart_epochs": comparison_restart_epochs,
        "full_walrus_runs": {
            key: tuple(value) for key, value in full_walrus_runs.items()
        },
        "full_walrus_restart_epochs": restart_epochs,
        "aggregation": (
            "Panel a shallow U-Net and panel b shallow Y-Net + L2 prior "
            "(calibrated training): epoch-wise mean ± sample SD across three "
            "seeds; panel a calibrated shallow U-Net: "
            "single S1 run; "
            "deep U-Net (calibrated training) panels and training-set "
            "comparison: chronological concatenation of warm-start runs"
        ),
    }
    return figure, metadata
