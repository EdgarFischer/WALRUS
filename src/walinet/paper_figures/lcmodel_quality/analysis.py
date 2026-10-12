"""Subject-level LCModel CRLB fit-rate comparison."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from walinet.config.build import build_config


DEFAULT_METHODS = {
    "HLSVD + L2": "LCModelHSVDL2",
    "WALRUS": "LCModelAfterWalinet",
}
DEFAULT_METABOLITES = {
    "tNAA": "NAA_NAAG",
    "tCr": "Cr_PCr",
    "tCho": "GPC_PCh",
    "Glx": "Glu_Gln",
}


def _load_training_config(path: Path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return build_config(raw, config_dir=path.parent)


def collect_fit_rates(
    *,
    project_root: str | Path,
    crlb_threshold: float = 20.0,
    training_config: str | Path = "configs/Training/train_3T.yaml",
    methods: Mapping[str, str] = DEFAULT_METHODS,
    metabolites: Mapping[str, str] = DEFAULT_METABOLITES,
):
    """Return one CRLB fit rate per subject, method, and metabolite.

    A successful voxel has a finite CRLB satisfying
    ``0 < CRLB <= crlb_threshold``. The denominator is always the complete
    brain-mask voxel count, so missing/zero/nonfinite CRLB values are failures.
    """
    import pandas as pd

    if not np.isfinite(crlb_threshold) or crlb_threshold <= 0:
        raise ValueError("crlb_threshold must be finite and greater than zero")
    root = Path(project_root).expanduser().resolve()
    config_path = Path(training_config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = _load_training_config(config_path.resolve())
    rows = []
    for subject in cfg.data.val_subjects:
        subject_dir = Path(cfg.data.base_dir) / subject
        mask_path = subject_dir / "masks/brain_mask.npy"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        brain_mask = np.asarray(np.load(mask_path, allow_pickle=False)).squeeze() > 0
        if brain_mask.ndim != 3:
            raise ValueError(f"{mask_path}: expected 3-D mask, got {brain_mask.shape}")
        n_brain = int(np.count_nonzero(brain_mask))
        if n_brain == 0:
            raise ValueError(f"{mask_path}: brain mask is empty")
        for method, folder in methods.items():
            map_root = subject_dir / folder / "maps/Orig"
            for metabolite, filename in metabolites.items():
                map_path = map_root / f"{filename}_sd_map.nii.gz"
                if not map_path.is_file():
                    raise FileNotFoundError(map_path)
                crlb = np.asarray(nib.load(map_path).dataobj, dtype=np.float64).squeeze()
                if crlb.shape != brain_mask.shape:
                    raise ValueError(
                        f"{map_path}: map shape {crlb.shape} != mask {brain_mask.shape}"
                    )
                successful = (
                    brain_mask & np.isfinite(crlb)
                    & (crlb > 0) & (crlb <= crlb_threshold)
                )
                n_successful = int(np.count_nonzero(successful))
                fit_rate = n_successful / n_brain
                rows.append({
                    "subject": subject,
                    "method": method,
                    "metabolite": metabolite,
                    "crlb_threshold": float(crlb_threshold),
                    "n_successful_voxels": n_successful,
                    "n_brain_voxels": n_brain,
                    "fit_rate": fit_rate,
                    "fit_rate_percent": 100.0 * fit_rate,
                })
    return pd.DataFrame.from_records(rows)


def summary_table(subject_fit_rates):
    """Mean and sample SD across subject-level fit rates."""
    return (
        subject_fit_rates.groupby(["metabolite", "method"], sort=False)
        ["fit_rate_percent"]
        .agg(
            mean_fit_rate_percent="mean",
            std_fit_rate_percent="std",
            n_subjects="count",
        )
        .reset_index()
    )


def plot_fit_rates(subject_fit_rates, *, figsize=(7.0, 4.2), axis=None):
    """Plot all metabolite fit rates in one quantitative panel."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = list(dict.fromkeys(subject_fit_rates["method"]))
    subjects = list(dict.fromkeys(subject_fit_rates["subject"]))
    metabolites = list(dict.fromkeys(subject_fit_rates["metabolite"]))
    thresholds = subject_fit_rates["crlb_threshold"].unique()
    if len(thresholds) != 1:
        raise ValueError("Expected exactly one CRLB threshold")
    threshold = float(thresholds[0])
    metabolite_colors = dict(zip(
        metabolites, ("#0072b2", "#e69f00", "#009e73", "#cc79a7")
    ))
    method_markers = {"HLSVD + L2": "o", "WALRUS": "^"}
    unknown = set(methods) - set(method_markers)
    if unknown:
        raise ValueError(f"No marker configured for methods: {sorted(unknown)}")
    if axis is None:
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        figure = axis.figure
    x = np.arange(len(metabolites), dtype=float)
    offsets = np.linspace(-0.10, 0.10, len(methods)) if len(methods) > 1 else [0.0]
    for metabolite_index, metabolite in enumerate(metabolites):
        subset = subject_fit_rates[subject_fit_rates.metabolite == metabolite]
        for method_index, method in enumerate(methods):
            indexed = subset[subset.method == method].set_index("subject")
            missing = set(subjects) - set(indexed.index)
            if missing:
                raise ValueError(f"{metabolite}/{method}: missing subjects {missing}")
            values = indexed.loc[subjects, "fit_rate_percent"].to_numpy()
            axis.errorbar(
                metabolite_index + offsets[method_index], values.mean(),
                yerr=values.std(ddof=1), fmt=method_markers[method],
                markersize=8, capsize=4, linewidth=1.6,
                markerfacecolor=metabolite_colors[metabolite],
                markeredgecolor=metabolite_colors[metabolite],
                color=metabolite_colors[metabolite], zorder=2,
            )
    axis.set_xticks(x, metabolites)
    axis.set_ylabel("Fit rate [%]")
    axis.set_title(f"LCModel fit rate (CRLB ≤ {threshold:g}%)")
    axis.margins(x=0.12, y=0.15)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    method_handles = [
        Line2D(
            [0], [0], marker=method_markers[method], linestyle="none",
            markerfacecolor="0.35", markeredgecolor="0.35",
            markersize=8, label=method,
        )
        for method in methods
    ]
    axis.legend(handles=method_handles, frameon=False, title="Method")
    return figure


def plot_fit_rates_by_method(
    subject_fit_rates,
    *,
    method_colors,
    display_labels=None,
    figsize=(7.0, 4.2),
    axis=None,
):
    """Plot method-colored fit rates with metabolite-specific markers."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = list(dict.fromkeys(subject_fit_rates["method"]))
    subjects = list(dict.fromkeys(subject_fit_rates["subject"]))
    metabolites = list(dict.fromkeys(subject_fit_rates["metabolite"]))
    thresholds = subject_fit_rates["crlb_threshold"].unique()
    if len(thresholds) != 1:
        raise ValueError("Expected exactly one CRLB threshold")
    missing_colors = set(methods) - set(method_colors)
    if missing_colors:
        raise ValueError(f"No color configured for methods: {sorted(missing_colors)}")
    if display_labels is None:
        display_labels = {method: method for method in methods}

    marker_sequence = ("o", "s", "D", "^")
    if len(metabolites) > len(marker_sequence):
        raise ValueError("At most four metabolite markers are currently supported")
    metabolite_markers = dict(zip(metabolites, marker_sequence))

    if axis is None:
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        figure = axis.figure

    x = np.arange(len(methods), dtype=float)
    offsets = np.linspace(-0.12, 0.12, len(metabolites))
    for method_index, method in enumerate(methods):
        subset = subject_fit_rates[subject_fit_rates.method == method]
        for metabolite_index, metabolite in enumerate(metabolites):
            indexed = subset[subset.metabolite == metabolite].set_index("subject")
            values = indexed["fit_rate_percent"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            yerr = values.std(ddof=1) if values.size > 1 else 0.0
            axis.errorbar(
                method_index + offsets[metabolite_index],
                values.mean(),
                yerr=yerr,
                fmt=metabolite_markers[metabolite],
                markersize=8,
                capsize=4,
                linewidth=1.6,
                color=method_colors[method],
                markerfacecolor=method_colors[method],
                markeredgecolor=method_colors[method],
                zorder=2,
            )

    axis.set_xticks(x, [display_labels.get(method, method) for method in methods])
    axis.set_ylabel("Fit rate [%]")
    axis.set_title(f"LCModel fit rate (CRLB ≤ {float(thresholds[0]):g}%)")
    axis.margins(x=0.15, y=0.15)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    metabolite_handles = [
        Line2D(
            [0],
            [0],
            marker=metabolite_markers[metabolite],
            linestyle="none",
            markerfacecolor="0.35",
            markeredgecolor="0.35",
            markersize=8,
            label=metabolite,
        )
        for metabolite in metabolites
    ]
    axis.legend(
        handles=metabolite_handles,
        frameon=False,
        title="Metabolite",
        ncols=2,
    )
    return figure
