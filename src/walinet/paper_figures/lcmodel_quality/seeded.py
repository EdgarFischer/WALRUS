"""Seed-aware LCModel fit-rate analysis for the 7T UCSF cohort."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import warnings

import nibabel as nib
import numpy as np


DEFAULT_7T_METHOD_GROUPS = {
    "U16": ("LCModelU16_S1", "LCModelU16_S2", "LCModelU16_S3"),
    "U20": ("LCModelU20_S1", "LCModelU20_S2", "LCModelU20_S3"),
    "Y16": ("LCModelY16_S1", "LCModelY16_S2", "LCModelY16_S3"),
    "Full WALRUS": ("LCModelAfterWalrus",),
}

DEFAULT_7T_UCSF_SUBJECTS = tuple(f"Vol{index}_UCSF" for index in range(1, 6))

DEFAULT_METABOLITES = {
    "tNAA": "NAA_NAAG",
    "tCr": "Cr_PCr",
    "tCho": "GPC_PCh",
    "Glx": "Glu_Gln",
}


def collect_seeded_fit_rates(
    *,
    base_dir: str | Path,
    subjects: Sequence[str] = DEFAULT_7T_UCSF_SUBJECTS,
    method_groups: Mapping[str, Sequence[str]] = DEFAULT_7T_METHOD_GROUPS,
    metabolites: Mapping[str, str] = DEFAULT_METABOLITES,
    crlb_threshold: float = 20.0,
    allow_missing: bool = False,
):
    """Return raw seed-level and seed-averaged subject-level fit rates.

    A successful brain-mask voxel has a finite CRLB satisfying
    ``0 < CRLB <= crlb_threshold``. Missing, zero, nonfinite, or excessive
    CRLB values count as failures because the denominator is the complete
    brain-mask voxel count.
    """
    import pandas as pd

    if not np.isfinite(crlb_threshold) or crlb_threshold <= 0:
        raise ValueError("crlb_threshold must be finite and greater than zero")

    root = Path(base_dir).expanduser().resolve()
    rows: list[dict[str, object]] = []
    missing_paths: list[Path] = []
    for subject in subjects:
        subject_dir = root / subject
        mask_path = subject_dir / "masks/brain_mask.npy"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        brain_mask = np.asarray(np.load(mask_path, allow_pickle=False)).squeeze() > 0
        if brain_mask.ndim != 3:
            raise ValueError(f"{mask_path}: expected a 3-D mask, got {brain_mask.shape}")
        n_brain = int(np.count_nonzero(brain_mask))
        if n_brain == 0:
            raise ValueError(f"{mask_path}: brain mask is empty")

        for method, folders in method_groups.items():
            if not folders:
                raise ValueError(f"{method}: no LCModel folders configured")
            for seed_index, folder in enumerate(folders, start=1):
                map_root = subject_dir / folder / "maps/Orig"
                for metabolite, filename in metabolites.items():
                    map_path = map_root / f"{filename}_sd_map.nii.gz"
                    if not map_path.is_file():
                        if allow_missing:
                            missing_paths.append(map_path)
                            continue
                        raise FileNotFoundError(map_path)
                    crlb = np.asarray(nib.load(map_path).dataobj, dtype=np.float64).squeeze()
                    if crlb.shape != brain_mask.shape:
                        raise ValueError(
                            f"{map_path}: map shape {crlb.shape} != "
                            f"mask shape {brain_mask.shape}"
                        )
                    successful = (
                        brain_mask
                        & np.isfinite(crlb)
                        & (crlb > 0)
                        & (crlb <= crlb_threshold)
                    )
                    n_successful = int(np.count_nonzero(successful))
                    rows.append(
                        {
                            "subject": subject,
                            "method": method,
                            "seed": seed_index,
                            "fit_folder": folder,
                            "metabolite": metabolite,
                            "crlb_threshold": float(crlb_threshold),
                            "n_successful_voxels": n_successful,
                            "n_brain_voxels": n_brain,
                            "fit_rate": n_successful / n_brain,
                            "fit_rate_percent": 100.0 * n_successful / n_brain,
                        }
                    )

    if missing_paths:
        examples = "\n".join(f"  - {path}" for path in missing_paths[:8])
        remainder = len(missing_paths) - min(len(missing_paths), 8)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        warnings.warn(
            "LCModel fitting is incomplete: "
            f"{len(missing_paths)} expected CRLB map(s) are not available yet. "
            "The calculation and figure use every currently available map.\n"
            f"{examples}{suffix}",
            RuntimeWarning,
            stacklevel=2,
        )
    if not rows:
        raise FileNotFoundError("No requested LCModel CRLB maps are available")

    seed_level = pd.DataFrame.from_records(rows)
    subject_level = (
        seed_level.groupby(
            ["subject", "method", "metabolite", "crlb_threshold"],
            as_index=False,
            sort=False,
        )
        .agg(
            fit_rate=("fit_rate", "mean"),
            fit_rate_percent=("fit_rate_percent", "mean"),
            n_seeds=("seed", "count"),
        )
    )
    return seed_level, subject_level


def seeded_summary_table(subject_fit_rates):
    """Mean and sample SD across subjects after averaging seeds per subject."""
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


def plot_seeded_fit_rates(
    subject_fit_rates,
    *,
    figsize: tuple[float, float] = (8.0, 4.6),
    axis=None,
    display_labels: Mapping[str, str] | None = None,
):
    """Plot mean ± subject-level sample SD for all methods and metabolites."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = list(dict.fromkeys(subject_fit_rates["method"]))
    if display_labels is None:
        display_labels = {method: method for method in methods}
    subjects = list(dict.fromkeys(subject_fit_rates["subject"]))
    metabolites = list(dict.fromkeys(subject_fit_rates["metabolite"]))
    thresholds = subject_fit_rates["crlb_threshold"].unique()
    if len(thresholds) != 1:
        raise ValueError("Expected exactly one CRLB threshold")

    colors = dict(zip(metabolites, ("#0072b2", "#e69f00", "#009e73", "#cc79a7")))
    markers = {"U16": "o", "U20": "s", "Y16": "D", "Full WALRUS": "^"}
    missing_markers = set(methods) - set(markers)
    if missing_markers:
        raise ValueError(f"No marker configured for methods: {sorted(missing_markers)}")

    if axis is None:
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        figure = axis.figure

    x = np.arange(len(metabolites), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(methods))
    for metabolite_index, metabolite in enumerate(metabolites):
        subset = subject_fit_rates[subject_fit_rates.metabolite == metabolite]
        for method_index, method in enumerate(methods):
            indexed = subset[subset.method == method].set_index("subject")
            missing = set(subjects) - set(indexed.index)
            if missing:
                raise ValueError(f"{metabolite}/{method}: missing subjects {sorted(missing)}")
            values = indexed.loc[subjects, "fit_rate_percent"].to_numpy(dtype=float)
            axis.errorbar(
                metabolite_index + offsets[method_index],
                values.mean(),
                yerr=values.std(ddof=1),
                fmt=markers[method],
                markersize=8,
                capsize=4,
                linewidth=1.6,
                color=colors[metabolite],
                markerfacecolor=colors[metabolite],
                markeredgecolor=colors[metabolite],
            )

    axis.set_xticks(x, metabolites)
    axis.set_ylabel("Fit rate [%]", fontsize=12)
    axis.set_title(
        f"7T LCModel fit rate (CRLB ≤ {float(thresholds[0]):g}%)",
        fontsize=12,
    )
    axis.tick_params(axis="both", labelsize=11)
    axis.margins(x=0.12, y=0.15)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[method],
            linestyle="none",
            markerfacecolor="0.35",
            markeredgecolor="0.35",
            markersize=8,
            label=display_labels.get(method, method),
        )
        for method in methods
    ]
    axis.legend(handles=handles, frameon=False, title="Method", fontsize=10)
    return figure
