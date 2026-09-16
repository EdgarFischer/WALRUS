from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from walinet.paper_figures.paper_fig1.metrics import ALL_METRICS


def results_dataframe(path: str | Path):
    """Return per-subject, model, and dilation summary statistics."""
    import pandas as pd

    rows = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        for subject, subject_group in h5["subjects"].items():
            for _, group in subject_group["dilations"].items():
                model_groups = (
                    group["models"].items()
                    if "models" in group
                    else [("model", group)]
                )
                for model, model_group in model_groups:
                    for metric in ALL_METRICS:
                        values = model_group[metric][:]
                        rows.append({
                            "subject": subject,
                            "model": model,
                            "dilation": int(group.attrs["n_voxels"]),
                            "transferred_voxels": int(group.attrs["transferred_voxels"]),
                            "lipid_mask_remaining_percent": float(
                                group.attrs["lipid_mask_remaining_percent"]
                            ),
                            "metric": metric,
                            "mean": float(np.mean(values)),
                            "median": float(np.median(values)),
                            "q25": float(np.quantile(values, 0.25)),
                            "q75": float(np.quantile(values, 0.75)),
                        })
    return pd.DataFrame.from_records(rows).sort_values(
        ["subject", "model", "dilation", "metric"]
    ).reset_index(drop=True)


def aggregates_dataframe(path: str | Path):
    """Return mean and SD across subjects after averaging model seeds."""
    import pandas as pd

    rows = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        if "aggregates" not in h5:
            raise ValueError("Result file has no v2 aggregate results.")
        for _, dilation_group in h5["aggregates"].items():
            for metric in ALL_METRICS:
                metric_group = dilation_group[metric]
                rows.append({
                    "dilation": int(dilation_group.attrs["n_voxels"]),
                    "metric": metric,
                    "mean": float(metric_group["mean_over_subjects"][()]),
                    "std": float(metric_group["std_over_subjects"][()]),
                    "n_subjects": int(metric_group["subject_means_over_seeds"].size),
                    "n_seeds": int(metric_group["subject_seed_means"].shape[1]),
                })
    return pd.DataFrame.from_records(rows).sort_values(
        ["dilation", "metric"]
    ).reset_index(drop=True)


def subject_means_dataframe(path: str | Path):
    """Return the five stored subject means after averaging model seeds."""
    import pandas as pd

    rows = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        subjects = [value.decode() if isinstance(value, bytes) else str(value)
                    for value in h5["subject_labels"][:]]
        for dilation_group in h5["aggregates"].values():
            for metric in ALL_METRICS:
                values = dilation_group[metric]["subject_means_over_seeds"][:]
                rows.extend({
                    "subject": subject,
                    "dilation": int(dilation_group.attrs["n_voxels"]),
                    "metric": metric,
                    "mean": float(value),
                } for subject, value in zip(subjects, values))
    return pd.DataFrame.from_records(rows).sort_values(
        ["subject", "dilation", "metric"]
    ).reset_index(drop=True)
