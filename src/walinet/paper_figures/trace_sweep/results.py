from __future__ import annotations

from pathlib import Path

import h5py


def results_dataframe(path: str | Path):
    """Return spectrum-averaged values for every subject, seed, and target."""
    import pandas as pd

    rows = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        for subject, subject_group in h5["subjects"].items():
            for target_group in subject_group["targets"].values():
                target = float(target_group.attrs["target"])
                for model, model_group in target_group["models"].items():
                    for metric in ("nuisance_relative_l2", "metabolite_relative_l2"):
                        values = model_group[metric][:]
                        rows.append({"subject": subject, "model": model, "target": target,
                                     "metric": metric, "mean": float(values.mean())})
    return pd.DataFrame.from_records(rows).sort_values(
        ["subject", "model", "target", "metric"]
    ).reset_index(drop=True)


def aggregates_dataframe(path: str | Path):
    """Return mean and SD across subjects after averaging model seeds."""
    import pandas as pd

    rows = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        for target_group in h5["aggregates"].values():
            for metric in ("nuisance_relative_l2", "metabolite_relative_l2"):
                group = target_group[metric]
                rows.append({
                    "target": float(target_group.attrs["target"]),
                    "metric": metric,
                    "mean": float(group["mean_over_subjects"][()]),
                    "std": float(group["std_over_subjects"][()]),
                    "n_subjects": int(group["subject_means_over_seeds"].size),
                    "n_seeds": int(group["subject_seed_means"].shape[1]),
                })
    return pd.DataFrame.from_records(rows).sort_values(
        ["target", "metric"]
    ).reset_index(drop=True)


def subject_means_dataframe(path: str | Path):
    """Return the five stored subject means after averaging model seeds."""
    import pandas as pd

    rows = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        subjects = [value.decode() if isinstance(value, bytes) else str(value)
                    for value in h5["subject_labels"][:]]
        for target_group in h5["aggregates"].values():
            for metric in ("nuisance_relative_l2", "metabolite_relative_l2"):
                values = target_group[metric]["subject_means_over_seeds"][:]
                rows.extend({
                    "subject": subject,
                    "target": float(target_group.attrs["target"]),
                    "metric": metric,
                    "mean": float(value),
                } for subject, value in zip(subjects, values))
    return pd.DataFrame.from_records(rows).sort_values(
        ["subject", "target", "metric"]
    ).reset_index(drop=True)
