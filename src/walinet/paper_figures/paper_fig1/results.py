"""Read Paper Figure 1 evaluation results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from walinet.paper_figures.paper_fig1.metrics import ALL_METRICS


def load_results(path: str | Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    result = {}
    with h5py.File(Path(path).expanduser(), "r") as h5:
        for subject, subject_group in h5["subjects"].items():
            result[subject] = {}
            for model, model_group in subject_group["models"].items():
                result[subject][model] = {
                    metric: model_group[metric][:] for metric in ALL_METRICS
                }
    return result


def results_dataframe(path: str | Path | Sequence[str | Path]):
    """Return compact per-subject/model summary statistics."""
    import pandas as pd

    rows = []
    paths = (path,) if isinstance(path, (str, Path)) else tuple(path)
    for result_path in paths:
        with h5py.File(Path(result_path).expanduser(), "r") as h5:
            for subject, subject_group in h5["subjects"].items():
                for model, model_group in subject_group["models"].items():
                    family = str(model_group.attrs["family"])
                    for metric in ALL_METRICS:
                        values = model_group[metric][:]
                        rows.append({
                            "subject": subject,
                            "model": model,
                            "family": family,
                            "metric": metric,
                            "mean": float(np.mean(values)),
                            "median": float(np.median(values)),
                            "q25": float(np.quantile(values, 0.25)),
                            "q75": float(np.quantile(values, 0.75)),
                        })
    frame = pd.DataFrame.from_records(rows)
    duplicates = frame.duplicated(["subject", "model", "metric"], keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, ["subject", "model", "metric"]]
        raise ValueError(f"Duplicate result rows across HDF5 inputs:\n{values}")
    return frame
