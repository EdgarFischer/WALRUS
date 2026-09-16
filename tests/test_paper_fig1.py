from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from walinet.paper_figures.paper_fig1.config import load_evaluation_config
from walinet.paper_figures.paper_fig1.metrics import (
    ALL_METRICS,
    NUISANCE_RESIDUAL_METRIC,
    calculate_errors,
)
from walinet.paper_figures.paper_fig1.results import load_results, results_dataframe


def test_metrics_are_zero_for_perfect_nuisance_prediction():
    generator = torch.Generator().manual_seed(7)
    clean_metabolite = torch.randn(4, 2, 8, generator=generator)
    target = torch.randn(4, 2, 8, generator=generator)
    nuisance, metabolite, reference = calculate_errors(
        true_nuisance=target,
        predicted_nuisance=target.clone(),
        clean_metabolite=clean_metabolite,
    )
    torch.testing.assert_close(nuisance, torch.zeros_like(nuisance))
    torch.testing.assert_close(metabolite, torch.zeros_like(metabolite))
    assert torch.all(reference > 0)


def test_config_ignores_disabled_models_and_resolves_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "training_config": "train.yaml",
            "device": "cpu",
            "simulation": {"subjects": ["S1"], "n_spectra_per_subject": 2},
            "models": [
                {"name": "active", "family": "unet", "path": "model"},
                {"name": "future", "family": "ynet", "path": "future", "enabled": False},
            ],
            "output": {"path": "result.h5"},
        }),
        encoding="utf-8",
    )
    config = load_evaluation_config(config_path)
    assert [model.name for model in config.models] == ["active"]
    assert config.models[0].path == (tmp_path / "model").resolve()
    assert config.output_path == (tmp_path / "result.h5").resolve()


def test_result_readers(tmp_path):
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("subjects/S1/models/M1")
        group.attrs["family"] = "unet"
        for index, metric in enumerate(ALL_METRICS, start=1):
            group.create_dataset(metric, data=[float(index), float(index + 2)])

    nested = load_results(path)
    np.testing.assert_array_equal(
        nested["S1"]["M1"][NUISANCE_RESIDUAL_METRIC], [1.0, 3.0]
    )
    summary = results_dataframe(path)
    assert set(summary["metric"]) == set(ALL_METRICS)
    assert summary.loc[
        summary["metric"] == NUISANCE_RESIDUAL_METRIC, "median"
    ].item() == 2.0
