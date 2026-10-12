"""Compact grid of simulated full, nuisance, and metabolite spectra."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def _ppm_axis(simulation_config) -> np.ndarray:
    acquisition = simulation_config.acquisition
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            acquisition.n_timepoints,
            d=1.0 / acquisition.bandwidth_hz,
        )
    )
    return 4.7 - frequencies / acquisition.hz_per_ppm


def simulate_examples(
    *,
    training_config: str | Path,
    n_examples: int = 8,
    seed: int = 12345,
    subject: str | None = None,
) -> dict[str, np.ndarray]:
    """Simulate examples after loading exactly one training subject."""
    import torch
    import yaml

    from walinet.config.build import build_config
    from walinet.config.build_simulation import build_simulation_config
    from walinet.training_data.lcmodel_basis.acquisition import (
        prepare_basis_for_acquisition,
    )
    from walinet.training_data.metabolite_simulation import MetaboliteSimulator
    from walinet.training_data.simulation_resources import load_simulation_pool
    from walinet.training_data.spectrum_simulator import SpectrumSimulator

    if n_examples <= 0:
        raise ValueError("n_examples must be greater than zero")
    training_path = Path(training_config).expanduser().resolve()
    raw_training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    train_cfg = build_config(raw_training, config_dir=training_path.parent)
    simulation_path = Path(train_cfg.data.simulation_config).resolve()
    raw_simulation = yaml.safe_load(simulation_path.read_text(encoding="utf-8"))
    simulation_cfg = build_simulation_config(
        raw_simulation, config_dir=simulation_path.parent
    )
    selected_subject = subject or train_cfg.data.train_subjects[0]
    if selected_subject not in train_cfg.data.train_subjects:
        raise ValueError(
            f"{selected_subject!r} is not a configured training subject"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the simulation configuration")
    device = torch.device(f"cuda:{int(train_cfg.run.gpu)}")
    torch.cuda.set_device(device)
    basis = prepare_basis_for_acquisition(
        simulation_cfg.basis.library,
        target_bandwidth=simulation_cfg.acquisition.bandwidth_hz,
        target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
        dataset_name="clean_fid",
    )
    resource_filename = train_cfg.data.resources.filename.format(
        version=train_cfg.data.resources.version
    )
    pool = load_simulation_pool(
        base_dir=train_cfg.data.base_dir,
        subjects=[selected_subject],
        resource_filename=resource_filename,
        target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
        expected_bandwidth_hz=simulation_cfg.acquisition.bandwidth_hz,
        load_projection_operator=simulation_cfg.lipid_projection.enabled,
    ).to(device)
    metabolite_simulator = MetaboliteSimulator(
        prepared_basis=basis, config=simulation_cfg, device=device
    )
    simulator = SpectrumSimulator(
        pool=pool, metabolite_simulator=metabolite_simulator,
        config=simulation_cfg,
    )
    print(f"Using one training subject: {selected_subject}", flush=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    batch = simulator.simulate(
        batch_size=n_examples,
        generator=generator,
    )
    result = {
        "ppm": _ppm_axis(simulation_cfg),
        "full": batch.normalized_input_spectra.detach().cpu().numpy(),
        "nuisance": batch.normalized_target_spectra.detach().cpu().numpy(),
        "metabolites": (
            batch.normalized_clean_metabolite_spectra.detach().cpu().numpy()
        ),
    }
    del batch, simulator, metabolite_simulator, pool
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def create_simulation_examples_figure(
    datasets: Mapping[str, Mapping[str, np.ndarray]],
    *,
    figsize: tuple[float, float] = (11.5, 9.0),
):
    """Plot an N×6 grid for 3T and 7T simulation components."""
    import matplotlib.pyplot as plt

    fields = ("3T", "7T")
    components = (
        ("full", "Full spectrum", "black"),
        ("nuisance", "Nuisance", "#D55E00"),
        ("metabolites", "Metabolites", "#0072B2"),
    )
    missing = set(fields) - set(datasets)
    if missing:
        raise ValueError(f"Missing field-strength datasets: {sorted(missing)}")
    n_examples = int(np.asarray(datasets["3T"]["full"]).shape[0])
    if n_examples < 1:
        raise ValueError("No examples supplied")
    for field in fields:
        if np.asarray(datasets[field]["full"]).shape[0] != n_examples:
            raise ValueError("3T and 7T must contain the same number of examples")

    figure, axes = plt.subplots(
        n_examples, 6, figsize=figsize, squeeze=False,
        gridspec_kw={"wspace": 0.08, "hspace": 0.08},
    )
    figure.subplots_adjust(
        left=0.05, right=0.995, bottom=0.07, top=0.89,
        wspace=0.08, hspace=0.08,
    )
    for field_index, field in enumerate(fields):
        first = axes[0, 3 * field_index].get_position()
        last = axes[0, 3 * field_index + 2].get_position()
        figure.text(
            (first.x0 + last.x1) / 2,
            0.965,
            field,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
    for field_index, field in enumerate(fields):
        data = datasets[field]
        ppm = np.asarray(data["ppm"], dtype=float)
        for component_index, (key, title, color) in enumerate(components):
            column = 3 * field_index + component_index
            spectra = np.asarray(data[key])
            if spectra.shape != (n_examples, ppm.size):
                raise ValueError(
                    f"{field}/{key}: shape {spectra.shape} is incompatible "
                    f"with {(n_examples, ppm.size)}"
                )
            for row in range(n_examples):
                axis = axes[row, column]
                axis.plot(ppm, spectra[row].real, color=color, linewidth=0.72)
                axis.set_xlim(float(ppm.max()), float(ppm.min()))
                axis.set_yticks([])
                axis.spines[["top", "right", "left"]].set_visible(False)
                axis.tick_params(axis="x", labelsize=8, length=2.5)
                if row == 0:
                    axis.set_title(
                        title, fontsize=10, fontweight="bold", pad=5
                    )
                if row != n_examples - 1:
                    axis.tick_params(axis="x", labelbottom=False)
                else:
                    axis.set_xlabel("ppm", fontsize=9)
                    axis.set_xticks([
                        tick for tick in (8, 4, 0)
                        if float(ppm.min()) <= tick <= float(ppm.max())
                    ])
                if column == 0:
                    axis.set_ylabel(
                        str(row + 1), rotation=0, labelpad=8,
                        va="center", fontsize=9, fontweight="bold",
                    )
    figure.align_ylabels(axes[:, 0])
    return figure
