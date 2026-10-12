"""Streaming, reproducible comparison of WALINET model families."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from walinet.config.build import build_config
from walinet.config.build_simulation import build_simulation_config
from walinet.inference.inference import (
    _load_checkpoint_state_dict,
    _load_model_and_params,
)
from walinet.paper_figures.paper_fig1.config import (
    EvaluationConfig,
    ModelSpec,
    load_evaluation_config,
)
from walinet.paper_figures.paper_fig1.metrics import ALL_METRICS, calculate_errors
from walinet.training.training import forward_model
from walinet.training_data.lcmodel_basis.acquisition import prepare_basis_for_acquisition
from walinet.training_data.metabolite_simulation import MetaboliteSimulator
from walinet.training_data.simulation_resources import load_simulation_pool
from walinet.training_data.spectrum_simulator import SpectrumSimulator


@dataclass(frozen=True)
class LoadedModel:
    spec: ModelSpec
    architecture: str
    model: torch.nn.Module


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _load_models(config: EvaluationConfig, device: torch.device) -> tuple[LoadedModel, ...]:
    loaded = []
    for spec in config.models:
        model_class, params, architecture, model_dir = _load_model_and_params(
            exp=spec.path.name,
            model_root=spec.path.parent,
            architecture="auto",
        )
        expected_architecture = "ynet" if spec.family == "ynet" else "unet"
        if architecture != expected_architecture:
            raise ValueError(
                f"{spec.name}: configured family {spec.family!r}, but model "
                f"metadata reports {architecture!r}."
            )
        model = model_class(
            nLayers=int(params["nLayers"]),
            nFilters=int(params["nFilters"]),
            dropout=float(params.get("dropout", 0.0)),
            in_channels=int(params["in_channels"]),
            out_channels=int(params["out_channels"]),
        )
        checkpoint_path = model_dir / spec.checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        model.load_state_dict(_load_checkpoint_state_dict(checkpoint_path, device))
        model.to(device).eval()
        loaded.append(LoadedModel(spec=spec, architecture=architecture, model=model))
    return tuple(loaded)


def _build_common_simulation(config: EvaluationConfig, device: torch.device):
    if not config.training_config.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config.training_config}")
    train_cfg = build_config(
        _load_yaml(config.training_config),
        config_dir=config.training_config.parent,
    )
    simulation_path = Path(train_cfg.data.simulation_config).resolve()
    simulation_raw = _load_yaml(simulation_path)
    # Model snapshots retain paths relative to the original central config
    # directory. Relocate files only when those archived relative paths no
    # longer resolve; all numerical simulation settings remain untouched.
    basis_path = Path(simulation_raw["basis"]["library"])
    if not basis_path.is_absolute() and not (simulation_path.parent / basis_path).resolve().is_file():
        project_root = Path(__file__).resolve().parents[4]
        relocated_basis = project_root / "MetabModes/LCModelBasis/processed" / basis_path.name
        if relocated_basis.is_file():
            simulation_raw["basis"]["library"] = str(relocated_basis)
    for profile in simulation_raw.get("metabolites", {}).get("profiles", []):
        profile_path = Path(profile["config"])
        if profile_path.is_absolute() or (simulation_path.parent / profile_path).resolve().is_file():
            continue
        relocated_profile = simulation_path.parent.parent / "dependencies" / profile_path.name
        if relocated_profile.is_file():
            profile["config"] = str(relocated_profile)
    simulation_cfg = build_simulation_config(
        simulation_raw,
        config_dir=simulation_path.parent,
    )
    if not simulation_cfg.lipid_projection.enabled:
        raise ValueError(
            "PaperFig1 requires lipid_projection.enabled=true so the same "
            "simulated batches can be evaluated by YNet and U-Net."
        )
    basis = prepare_basis_for_acquisition(
        simulation_cfg.basis.library,
        target_bandwidth=simulation_cfg.acquisition.bandwidth_hz,
        target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
        dataset_name="clean_fid",
    )
    metabolite_simulator = MetaboliteSimulator(
        prepared_basis=basis,
        config=simulation_cfg,
        device=device,
    )
    resource_filename = train_cfg.data.resources.filename.format(
        version=train_cfg.data.resources.version
    )
    return train_cfg, simulation_cfg, metabolite_simulator, resource_filename


def _subject_simulator(
    *,
    subject: str,
    train_cfg,
    simulation_cfg,
    metabolite_simulator,
    resource_filename: str,
    device: torch.device,
) -> SpectrumSimulator:
    resource_subject = subject
    expected_path = Path(train_cfg.data.base_dir) / subject / resource_filename
    if not expected_path.is_file() and "_" in subject:
        subject_name, site = subject.rsplit("_", 1)
        current_name = f"{site}/{subject_name}"
        current_path = Path(train_cfg.data.base_dir) / current_name / resource_filename
        if current_path.is_file():
            resource_subject = current_name
    pool = load_simulation_pool(
        base_dir=train_cfg.data.base_dir,
        subjects=[resource_subject],
        resource_filename=resource_filename,
        target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
        expected_bandwidth_hz=simulation_cfg.acquisition.bandwidth_hz,
        load_projection_operator=True,
    ).to(device)
    return SpectrumSimulator(
        pool=pool,
        metabolite_simulator=metabolite_simulator,
        config=simulation_cfg,
    )


def _prepare_output(h5: h5py.File, config: EvaluationConfig) -> None:
    h5.attrs["format"] = "walinet.paper_fig1.model_comparison.v2"
    h5.attrs["n_spectra_per_subject"] = config.n_spectra_per_subject
    h5.attrs["batch_size"] = config.batch_size
    h5.attrs["seed"] = config.seed
    h5.attrs["device"] = config.device
    h5.create_dataset(
        "evaluation_config_yaml",
        data=config.source_path.read_text(encoding="utf-8"),
    )
    subjects_group = h5.create_group("subjects")
    chunk_size = min(config.batch_size, config.n_spectra_per_subject)
    for subject in config.subjects:
        models_group = subjects_group.create_group(subject).create_group("models")
        for spec in config.models:
            group = models_group.create_group(spec.name)
            group.attrs["family"] = spec.family
            group.attrs["model_path"] = str(spec.path)
            group.attrs["checkpoint"] = spec.checkpoint
            for metric in ALL_METRICS:
                group.create_dataset(
                    metric,
                    shape=(config.n_spectra_per_subject,),
                    dtype=np.float32,
                    chunks=(chunk_size,),
                    compression="gzip",
                    compression_opts=4,
                )


def run_evaluation(config_path: str | Path, *, overwrite: bool = False) -> Path:
    """Evaluate all configured models on identical subject-specific simulations."""
    config = load_evaluation_config(config_path)
    if config.output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {config.output_path}. Pass overwrite=True to replace it."
        )
    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError(f"CUDA device requested but unavailable: {config.device}")

    device = torch.device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config.seed)

    models = _load_models(config, device)
    common = _build_common_simulation(config, device)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config.output_path.with_name(f".{config.output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    try:
        with h5py.File(temporary_path, "w") as h5:
            _prepare_output(h5, config)
            for subject_index, subject in enumerate(config.subjects):
                print(f"[PaperFig1] Subject {subject_index + 1}/{len(config.subjects)}: {subject}")
                simulator = _subject_simulator(
                    subject=subject,
                    train_cfg=common[0],
                    simulation_cfg=common[1],
                    metabolite_simulator=common[2],
                    resource_filename=common[3],
                    device=device,
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(config.seed + subject_index)
                written = 0
                while written < config.n_spectra_per_subject:
                    current_size = min(
                        config.batch_size,
                        config.n_spectra_per_subject - written,
                    )
                    batch = simulator.simulate(
                        batch_size=current_size,
                        generator=generator,
                    )
                    if batch.network_l2 is None:
                        raise RuntimeError(f"{subject}: simulator produced no YNet L2 input")

                    for loaded in models:
                        with torch.inference_mode():
                            prediction = forward_model(
                                model=loaded.model,
                                network_input=batch.network_input,
                                network_l2=batch.network_l2,
                                architecture=loaded.architecture,
                            )[:, :2, :]
                            errors = calculate_errors(
                                true_nuisance=batch.network_target,
                                predicted_nuisance=prediction,
                                clean_metabolite=(
                                    batch.normalized_clean_metabolite_spectra
                                ),
                            )
                        group = h5[f"subjects/{subject}/models/{loaded.spec.name}"]
                        stop = written + current_size
                        for metric, values in zip(ALL_METRICS, errors):
                            group[metric][written:stop] = (
                                values.detach().cpu().numpy().astype(np.float32)
                            )

                    written += current_size
                    h5.flush()
                    print(
                        f"[PaperFig1] {subject}: {written}/"
                        f"{config.n_spectra_per_subject}",
                        flush=True,
                    )
                    del batch

                h5[f"subjects/{subject}"].attrs["complete"] = True

            aggregates = h5.create_group("aggregates")
            aggregates.attrs["aggregation_unit"] = "subject_mean"
            aggregates.attrs["std_ddof"] = 1
            aggregates.attrs["n_subjects"] = len(config.subjects)
            for spec in config.models:
                model_group = aggregates.create_group(spec.name)
                for metric in ALL_METRICS:
                    subject_means = np.asarray(
                        [
                            np.mean(
                                h5[f"subjects/{subject}/models/{spec.name}/{metric}"][:],
                                dtype=np.float64,
                            )
                            for subject in config.subjects
                        ],
                        dtype=np.float64,
                    )
                    metric_group = model_group.create_group(metric)
                    metric_group.create_dataset("subject_means", data=subject_means)
                    metric_group.create_dataset(
                        "mean_over_subjects", data=subject_means.mean()
                    )
                    metric_group.create_dataset(
                        "std_over_subjects", data=subject_means.std(ddof=1)
                    )

        os.replace(temporary_path, config.output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return config.output_path
