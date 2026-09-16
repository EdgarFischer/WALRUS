"""Evaluate YNet sensitivity to the lipid-operator trace target."""

from __future__ import annotations

import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from walinet.config.build import build_config
from walinet.config.build_simulation import build_simulation_config
from walinet.paper_figures.mask_degradation.pipeline import _load_ynet
from walinet.paper_figures.paper_fig1.metrics import ALL_METRICS, calculate_errors
from walinet.paper_figures.trace_sweep.config import load_config
from walinet.training.training import forward_model
from walinet.training_data.lcmodel_basis.acquisition import prepare_basis_for_acquisition
from walinet.training_data.lipid_removal import (
    DEFAULT_TRACE_TOLERANCE,
    compute_lipid_projection_operator,
)
from walinet.training_data.metabolite_simulation import MetaboliteSimulator
from walinet.training_data.simulation_resources import load_simulation_pool
from walinet.training_data.spectrum_simulator import (
    SpectrumSimulator,
    complex_spectra_to_channels,
)


METRICS = ALL_METRICS


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _target_key(target: float) -> str:
    return f"target_{round(target * 100):03d}"


def _load_lipid_spectra(
    subject_dir: Path,
    input_data: str,
    lipid_mask_path: str,
    resource_path: str,
    n_timepoints: int,
) -> np.ndarray:
    lipid_mask = np.load(subject_dir / lipid_mask_path) > 0
    csi = np.load(subject_dir / input_data, mmap_mode="r")
    if csi.shape[:-1] != lipid_mask.shape:
        raise ValueError(
            f"{subject_dir.name}: CSI/mask shape mismatch: "
            f"{csi.shape[:-1]} versus {lipid_mask.shape}."
        )
    with h5py.File(subject_dir / resource_path, "r") as resources:
        lipid_fids = np.asarray(resources["lipid_fids"], dtype=np.complex64)
    if lipid_fids.shape[0] != int(np.count_nonzero(lipid_mask)):
        raise ValueError(
            f"{subject_dir.name}: lipid_fids count does not match lipid mask."
        )
    if lipid_fids.shape[1] < n_timepoints:
        padded = np.zeros((lipid_fids.shape[0], n_timepoints), dtype=np.complex64)
        padded[:, :lipid_fids.shape[1]] = lipid_fids
        lipid_fids = padded
    lipid_fids = lipid_fids[:, :n_timepoints]
    return np.fft.fftshift(np.fft.fft(lipid_fids, axis=-1), axes=-1).astype(
        np.complex64, copy=False
    )


def run_evaluation(config_path: str | Path, *, overwrite: bool = False) -> Path:
    config = load_config(config_path)
    if config.output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {config.output_path}. Pass overwrite=True to replace it."
        )
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {config.device}")
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    train_cfg = build_config(
        _load_yaml(config.training_config), config_dir=config.training_config.parent
    )
    simulation_path = Path(train_cfg.data.simulation_config)
    simulation_cfg = build_simulation_config(
        _load_yaml(simulation_path), config_dir=simulation_path.parent
    )
    if not simulation_cfg.lipid_projection.enabled:
        raise ValueError("lipid_projection must be enabled for this evaluation.")
    basis = prepare_basis_for_acquisition(
        simulation_cfg.basis.library,
        target_bandwidth=simulation_cfg.acquisition.bandwidth_hz,
        target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
        dataset_name="clean_fid",
    )
    metabolite_simulator = MetaboliteSimulator(
        prepared_basis=basis, config=simulation_cfg, device=device
    )
    resource_filename = train_cfg.data.resources.filename.format(
        version=train_cfg.data.resources.version
    )
    models = [
        (item.label, _load_ynet(item.path, item.checkpoint, device))
        for item in config.models
    ]

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config.output_path.with_name(f".{config.output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    chunk_size = min(config.batch_size, config.n_spectra_per_subject)
    combination_means = {
        metric: {
            target: np.empty((len(config.subjects), len(models)), dtype=np.float64)
            for target in config.targets
        }
        for metric in METRICS
    }

    try:
        with h5py.File(temporary_path, "w") as h5:
            h5.attrs["format"] = "walinet.trace_sweep.v2"
            h5.attrs["seed"] = config.seed
            h5.attrs["n_spectra_per_subject"] = config.n_spectra_per_subject
            h5.attrs["operator_tolerance"] = DEFAULT_TRACE_TOLERANCE
            text_dtype = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset("evaluation_config_yaml", data=config.source_path.read_text())
            h5.create_dataset("targets", data=config.targets)
            h5.create_dataset("subject_labels", data=config.subjects, dtype=text_dtype)
            h5.create_dataset("model_labels", data=[x.label for x in config.models],
                              dtype=text_dtype)
            h5.create_dataset("model_paths", data=[str(x.path) for x in config.models],
                              dtype=text_dtype)
            subjects_group = h5.create_group("subjects")

            for subject_index, subject in enumerate(config.subjects):
                print(
                    f"[Trace sweep] Subject {subject_index + 1}/{len(config.subjects)}: "
                    f"{subject}", flush=True
                )
                subject_dir = Path(train_cfg.data.base_dir) / subject
                lipid_spectra = _load_lipid_spectra(
                    subject_dir, config.input_data, config.lipid_mask,
                    resource_filename, simulation_cfg.acquisition.n_timepoints,
                )
                compact = lipid_spectra[:, None, None, :]
                compact_mask = np.ones(compact.shape[:-1], dtype=bool)
                subject_group = subjects_group.create_group(subject)
                target_groups = subject_group.create_group("targets")
                operators = {}
                for target in config.targets:
                    print(f"  Computing operator for target {target:.2f}...", flush=True)
                    operator = compute_lipid_projection_operator(
                        compact,
                        compact_mask,
                        target=target,
                        method="gpu" if device.type == "cuda" else "eigenvalue",
                        gpu_index=0 if device.index is None else device.index,
                    ).astype(np.complex64, copy=False)
                    operators[target] = torch.as_tensor(operator, device=device)
                    group = target_groups.create_group(_target_key(target))
                    group.attrs["target"] = target
                    group.attrs["operator_tolerance"] = DEFAULT_TRACE_TOLERANCE
                    group.create_dataset("operator", data=operator, compression="gzip")
                    model_groups = group.create_group("models")
                    for model_label, _ in models:
                        model_group = model_groups.create_group(model_label)
                        for metric in METRICS:
                            model_group.create_dataset(
                                metric, shape=(config.n_spectra_per_subject,),
                                dtype=np.float32, chunks=(chunk_size,), compression="gzip",
                                compression_opts=4,
                            )
                del compact, compact_mask, lipid_spectra

                pool = load_simulation_pool(
                    base_dir=train_cfg.data.base_dir,
                    subjects=[subject],
                    resource_filename=resource_filename,
                    target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
                    expected_bandwidth_hz=simulation_cfg.acquisition.bandwidth_hz,
                    load_projection_operator=True,
                ).to(device)
                simulator = SpectrumSimulator(
                    pool=pool, metabolite_simulator=metabolite_simulator,
                    config=simulation_cfg,
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(config.seed + subject_index)
                sums = {
                    metric: {target: np.zeros(len(models)) for target in config.targets}
                    for metric in METRICS
                }
                written = 0
                while written < config.n_spectra_per_subject:
                    current_size = min(config.batch_size,
                                       config.n_spectra_per_subject - written)
                    batch = simulator.simulate(
                        batch_size=current_size, generator=generator
                    )
                    stop = written + current_size
                    for target, operator in operators.items():
                        projected = batch.raw_input_spectra @ operator
                        network_l2 = complex_spectra_to_channels(
                            projected / batch.normalization_scale
                        )
                        target_group = target_groups[_target_key(target)]
                        for model_index, (model_label, model) in enumerate(models):
                            with torch.inference_mode():
                                prediction = forward_model(
                                    model=model,
                                    network_input=batch.network_input,
                                    network_l2=network_l2,
                                    architecture="ynet",
                                )[:, :2, :]
                                errors = calculate_errors(
                                    true_nuisance=batch.network_target,
                                    predicted_nuisance=prediction,
                                    clean_metabolite=(
                                        batch.normalized_clean_metabolite_spectra
                                    ),
                                )
                            for metric, tensor in zip(METRICS, errors):
                                values = tensor.cpu().numpy().astype(np.float32)
                                target_group["models"][model_label][metric][written:stop] = values
                                sums[metric][target][model_index] += np.sum(
                                    values, dtype=np.float64
                                )
                    written = stop
                    h5.flush()
                    print(f"  {subject}: {written}/{config.n_spectra_per_subject}",
                          flush=True)
                    del batch
                for metric in METRICS:
                    for target in config.targets:
                        combination_means[metric][target][subject_index] = (
                            sums[metric][target] / config.n_spectra_per_subject
                        )
                subject_group.attrs["complete"] = True

            aggregates = h5.create_group("aggregates")
            aggregates.attrs["aggregation_unit"] = "subject_mean_over_seeds"
            aggregates.attrs["n_subjects"] = len(config.subjects)
            aggregates.attrs["n_seeds"] = len(models)
            aggregates.attrs["std_ddof"] = 1
            for target in config.targets:
                target_group = aggregates.create_group(_target_key(target))
                target_group.attrs["target"] = target
                for metric in METRICS:
                    values = combination_means[metric][target]
                    subject_means = values.mean(axis=1)
                    group = target_group.create_group(metric)
                    group.create_dataset("subject_seed_means", data=values)
                    group.create_dataset("subject_means_over_seeds", data=subject_means)
                    group.create_dataset("mean_over_subjects", data=subject_means.mean())
                    group.create_dataset("std_over_subjects", data=subject_means.std(ddof=1))
            h5.flush()
        os.replace(temporary_path, config.output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return config.output_path
