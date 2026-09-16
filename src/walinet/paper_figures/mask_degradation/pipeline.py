"""Evaluate multiple YNet seeds with operators from degraded masks."""

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
from walinet.inference.inference import _load_checkpoint_state_dict, _load_model_and_params
from walinet.paper_figures.mask_degradation.config import load_config
from walinet.paper_figures.mask_degradation.masks import dilate_lipid_inward
from walinet.paper_figures.paper_fig1.metrics import ALL_METRICS, calculate_errors
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


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _load_ynet(model_path: Path, checkpoint: str, device: torch.device):
    model_class, params, architecture, model_dir = _load_model_and_params(
        exp=model_path.name,
        model_root=model_path.parent,
        architecture="auto",
    )
    if architecture != "ynet":
        raise ValueError(f"Mask degradation requires a YNet, found {architecture!r}.")
    model = model_class(
        nLayers=int(params["nLayers"]),
        nFilters=int(params["nFilters"]),
        dropout=float(params.get("dropout", 0.0)),
        in_channels=int(params["in_channels"]),
        out_channels=int(params["out_channels"]),
    )
    checkpoint_path = model_dir / checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    model.load_state_dict(_load_checkpoint_state_dict(checkpoint_path, device))
    return model.to(device).eval()


def _compute_operator(
    *,
    subject_dir: Path,
    input_data: str,
    brain_mask_path: str,
    lipid_mask_path: str,
    resource_path: str = "TrainData/SimulationResources_water_lipid_v1.h5",
    n_timepoints: int,
    n_voxels: int,
    device: torch.device,
    target: float = 0.938,
) -> tuple[np.ndarray, int, int, int]:
    brain = np.load(subject_dir / brain_mask_path) > 0
    lipid = np.load(subject_dir / lipid_mask_path) > 0
    _, degraded_lipid, transferred = dilate_lipid_inward(
        brain,
        lipid,
        n_voxels,
    )
    csi = np.load(subject_dir / input_data, mmap_mode="r")
    if csi.shape[:-1] != degraded_lipid.shape:
        raise ValueError(
            f"{subject_dir.name}: CSI and mask spatial shapes differ: "
            f"{csi.shape[:-1]} versus {degraded_lipid.shape}."
        )
    resource_file = subject_dir / resource_path
    with h5py.File(resource_file, "r") as resources:
        stored_lipid_fids = np.asarray(resources["lipid_fids"], dtype=np.complex64)
        if stored_lipid_fids.shape[0] != int(np.count_nonzero(lipid)):
            raise ValueError(
                f"{subject_dir.name}: cannot map compact water-suppressed lipid FIDs "
                "back to mask voxels because their counts differ."
            )
        retained_original_lipid = degraded_lipid[lipid]
        added_brain_voxels = degraded_lipid & brain
        if not np.any(added_brain_voxels):
            lipid_fids = stored_lipid_fids[
                retained_original_lipid, :n_timepoints
            ].copy()
        else:
            isolated_brain_water = np.asarray(
                resources["water_fids"], dtype=np.complex64
            )
            lipid_fids = (
                np.asarray(csi[degraded_lipid, :n_timepoints], dtype=np.complex64)
                - isolated_brain_water[degraded_lipid, :n_timepoints]
            )
            selected_original_lipid = lipid[degraded_lipid]
            lipid_fids[selected_original_lipid] = stored_lipid_fids[
                retained_original_lipid, :n_timepoints
            ]
    if lipid_fids.shape[0] == 0:
        raise ValueError(
            f"{subject_dir.name}: dilation {n_voxels} removes the complete "
            "lipid mask, so no projection operator can be computed."
        )
    if lipid_fids.shape[-1] < n_timepoints:
        prepared = np.zeros((lipid_fids.shape[0], n_timepoints), dtype=np.complex64)
        prepared[:, :lipid_fids.shape[-1]] = lipid_fids
        lipid_fids = prepared
    lipid_spectra = np.fft.fftshift(
        np.fft.fft(lipid_fids, axis=-1),
        axes=-1,
    ).astype(np.complex64, copy=False)
    compact_spectra = lipid_spectra[:, None, None, :]
    compact_mask = np.ones(compact_spectra.shape[:-1], dtype=bool)
    operator = compute_lipid_projection_operator(
        compact_spectra,
        compact_mask,
        target=target,
        method="gpu" if device.type == "cuda" else "eigenvalue",
        gpu_index=0 if device.index is None else device.index,
    )
    return (
        np.ascontiguousarray(operator, dtype=np.complex64),
        int(np.count_nonzero(lipid)),
        int(np.count_nonzero(degraded_lipid)),
        int(np.count_nonzero(transferred)),
    )


def run_evaluation(config_path: str | Path, *, overwrite: bool = False) -> Path:
    """Run the subject-wise mask-degradation evaluation."""
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
        _load_yaml(config.training_config),
        config_dir=config.training_config.parent,
    )
    simulation_path = Path(train_cfg.data.simulation_config)
    simulation_cfg = build_simulation_config(
        _load_yaml(simulation_path),
        config_dir=simulation_path.parent,
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
        prepared_basis=basis,
        config=simulation_cfg,
        device=device,
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

    try:
        with h5py.File(temporary_path, "w") as h5:
            h5.attrs["format"] = "walinet.mask_degradation.v3"
            h5.attrs["seed"] = config.seed
            h5.attrs["n_spectra_per_subject"] = config.n_spectra_per_subject
            h5.attrs["operator_target"] = config.operator_target
            h5.attrs["operator_tolerance"] = DEFAULT_TRACE_TOLERANCE
            h5.create_dataset(
                "evaluation_config_yaml",
                data=config.source_path.read_text(encoding="utf-8"),
            )
            string_dtype = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset("model_labels", data=[item.label for item in config.models],
                              dtype=string_dtype)
            h5.create_dataset("model_paths", data=[str(item.path) for item in config.models],
                              dtype=string_dtype)
            h5.create_dataset("checkpoints", data=[item.checkpoint for item in config.models],
                              dtype=string_dtype)
            h5.create_dataset("subject_labels", data=list(config.subjects), dtype=string_dtype)
            subjects_group = h5.create_group("subjects")
            metrics = ALL_METRICS
            combination_means = {
                metric: {
                    n_voxels: np.empty(
                        (len(config.subjects), len(config.models)), dtype=np.float64
                    )
                    for n_voxels in config.dilations
                }
                for metric in metrics
            }

            for subject_index, subject in enumerate(config.subjects):
                print(
                    f"[Mask degradation] Subject {subject_index + 1}/"
                    f"{len(config.subjects)}: {subject}",
                    flush=True,
                )
                subject_dir = Path(train_cfg.data.base_dir) / subject
                subject_group = subjects_group.create_group(subject)
                dilations_group = subject_group.create_group("dilations")
                operators: dict[int, torch.Tensor] = {}

                for n_voxels in config.dilations:
                    print(f"  Computing operator for dilation {n_voxels}...", flush=True)
                    operator, original_count, degraded_count, transferred_count = (
                        _compute_operator(
                            subject_dir=subject_dir,
                            input_data=config.input_data,
                            brain_mask_path=config.brain_mask,
                            lipid_mask_path=config.lipid_mask,
                            resource_path=resource_filename,
                            n_timepoints=simulation_cfg.acquisition.n_timepoints,
                            n_voxels=n_voxels,
                            device=device,
                            target=config.operator_target,
                        )
                    )
                    operators[n_voxels] = torch.as_tensor(operator, device=device)
                    group = dilations_group.create_group(f"n_voxels_{n_voxels:03d}")
                    group.attrs["n_voxels"] = n_voxels
                    group.attrs["operator_tolerance"] = DEFAULT_TRACE_TOLERANCE
                    group.attrs["original_lipid_voxels"] = original_count
                    group.attrs["degraded_lipid_voxels"] = degraded_count
                    group.attrs["transferred_voxels"] = transferred_count
                    remaining_percent = 100.0 * min(
                        degraded_count, original_count
                    ) / original_count
                    group.attrs["lipid_mask_remaining_percent"] = remaining_percent
                    print(
                        f"    Lipid mask remaining: {remaining_percent:.2f}% "
                        f"({degraded_count}/{original_count} voxels)",
                        flush=True,
                    )
                    group.create_dataset("operator", data=operator, compression="gzip")
                    model_group = group.create_group("models")
                    for model_label, _ in models:
                        result_group = model_group.create_group(model_label)
                        for metric in metrics:
                            result_group.create_dataset(
                                metric,
                                shape=(config.n_spectra_per_subject,),
                                dtype=np.float32,
                                chunks=(chunk_size,),
                                compression="gzip",
                                compression_opts=4,
                            )

                pool = load_simulation_pool(
                    base_dir=train_cfg.data.base_dir,
                    subjects=[subject],
                    resource_filename=resource_filename,
                    target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
                    expected_bandwidth_hz=simulation_cfg.acquisition.bandwidth_hz,
                    load_projection_operator=True,
                ).to(device)
                simulator = SpectrumSimulator(
                    pool=pool,
                    metabolite_simulator=metabolite_simulator,
                    config=simulation_cfg,
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(config.seed + subject_index)
                error_sums = {
                    metric: {
                        n_voxels: np.zeros(len(models), dtype=np.float64)
                        for n_voxels in config.dilations
                    }
                    for metric in metrics
                }
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
                    stop = written + current_size
                    for n_voxels, operator in operators.items():
                        projected = batch.raw_input_spectra @ operator
                        network_l2 = complex_spectra_to_channels(
                            projected / batch.normalization_scale
                        )
                        group = dilations_group[f"n_voxels_{n_voxels:03d}"]
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
                            result_group = group["models"][model_label]
                            values = {
                                metric: value.cpu().numpy()
                                for metric, value in zip(metrics, errors)
                            }
                            for metric, metric_values in values.items():
                                metric_values = metric_values.astype(np.float32)
                                result_group[metric][written:stop] = metric_values
                                error_sums[metric][n_voxels][model_index] += float(
                                    np.sum(metric_values, dtype=np.float64)
                                )
                    written = stop
                    h5.flush()
                    print(
                        f"  {subject}: {written}/{config.n_spectra_per_subject}",
                        flush=True,
                    )
                    del batch
                for metric in metrics:
                    for n_voxels in config.dilations:
                        combination_means[metric][n_voxels][subject_index, :] = (
                            error_sums[metric][n_voxels]
                            / config.n_spectra_per_subject
                        )
                subject_group.attrs["complete"] = True

            aggregates = h5.create_group("aggregates")
            aggregates.attrs["aggregation_unit"] = "subject_mean_over_seeds"
            aggregates.attrs["n_combinations"] = len(config.subjects) * len(config.models)
            aggregates.attrs["n_subjects"] = len(config.subjects)
            aggregates.attrs["n_seeds"] = len(config.models)
            aggregates.attrs["std_ddof"] = 1
            for n_voxels in config.dilations:
                dilation_group = aggregates.create_group(f"n_voxels_{n_voxels:03d}")
                dilation_group.attrs["n_voxels"] = n_voxels
                for metric in metrics:
                    metric_group = dilation_group.create_group(metric)
                    values = combination_means[metric][n_voxels]
                    subject_means = np.mean(values, axis=1)
                    metric_group.create_dataset("subject_seed_means", data=values)
                    metric_group.create_dataset(
                        "subject_means_over_seeds", data=subject_means
                    )
                    metric_group.create_dataset(
                        "mean_over_subjects", data=np.mean(subject_means)
                    )
                    metric_group.create_dataset(
                        "std_over_subjects", data=np.std(subject_means, ddof=1)
                    )
            h5.flush()

        os.replace(temporary_path, config.output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return config.output_path
