"""Resumable paired 3T comparison of WALRUS and HLSVD+L2."""

from __future__ import annotations

import json
from pathlib import Path
import random

import h5py
import numpy as np
import torch
import yaml

from walinet.config.build import build_config
from walinet.config.build_simulation import build_simulation_config
from walinet.config.build_water_lipid import build_water_lipid_extraction_config
from walinet.inference.inference import (
    _load_checkpoint_state_dict,
    _load_model_and_params,
)
from walinet.paper_figures.paper_fig1.metrics import ALL_METRICS, calculate_errors
from walinet.training.training import forward_model
from walinet.training_data.hsvd_water import estimate_water_hsvd
from walinet.training_data.lcmodel_basis.acquisition import (
    prepare_basis_for_acquisition,
)
from walinet.training_data.metabolite_simulation import MetaboliteSimulator
from walinet.training_data.simulation_resources import load_simulation_pool
from walinet.training_data.spectrum_simulator import (
    SpectrumSimulator,
    complex_spectra_to_channels,
)


METHODS = ("HLSVD + L2", "WALRUS")
METRICS = ALL_METRICS


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _load_model(model_dir: Path, checkpoint: str, device: torch.device):
    model_class, params, architecture, resolved_dir = _load_model_and_params(
        exp=model_dir.name,
        model_root=model_dir.parent,
        architecture="auto",
    )
    model = model_class(
        nLayers=int(params["nLayers"]),
        nFilters=int(params["nFilters"]),
        dropout=float(params.get("dropout", 0.0)),
        in_channels=int(params["in_channels"]),
        out_channels=int(params["out_channels"]),
    )
    checkpoint_path = resolved_dir / checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model.load_state_dict(_load_checkpoint_state_dict(checkpoint_path, device))
    return model.to(device).eval(), architecture, checkpoint_path


def _signature(**values) -> str:
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }
    return json.dumps(serializable, sort_keys=True)


def _save_generator_state(group: h5py.Group, generator: torch.Generator) -> None:
    state = generator.get_state().cpu().numpy()
    if "generator_state" in group:
        del group["generator_state"]
    group.create_dataset("generator_state", data=state, dtype=np.uint8)


def _restore_generator(
    group: h5py.Group, *, device: torch.device, seed: int
) -> torch.Generator:
    generator = torch.Generator(device=device)
    if "generator_state" in group:
        state = torch.as_tensor(group["generator_state"][:], dtype=torch.uint8)
        generator.set_state(state)
    else:
        generator.manual_seed(seed)
    return generator


def _prepare_subject_group(
    h5: h5py.File,
    *,
    index: int,
    subject: str,
    n_spectra: int,
    chunk_size: int,
) -> h5py.Group:
    root = h5.require_group("subjects")
    key = f"subject_{index:03d}"
    if key in root:
        group = root[key]
        if str(group.attrs["name"]) != subject:
            raise RuntimeError(f"Subject mismatch in existing result group {key}")
        return group
    group = root.create_group(key)
    group.attrs["name"] = subject
    group.attrs["written"] = 0
    group.attrs["complete"] = False
    methods = group.create_group("methods")
    for method in METHODS:
        method_group = methods.create_group(method)
        for metric in METRICS:
            method_group.create_dataset(
                metric,
                shape=(n_spectra,),
                dtype=np.float32,
                chunks=(min(chunk_size, n_spectra),),
                compression="gzip",
                compression_opts=4,
            )
    return group


def _write_aggregates(h5: h5py.File) -> None:
    if "aggregates" in h5:
        del h5["aggregates"]
    subject_groups = list(h5["subjects"].values())
    if not all(bool(group.attrs.get("complete", False)) for group in subject_groups):
        return
    aggregates = h5.create_group("aggregates")
    aggregates.attrs["aggregation_unit"] = "subject_mean"
    aggregates.attrs["std_ddof"] = 1
    aggregates.attrs["n_subjects"] = len(subject_groups)
    for method in METHODS:
        method_group = aggregates.create_group(method)
        for metric in METRICS:
            subject_means = np.asarray(
                [
                    np.mean(group[f"methods/{method}/{metric}"][:], dtype=np.float64)
                    for group in subject_groups
                ],
                dtype=np.float64,
            )
            metric_group = method_group.create_group(metric)
            metric_group.create_dataset("subject_means", data=subject_means)
            metric_group.create_dataset("mean_over_subjects", data=subject_means.mean())
            metric_group.create_dataset(
                "std_over_subjects", data=subject_means.std(ddof=1)
            )


def run_evaluation(
    *,
    project_root: str | Path,
    training_config: str | Path = "configs/Training/train_3T.yaml",
    water_config: str | Path = "configs/Simulation/3T_water_lipid_v1.yaml",
    model_dir: str | Path = "models/3T_Final",
    checkpoint: str = "model_last.pt",
    output_path: str | Path = "results/HLSVD_L2_Comparison/3T_validation_clean_metabolite.h5",
    n_spectra_per_subject: int = 30_000,
    simulation_batch_size: int = 512,
    hsvd_batch_size: int = 128,
    seed: int = 20260915,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> Path:
    """Run or resume the paired comparison and return its HDF5 path."""
    root = Path(project_root).expanduser().resolve()
    training_path = _resolve(training_config, root)
    water_path = _resolve(water_config, root)
    model_path = _resolve(model_dir, root)
    result_path = _resolve(output_path, root)
    if n_spectra_per_subject <= 0 or simulation_batch_size <= 0:
        raise ValueError("Spectrum count and simulation batch size must be positive")
    if hsvd_batch_size <= 0:
        raise ValueError("HSVD batch size must be positive")
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("This evaluation requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    torch.cuda.set_device(torch_device)

    train_cfg = build_config(_yaml(training_path), config_dir=training_path.parent)
    simulation_path = Path(train_cfg.data.simulation_config).resolve()
    simulation_cfg = build_simulation_config(
        _yaml(simulation_path), config_dir=simulation_path.parent
    )
    water_cfg = build_water_lipid_extraction_config(
        _yaml(water_path), config_dir=water_path.parent
    )
    if simulation_cfg.acquisition.n_timepoints != 288:
        raise ValueError(
            f"Expected fixed 288-point 3T simulation, got "
            f"{simulation_cfg.acquisition.n_timepoints}"
        )
    if abs(simulation_cfg.acquisition.bandwidth_hz - water_cfg.water_extraction.bandwidth) > 1e-6:
        raise ValueError("Simulation and HLSVD bandwidths differ")

    subjects = tuple(train_cfg.data.val_subjects)
    if not subjects:
        raise ValueError("Training config contains no validation subjects")
    run_signature = _signature(
        training_config=training_path,
        water_config=water_path,
        model_dir=model_path,
        checkpoint=checkpoint,
        subjects=subjects,
        n_spectra_per_subject=n_spectra_per_subject,
        simulation_batch_size=simulation_batch_size,
        hsvd_batch_size=hsvd_batch_size,
        seed=seed,
        device=device,
        metric_definition="clean_metabolite_v2",
    )
    if overwrite and result_path.exists():
        result_path.unlink()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, architecture, checkpoint_path = _load_model(
        model_path, checkpoint, torch_device
    )
    basis = prepare_basis_for_acquisition(
        simulation_cfg.basis.library,
        target_bandwidth=simulation_cfg.acquisition.bandwidth_hz,
        target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
        dataset_name="clean_fid",
    )
    metabolite_simulator = MetaboliteSimulator(
        prepared_basis=basis, config=simulation_cfg, device=torch_device
    )
    resource_filename = train_cfg.data.resources.filename.format(
        version=train_cfg.data.resources.version
    )
    hsvd_kwargs = dict(
        sampling_frequency_hz=water_cfg.water_extraction.bandwidth,
        n_components=water_cfg.water_extraction.hsvd_components,
        min_frequency_hz=water_cfg.water_extraction.min_freq,
        max_frequency_hz=water_cfg.water_extraction.max_freq,
        device=torch_device,
        batch_size=hsvd_batch_size,
        cuda_dtype=torch.complex64,
    )

    mode = "a" if result_path.exists() else "w"
    with h5py.File(result_path, mode) as h5:
        if mode == "w":
            h5.attrs["format"] = "walinet.hlsvd_l2_comparison.v2"
            h5.attrs["signature"] = run_signature
            h5.attrs["checkpoint"] = str(checkpoint_path)
            h5.attrs["architecture"] = architecture
            h5.attrs["n_subjects"] = len(subjects)
            h5.attrs["n_spectra_per_subject"] = n_spectra_per_subject
        elif str(h5.attrs.get("signature", "")) != run_signature:
            raise RuntimeError(
                "Existing result was created with different settings. Change "
                "output_path or pass overwrite=True."
            )

        for subject_index, subject in enumerate(subjects):
            group = _prepare_subject_group(
                h5,
                index=subject_index,
                subject=subject,
                n_spectra=n_spectra_per_subject,
                chunk_size=simulation_batch_size,
            )
            if bool(group.attrs.get("complete", False)):
                print(f"[Skip {subject_index + 1}/{len(subjects)}] {subject}", flush=True)
                continue
            written = int(group.attrs["written"])
            print(
                f"[Subject {subject_index + 1}/{len(subjects)}] {subject} "
                f"(resume at {written}/{n_spectra_per_subject})",
                flush=True,
            )
            pool = load_simulation_pool(
                base_dir=train_cfg.data.base_dir,
                subjects=[subject],
                resource_filename=resource_filename,
                target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
                expected_bandwidth_hz=simulation_cfg.acquisition.bandwidth_hz,
                load_projection_operator=True,
            ).to(torch_device)
            simulator = SpectrumSimulator(
                pool=pool,
                metabolite_simulator=metabolite_simulator,
                config=simulation_cfg,
            )
            if pool.lipid_projection_operators is None:
                raise RuntimeError(f"{subject}: no stored L2 operator was loaded")
            if tuple(pool.lipid_projection_operators.shape) != (1, 288, 288):
                raise RuntimeError(
                    f"{subject}: unexpected operator shape "
                    f"{tuple(pool.lipid_projection_operators.shape)}"
                )
            lipid_operator = pool.lipid_projection_operators[0]
            generator = _restore_generator(
                group, device=torch_device, seed=seed + subject_index
            )
            while written < n_spectra_per_subject:
                current = min(
                    simulation_batch_size, n_spectra_per_subject - written
                )
                batch = simulator.simulate(batch_size=current, generator=generator)

                with torch.inference_mode():
                    walrus_prediction = forward_model(
                        model=model,
                        network_input=batch.network_input,
                        network_l2=batch.network_l2,
                        architecture=architecture,
                    )[:, :2, :]
                    walrus_errors = calculate_errors(
                        true_nuisance=batch.network_target,
                        predicted_nuisance=walrus_prediction,
                        clean_metabolite=(
                            batch.normalized_clean_metabolite_spectra
                        ),
                    )

                    full_fids = torch.fft.ifft(
                        torch.fft.ifftshift(batch.raw_input_spectra, dim=-1),
                        dim=-1,
                    ).cpu().numpy()
                    water_fids = estimate_water_hsvd(full_fids, **hsvd_kwargs)
                    water_spectra = torch.fft.fftshift(
                        torch.fft.fft(
                            torch.as_tensor(water_fids, device=torch_device), dim=-1
                        ),
                        dim=-1,
                    )
                    # Apply the stored subject-specific operator directly to
                    # the FULL noisy spectrum, independent of the simulator's
                    # optional legacy YNet-L2-input switch.
                    l2_spectra = batch.raw_input_spectra @ lipid_operator
                    baseline_nuisance = (
                        water_spectra + l2_spectra
                    ) / batch.normalization_scale
                    baseline_prediction = complex_spectra_to_channels(
                        baseline_nuisance
                    )
                    baseline_errors = calculate_errors(
                        true_nuisance=batch.network_target,
                        predicted_nuisance=baseline_prediction,
                        clean_metabolite=(
                            batch.normalized_clean_metabolite_spectra
                        ),
                    )

                stop = written + current
                for method, errors in zip(METHODS, (baseline_errors, walrus_errors)):
                    for metric, values in zip(METRICS, errors):
                        group[f"methods/{method}/{metric}"][written:stop] = (
                            values.detach().cpu().numpy().astype(np.float32)
                        )
                written = stop
                group.attrs.modify("written", written)
                _save_generator_state(group, generator)
                h5.flush()
                print(
                    f"  {subject}: {written}/{n_spectra_per_subject}", flush=True
                )
                del batch, full_fids, water_fids, water_spectra, l2_spectra

            group.attrs.modify("complete", True)
            h5.flush()
        _write_aggregates(h5)
        h5.attrs["complete"] = all(
            bool(group.attrs.get("complete", False))
            for group in h5["subjects"].values()
        )
        h5.flush()
    return result_path
