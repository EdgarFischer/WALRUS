#!/usr/bin/env python3
"""Compare WALINET-style and WALRUS simulator parameters with 5 UCSF subjects.

The script extracts one multivariate feature vector per valid LCModel voxel,
simulates the same parameterization from both simulator configurations, applies
one robust in-vivo scaler, and computes mean 5-nearest-neighbour distances.

No dense query-by-reference distance matrix is formed. ``scipy.cKDTree`` is
queried in parallel and in bounded chunks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from nibabel.processing import resample_from_to  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.config.build_simulation import build_simulation_config  # noqa: E402
from walinet.training_data.distributions import sample_distribution  # noqa: E402
from walinet.training_data.lcmodel_basis.acquisition import (  # noqa: E402
    prepare_basis_for_acquisition,
)
from walinet.training_data.metabolite_simulation import (  # noqa: E402
    MetaboliteSimulator,
)


EXTRA_DATA_ROOT = (
    Path("/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home")
    / "bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy"
    / "LargeData_d3hj/Results_Dicom_LCM/UCSF"
)
METABOLITE_DATA_ROOT = (
    PROJECT_ROOT.parent
    / "Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/UCSF"
)
SUBJECTS = tuple(f"Vol{index}" for index in range(1, 6))
# Both requested profile files contain these 16 enabled basis components.
PROFILE_METABOLITES = (
    "Asp", "Cr", "GABA", "Glc", "Gln", "Glu", "GPC", "GSH", "Ins",
    "NAA", "NAAG", "PCh", "PCr", "Scyllo", "Tau", "TwoHG",
)
# PCh is excluded a priori. TwoHG is taken from the primary WALRUS LCModel fits;
# the complete phase maps are taken from the historical LCModel fits.
HEALTHY_METABOLITES = (
    "Asp",
    "Cr",
    "GABA",
    "Glc",
    "Gln",
    "Glu",
    "GPC",
    "GSH",
    "Ins",
    "NAA",
    "NAAG",
    "PCr",
    "Scyllo",
    "Tau",
    "TwoHG",
)
FEATURE_NAMES = (
    *HEALTHY_METABOLITES,
    "SNR",
    "shift_ppm",
    "phase0_deg",
    "phase1_deg_per_ppm",
)
WALRUS_PROFILE = PROJECT_ROOT / "configs/Simulation/Metabos_healthy_7T_2sigma.yaml"
WALINET_PROFILE = (
    PROJECT_ROOT / "configs/Simulation/Metabos_healthy_7T_WALINET_Style.yaml"
)
DEFAULT_WALRUS_CONFIG = (
    PROJECT_ROOT
    / "configs/Simulation/7T_UNet_32_6_fixed_trainingset_snapshot.yaml"
)
DEFAULT_WALINET_CONFIG = (
    PROJECT_ROOT / "configs/Simulation/7T_on_the_fly_WALINET_STYLE.yaml"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/SimulatorRealism7T"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metabolite-root", type=Path, default=METABOLITE_DATA_ROOT)
    parser.add_argument("--extra-root", type=Path, default=EXTRA_DATA_ROOT)
    parser.add_argument("--walrus-config", type=Path, default=DEFAULT_WALRUS_CONFIG)
    parser.add_argument("--walinet-config", type=Path, default=DEFAULT_WALINET_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-simulations", type=int, default=100_000)
    parser.add_argument("--simulation-batch-size", type=int, default=4096)
    parser.add_argument("--query-chunk-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse cached in_vivo_features.npz and simulated_features.npz.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return raw


def _load_simulation_config(path: Path, *, profile_override: Path | None = None):
    path = path.expanduser().resolve()
    raw = _load_yaml(path)
    if profile_override is not None:
        profile_override = profile_override.expanduser().resolve()
        raw["metabolites"]["profiles"] = [
            {"config": str(profile_override), "probability": 1.0}
        ]
    return build_simulation_config(raw, config_dir=path.parent)


def _name_key(name: str) -> str:
    name = re.sub(r"^\d+[_-]+", "", str(name))
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _prepare_basis(simulation_cfg):
    return prepare_basis_for_acquisition(
        simulation_cfg.basis.library,
        target_bandwidth=simulation_cfg.acquisition.bandwidth_hz,
        target_n_timepoints=simulation_cfg.acquisition.n_timepoints,
        dataset_name="clean_fid",
    )


def _enabled_profile_components(path: Path) -> tuple[str, ...]:
    raw = _load_yaml(path)
    components = []
    for entry in raw.get("metabolites", {}).values():
        if entry.get("enabled", False):
            components.append(str(entry["basis_component"]))
    return tuple(components)


def validate_profiles() -> None:
    expected = set(PROFILE_METABOLITES)
    for path in (WALRUS_PROFILE, WALINET_PROFILE):
        actual = set(_enabled_profile_components(path))
        if actual != expected:
            raise ValueError(
                f"Unexpected enabled metabolites in {path}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    print(
        "Validated both healthy profiles: 16 enabled; excluding only PCh a priori.",
        flush=True,
    )


def _feature_matrix(
    metabolite_values: np.ndarray,
    snr: np.ndarray,
    shift_ppm: np.ndarray,
    phase0_deg: np.ndarray,
    phase1_deg_per_ppm: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack(
        (
            metabolite_values,
            snr,
            shift_ppm,
            phase0_deg,
            phase1_deg_per_ppm,
        )
    ).astype(np.float32, copy=False)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature count mismatch: {matrix.shape[1]} vs {len(FEATURE_NAMES)}"
        )
    return np.ascontiguousarray(matrix)


def extract_subject_features(
    subject: str,
    *,
    metabolite_root: Path,
    extra_root: Path,
    basis,
    batch_size: int = 4096,
) -> np.ndarray:
    subject_number = int(subject.removeprefix("Vol"))
    primary_dir = metabolite_root / subject
    maps_path = primary_dir / "LCModelB4Walrus/metabolite_maps.npz"
    mask_path = primary_dir / "masks/brain_mask.npy"
    extra_maps_dir = extra_root / f"Vol{subject_number:02d}_DicomNew/maps/Extra"
    if not maps_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Missing primary maps or mask for {subject}")

    def load_map(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(path)
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)

    mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    with np.load(maps_path, allow_pickle=False) as archive:
        concentrations = np.asarray(archive["concentrations"], dtype=np.float32)
        metabolite_names = tuple(str(x) for x in archive["metabolite_names"])
    if concentrations.shape[:3] != mask.shape:
        raise ValueError(f"{subject}: incompatible primary maps/mask shapes")
    map_indices = {_name_key(name): index for index, name in enumerate(metabolite_names)}

    basis_names = tuple(str(name) for name in basis.names)
    basis_fids = np.asarray(basis.fids, dtype=np.complex64)
    coefficients = np.zeros((*mask.shape, len(basis_names)), dtype=np.float32)
    for basis_index, name in enumerate(basis_names):
        map_index = map_indices.get(_name_key(name))
        if map_index is not None:
            coefficients[..., basis_index] = concentrations[..., map_index]

    selected_basis_indices = []
    for name in HEALTHY_METABOLITES:
        matches = [i for i, basis_name in enumerate(basis_names) if _name_key(basis_name) == _name_key(name)]
        if len(matches) != 1:
            raise KeyError(f"{subject}: expected one basis match for {name}, got {matches}")
        selected_basis_indices.append(matches[0])

    def old_to_primary(array: np.ndarray) -> np.ndarray:
        # Verified for all five subjects against their magnitude images:
        # Pearson r=1.0 and brain-mask Dice=1.0.
        return np.flip(array.swapaxes(0, 1), axis=(0, 1)).copy()

    # Reproduce the orientation check from the two independently stored
    # magnitude images and masks before combining maps from the two sources.
    old_maps_dir = extra_maps_dir.parent
    old_mask_image = nib.load(str(old_maps_dir / "mask.nii.gz"))
    transformed_old_mask = old_to_primary(
        np.asarray(old_mask_image.dataobj, dtype=np.float32)
    ) > 0
    dice = (
        2.0 * np.count_nonzero(mask & transformed_old_mask)
        / (np.count_nonzero(mask) + np.count_nonzero(transformed_old_mask))
    )
    old_magnitude_on_fit_grid = resample_from_to(
        nib.load(str(old_maps_dir / "magnitude.nii.gz")),
        old_mask_image,
        order=1,
    ).get_fdata(dtype=np.float32)
    transformed_old_magnitude = old_to_primary(old_magnitude_on_fit_grid)
    primary_magnitude = np.asarray(
        np.load(primary_dir / "OriginalData/magnitude.npy", allow_pickle=False),
        dtype=np.float32,
    )
    correlation = float(
        np.corrcoef(
            primary_magnitude[mask], transformed_old_magnitude[mask]
        )[0, 1]
    )
    if not np.isfinite(correlation) or correlation < 0.999999 or dice < 0.999999:
        raise RuntimeError(
            f"{subject}: failed cross-source orientation validation: "
            f"magnitude r={correlation:.9f}, mask Dice={dice:.9f}"
        )
    print(
        f"{subject}: orientation verified (magnitude r={correlation:.9f}, "
        f"mask Dice={dice:.9f})",
        flush=True,
    )

    fwhm_ppm = old_to_primary(load_map(extra_maps_dir / "FWHM_map.nii.gz"))
    snr_map = old_to_primary(load_map(extra_maps_dir / "SNR_map.nii.gz"))
    shift_map = old_to_primary(load_map(extra_maps_dir / "shift_map.nii.gz"))
    phase0_map = old_to_primary(load_map(extra_maps_dir / "0_pha_map.nii.gz"))
    phase1_map = old_to_primary(load_map(extra_maps_dir / "1_pha_map.nii.gz"))
    for name, current_map in (
        ("FWHM", fwhm_ppm), ("SNR", snr_map), ("shift", shift_map),
        ("0_pha", phase0_map), ("1_pha", phase1_map),
    ):
        if current_map.shape != mask.shape:
            raise ValueError(f"{subject}: incompatible {name} shape {current_map.shape}")

    initial_valid = (
        mask
        & np.all(np.isfinite(coefficients), axis=-1)
        & np.isfinite(fwhm_ppm)
        & (fwhm_ppm >= 0)
        & np.isfinite(snr_map)
        & np.isfinite(shift_map)
        & np.isfinite(phase0_map)
        & np.isfinite(phase1_map)
        & np.any(coefficients != 0, axis=-1)
    )
    flat_indices = np.flatnonzero(initial_valid.reshape(-1))
    flat_coefficients = coefficients.reshape(-1, len(basis_names))
    flat_fwhm_hz = fwhm_ppm.reshape(-1) * float(basis.hz_per_ppm)
    time_axis = np.arange(basis.n_timepoints, dtype=np.float32) * float(basis.dwell_time)
    normalized = np.full((len(flat_indices), len(HEALTHY_METABOLITES)), np.nan, dtype=np.float32)

    for start in range(0, len(flat_indices), batch_size):
        stop = min(start + batch_size, len(flat_indices))
        voxel_indices = flat_indices[start:stop]
        current = flat_coefficients[voxel_indices]
        combined_fids = current @ basis_fids
        decay = np.exp(
            -np.pi * flat_fwhm_hz[voxel_indices, None] * time_axis[None, :]
        ).astype(np.float32, copy=False)
        spectra = np.fft.fftshift(np.fft.fft(combined_fids * decay, axis=-1), axes=-1)
        maximum = np.max(np.abs(spectra), axis=-1)
        valid = np.isfinite(maximum) & (maximum > 0)
        normalized[start:stop][valid] = (
            current[valid][:, selected_basis_indices] / maximum[valid, None]
        ).astype(np.float32, copy=False)

    flat_snr = snr_map.reshape(-1)[flat_indices]
    flat_shift = shift_map.reshape(-1)[flat_indices]
    flat_phase0 = phase0_map.reshape(-1)[flat_indices]
    flat_phase1 = phase1_map.reshape(-1)[flat_indices]
    features = _feature_matrix(
        normalized,
        flat_snr,
        flat_shift,
        flat_phase0,
        flat_phase1,
    )
    valid_rows = np.all(np.isfinite(features), axis=1)
    features = features[valid_rows]
    if features.size == 0:
        raise RuntimeError(f"{subject}: no valid feature rows")
    print(f"{subject}: {len(features):,} valid LCModel voxels", flush=True)
    return features


def extract_all_in_vivo_features(
    *, metabolite_root: Path, extra_root: Path, walrus_config: Path
) -> dict[str, np.ndarray]:
    simulation_cfg = _load_simulation_config(walrus_config)
    basis = _prepare_basis(simulation_cfg)
    return {
        subject: extract_subject_features(
            subject,
            metabolite_root=metabolite_root,
            extra_root=extra_root,
            basis=basis,
        )
        for subject in SUBJECTS
    }


def simulate_features(
    config_path: Path,
    *,
    n_spectra: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    profile_path: Path,
) -> np.ndarray:
    config = _load_simulation_config(config_path, profile_override=profile_path)
    basis = _prepare_basis(config)
    simulator = MetaboliteSimulator(
        prepared_basis=basis,
        config=config,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    basis_names = tuple(str(name) for name in basis.names)
    selected_indices = []
    for name in HEALTHY_METABOLITES:
        matches = [i for i, basis_name in enumerate(basis_names) if _name_key(basis_name) == _name_key(name)]
        if len(matches) != 1:
            raise KeyError(f"Expected one basis match for {name}, got {matches}")
        selected_indices.append(matches[0])

    features = np.empty((n_spectra, len(FEATURE_NAMES)), dtype=np.float32)
    hz_per_ppm = float(basis.hz_per_ppm)
    for start in range(0, n_spectra, batch_size):
        stop = min(start + batch_size, n_spectra)
        metabolites = simulator.simulate(batch_size=stop - start, generator=generator)
        snr = sample_distribution(
            distribution=config.noise.snr.distribution,
            shape=(stop - start,),
            device=device,
            dtype=metabolites.clean_spectra.real.dtype,
            generator=generator,
        )
        maximum = torch.amax(torch.abs(metabolites.clean_spectra), dim=-1)
        normalized = metabolites.concentrations[:, selected_indices] / maximum[:, None]
        batch_features = _feature_matrix(
            normalized.detach().cpu().numpy(),
            snr.detach().cpu().numpy(),
            (metabolites.frequency_shifts_hz / hz_per_ppm).detach().cpu().numpy(),
            torch.rad2deg(metabolites.zero_order_phases_radians).detach().cpu().numpy(),
            (
                torch.rad2deg(metabolites.first_order_phases_rad_per_hz)
                * hz_per_ppm
            ).detach().cpu().numpy(),
        )
        features[start:stop] = batch_features
        print(f"  simulated {stop:,}/{n_spectra:,}", flush=True)

    if not np.isfinite(features).all():
        raise RuntimeError(f"Non-finite simulated features from {config_path}")
    return features


def robust_scaler(
    in_vivo: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.concatenate([in_vivo[subject] for subject in SUBJECTS], axis=0)
    median = np.median(pooled, axis=0)
    q25, q75 = np.percentile(pooled, [25.0, 75.0], axis=0)
    iqr = q75 - q25
    scale = iqr / 1.349
    invalid = ~np.isfinite(scale) | (scale <= np.finfo(np.float32).eps)
    if np.any(invalid):
        bad = [FEATURE_NAMES[i] for i in np.flatnonzero(invalid)]
        print(
            "Excluding features with zero/invalid in-vivo IQR from kNN: "
            f"{bad}",
            flush=True,
        )
    active = ~invalid
    if not np.any(active):
        raise ValueError("No feature has a valid robust scale")
    return (
        median.astype(np.float32),
        iqr.astype(np.float32),
        scale.astype(np.float32),
        active,
    )


def standardize(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (values - median[None, :]) / scale[None, :]
    return np.ascontiguousarray(result, dtype=np.float32)


def query_mean_knn(
    tree: cKDTree,
    queries: np.ndarray,
    *,
    k: int,
    workers: int,
    chunk_size: int,
) -> np.ndarray:
    output = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        distances, _ = tree.query(queries[start:stop], k=k, workers=workers)
        output[start:stop] = np.mean(distances, axis=1, dtype=np.float64)
    return output


def evaluate_distances(
    in_vivo: dict[str, np.ndarray],
    walinet: np.ndarray,
    walrus: np.ndarray,
    *,
    workers: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    median, iqr, scale, active = robust_scaler(in_vivo)
    standardized_real = {
        subject: standardize(
            in_vivo[subject][:, active], median[active], scale[active]
        )
        for subject in SUBJECTS
    }
    standardized_walinet = standardize(
        walinet[:, active], median[active], scale[active]
    )
    standardized_walrus = standardize(
        walrus[:, active], median[active], scale[active]
    )

    real_distances = []
    real_subject_index = []
    for subject_index, held_out in enumerate(SUBJECTS):
        reference = np.concatenate(
            [standardized_real[s] for s in SUBJECTS if s != held_out], axis=0
        )
        print(
            f"kNN fold {held_out}: reference={len(reference):,}, "
            f"held-out={len(standardized_real[held_out]):,}",
            flush=True,
        )
        tree = cKDTree(reference, compact_nodes=True, balanced_tree=True)
        current_real = query_mean_knn(
            tree,
            standardized_real[held_out],
            k=5,
            workers=workers,
            chunk_size=chunk_size,
        )
        real_distances.append(current_real)
        real_subject_index.append(
            np.full(len(current_real), subject_index, dtype=np.int8)
        )
    pooled_reference = np.concatenate(
        [standardized_real[subject] for subject in SUBJECTS], axis=0
    )
    print(
        f"Simulation kNN: pooled five-subject reference={len(pooled_reference):,}",
        flush=True,
    )
    pooled_tree = cKDTree(
        pooled_reference,
        compact_nodes=True,
        balanced_tree=True,
    )
    walinet_distances = query_mean_knn(
        pooled_tree,
        standardized_walinet,
        k=5,
        workers=workers,
        chunk_size=chunk_size,
    )
    walrus_distances = query_mean_knn(
        pooled_tree,
        standardized_walrus,
        k=5,
        workers=workers,
        chunk_size=chunk_size,
    )
    return {
        "feature_names": np.asarray(FEATURE_NAMES),
        "active_feature_mask": active,
        "active_feature_names": np.asarray(FEATURE_NAMES)[active],
        "excluded_zero_iqr_feature_names": np.asarray(FEATURE_NAMES)[~active],
        "subjects": np.asarray(SUBJECTS),
        "median": median,
        "iqr": iqr,
        "robust_std": scale,
        "in_vivo": np.concatenate(real_distances),
        "in_vivo_subject_index": np.concatenate(real_subject_index),
        "walinet": walinet_distances,
        "walrus": walrus_distances,
    }


def save_histogram(results: dict[str, np.ndarray], output_path: Path) -> None:
    groups = {
        "In vivo (LOSO)": results["in_vivo"],
        "WALINET style": results["walinet"],
        "WALRUS": results["walrus"],
    }
    pooled = np.concatenate(list(groups.values()))
    upper = float(np.percentile(pooled, 99.5))
    bins = np.linspace(0.0, upper, 101)
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for label, values in groups.items():
        axis.hist(
            np.clip(values, 0.0, upper),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=f"{label} (median={np.median(values):.3f})",
        )
    axis.set_xlabel("Mean Euclidean distance to 5 nearest in-vivo neighbours")
    axis.set_ylabel("Density")
    axis.set_title("7 T simulator realism — five UCSF subjects")
    axis.legend(frameon=False)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def summarize(results: dict[str, np.ndarray]) -> dict:
    summary = {}
    baseline = float(np.median(results["in_vivo"]))
    for key in ("in_vivo", "walinet", "walrus"):
        values = np.asarray(results[key])
        summary[key] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "median_over_in_vivo": float(np.median(values) / baseline),
        }
    return summary


def main() -> None:
    args = parse_args()
    validate_profiles()
    if args.n_simulations <= 0 or args.simulation_batch_size <= 0:
        raise ValueError("Simulation sizes must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    in_vivo_path = output_dir / "in_vivo_features.npz"
    simulated_path = output_dir / "simulated_features.npz"

    if args.reuse_features:
        with np.load(in_vivo_path, allow_pickle=False) as archive:
            in_vivo = {subject: archive[subject] for subject in SUBJECTS}
        with np.load(simulated_path, allow_pickle=False) as archive:
            walinet = archive["walinet"]
            walrus = archive["walrus"]
        print("Reused cached feature matrices.", flush=True)
    else:
        print("Extracting in-vivo LCModel features...", flush=True)
        in_vivo = extract_all_in_vivo_features(
            metabolite_root=args.metabolite_root.expanduser().resolve(),
            extra_root=args.extra_root.expanduser().resolve(),
            walrus_config=args.walrus_config,
        )
        np.savez_compressed(
            in_vivo_path,
            feature_names=np.asarray(FEATURE_NAMES),
            **in_vivo,
        )

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        print("Simulating WALINET-style features...", flush=True)
        walinet = simulate_features(
            args.walinet_config,
            n_spectra=args.n_simulations,
            batch_size=args.simulation_batch_size,
            seed=args.seed,
            device=device,
            profile_path=WALINET_PROFILE,
        )
        print("Simulating WALRUS features...", flush=True)
        walrus = simulate_features(
            args.walrus_config,
            n_spectra=args.n_simulations,
            batch_size=args.simulation_batch_size,
            seed=args.seed + 1,
            device=device,
            profile_path=WALRUS_PROFILE,
        )
        np.savez_compressed(
            simulated_path,
            feature_names=np.asarray(FEATURE_NAMES),
            walinet=walinet,
            walrus=walrus,
            walinet_seed=np.asarray(args.seed),
            walrus_seed=np.asarray(args.seed + 1),
        )

    print("Computing 5-NN distances...", flush=True)
    results = evaluate_distances(
        in_vivo,
        walinet,
        walrus,
        workers=args.workers,
        chunk_size=args.query_chunk_size,
    )
    np.savez_compressed(output_dir / "knn_distances.npz", **results)
    summary = summarize(results)
    metadata = {
        "metabolite_root": str(args.metabolite_root.expanduser().resolve()),
        "extra_root": str(args.extra_root.expanduser().resolve()),
        "subjects": list(SUBJECTS),
        "features": list(FEATURE_NAMES),
        "excluded": [
            "water/metabolite",
            "lipid/metabolite",
            "FWHM",
            "PCh (a-priori exclusion requested for the paper)",
        ],
        "spatial_alignment": (
            "historical Extra maps transformed to primary grid by swap(x,y), "
            "flip(x), flip(y); verified per subject with magnitude Pearson r=1.0 "
            "and brain-mask Dice=1.0"
        ),
        "simulation_profiles": {
            "walinet": str(WALINET_PROFILE.resolve()),
            "walrus": str(WALRUS_PROFILE.resolve()),
            "probability": 1.0,
        },
        "profile_metabolites": list(PROFILE_METABOLITES),
        "k": 5,
        "scaler": "pooled in-vivo median and IQR/1.349",
        "in_vivo_reference": "subject-wise leave-one-subject-out",
        "simulation_reference": "pooled voxels from all five in-vivo subjects",
        "walinet_config": str(args.walinet_config.expanduser().resolve()),
        "walrus_config": str(args.walrus_config.expanduser().resolve()),
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    save_histogram(results, output_dir / "knn_distance_histogram")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DONE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
