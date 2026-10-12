# src/walinet/config/build_simulation.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .schema_simulation import (
    AcquisitionCfg,
    BasisCfg,
    FirstOrderPhaseCfg,
    FrequencyShiftCfg,
    FWHMCfg,
    LipidCfg,
    LipidProjectionCfg,
    LogUniformDistributionCfg,
    LogNormalComponentCfg,
    MetaboliteCfg,
    MetaboliteProfileCfg,
    NoiseCfg,
    NormalComponentCfg,
    PositiveMixtureDistributionCfg,
    PositiveScalingDistributionCfg,
    SimulationConfig,
    SNRDistributionCfg,
    SubjectSamplingCfg,
    SymmetricMixtureDistributionCfg,
    UniformDistributionCfg,
    WaterCfg,
    ZeroOrderPhaseCfg,
)


# -----------------------------------------------------------------------------
# Generic configuration readers
# -----------------------------------------------------------------------------


def _require_mapping(
    mapping: dict[str, Any],
    key: str,
    *,
    parameter_path: str,
) -> dict[str, Any]:
    """Read one required nested YAML mapping."""
    if key not in mapping:
        raise KeyError(
            "Missing required configuration section: "
            f"{parameter_path}.{key}"
        )

    value = mapping[key]

    if not isinstance(value, dict):
        raise TypeError(
            f"{parameter_path}.{key} must be a mapping, "
            f"but found {type(value).__name__}."
        )

    return value


def _require_float(
    mapping: dict[str, Any],
    key: str,
    *,
    parameter_path: str,
) -> float:
    """Read one required finite float from a YAML mapping."""
    if key not in mapping:
        raise KeyError(
            "Missing required configuration value: "
            f"{parameter_path}.{key}"
        )

    value = mapping[key]

    if value is None:
        raise ValueError(
            f"{parameter_path}.{key} must not be null."
        )

    try:
        value_float = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{parameter_path}.{key} must be numeric, "
            f"but found {value!r}."
        ) from error

    if not math.isfinite(value_float):
        raise ValueError(
            f"{parameter_path}.{key} must be finite."
        )

    return value_float


def _require_int(
    mapping: dict[str, Any],
    key: str,
    *,
    parameter_path: str,
) -> int:
    """Read one required integer without silently truncating floats."""
    if key not in mapping:
        raise KeyError(
            "Missing required configuration value: "
            f"{parameter_path}.{key}"
        )

    value = mapping[key]

    if value is None:
        raise ValueError(
            f"{parameter_path}.{key} must not be null."
        )

    if isinstance(value, bool):
        raise TypeError(
            f"{parameter_path}.{key} must be an integer, not bool."
        )

    try:
        value_float = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{parameter_path}.{key} must be an integer, "
            f"but found {value!r}."
        ) from error

    if (
        not math.isfinite(value_float)
        or not value_float.is_integer()
    ):
        raise ValueError(
            f"{parameter_path}.{key} must be an integer, "
            f"but found {value!r}."
        )

    return int(value_float)




def _read_optional_bool(
    mapping: dict[str, Any],
    key: str,
    *,
    default: bool,
    parameter_path: str,
) -> bool:
    """Read an optional YAML boolean without coercing strings."""
    if key not in mapping:
        return default

    value = mapping[key]

    if not isinstance(value, bool):
        raise TypeError(
            f"{parameter_path}.{key} must be a boolean, "
            f"but found {value!r}."
        )

    return value

def _require_string(
    mapping: dict[str, Any],
    key: str,
    *,
    parameter_path: str,
) -> str:
    """Read one required non-empty string."""
    if key not in mapping:
        raise KeyError(
            "Missing required configuration value: "
            f"{parameter_path}.{key}"
        )

    value = mapping[key]

    if value is None:
        raise ValueError(
            f"{parameter_path}.{key} must not be null."
        )

    value_string = str(value).strip()

    if not value_string:
        raise ValueError(
            f"{parameter_path}.{key} must not be empty."
        )

    return value_string


def _read_distribution_type(
    distribution_raw: dict[str, Any],
    *,
    expected: str,
    parameter_path: str,
) -> str:
    """Read and validate ``distribution.type``."""
    distribution_type = _require_string(
        distribution_raw,
        "type",
        parameter_path=f"{parameter_path}.distribution",
    ).lower()

    if distribution_type != expected:
        raise ValueError(
            f"{parameter_path}.distribution.type must be "
            f"{expected!r}, but found {distribution_type!r}."
        )

    return distribution_type


def _resolve_path(
    path: str,
    config_dir: Path | None,
) -> str:
    """Resolve a path relative to the simulation YAML."""
    path = path.strip()

    if not path:
        return ""

    resolved_path = Path(path).expanduser()

    if (
        config_dir is not None
        and not resolved_path.is_absolute()
    ):
        resolved_path = config_dir / resolved_path

    return str(resolved_path.resolve())


# -----------------------------------------------------------------------------
# Unit conversions
# -----------------------------------------------------------------------------


def _calculate_hz_per_ppm(
    nmr_frequency_hz: float,
) -> float:
    if (
        not math.isfinite(nmr_frequency_hz)
        or nmr_frequency_hz <= 0
    ):
        raise ValueError(
            "acquisition.nmr_frequency_hz must be finite and > 0."
        )

    return nmr_frequency_hz / 1e6


def _scale_linear_value(
    value: float,
    *,
    scale: float,
    parameter_path: str,
) -> float:
    """Convert a linearly scaled quantity to the internal unit."""
    if (
        not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            f"Internal conversion scale for {parameter_path} "
            "must be finite and > 0."
        )

    return float(value) * scale


def _scale_lognormal_location(
    log_mu: float,
    *,
    scale: float,
    parameter_path: str,
) -> float:
    """
    Convert a lognormal location parameter under ``Y = scale * X``.

        log_mu_Y = log_mu_X + log(scale)
    """
    if (
        not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            f"Internal lognormal conversion scale for {parameter_path} "
            "must be finite and > 0."
        )

    return float(log_mu) + math.log(scale)


# -----------------------------------------------------------------------------
# Generic distribution builders
# -----------------------------------------------------------------------------


def _build_positive_mixture_distribution(
    raw: dict[str, Any],
    *,
    parameter_path: str,
    normal_mean_key: str,
    normal_std_key: str,
    minimum_key: str,
    linear_scale: float = 1.0,
) -> PositiveMixtureDistributionCfg:
    """
    Build a positive 50/50 truncated-normal/lognormal mixture.

    ``linear_scale`` converts numerical values from the YAML unit to the
    simulator's internal unit. The same scale is applied to the normal mean,
    normal standard deviation, lower bound, and lognormal location.
    ``log_sigma`` is unchanged.
    """
    distribution_raw = _require_mapping(
        raw,
        "distribution",
        parameter_path=parameter_path,
    )

    distribution_type = _read_distribution_type(
        distribution_raw,
        expected="positive_mixture",
        parameter_path=parameter_path,
    )

    normal_raw = _require_mapping(
        distribution_raw,
        "normal",
        parameter_path=f"{parameter_path}.distribution",
    )

    lognormal_raw = _require_mapping(
        distribution_raw,
        "lognormal",
        parameter_path=f"{parameter_path}.distribution",
    )

    normal_mean = _scale_linear_value(
        _require_float(
            normal_raw,
            normal_mean_key,
            parameter_path=f"{parameter_path}.distribution.normal",
        ),
        scale=linear_scale,
        parameter_path=parameter_path,
    )

    normal_std = _scale_linear_value(
        _require_float(
            normal_raw,
            normal_std_key,
            parameter_path=f"{parameter_path}.distribution.normal",
        ),
        scale=linear_scale,
        parameter_path=parameter_path,
    )

    log_mu = _scale_lognormal_location(
        _require_float(
            lognormal_raw,
            "log_mu",
            parameter_path=f"{parameter_path}.distribution.lognormal",
        ),
        scale=linear_scale,
        parameter_path=parameter_path,
    )

    log_sigma = _require_float(
        lognormal_raw,
        "log_sigma",
        parameter_path=f"{parameter_path}.distribution.lognormal",
    )

    minimum = _scale_linear_value(
        _require_float(
            distribution_raw,
            minimum_key,
            parameter_path=f"{parameter_path}.distribution",
        ),
        scale=linear_scale,
        parameter_path=parameter_path,
    )

    return PositiveMixtureDistributionCfg(
        type=distribution_type,
        normal=NormalComponentCfg(
            mean=normal_mean,
            std=normal_std,
        ),
        lognormal=LogNormalComponentCfg(
            log_mu=log_mu,
            log_sigma=log_sigma,
        ),
        minimum=minimum,
    )


def _build_positive_scaling_distribution(
    raw: dict[str, Any],
    *,
    parameter_path: str,
):
    """Build either the calibrated mixture or optional log-uniform scaling."""
    distribution_raw = _require_mapping(
        raw,
        "distribution",
        parameter_path=parameter_path,
    )
    distribution_type = _require_string(
        distribution_raw,
        "type",
        parameter_path=f"{parameter_path}.distribution",
    ).lower()

    if distribution_type == "positive_mixture":
        return _build_positive_mixture_distribution(
            raw,
            parameter_path=parameter_path,
            normal_mean_key="mean",
            normal_std_key="std",
            minimum_key="minimum",
        )

    if distribution_type == "log_uniform":
        return LogUniformDistributionCfg(
            type=distribution_type,
            minimum=_require_float(
                distribution_raw,
                "minimum",
                parameter_path=f"{parameter_path}.distribution",
            ),
            maximum=_require_float(
                distribution_raw,
                "maximum",
                parameter_path=f"{parameter_path}.distribution",
            ),
        )

    raise ValueError(
        f"{parameter_path}.distribution.type must be 'positive_mixture' "
        f"or 'log_uniform', but found {distribution_type!r}."
    )


def _build_symmetric_mixture_distribution(
    raw: dict[str, Any],
    *,
    parameter_path: str,
    center_key: str,
    normal_std_key: str,
    linear_scale: float,
) -> SymmetricMixtureDistributionCfg:
    """
    Build a signed normal-core/symmetric-lognormal-tail mixture.

    ``linear_scale`` converts the center, normal standard deviation, and tail
    magnitudes from the YAML unit to the simulator's internal unit.
    """
    distribution_raw = _require_mapping(
        raw,
        "distribution",
        parameter_path=parameter_path,
    )

    distribution_type = _read_distribution_type(
        distribution_raw,
        expected="symmetric_mixture",
        parameter_path=parameter_path,
    )

    normal_raw = _require_mapping(
        distribution_raw,
        "normal",
        parameter_path=f"{parameter_path}.distribution",
    )

    tail_raw = _require_mapping(
        distribution_raw,
        "lognormal_tail",
        parameter_path=f"{parameter_path}.distribution",
    )

    center = _scale_linear_value(
        _require_float(
            distribution_raw,
            center_key,
            parameter_path=f"{parameter_path}.distribution",
        ),
        scale=linear_scale,
        parameter_path=parameter_path,
    )

    normal_std = _scale_linear_value(
        _require_float(
            normal_raw,
            normal_std_key,
            parameter_path=f"{parameter_path}.distribution.normal",
        ),
        scale=linear_scale,
        parameter_path=parameter_path,
    )

    tail_log_mu = _scale_lognormal_location(
        _require_float(
            tail_raw,
            "log_mu",
            parameter_path=(
                f"{parameter_path}.distribution.lognormal_tail"
            ),
        ),
        scale=linear_scale,
        parameter_path=parameter_path,
    )

    tail_log_sigma = _require_float(
        tail_raw,
        "log_sigma",
        parameter_path=(
            f"{parameter_path}.distribution.lognormal_tail"
        ),
    )

    return SymmetricMixtureDistributionCfg(
        type=distribution_type,
        normal=NormalComponentCfg(
            mean=center,
            std=normal_std,
        ),
        lognormal_tail=LogNormalComponentCfg(
            log_mu=tail_log_mu,
            log_sigma=tail_log_sigma,
        ),
    )


def _build_uniform_distribution(
    raw: dict[str, Any],
    *,
    parameter_path: str,
    minimum_key: str,
    maximum_key: str,
    linear_scale: float = 1.0,
) -> UniformDistributionCfg:
    """Build a continuous uniform distribution and convert its unit."""
    distribution_raw = _require_mapping(
        raw,
        "distribution",
        parameter_path=parameter_path,
    )
    distribution_type = _read_distribution_type(
        distribution_raw,
        expected="uniform",
        parameter_path=parameter_path,
    )
    return UniformDistributionCfg(
        type=distribution_type,
        minimum=_scale_linear_value(
            _require_float(
                distribution_raw,
                minimum_key,
                parameter_path=f"{parameter_path}.distribution",
            ),
            scale=linear_scale,
            parameter_path=parameter_path,
        ),
        maximum=_scale_linear_value(
            _require_float(
                distribution_raw,
                maximum_key,
                parameter_path=f"{parameter_path}.distribution",
            ),
            scale=linear_scale,
            parameter_path=parameter_path,
        ),
    )


# -----------------------------------------------------------------------------
# Cross-component validation
# -----------------------------------------------------------------------------


def validate_simulation_config(
    cfg: SimulationConfig,
) -> None:
    """Validate constraints involving more than one config section."""
    basis_library = Path(cfg.basis.library)

    if not basis_library.is_file():
        raise FileNotFoundError(
            "Prepared basis library not found:\n"
            f"  {basis_library}"
        )

    for profile in cfg.metabolites.profiles:
        profile_path = Path(profile.config)

        if not profile_path.is_file():
            raise FileNotFoundError(
                "Metabolite profile configuration not found:\n"
                f"  {profile_path}"
            )

    if (
        cfg.lipid_projection.enabled
        and cfg.subject_sampling.mixing != "same_subject"
    ):
        raise ValueError(
            "lipid_projection.enabled may currently only be used with "
            "subject_sampling.mixing='same_subject'."
        )


# -----------------------------------------------------------------------------
# Public builder
# -----------------------------------------------------------------------------


def build_simulation_config(
    raw: dict,
    config_dir: Path | None = None,
) -> SimulationConfig:
    """
    Build the typed runtime configuration from the nested simulation YAML.

    YAML units
    ----------
    frequency shift:
        ppm
    FWHM:
        ppm
    zero-order phase:
        degrees
    first-order phase:
        degrees per ppm

    Internal units
    --------------
    frequency shift:
        Hz
    FWHM:
        Hz
    zero-order phase:
        radians
    first-order phase:
        radians per Hz

    For every lognormal magnitude parameter transformed as ``Y = scale * X``:

        log_mu_Y = log_mu_X + log(scale)
        log_sigma_Y = log_sigma_X
    """
    if not isinstance(raw, dict):
        raise TypeError(
            "Simulation configuration must be a top-level mapping."
        )

    version = _require_string(
        raw,
        "version",
        parameter_path="simulation",
    )

    # Acquisition -------------------------------------------------------------
    acquisition_raw = _require_mapping(
        raw,
        "acquisition",
        parameter_path="simulation",
    )

    acquisition = AcquisitionCfg(
        bandwidth_hz=_require_float(
            acquisition_raw,
            "bandwidth_hz",
            parameter_path="acquisition",
        ),
        n_timepoints=_require_int(
            acquisition_raw,
            "n_timepoints",
            parameter_path="acquisition",
        ),
        min_acquired_n_timepoints=_require_int(
            acquisition_raw,
            "min_acquired_n_timepoints",
            parameter_path="acquisition",
        ),
        max_acquired_n_timepoints=_require_int(
            acquisition_raw,
            "max_acquired_n_timepoints",
            parameter_path="acquisition",
        ),
        zero_filling=_read_optional_bool(
            acquisition_raw,
            "zero_filling",
            default=True,
            parameter_path="acquisition",
        ),
        nmr_frequency_hz=_require_float(
            acquisition_raw,
            "nmr_frequency_hz",
            parameter_path="acquisition",
        ),
    )

    hz_per_ppm = _calculate_hz_per_ppm(
        acquisition.nmr_frequency_hz
    )

    # Basis -------------------------------------------------------------------
    basis_raw = _require_mapping(
        raw,
        "basis",
        parameter_path="simulation",
    )

    basis = BasisCfg(
        library=_resolve_path(
            _require_string(
                basis_raw,
                "library",
                parameter_path="basis",
            ),
            config_dir,
        )
    )

    # Metabolite profiles -----------------------------------------------------
    metabolites_raw = _require_mapping(
        raw,
        "metabolites",
        parameter_path="simulation",
    )

    profiles_raw = metabolites_raw.get("profiles")

    if not isinstance(profiles_raw, list):
        raise TypeError(
            "metabolites.profiles must be a list."
        )

    profiles: list[MetaboliteProfileCfg] = []

    for profile_index, profile_raw in enumerate(profiles_raw):
        parameter_path = f"metabolites.profiles[{profile_index}]"

        if not isinstance(profile_raw, dict):
            raise TypeError(
                f"{parameter_path} must be a mapping."
            )

        profiles.append(
            MetaboliteProfileCfg(
                config=_resolve_path(
                    _require_string(
                        profile_raw,
                        "config",
                        parameter_path=parameter_path,
                    ),
                    config_dir,
                ),
                probability=_require_float(
                    profile_raw,
                    "probability",
                    parameter_path=parameter_path,
                ),
            )
        )

    # Signed metabolite parameters -------------------------------------------
    frequency_shift_raw = _require_mapping(
        metabolites_raw,
        "frequency_shift",
        parameter_path="metabolites",
    )
    frequency_shift_type = _require_string(
        _require_mapping(
            frequency_shift_raw,
            "distribution",
            parameter_path="metabolites.frequency_shift",
        ),
        "type",
        parameter_path="metabolites.frequency_shift.distribution",
    ).lower()
    frequency_shift = FrequencyShiftCfg(
        distribution=(
            _build_uniform_distribution(
                frequency_shift_raw,
                parameter_path="metabolites.frequency_shift",
                minimum_key="minimum_ppm",
                maximum_key="maximum_ppm",
                linear_scale=hz_per_ppm,
            )
            if frequency_shift_type == "uniform"
            else _build_symmetric_mixture_distribution(
                frequency_shift_raw,
                parameter_path="metabolites.frequency_shift",
                center_key="center_ppm",
                normal_std_key="std_ppm",
                linear_scale=hz_per_ppm,
            )
        )
    )

    degrees_to_radians_scale = math.pi / 180.0

    zero_order_phase_raw = _require_mapping(
        metabolites_raw,
        "zero_order_phase",
        parameter_path="metabolites",
    )
    zero_order_phase_type = _require_string(
        _require_mapping(
            zero_order_phase_raw,
            "distribution",
            parameter_path="metabolites.zero_order_phase",
        ),
        "type",
        parameter_path="metabolites.zero_order_phase.distribution",
    ).lower()
    zero_order_phase = ZeroOrderPhaseCfg(
        distribution=(
            _build_uniform_distribution(
                zero_order_phase_raw,
                parameter_path="metabolites.zero_order_phase",
                minimum_key="minimum_rad",
                maximum_key="maximum_rad",
            )
            if zero_order_phase_type == "uniform"
            else _build_symmetric_mixture_distribution(
                zero_order_phase_raw,
                parameter_path="metabolites.zero_order_phase",
                center_key="center_deg",
                normal_std_key="std_deg",
                linear_scale=degrees_to_radians_scale,
            )
        )
    )

    degrees_per_ppm_to_radians_per_hz_scale = (
        degrees_to_radians_scale / hz_per_ppm
    )

    first_order_phase = FirstOrderPhaseCfg(
        distribution=_build_symmetric_mixture_distribution(
            _require_mapping(
                metabolites_raw,
                "first_order_phase",
                parameter_path="metabolites",
            ),
            parameter_path="metabolites.first_order_phase",
            center_key="center_deg_per_ppm",
            normal_std_key="std_deg_per_ppm",
            linear_scale=(
                degrees_per_ppm_to_radians_per_hz_scale
            ),
        )
    )

    # Positive metabolite FWHM ------------------------------------------------
    fwhm_raw = _require_mapping(
        metabolites_raw,
        "fwhm",
        parameter_path="metabolites",
    )
    fwhm_type = _require_string(
        _require_mapping(
            fwhm_raw,
            "distribution",
            parameter_path="metabolites.fwhm",
        ),
        "type",
        parameter_path="metabolites.fwhm.distribution",
    ).lower()
    fwhm = FWHMCfg(
        distribution=(
            _build_uniform_distribution(
                fwhm_raw,
                parameter_path="metabolites.fwhm",
                minimum_key="minimum_ppm",
                maximum_key="maximum_ppm",
                linear_scale=hz_per_ppm,
            )
            if fwhm_type == "uniform"
            else _build_positive_mixture_distribution(
                fwhm_raw,
                parameter_path="metabolites.fwhm",
                normal_mean_key="mean_ppm",
                normal_std_key="std_ppm",
                minimum_key="minimum_ppm",
                linear_scale=hz_per_ppm,
            )
        )
    )

    metabolites = MetaboliteCfg(
        profiles=tuple(profiles),
        frequency_shift=frequency_shift,
        fwhm=fwhm,
        zero_order_phase=zero_order_phase,
        first_order_phase=first_order_phase,
    )

    # Noise -------------------------------------------------------------------
    noise_raw = _require_mapping(
        raw,
        "noise",
        parameter_path="simulation",
    )

    snr = SNRDistributionCfg(
        distribution=_build_positive_mixture_distribution(
            _require_mapping(
                noise_raw,
                "snr",
                parameter_path="noise",
            ),
            parameter_path="noise.snr",
            normal_mean_key="mean",
            normal_std_key="std",
            minimum_key="min",
        )
    )

    noise = NoiseCfg(snr=snr)

    # Water -------------------------------------------------------------------
    water_raw = _require_mapping(
        raw,
        "water",
        parameter_path="simulation",
    )

    water = WaterCfg(
        scaling=PositiveScalingDistributionCfg(
            distribution=_build_positive_scaling_distribution(
                _require_mapping(
                    water_raw,
                    "scaling",
                    parameter_path="water",
                ),
                parameter_path="water.scaling",
            )
        )
    )

    # Lipids ------------------------------------------------------------------
    lipids_raw = _require_mapping(
        raw,
        "lipids",
        parameter_path="simulation",
    )

    lipids = LipidCfg(
        n_random_fids=_require_int(
            lipids_raw,
            "n_random_fids",
            parameter_path="lipids",
        ),
        scaling=PositiveScalingDistributionCfg(
            distribution=_build_positive_scaling_distribution(
                _require_mapping(
                    lipids_raw,
                    "scaling",
                    parameter_path="lipids",
                ),
                parameter_path="lipids.scaling",
            )
        ),
    )

    # Subject sampling --------------------------------------------------------
    subject_sampling_raw = raw.get("subject_sampling", {})

    if not isinstance(subject_sampling_raw, dict):
        raise TypeError(
            "subject_sampling must be a mapping."
        )

    subject_sampling = SubjectSamplingCfg(
        mixing=str(
            subject_sampling_raw.get(
                "mixing",
                "same_subject",
            )
        ).strip()
    )

    # Optional lipid projection ----------------------------------------------
    lipid_projection_raw = raw.get("lipid_projection", {})

    if not isinstance(lipid_projection_raw, dict):
        raise TypeError(
            "lipid_projection must be a mapping."
        )

    lipid_projection = LipidProjectionCfg(
        enabled=_read_optional_bool(
            lipid_projection_raw,
            "enabled",
            default=False,
            parameter_path="lipid_projection",
        )
    )

    cfg = SimulationConfig(
        version=version,
        acquisition=acquisition,
        basis=basis,
        metabolites=metabolites,
        noise=noise,
        water=water,
        lipids=lipids,
        subject_sampling=subject_sampling,
        lipid_projection=lipid_projection,
    )

    validate_simulation_config(cfg)

    return cfg
