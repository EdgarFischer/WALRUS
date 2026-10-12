# src/walinet/config/schema_simulation.py

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias


PositiveDistributionType = Literal[
    "positive_mixture",
    "log_uniform",
    "uniform",
]

SignedDistributionType = Literal[
    "symmetric_mixture",
    "uniform",
]


# -----------------------------------------------------------------------------
# Generic distribution configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalComponentCfg:
    """
    One normal-distribution component in the simulator's internal unit.

    ``mean`` and ``std`` may therefore represent Hz, radians, rad/Hz, or a
    dimensionless scaling factor, depending on the parameter using the
    component.
    """

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError(
                "NormalComponentCfg.mean must be finite."
            )

        if (
            not math.isfinite(self.std)
            or self.std < 0
        ):
            raise ValueError(
                "NormalComponentCfg.std must be finite and >= 0."
            )


@dataclass(frozen=True)
class LogNormalComponentCfg:
    """
    One lognormal-distribution component.

    Sampling is defined by

        exp(N(log_mu, log_sigma**2)).

    ``log_mu`` refers to the logarithm of numerical values expressed in the
    simulator's internal unit. ``log_sigma`` is dimensionless.
    """

    log_mu: float
    log_sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_mu):
            raise ValueError(
                "LogNormalComponentCfg.log_mu must be finite."
            )

        if (
            not math.isfinite(self.log_sigma)
            or self.log_sigma < 0
        ):
            raise ValueError(
                "LogNormalComponentCfg.log_sigma must be "
                "finite and >= 0."
            )


@dataclass(frozen=True)
class PositiveMixtureDistributionCfg:
    """
    Positive 50/50 mixture distribution.

    Model
    -----

        0.5 * truncated Normal
        +
        0.5 * LogNormal

    Both components use the same lower bound. Draws less than or equal to
    ``minimum`` are rejected and resampled. The fixed mixture weights are
    defined centrally by the sampling implementation and are deliberately not
    configurable per parameter.
    """

    normal: NormalComponentCfg
    lognormal: LogNormalComponentCfg
    minimum: float
    type: PositiveDistributionType = "positive_mixture"

    def __post_init__(self) -> None:
        if self.type != "positive_mixture":
            raise ValueError(
                "PositiveMixtureDistributionCfg.type must be "
                "'positive_mixture'."
            )

        if not math.isfinite(self.minimum):
            raise ValueError(
                "PositiveMixtureDistributionCfg.minimum must be finite."
            )

        if self.minimum < 0:
            raise ValueError(
                "PositiveMixtureDistributionCfg.minimum must be >= 0."
            )

        if (
            self.normal.std == 0
            and self.normal.mean <= self.minimum
        ):
            raise ValueError(
                "The truncated-normal component cannot be sampled because "
                "std == 0 and mean <= minimum."
            )


@dataclass(frozen=True)
class LogUniformDistributionCfg:
    """Positive distribution uniform in natural-log space."""

    minimum: float
    maximum: float
    type: PositiveDistributionType = "log_uniform"

    def __post_init__(self) -> None:
        if self.type != "log_uniform":
            raise ValueError(
                "LogUniformDistributionCfg.type must be 'log_uniform'."
            )
        if not math.isfinite(self.minimum) or self.minimum <= 0:
            raise ValueError(
                "LogUniformDistributionCfg.minimum must be finite and > 0."
            )
        if not math.isfinite(self.maximum) or self.maximum < self.minimum:
            raise ValueError(
                "LogUniformDistributionCfg.maximum must be finite and >= "
                "minimum."
            )


@dataclass(frozen=True)
class UniformDistributionCfg:
    """Continuous uniform distribution in the simulator's internal unit."""

    minimum: float
    maximum: float
    type: Literal["uniform"] = "uniform"

    def __post_init__(self) -> None:
        if self.type != "uniform":
            raise ValueError(
                "UniformDistributionCfg.type must be 'uniform'."
            )
        if not math.isfinite(self.minimum):
            raise ValueError(
                "UniformDistributionCfg.minimum must be finite."
            )
        if not math.isfinite(self.maximum):
            raise ValueError(
                "UniformDistributionCfg.maximum must be finite."
            )
        if self.maximum < self.minimum:
            raise ValueError(
                "UniformDistributionCfg.maximum must be >= minimum."
            )


@dataclass(frozen=True)
class SymmetricMixtureDistributionCfg:
    """
    Signed mixture with a normal core and symmetric lognormal tails.

    Model
    -----

        0.50 * Normal(center, normal_std)
        +
        0.25 * (center + LogNormal(tail_log_mu, tail_log_sigma))
        +
        0.25 * (center - LogNormal(tail_log_mu, tail_log_sigma))

    The center is stored as ``normal.mean`` so the normal component can be
    handled by the same generic normal-component type used elsewhere. The
    fixed mixture weights are defined centrally by the sampling
    implementation.
    """

    normal: NormalComponentCfg
    lognormal_tail: LogNormalComponentCfg
    type: SignedDistributionType = "symmetric_mixture"

    def __post_init__(self) -> None:
        if self.type != "symmetric_mixture":
            raise ValueError(
                "SymmetricMixtureDistributionCfg.type must be "
                "'symmetric_mixture'."
            )

    @property
    def center(self) -> float:
        return self.normal.mean

    @property
    def normal_std(self) -> float:
        return self.normal.std


PositiveDistributionCfg: TypeAlias = (
    PositiveMixtureDistributionCfg
    | LogUniformDistributionCfg
    | UniformDistributionCfg
)
SignedDistributionCfg: TypeAlias = (
    SymmetricMixtureDistributionCfg | UniformDistributionCfg
)


# -----------------------------------------------------------------------------
# Acquisition and resource configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AcquisitionCfg:
    """
    Acquisition parameters of the simulated spectra.

    ``n_timepoints`` is the fixed internal maximum FID length.

    With zero-filling enabled, one acquisition length is sampled per
    spectrum and all spectra retain ``n_timepoints`` samples.

    Without zero-filling, one acquisition length is sampled for the
    complete batch and the batch is returned at this native length.
    """

    bandwidth_hz: float
    n_timepoints: int

    min_acquired_n_timepoints: int
    max_acquired_n_timepoints: int

    zero_filling: bool

    nmr_frequency_hz: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.bandwidth_hz)
            or self.bandwidth_hz <= 0
        ):
            raise ValueError(
                "acquisition.bandwidth_hz must be finite and > 0."
            )

        if self.n_timepoints <= 0:
            raise ValueError(
                "acquisition.n_timepoints must be > 0."
            )

        if self.min_acquired_n_timepoints <= 0:
            raise ValueError(
                "acquisition.min_acquired_n_timepoints must be > 0."
            )

        if (
            self.max_acquired_n_timepoints
            < self.min_acquired_n_timepoints
        ):
            raise ValueError(
                "acquisition.max_acquired_n_timepoints must be >= "
                "acquisition.min_acquired_n_timepoints."
            )

        if self.max_acquired_n_timepoints > self.n_timepoints:
            raise ValueError(
                "acquisition.max_acquired_n_timepoints must be <= "
                "acquisition.n_timepoints."
            )

        if (
            not math.isfinite(self.nmr_frequency_hz)
            or self.nmr_frequency_hz <= 0
        ):
            raise ValueError(
                "acquisition.nmr_frequency_hz must be finite and > 0."
            )

    @property
    def dwell_time_seconds(self) -> float:
        return 1.0 / self.bandwidth_hz

    @property
    def hz_per_ppm(self) -> float:
        return self.nmr_frequency_hz / 1e6


@dataclass(frozen=True)
class BasisCfg:
    """Path to the prepared LCModel basis library."""

    library: str

    def __post_init__(self) -> None:
        if not self.library.strip():
            raise ValueError(
                "basis.library must not be empty."
            )


@dataclass(frozen=True)
class MetaboliteProfileCfg:
    """One metabolite concentration profile and its selection probability."""

    config: str
    probability: float

    def __post_init__(self) -> None:
        if not self.config.strip():
            raise ValueError(
                "metabolites.profiles[].config must not be empty."
            )

        if (
            not math.isfinite(self.probability)
            or self.probability < 0
        ):
            raise ValueError(
                "metabolites.profiles[].probability must be "
                "finite and >= 0."
            )


# -----------------------------------------------------------------------------
# Parameter-specific wrappers around generic distributions
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyShiftCfg:
    """
    Global frequency-shift distribution in the internal unit Hz.
    """

    distribution: SignedDistributionCfg

    @property
    def center_hz(self) -> float:
        return self.distribution.center

    @property
    def normal_std_hz(self) -> float:
        return self.distribution.normal_std

    @property
    def tail_log_mu_hz(self) -> float:
        return self.distribution.lognormal_tail.log_mu

    @property
    def tail_log_sigma(self) -> float:
        return self.distribution.lognormal_tail.log_sigma


@dataclass(frozen=True)
class FWHMCfg:
    """
    Positive mixture distribution of the total Voigt FWHM in Hz.
    """

    distribution: PositiveDistributionCfg

    @property
    def normal_mean_hz(self) -> float:
        return self.distribution.normal.mean

    @property
    def normal_std_hz(self) -> float:
        return self.distribution.normal.std

    @property
    def log_mu_hz(self) -> float:
        return self.distribution.lognormal.log_mu

    @property
    def log_sigma(self) -> float:
        return self.distribution.lognormal.log_sigma

    @property
    def minimum_hz(self) -> float:
        return self.distribution.minimum


@dataclass(frozen=True)
class ZeroOrderPhaseCfg:
    """
    Frequency-independent phase distribution in radians.
    """

    distribution: SignedDistributionCfg

    @property
    def center_rad(self) -> float:
        return self.distribution.center

    @property
    def normal_std_rad(self) -> float:
        return self.distribution.normal_std

    @property
    def tail_log_mu_rad(self) -> float:
        return self.distribution.lognormal_tail.log_mu

    @property
    def tail_log_sigma(self) -> float:
        return self.distribution.lognormal_tail.log_sigma


@dataclass(frozen=True)
class FirstOrderPhaseCfg:
    """
    Linear phase-slope distribution in radians per Hz.
    """

    distribution: SymmetricMixtureDistributionCfg

    @property
    def center_rad_per_hz(self) -> float:
        return self.distribution.center

    @property
    def normal_std_rad_per_hz(self) -> float:
        return self.distribution.normal_std

    @property
    def tail_log_mu_rad_per_hz(self) -> float:
        return self.distribution.lognormal_tail.log_mu

    @property
    def tail_log_sigma(self) -> float:
        return self.distribution.lognormal_tail.log_sigma


@dataclass(frozen=True)
class MetaboliteCfg:
    """Metabolite simulation parameters."""

    profiles: tuple[MetaboliteProfileCfg, ...]

    frequency_shift: FrequencyShiftCfg
    fwhm: FWHMCfg

    zero_order_phase: ZeroOrderPhaseCfg
    first_order_phase: FirstOrderPhaseCfg

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError(
                "metabolites.profiles must contain at least one profile."
            )

        probability_sum = sum(
            profile.probability
            for profile in self.profiles
        )

        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "metabolites.profiles probabilities must sum to 1. "
                f"Found: {probability_sum:.12g}"
            )


@dataclass(frozen=True)
class SNRDistributionCfg:
    """Positive mixture distribution of the target LCModel-compatible SNR."""

    distribution: PositiveMixtureDistributionCfg

    @property
    def minimum(self) -> float:
        return self.distribution.minimum


@dataclass(frozen=True)
class NoiseCfg:
    """Receiver-noise simulation parameters."""

    snr: SNRDistributionCfg


@dataclass(frozen=True)
class PositiveScalingDistributionCfg:
    """Positive distribution of a dimensionless scaling factor."""

    distribution: PositiveDistributionCfg

    @property
    def minimum(self) -> float:
        return self.distribution.minimum


@dataclass(frozen=True)
class WaterCfg:
    """Water scaling relative to the maximum metabolite amplitude."""

    scaling: PositiveScalingDistributionCfg


@dataclass(frozen=True)
class LipidCfg:
    """Lipid baseline simulation parameters."""

    n_random_fids: int
    scaling: PositiveScalingDistributionCfg

    def __post_init__(self) -> None:
        if self.n_random_fids <= 0:
            raise ValueError(
                "lipids.n_random_fids must be > 0."
            )


@dataclass(frozen=True)
class SubjectSamplingCfg:
    """Controls how water and lipid resources are associated with subjects."""

    mixing: str

    def __post_init__(self) -> None:
        supported_modes = {
            "same_subject",
            "separate_water_lipid_subjects",
            "independent_lipid_fids",
        }

        if self.mixing not in supported_modes:
            raise ValueError(
                "Unsupported subject_sampling.mixing: "
                f"{self.mixing!r}. Supported modes: "
                f"{sorted(supported_modes)}"
            )


@dataclass(frozen=True)
class LipidProjectionCfg:
    """Optional legacy lipid-projection output."""

    enabled: bool


@dataclass(frozen=True)
class SimulationConfig:
    """Fully validated internal simulation configuration."""

    version: str

    acquisition: AcquisitionCfg
    basis: BasisCfg
    metabolites: MetaboliteCfg
    noise: NoiseCfg
    water: WaterCfg
    lipids: LipidCfg
    subject_sampling: SubjectSamplingCfg
    lipid_projection: LipidProjectionCfg

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError(
                "version must not be empty."
            )
