# src/walinet/training_data/metabolite_simulation.py
# Distribution refactor: mixture_v2

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch
import yaml

from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.distributions import (
    sample_distribution,
    sample_positive_mixture_parameters,
    sample_truncated_normal,
    sample_uniform,
    validate_generator_device,
)
from walinet.training_data.lcmodel_basis.acquisition import (
    PreparedBasis,
)


VOIGT_LORENTZ_COEFFICIENT = 0.5346
VOIGT_LORENTZ_SQUARED_COEFFICIENT = 0.2166


@dataclass(frozen=True)
class MetaboliteSamplingTable:
    """
    One metabolite sampling profile aligned with PreparedBasis.names.

    All parameter tensors have shape ``(n_basis_components,)``. Disabled
    basis components contain safe placeholder parameters and are tracked by
    ``enabled_mask``; their sampled concentrations are set to exactly zero.
    """

    config_path: str

    basis_names: tuple[str, ...]

    active_config_names: tuple[str, ...]
    active_basis_names: tuple[str, ...]

    means: torch.Tensor
    stds: torch.Tensor

    log_mus: torch.Tensor
    log_sigmas: torch.Tensor

    minimums: torch.Tensor
    truncated_normal_mask: torch.Tensor
    enabled_mask: torch.Tensor

    @property
    def n_basis_components(self) -> int:
        return len(self.basis_names)

    @property
    def n_active_components(self) -> int:
        return len(self.active_basis_names)

    @property
    def device(self) -> torch.device:
        return self.means.device


@dataclass(frozen=True)
class SimulatedMetabolites:
    """One batch of noise-free simulated metabolites."""

    clean_fids: torch.Tensor
    clean_spectra: torch.Tensor

    concentrations: torch.Tensor
    profile_indices: torch.Tensor

    zero_order_phases_radians: torch.Tensor
    first_order_phases_rad_per_hz: torch.Tensor
    frequency_shifts_hz: torch.Tensor

    voigt_fwhm_hz: torch.Tensor
    lorentzian_fractions: torch.Tensor
    gaussian_fwhm_hz: torch.Tensor
    lorentzian_fwhm_hz: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.clean_fids.shape[0])

    @property
    def n_timepoints(self) -> int:
        return int(self.clean_fids.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.clean_fids.device


def _load_yaml_mapping(
    path: Path,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            "Metabolite profile configuration not found:\n"
            f"  {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise TypeError(
            "Metabolite profile configuration must contain "
            "a YAML mapping."
        )

    return raw


def _require_profile_float(
    mapping: dict,
    key: str,
    *,
    parameter_path: str,
    profile_path: Path,
) -> float:
    if key not in mapping:
        raise KeyError(
            f"Missing required value {parameter_path}.{key}.\n"
            f"Profile: {profile_path}"
        )

    value = mapping[key]

    if value is None:
        raise ValueError(
            f"{parameter_path}.{key} must not be null.\n"
            f"Profile: {profile_path}"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{parameter_path}.{key} must be numeric, "
            f"but found {value!r}.\n"
            f"Profile: {profile_path}"
        ) from error

    if not np.isfinite(result):
        raise ValueError(
            f"{parameter_path}.{key} must be finite.\n"
            f"Profile: {profile_path}"
        )

    return result


def load_metabolite_sampling_table(
    *,
    path: str | Path,
    prepared_basis: PreparedBasis,
    device: torch.device | str,
) -> MetaboliteSamplingTable:
    """
    Load one metabolite profile and align it with the basis.

    Required distribution layout for every enabled metabolite::

        distribution:
          type: positive_mixture
          normal:
            mean: ...
            std: ...
          lognormal:
            log_mu: ...
            log_sigma: ...
          minimum: 0.0
    """
    path = Path(path).resolve()
    device = torch.device(device)

    raw = _load_yaml_mapping(path)

    sampling_raw = raw.get("sampling", {})

    if not isinstance(sampling_raw, dict):
        raise TypeError(
            "sampling must be a mapping.\n"
            f"Profile: {path}"
        )

    independent = bool(
        sampling_raw.get("independent", True)
    )

    if not independent:
        raise ValueError(
            "Only independent metabolite concentration sampling is "
            "currently supported.\n"
            f"Profile: {path}"
        )

    default_distribution = str(
        sampling_raw.get(
            "default_distribution",
            "positive_mixture",
        )
    ).strip().lower()

    supported_distributions = {
        "positive_mixture",
        "truncated_normal",
    }

    if default_distribution not in supported_distributions:
        raise ValueError(
            "sampling.default_distribution must be one of "
            f"{sorted(supported_distributions)}, but found "
            f"{default_distribution!r}.\n"
            f"Profile: {path}"
        )

    metabolites_raw = raw.get("metabolites")

    if not isinstance(metabolites_raw, dict):
        raise TypeError(
            "Metabolite profile must contain a 'metabolites' mapping.\n"
            f"Profile: {path}"
        )

    n_basis_components = prepared_basis.n_metabolites

    # Safe placeholders for disabled or unspecified basis components.
    # They make vectorized full-table sampling valid before those entries are
    # overwritten with exactly zero through enabled_mask.
    means = np.ones(n_basis_components, dtype=np.float32)
    stds = np.zeros(n_basis_components, dtype=np.float32)
    log_mus = np.zeros(n_basis_components, dtype=np.float32)
    log_sigmas = np.zeros(n_basis_components, dtype=np.float32)
    minimums = np.zeros(n_basis_components, dtype=np.float32)
    truncated_normal_mask = np.zeros(n_basis_components, dtype=bool)
    enabled_mask = np.zeros(n_basis_components, dtype=bool)

    active_config_names: list[str] = []
    active_basis_names: list[str] = []
    used_basis_components: set[str] = set()

    for config_name, metabolite_raw in metabolites_raw.items():
        if not isinstance(metabolite_raw, dict):
            raise TypeError(
                f"Metabolite entry {config_name!r} must be a mapping.\n"
                f"Profile: {path}"
            )

        enabled = bool(metabolite_raw.get("enabled", True))

        if not enabled:
            continue

        basis_component_raw = metabolite_raw.get("basis_component")

        if (
            basis_component_raw is None
            or not str(basis_component_raw).strip()
        ):
            raise ValueError(
                f"Enabled metabolite {config_name!r} has no "
                "basis_component.\n"
                f"Profile: {path}"
            )

        basis_component = str(basis_component_raw).strip()

        if basis_component in used_basis_components:
            raise ValueError(
                "Multiple enabled metabolite entries map to the same "
                "basis component:\n"
                f"  {basis_component}\n"
                f"Profile: {path}"
            )

        try:
            basis_index = prepared_basis.index(basis_component)
        except KeyError as error:
            raise KeyError(
                f"Metabolite {config_name!r} references a missing "
                "basis component:\n"
                f"  {basis_component}\n"
                f"Profile: {path}"
            ) from error

        distribution_raw = metabolite_raw.get("distribution")

        if not isinstance(distribution_raw, dict):
            raise TypeError(
                f"Metabolite {config_name!r} has no valid distribution "
                "mapping.\n"
                f"Profile: {path}"
            )

        parameter_path = f"metabolites.{config_name}.distribution"

        distribution_type = str(
            distribution_raw.get("type", default_distribution)
        ).strip().lower()

        if distribution_type not in supported_distributions:
            raise ValueError(
                f"{parameter_path}.type must be one of "
                f"{sorted(supported_distributions)}, "
                f"but found {distribution_type!r}.\n"
                f"Profile: {path}"
            )

        if distribution_type == "positive_mixture":
            normal_raw = distribution_raw.get("normal")
        else:
            normal_raw = distribution_raw

        if not isinstance(normal_raw, dict):
            raise TypeError(
                f"{parameter_path}.normal must be a mapping.\n"
                f"Profile: {path}"
            )

        normal_mean = _require_profile_float(
            normal_raw,
            "mean",
            parameter_path=(
                f"{parameter_path}.normal"
                if distribution_type == "positive_mixture"
                else parameter_path
            ),
            profile_path=path,
        )
        normal_std = _require_profile_float(
            normal_raw,
            "std",
            parameter_path=(
                f"{parameter_path}.normal"
                if distribution_type == "positive_mixture"
                else parameter_path
            ),
            profile_path=path,
        )
        if distribution_type == "positive_mixture":
            lognormal_raw = distribution_raw.get("lognormal")
            if not isinstance(lognormal_raw, dict):
                raise TypeError(
                    f"{parameter_path}.lognormal must be a mapping.\n"
                    f"Profile: {path}"
                )
            log_mu = _require_profile_float(
                lognormal_raw,
                "log_mu",
                parameter_path=f"{parameter_path}.lognormal",
                profile_path=path,
            )
            log_sigma = _require_profile_float(
                lognormal_raw,
                "log_sigma",
                parameter_path=f"{parameter_path}.lognormal",
                profile_path=path,
            )
        else:
            # Safe placeholders; the truncated-normal sampling path does not
            # use lognormal parameters.
            log_mu = 0.0
            log_sigma = 0.0
        minimum = _require_profile_float(
            distribution_raw,
            "minimum",
            parameter_path=parameter_path,
            profile_path=path,
        )

        if normal_std < 0:
            raise ValueError(
                f"{parameter_path}.normal.std must be >= 0.\n"
                f"Profile: {path}"
            )

        if minimum < 0:
            raise ValueError(
                f"{parameter_path}.minimum must be >= 0.\n"
                f"Profile: {path}"
            )

        if normal_std == 0 and normal_mean <= minimum:
            raise ValueError(
                f"{parameter_path}.normal cannot be sampled because "
                "std == 0 and mean <= minimum.\n"
                f"Profile: {path}"
            )

        if distribution_type == "positive_mixture" and log_sigma < 0:
            raise ValueError(
                f"{parameter_path}.lognormal.log_sigma must be >= 0.\n"
                f"Profile: {path}"
            )

        if distribution_type == "positive_mixture" and log_sigma == 0:
            try:
                constant_lognormal_value = math.exp(log_mu)
            except OverflowError as error:
                raise ValueError(
                    f"{parameter_path}.lognormal.log_mu overflows.\n"
                    f"Profile: {path}"
                ) from error

            if (
                not math.isfinite(constant_lognormal_value)
                or constant_lognormal_value <= minimum
            ):
                raise ValueError(
                    f"{parameter_path}.lognormal cannot be sampled because "
                    "log_sigma == 0 and exp(log_mu) is not above minimum.\n"
                    f"Profile: {path}"
                )

        means[basis_index] = normal_mean
        stds[basis_index] = normal_std
        log_mus[basis_index] = log_mu
        log_sigmas[basis_index] = log_sigma
        minimums[basis_index] = minimum
        truncated_normal_mask[basis_index] = (
            distribution_type == "truncated_normal"
        )
        enabled_mask[basis_index] = True

        active_config_names.append(str(config_name))
        active_basis_names.append(basis_component)
        used_basis_components.add(basis_component)

    if not active_basis_names:
        raise ValueError(
            "No enabled metabolites were found.\n"
            f"Profile: {path}"
        )

    return MetaboliteSamplingTable(
        config_path=str(path),
        basis_names=tuple(prepared_basis.names),
        active_config_names=tuple(active_config_names),
        active_basis_names=tuple(active_basis_names),
        means=torch.from_numpy(means).to(device=device),
        stds=torch.from_numpy(stds).to(device=device),
        log_mus=torch.from_numpy(log_mus).to(device=device),
        log_sigmas=torch.from_numpy(log_sigmas).to(device=device),
        minimums=torch.from_numpy(minimums).to(device=device),
        truncated_normal_mask=torch.from_numpy(
            truncated_normal_mask
        ).to(device=device),
        enabled_mask=torch.from_numpy(enabled_mask).to(device=device),
    )


class MetaboliteSimulator:
    """
    Vectorized noise-free metabolite simulator.

    Processing order:

        1. Sample one metabolite profile per spectrum.
        2. Sample non-negative concentrations from that profile.
        3. Combine PreparedBasis FIDs.
        4. Sample total Voigt FWHM and line-shape mixture.
        5. Sample frequency shift and zero-order phase.
        6. Apply frequency shift, zero-order phase, and
           Gaussian/Lorentzian FID broadening.
        7. Sample and apply first-order phase in the frequency
           domain.
        8. Transform to fftshifted spectra.

    The zero-order and first-order phases are applied only to the
    simulated metabolite signal. Measured water and lipid signals
    retain their original complex phases.

    The simulator has no internal random state. An explicit
    torch.Generator must be supplied for every simulation call.
    """

    def __init__(
        self,
        *,
        prepared_basis: PreparedBasis,
        config: SimulationConfig,
        device: torch.device | str,
        max_relative_bandwidth_error: float = 1e-3,
    ) -> None:
        self.prepared_basis = (
            prepared_basis
        )

        self.config = config

        self.device = torch.device(
            device
        )

        self._validate_prepared_basis(
            max_relative_bandwidth_error=(
                max_relative_bandwidth_error
            )
        )

        basis_array = np.ascontiguousarray(
            prepared_basis.fids,
            dtype=np.complex64,
        )

        self.basis_fids = torch.from_numpy(
            basis_array
        ).to(
            device=self.device
        ).contiguous()

        self.sampling_tables = tuple(
            load_metabolite_sampling_table(
                path=profile.config,
                prepared_basis=prepared_basis,
                device=self.device,
            )
            for profile in (
                config
                .metabolites
                .profiles
            )
        )

        self.profile_config_paths = tuple(
            table.config_path
            for table in self.sampling_tables
        )

        self.profile_probabilities = torch.tensor(
            [
                profile.probability
                for profile in (
                    config
                    .metabolites
                    .profiles
                )
            ],
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        self.profile_means = torch.stack(
            [
                table.means
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.profile_stds = torch.stack(
            [
                table.stds
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.profile_log_mus = torch.stack(
            [
                table.log_mus
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.profile_log_sigmas = torch.stack(
            [
                table.log_sigmas
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.profile_minimums = torch.stack(
            [
                table.minimums
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.profile_truncated_normal_masks = torch.stack(
            [
                table.truncated_normal_mask
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.profile_enabled_masks = torch.stack(
            [
                table.enabled_mask
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.time_axis_seconds = (
            torch.arange(
                prepared_basis.n_timepoints,
                device=self.device,
                dtype=torch.float32,
            )
            * float(
                prepared_basis.dwell_time
            )
        )

        # Unshifted FFT frequency offsets in Hz.
        #
        # The first-order phase is defined relative to 0 Hz:
        #
        #     phase(f) = phi_1 * f
        #
        # where phi_1 is expressed in rad/Hz.
        self.frequency_axis_hz = (
            torch.fft.fftfreq(
                prepared_basis.n_timepoints,
                d=float(
                    prepared_basis.dwell_time
                ),
                device=self.device,
                dtype=torch.float32,
            )
        )

    @property
    def n_profiles(self) -> int:
        return len(
            self.sampling_tables
        )

    @property
    def n_basis_components(self) -> int:
        return int(
            self.basis_fids.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.basis_fids.shape[-1]
        )

    def simulate(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> SimulatedMetabolites:
        """
        Simulate one batch of noise-free metabolite FIDs and spectra.
        """
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        validate_generator_device(
            generator=generator,
            device=self.device,
        )

        profile_indices = (
            self._sample_profile_indices(
                batch_size=batch_size,
                generator=generator,
            )
        )

        concentrations = (
            self._sample_concentrations(
                profile_indices=(
                    profile_indices
                ),
                generator=generator,
            )
        )

        metabolite_fids = (
            concentrations.to(
                dtype=self.basis_fids.dtype
            )
            @ self.basis_fids
        )

        zero_order_phase_cfg = (
            self.config
            .metabolites
            .zero_order_phase
        )

        zero_order_phases = sample_distribution(
            distribution=zero_order_phase_cfg.distribution,
            shape=(batch_size,),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )

        first_order_phase_cfg = (
            self.config
            .metabolites
            .first_order_phase
        )

        first_order_phases = sample_distribution(
            distribution=first_order_phase_cfg.distribution,
            shape=(batch_size,),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )

        frequency_shift_cfg = (
            self.config
            .metabolites
            .frequency_shift
        )

        frequency_shifts = sample_distribution(
            distribution=frequency_shift_cfg.distribution,
            shape=(batch_size,),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )

        fwhm_cfg = (
            self.config
            .metabolites
            .fwhm
        )

        voigt_fwhm_hz = sample_distribution(
            distribution=fwhm_cfg.distribution,
            shape=(batch_size,),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )

        lorentzian_fractions = sample_uniform(
            minimum=0.0,
            maximum=1.0,
            shape=(batch_size,),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )

        (
            lorentzian_fwhm_hz,
            gaussian_fwhm_hz,
        ) = self._split_voigt_fwhm(
            voigt_fwhm_hz=voigt_fwhm_hz,
            lorentzian_fractions=(
                lorentzian_fractions
            ),
        )

        affected_fids = (
            self._apply_fid_effects(
                metabolite_fids=(
                    metabolite_fids
                ),
                zero_order_phases_radians=(
                    zero_order_phases
                ),
                frequency_shifts_hz=(
                    frequency_shifts
                ),
                gaussian_fwhm_hz=(
                    gaussian_fwhm_hz
                ),
                lorentzian_fwhm_hz=(
                    lorentzian_fwhm_hz
                ),
            )
        )

        clean_fids = (
            self._apply_first_order_phase(
                metabolite_fids=(
                    affected_fids
                ),
                first_order_phases_rad_per_hz=(
                    first_order_phases
                ),
            )
        )

        clean_spectra = (
            torch.fft.fftshift(
                torch.fft.fft(
                    clean_fids,
                    dim=-1,
                ),
                dim=-1,
            )
            .contiguous()
        )

        result = SimulatedMetabolites(
            clean_fids=clean_fids,
            clean_spectra=clean_spectra,
            concentrations=(
                concentrations
            ),
            profile_indices=(
                profile_indices
            ),
            zero_order_phases_radians=(
                zero_order_phases
            ),
            first_order_phases_rad_per_hz=(
                first_order_phases
            ),
            frequency_shifts_hz=(
                frequency_shifts
            ),
            voigt_fwhm_hz=(
                voigt_fwhm_hz
            ),
            lorentzian_fractions=(
                lorentzian_fractions
            ),
            gaussian_fwhm_hz=(
                gaussian_fwhm_hz
            ),
            lorentzian_fwhm_hz=(
                lorentzian_fwhm_hz
            ),
        )

        self._validate_result(
            result
        )

        return result

    def _sample_profile_indices(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Sample one metabolite profile index per spectrum.
        """
        if self.n_profiles == 1:
            return torch.zeros(
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )

        return torch.multinomial(
            self.profile_probabilities,
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        ).contiguous()

    def _sample_concentrations(
        self,
        *,
        profile_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Sample independent metabolite concentrations from the configured
        positive distribution.

        Existing profiles use

            0.5 * TruncatedNormal(normal_mean, normal_std)
            +
            0.5 * LogNormal(log_mu, log_sigma).

        A metabolite configured as ``truncated_normal`` instead uses only
        ``TruncatedNormal(normal_mean, normal_std)``.

        The parameter tensors are selected from the sampled metabolite
        profile and passed to the same centralized sampler used for FWHM,
        SNR, water scaling, and lipid scaling. Disabled basis components are
        set to exactly zero afterwards.
        """
        batch_size = int(
            profile_indices.shape[0]
        )

        output_shape = (
            batch_size,
            self.n_basis_components,
        )

        normal_means = self.profile_means.index_select(
            0,
            profile_indices,
        )

        normal_stds = self.profile_stds.index_select(
            0,
            profile_indices,
        )

        log_mus = self.profile_log_mus.index_select(
            0,
            profile_indices,
        )

        log_sigmas = self.profile_log_sigmas.index_select(
            0,
            profile_indices,
        )

        minimums = self.profile_minimums.index_select(
            0,
            profile_indices,
        )

        enabled_mask = self.profile_enabled_masks.index_select(
            0,
            profile_indices,
        )

        truncated_normal_mask = (
            self.profile_truncated_normal_masks.index_select(
                0,
                profile_indices,
            )
        )

        if bool(torch.all(truncated_normal_mask | ~enabled_mask)):
            concentrations = sample_truncated_normal(
                mean=normal_means,
                std=normal_stds,
                minimum=minimums,
                shape=output_shape,
                device=self.device,
                dtype=torch.float32,
                generator=generator,
            )
        else:
            concentrations = sample_positive_mixture_parameters(
                normal_mean=normal_means,
                normal_std=normal_stds,
                log_mu=log_mus,
                log_sigma=log_sigmas,
                minimum=minimums,
                shape=output_shape,
                device=self.device,
                dtype=torch.float32,
                generator=generator,
            )

            if bool(torch.any(truncated_normal_mask)):
                truncated_values = sample_truncated_normal(
                    mean=normal_means,
                    std=normal_stds,
                    minimum=minimums,
                    shape=output_shape,
                    device=self.device,
                    dtype=torch.float32,
                    generator=generator,
                )
                concentrations = torch.where(
                    truncated_normal_mask,
                    truncated_values,
                    concentrations,
                )

        concentrations = torch.where(
            enabled_mask,
            concentrations,
            torch.zeros_like(concentrations),
        )

        return concentrations.contiguous()

    def _split_voigt_fwhm(
        self,
        *,
        voigt_fwhm_hz: torch.Tensor,
        lorentzian_fractions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split total Voigt FWHM into Lorentzian and Gaussian FWHM.

        Uses the approximation:

            V = a*L + sqrt(b*L^2 + G^2)

        with:

            a = 0.5346
            b = 0.2166

        lorentzian_fractions is uniform in [0, 1] and interpolates
        between the pure Gaussian and pure Lorentzian endpoints.
        """
        if torch.any(
            voigt_fwhm_hz <= 0
        ):
            raise ValueError(
                "voigt_fwhm_hz must be > 0."
            )

        if torch.any(
            (lorentzian_fractions < 0)
            | (lorentzian_fractions > 1)
        ):
            raise ValueError(
                "lorentzian_fractions must be in [0, 1]."
            )

        a = VOIGT_LORENTZ_COEFFICIENT
        b = VOIGT_LORENTZ_SQUARED_COEFFICIENT

        maximum_lorentzian_fwhm = (
            voigt_fwhm_hz
            / (
                a
                + math.sqrt(b)
            )
        )

        lorentzian_fwhm_hz = (
            lorentzian_fractions
            * maximum_lorentzian_fwhm
        )

        gaussian_fwhm_squared = (
            (
                voigt_fwhm_hz
                - a
                * lorentzian_fwhm_hz
            ).square()
            - b
            * lorentzian_fwhm_hz.square()
        )

        gaussian_fwhm_hz = torch.sqrt(
            torch.clamp(
                gaussian_fwhm_squared,
                min=0.0,
            )
        )

        reconstructed_voigt_fwhm = (
            a
            * lorentzian_fwhm_hz
            + torch.sqrt(
                b
                * lorentzian_fwhm_hz.square()
                + gaussian_fwhm_hz.square()
            )
        )

        if not torch.allclose(
            reconstructed_voigt_fwhm,
            voigt_fwhm_hz,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Voigt FWHM decomposition failed its "
                "reconstruction check."
            )

        return (
            lorentzian_fwhm_hz.contiguous(),
            gaussian_fwhm_hz.contiguous(),
        )

    def _apply_first_order_phase(
        self,
        *,
        metabolite_fids: torch.Tensor,
        first_order_phases_rad_per_hz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a linear first-order phase in the unshifted
        frequency domain.

        The phase is defined relative to the spectral center
        at 0 Hz:

            phase(f) = phi_1 * f

        where:

            phi_1:
                First-order phase in rad/Hz.

            f:
                Frequency offset in Hz.

        The spectrum is multiplied by:

            exp(+i * phi_1 * f)

        The sign convention can later be validated against the
        corresponding LCModel first-order phase output. If LCModel
        reports the correction rather than the distortion, the
        calibration values must be negated before simulation.
        """
        if torch.all(
            first_order_phases_rad_per_hz == 0
        ):
            return metabolite_fids.contiguous()

        spectra = torch.fft.fft(
            metabolite_fids,
            dim=-1,
        )

        phase_angles = (
            first_order_phases_rad_per_hz[
                :,
                None,
            ]
            * self.frequency_axis_hz[
                None,
                :
            ]
        )

        phase_factor = torch.polar(
            torch.ones_like(
                phase_angles
            ),
            phase_angles,
        )

        phased_fids = torch.fft.ifft(
            spectra
            * phase_factor,
            dim=-1,
        )

        return phased_fids.contiguous()

    def _apply_fid_effects(
        self,
        *,
        metabolite_fids: torch.Tensor,
        zero_order_phases_radians: torch.Tensor,
        frequency_shifts_hz: torch.Tensor,
        gaussian_fwhm_hz: torch.Tensor,
        lorentzian_fwhm_hz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply zero-order phase, frequency shift, and Voigt
        broadening.

        Zero-order phase:

            exp(+i * phi_0)

        Frequency shift:

            exp(+i * 2*pi*delta_f*t)

        Lorentzian FID decay:

            exp(-pi * L * t)

        Gaussian FID decay:

            exp(-(pi * G * t)^2 / (4*ln(2)))

        where L and G are the Lorentzian and Gaussian FWHM in Hz.
        """
        time_axis = (
            self.time_axis_seconds[
                None,
                :
            ]
        )

        phase_angles = (
            zero_order_phases_radians[
                :,
                None,
            ]
            + 2.0
            * math.pi
            * frequency_shifts_hz[
                :,
                None,
            ]
            * time_axis
        )

        phase_factor = torch.polar(
            torch.ones_like(
                phase_angles
            ),
            phase_angles,
        )

        lorentzian_exponent = (
            math.pi
            * lorentzian_fwhm_hz[
                :,
                None,
            ]
            * time_axis
        )

        gaussian_exponent = (
            (
                math.pi
                * gaussian_fwhm_hz[
                    :,
                    None,
                ]
                * time_axis
            ).square()
            / (
                4.0
                * math.log(2.0)
            )
        )

        decay_factor = torch.exp(
            -lorentzian_exponent
            -gaussian_exponent
        )

        result = (
            metabolite_fids
            * phase_factor
            * decay_factor
        )

        return result.contiguous()

    def _validate_prepared_basis(
        self,
        *,
        max_relative_bandwidth_error: float,
    ) -> None:
        basis = self.prepared_basis

        if basis.fids.ndim != 2:
            raise ValueError(
                "PreparedBasis.fids must have shape "
                "(n_metabolites, n_timepoints)."
            )

        if (
            len(basis.names)
            != basis.fids.shape[0]
        ):
            raise ValueError(
                "PreparedBasis.names does not match "
                "PreparedBasis.fids."
            )

        if (
            basis.n_timepoints
            != basis.fids.shape[-1]
        ):
            raise ValueError(
                "PreparedBasis.n_timepoints does not "
                "match PreparedBasis.fids."
            )

        expected_n_timepoints = (
            self.config
            .acquisition
            .n_timepoints
        )

        if (
            basis.n_timepoints
            != expected_n_timepoints
        ):
            raise ValueError(
                "Prepared basis and simulation configuration "
                "use different FID lengths:\n"
                f"  basis:      {basis.n_timepoints}\n"
                f"  simulation: {expected_n_timepoints}"
            )

        if not np.all(
            np.isfinite(
                basis.fids
            )
        ):
            raise ValueError(
                "PreparedBasis.fids contains NaN or Inf."
            )

        expected_bandwidth = (
            self.config
            .acquisition
            .bandwidth_hz
        )

        relative_error = abs(
            basis.bandwidth
            - expected_bandwidth
        ) / expected_bandwidth

        if (
            relative_error
            > max_relative_bandwidth_error
        ):
            raise ValueError(
                "Prepared basis bandwidth differs too strongly "
                "from the simulation bandwidth:\n"
                f"  basis:          {basis.bandwidth}\n"
                f"  simulation:     {expected_bandwidth}\n"
                f"  relative error: {relative_error:.3e}\n"
                f"  maximum:        "
                f"{max_relative_bandwidth_error:.3e}"
            )

    def _validate_result(
        self,
        result: SimulatedMetabolites,
    ) -> None:
        expected_signal_shape = (
            result.batch_size,
            self.n_timepoints,
        )

        if (
            result.clean_fids.shape
            != expected_signal_shape
        ):
            raise RuntimeError(
                "Unexpected clean_fids shape."
            )

        if (
            result.clean_spectra.shape
            != expected_signal_shape
        ):
            raise RuntimeError(
                "Unexpected clean_spectra shape."
            )

        expected_concentration_shape = (
            result.batch_size,
            self.n_basis_components,
        )

        if (
            result.concentrations.shape
            != expected_concentration_shape
        ):
            raise RuntimeError(
                "Unexpected concentrations shape."
            )

        if tuple(
            result.profile_indices.shape
        ) != (
            result.batch_size,
        ):
            raise RuntimeError(
                "Unexpected profile_indices shape."
            )

        if result.profile_indices.dtype != torch.long:
            raise RuntimeError(
                "profile_indices must use torch.long dtype."
            )

        if torch.any(
            result.profile_indices < 0
        ):
            raise RuntimeError(
                "profile_indices contains negative values."
            )

        if torch.any(
            result.profile_indices >= self.n_profiles
        ):
            raise RuntimeError(
                "profile_indices contains an out-of-range value."
            )

        parameter_tensors = {
            "zero_order_phases_radians": (
                result.zero_order_phases_radians
            ),
            "first_order_phases_rad_per_hz": (
                result.first_order_phases_rad_per_hz
            ),
            "frequency_shifts_hz": (
                result.frequency_shifts_hz
            ),
            "voigt_fwhm_hz": (
                result.voigt_fwhm_hz
            ),
            "lorentzian_fractions": (
                result.lorentzian_fractions
            ),
            "gaussian_fwhm_hz": (
                result.gaussian_fwhm_hz
            ),
            "lorentzian_fwhm_hz": (
                result.lorentzian_fwhm_hz
            ),
        }

        for name, tensor in parameter_tensors.items():
            if tuple(tensor.shape) != (
                result.batch_size,
            ):
                raise RuntimeError(
                    f"Unexpected shape for {name}: "
                    f"{tuple(tensor.shape)}"
                )

            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"{name} contains non-finite values."
                )

        if not torch.isfinite(
            result.concentrations
        ).all():
            raise RuntimeError(
                "Metabolite concentrations contain "
                "non-finite values."
            )

        if torch.any(
            result.concentrations < 0
        ):
            raise RuntimeError(
                "Metabolite concentrations contain negative values."
            )

        selected_enabled_masks = (
            self.profile_enabled_masks
            .index_select(
                0,
                result.profile_indices,
            )
        )

        if torch.any(
            result.concentrations[
                ~selected_enabled_masks
            ]
            != 0
        ):
            raise RuntimeError(
                "Disabled metabolites contain non-zero "
                "concentrations."
            )

        if torch.any(
            result.voigt_fwhm_hz <= 0
        ):
            raise RuntimeError(
                "Voigt FWHM contains non-positive values."
            )

        if torch.any(
            result.gaussian_fwhm_hz < 0
        ):
            raise RuntimeError(
                "Gaussian FWHM contains negative values."
            )

        if torch.any(
            result.lorentzian_fwhm_hz < 0
        ):
            raise RuntimeError(
                "Lorentzian FWHM contains negative values."
            )

        if not torch.isfinite(
            result.clean_fids.real
        ).all():
            raise RuntimeError(
                "Metabolite FIDs contain non-finite "
                "real values."
            )

        if not torch.isfinite(
            result.clean_fids.imag
        ).all():
            raise RuntimeError(
                "Metabolite FIDs contain non-finite "
                "imaginary values."
            )

        if not torch.isfinite(
            result.clean_spectra.real
        ).all():
            raise RuntimeError(
                "Metabolite spectra contain non-finite "
                "real values."
            )

        if not torch.isfinite(
            result.clean_spectra.imag
        ).all():
            raise RuntimeError(
                "Metabolite spectra contain non-finite "
                "imaginary values."
            )
