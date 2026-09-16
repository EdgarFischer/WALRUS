# TARGET DEFINITION: clean water + lipids baseline
# Input: metabolites + water + lipids + receiver noise

# src/walinet/training_data/spectrum_assembly.py
# Distribution refactor: mixture_v2

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.distributions import (
    sample_distribution,
    validate_generator_device,
)
from walinet.training_data.acquisition_length import (
    simulate_acquisition_length,
)
from walinet.training_data.metabolite_noise import (
    SimulatedNoise,
)
from walinet.training_data.metabolite_simulation import (
    SimulatedMetabolites,
)
from walinet.training_data.simulation_resources import (
    SimulationPool,
)
from walinet.training_data.simulator import (
    LipidMixture,
    SampledResources,
)


@dataclass(frozen=True)
class AssembledSpectra:
    """
    Fully assembled frequency-domain simulation batch.

    Shapes
    ------
    water_spectra:
        (batch_size, n_timepoints)

        Frequency-shifted and scaled water spectra before
        acquisition-length simulation.

    lipid_spectra:
        (batch_size, n_timepoints)

        Scaled lipid spectra before acquisition-length simulation.

    clean_mixture_spectra:
        Clean metabolites + water + lipids before receiver noise
        and before acquisition-length simulation.

    mixture_spectra:
        Final noisy input after acquisition-length simulation.

    baseline_spectra:
        Final clean water-plus-lipid target after the same
        acquisition-length simulation as mixture_spectra.

    acquired_n_timepoints:
        (batch_size,)

        Number of actually acquired FID samples per batch item.

    projected_spectra:
        Optional result after applying the subject-specific
        frequency-domain lipid-projection operator to the final
        mixture_spectra.
    """

    metabolites: SimulatedMetabolites
    noise: SimulatedNoise

    water_spectra: torch.Tensor
    lipid_spectra: torch.Tensor

    clean_mixture_spectra: torch.Tensor

    mixture_spectra: torch.Tensor
    baseline_spectra: torch.Tensor
    acquired_clean_metabolite_spectra: torch.Tensor

    acquired_n_timepoints: torch.Tensor

    projected_spectra: torch.Tensor | None

    water_scaling: torch.Tensor
    lipid_scaling: torch.Tensor

    water_subject_indices: torch.Tensor
    lipid_subject_indices: torch.Tensor

    @property
    def clean_metabolite_spectra(
        self,
    ) -> torch.Tensor:
        """
        Clean metabolite spectrum before acquisition-length
        simulation.
        """
        return self.metabolites.clean_spectra

    @property
    def target_spectra(
        self,
    ) -> torch.Tensor:
        """
        Final clean baseline target: water + lipids.
        """
        return self.baseline_spectra

    @property
    def metabolite_spectra(
        self,
    ) -> torch.Tensor:
        """
        Backward-compatible alias for the training target.

        The target is now the clean water-plus-lipid baseline,
        not a metabolite spectrum. This alias keeps the current
        SpectrumSimulator interface working until its naming is
        cleaned up separately.
        """
        return self.baseline_spectra

    @property
    def input_spectra(
        self,
    ) -> torch.Tensor:
        """
        Final unprojected noisy network input.

        Contains:
            metabolites + water + lipids + receiver noise
        """
        return self.mixture_spectra

    @property
    def l2_spectra(
        self,
    ) -> torch.Tensor | None:
        """
        Optional lipid-projected spectrum used as the second
        input of the legacy YNet architecture.

        Returns None when lipid projection is disabled.
        """
        return self.projected_spectra

    @property
    def batch_size(self) -> int:
        return int(
            self.mixture_spectra.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.mixture_spectra.shape[-1]
        )

    @property
    def device(self) -> torch.device:
        return self.mixture_spectra.device


def _maximum_absolute_value(
    spectra: torch.Tensor,
    *,
    description: str,
) -> torch.Tensor:
    """
    Return one maximum magnitude per spectrum.

    Output shape:
        (batch_size, 1)
    """
    if spectra.ndim != 2:
        raise ValueError(
            f"{description} must have shape (B, T), "
            f"but found {tuple(spectra.shape)}."
        )

    maximum = torch.amax(
        torch.abs(
            spectra
        ),
        dim=-1,
        keepdim=True,
    )

    invalid_rows = (
        ~torch.isfinite(maximum)
        | (maximum <= 0)
    ).squeeze(-1)

    if torch.any(
        invalid_rows
    ):
        bad_indices = torch.nonzero(
            invalid_rows,
            as_tuple=False,
        ).squeeze(-1)

        raise RuntimeError(
            f"{description} contains invalid or zero-amplitude "
            "rows at batch indices:\n"
            f"  {bad_indices.detach().cpu().tolist()}"
        )

    return maximum


def _validate_complex_batch(
    *,
    tensor: torch.Tensor,
    name: str,
    expected_shape: tuple[int, int],
    expected_device: torch.device,
) -> None:
    if tuple(
        tensor.shape
    ) != expected_shape:
        raise ValueError(
            f"Unexpected shape for {name}:\n"
            f"  expected: {expected_shape}\n"
            f"  found:    {tuple(tensor.shape)}"
        )

    if tensor.device != expected_device:
        raise ValueError(
            f"{name} is on {tensor.device}, "
            f"but expected {expected_device}."
        )

    if not torch.is_complex(
        tensor
    ):
        raise TypeError(
            f"{name} must be complex-valued."
        )


def _apply_frequency_shift_to_shifted_spectra(
    *,
    spectra: torch.Tensor,
    frequency_shifts_hz: torch.Tensor,
    bandwidth_hz: float,
) -> torch.Tensor:
    """
    Apply one frequency shift per batch item to fft-shifted spectra.

    The operation uses the same sign convention as the metabolite
    simulation:

        FID(t) -> FID(t) * exp(+i * 2*pi*delta_f*t)

    Parameters
    ----------
    spectra:
        FFT-shifted complex spectra with shape (B, T).

    frequency_shifts_hz:
        Frequency shifts with shape (B,).

    bandwidth_hz:
        Acquisition bandwidth in Hz.
    """
    if spectra.ndim != 2:
        raise ValueError(
            "spectra must have shape (B, T), "
            f"but found {tuple(spectra.shape)}."
        )

    if not torch.is_complex(
        spectra
    ):
        raise TypeError(
            "spectra must be complex-valued."
        )

    batch_size, n_timepoints = (
        spectra.shape
    )

    if tuple(
        frequency_shifts_hz.shape
    ) != (batch_size,):
        raise ValueError(
            "frequency_shifts_hz must have shape "
            f"{(batch_size,)}, but found "
            f"{tuple(frequency_shifts_hz.shape)}."
        )

    if (
        frequency_shifts_hz.device
        != spectra.device
    ):
        raise ValueError(
            "frequency_shifts_hz and spectra must be "
            "on the same device."
        )

    if not torch.isfinite(
        frequency_shifts_hz
    ).all():
        raise ValueError(
            "frequency_shifts_hz contains non-finite values."
        )

    bandwidth_hz = float(
        bandwidth_hz
    )

    if (
        not math.isfinite(bandwidth_hz)
        or bandwidth_hz <= 0
    ):
        raise ValueError(
            "bandwidth_hz must be finite and > 0."
        )

    time_axis_seconds = (
        torch.arange(
            n_timepoints,
            device=spectra.device,
            dtype=spectra.real.dtype,
        )
        / bandwidth_hz
    )

    # Input spectra are fft-shifted. Convert them back to the
    # corresponding FIDs before applying the frequency shift.
    fids = torch.fft.ifft(
        torch.fft.ifftshift(
            spectra,
            dim=-1,
        ),
        dim=-1,
    )

    phase_angles = (
        2.0
        * math.pi
        * frequency_shifts_hz[:, None]
        * time_axis_seconds[None, :]
    )

    phase_factor = torch.polar(
        torch.ones_like(
            phase_angles
        ),
        phase_angles,
    )

    shifted_fids = (
        fids
        * phase_factor
    )

    shifted_spectra = torch.fft.fftshift(
        torch.fft.fft(
            shifted_fids,
            dim=-1,
        ),
        dim=-1,
    )

    return shifted_spectra.contiguous()


def assemble_spectra(
    *,
    sampled: SampledResources,
    lipid_mixture: LipidMixture,
    metabolites: SimulatedMetabolites,
    noise: SimulatedNoise,
    pool: SimulationPool,
    config: SimulationConfig,
    generator: torch.Generator,
    acquisition_length_override: int | None = None,
) -> AssembledSpectra:
    """
    Scale and combine frequency-domain simulation components.

    Processing
    ----------
    1. Use the clean metabolite spectrum as amplitude reference.
    2. Apply the metabolite frequency shift to the sampled water.
    3. Normalize each shifted water spectrum by its own maximum.
    4. Normalize each mixed lipid spectrum by its own maximum.
    5. Sample the configured positive water and lipid scaling
       distributions, then apply both factors.
    6. Add clean metabolites, shifted water, and unshifted lipids.
    7. Build the clean water-plus-lipid baseline target.
    8. Apply the same sampled acquisition length and zero-filling
       operation to all required components.
    9. Scale and add receiver noise to the complete input mixture.
    10. Optionally apply the frequency-domain lipid projection to
        the final input.

    Water and lipids are already fft-shifted spectra when entering
    this function. The water resources are assumed to be B0-centered
    before they are stored. Lipids keep their measured frequency
    distribution and are not shifted here.
    """
    device = sampled.device

    validate_generator_device(
        generator=generator,
        device=device,
    )

    batch_size = sampled.batch_size
    n_timepoints = sampled.n_timepoints

    expected_shape = (
        batch_size,
        n_timepoints,
    )

    _validate_complex_batch(
        tensor=sampled.water_spectra,
        name="sampled.water_spectra",
        expected_shape=expected_shape,
        expected_device=device,
    )

    _validate_complex_batch(
        tensor=lipid_mixture.mixed_spectra,
        name="lipid_mixture.mixed_spectra",
        expected_shape=expected_shape,
        expected_device=device,
    )

    _validate_complex_batch(
        tensor=metabolites.clean_spectra,
        name="metabolites.clean_spectra",
        expected_shape=expected_shape,
        expected_device=device,
    )

    _validate_complex_batch(
        tensor=noise.noise_spectra,
        name="noise.noise_spectra",
        expected_shape=expected_shape,
        expected_device=device,
    )

    if metabolites.batch_size != batch_size:
        raise ValueError(
            "Metabolite batch size does not match "
            "the sampled-resource batch size."
        )

    if metabolites.n_timepoints != n_timepoints:
        raise ValueError(
            "Metabolite spectrum length does not match "
            "the sampled resources."
        )

    if noise.batch_size != batch_size:
        raise ValueError(
            "Noise batch size does not match "
            "the sampled resources."
        )

    if noise.n_timepoints != n_timepoints:
        raise ValueError(
            "Noise spectrum length does not match "
            "the sampled resources."
        )

    if pool.device != device:
        raise ValueError(
            "SimulationPool and sampled resources must use "
            "the same device."
        )

    if pool.n_timepoints != n_timepoints:
        raise ValueError(
            "SimulationPool and sampled resources have "
            "different spectrum lengths."
        )

    raw_water_spectra = (
        sampled.water_spectra
    )

    raw_lipid_spectra = (
        lipid_mixture.mixed_spectra
    )

    clean_metabolite_spectra = (
        metabolites.clean_spectra
    )

    # The sampled global frequency shift has already been applied
    # to the metabolites. Apply the identical shift to the water,
    # while leaving the spatially leaked lipid signal unchanged.
    shifted_water_spectra = (
        _apply_frequency_shift_to_shifted_spectra(
            spectra=raw_water_spectra,
            frequency_shifts_hz=(
                metabolites
                .frequency_shifts_hz
            ),
            bandwidth_hz=(
                config
                .acquisition
                .bandwidth_hz
            ),
        )
    )

    metabolite_maximum = (
        _maximum_absolute_value(
            clean_metabolite_spectra,
            description=(
                "Clean metabolite spectra"
            ),
        )
    )

    water_maximum = (
        _maximum_absolute_value(
            shifted_water_spectra,
            description=(
                "Frequency-shifted water spectra"
            ),
        )
    )

    lipid_maximum = (
        _maximum_absolute_value(
            raw_lipid_spectra,
            description="Lipid spectra",
        )
    )

    real_dtype = (
        clean_metabolite_spectra
        .real
        .dtype
    )

    water_scaling_cfg = (
        config
        .water
        .scaling
    )

    water_scaling = sample_distribution(
        distribution=water_scaling_cfg.distribution,
        shape=(batch_size,),
        device=device,
        dtype=real_dtype,
        generator=generator,
    )

    lipid_scaling_cfg = (
        config
        .lipids
        .scaling
    )

    lipid_scaling = sample_distribution(
        distribution=lipid_scaling_cfg.distribution,
        shape=(batch_size,),
        device=device,
        dtype=real_dtype,
        generator=generator,
    )


    water_spectra = (
        shifted_water_spectra
        / water_maximum
        * metabolite_maximum
        * water_scaling[:, None]
    ).contiguous()

    lipid_spectra = (
        raw_lipid_spectra
        / lipid_maximum
        * metabolite_maximum
        * lipid_scaling[:, None]
    ).contiguous()

    clean_baseline_spectra = (
        water_spectra
        + lipid_spectra
    ).contiguous()

    clean_mixture_spectra = (
        clean_metabolite_spectra
        + clean_baseline_spectra
    ).contiguous()

    # Transform all required components with exactly the same
    # sampled acquisition length.
    #
    # Channels:
    #   0: clean metabolites + water + lipids
    #   1: clean water + lipids
    #   2: clean metabolites
    #   3: unscaled receiver noise
    stacked_spectra = torch.stack(
        (
            clean_mixture_spectra,
            clean_baseline_spectra,
            clean_metabolite_spectra,
            noise.noise_spectra,
        ),
        dim=1,
    )

    acquisition_result = simulate_acquisition_length(
        spectra=stacked_spectra,
        config=config,
        generator=generator,
        acquisition_length_override=(
            acquisition_length_override
        ),
    )

    acquired_clean_mixture_spectra = (
        acquisition_result.spectra[:, 0, :]
        .contiguous()
    )

    baseline_spectra = (
        acquisition_result.spectra[:, 1, :]
        .contiguous()
    )

    acquired_metabolite_spectra = (
        acquisition_result.spectra[:, 2, :]
        .contiguous()
    )

    acquired_unscaled_noise_spectra = (
        acquisition_result.spectra[:, 3, :]
        .contiguous()
    )

    acquired_n_timepoints = (
        acquisition_result.acquired_n_timepoints
        .contiguous()
    )

    # Phase-invariant metabolite peak.
    # Shape: (batch_size, 1)
    metabolite_peak = _maximum_absolute_value(
        acquired_metabolite_spectra,
        description="Acquired clean metabolite spectra",
    )

    # LCModel-compatible noise estimate based on the RMS of the
    # real receiver-noise component.
    #
    # Shape: (batch_size, 1)
    current_noise_rms = torch.sqrt(
        torch.mean(
            acquired_unscaled_noise_spectra.real.square(),
            dim=-1,
            keepdim=True,
        )
    )

    if torch.any(
        ~torch.isfinite(current_noise_rms)
        | (current_noise_rms <= 0)
    ):
        raise RuntimeError(
            "Acquired receiver noise contains invalid RMS values."
        )

    target_noise_rms = (
        metabolite_peak
        / (
            2.0
            * noise.snr[:, None]
        )
    )

    noise_scaling = (
        target_noise_rms
        / current_noise_rms
    )

    scaled_noise_spectra = (
        acquired_unscaled_noise_spectra
        * noise_scaling
    ).contiguous()

    mixture_spectra = (
        acquired_clean_mixture_spectra
        + scaled_noise_spectra
    ).contiguous()

    projected_spectra: (
        torch.Tensor | None
    )

    if config.lipid_projection.enabled:
        operators = (
            pool
            .lipid_projection_operators
        )

        if operators is None:
            raise RuntimeError(
                "Lipid projection is enabled, but the "
                "SimulationPool contains no projection operators."
            )

        if operators.device != device:
            raise ValueError(
                "Projection operators and simulated spectra "
                "must use the same device."
            )

        expected_operator_pool_shape = (
            pool.n_subjects,
            n_timepoints,
            n_timepoints,
        )

        if (
            tuple(operators.shape)
            != expected_operator_pool_shape
        ):
            raise RuntimeError(
                "Unexpected projection-operator pool shape:\n"
                f"  expected: {expected_operator_pool_shape}\n"
                f"  found:    {tuple(operators.shape)}"
            )

        # Apply each subject-specific projection operator only once
        # to all spectra belonging to that subject.
        #
        # This avoids materializing a tensor with shape
        #
        #     (batch_size, n_timepoints, n_timepoints)
        #
        # which would require approximately 20 GB for
        # batch_size=3500, n_timepoints=840, complex64.
        projected_spectra = torch.empty_like(
            mixture_spectra
        )

        unique_subject_indices = torch.unique(
            sampled.lipid_subject_indices
        )

        for subject_index_tensor in unique_subject_indices:
            subject_index = int(
                subject_index_tensor.item()
            )

            batch_indices = torch.nonzero(
                sampled.lipid_subject_indices
                == subject_index_tensor,
                as_tuple=False,
            ).squeeze(-1)

            if batch_indices.numel() == 0:
                continue

            subject_spectra = (
                mixture_spectra.index_select(
                    0,
                    batch_indices,
                )
            )

            subject_operator = (
                operators[subject_index]
            )

            expected_operator_shape = (
                n_timepoints,
                n_timepoints,
            )

            if (
                tuple(subject_operator.shape)
                != expected_operator_shape
            ):
                raise RuntimeError(
                    "Unexpected subject projection-operator shape:\n"
                    f"  subject index: {subject_index}\n"
                    f"  expected:      {expected_operator_shape}\n"
                    f"  found:         "
                    f"{tuple(subject_operator.shape)}"
                )

            # Row-vector convention:
            #
            #     projected = spectrum @ operator
            projected_subject_spectra = (
                subject_spectra
                @ subject_operator
            )

            projected_spectra.index_copy_(
                0,
                batch_indices,
                projected_subject_spectra,
            )

        projected_spectra = (
            projected_spectra.contiguous()
        )

    else:
        projected_spectra = None

    output_tensors = {
        "water_spectra":
            water_spectra,

        "lipid_spectra":
            lipid_spectra,

        "clean_mixture_spectra":
            clean_mixture_spectra,

        "mixture_spectra":
            mixture_spectra,

        "baseline_spectra":
            baseline_spectra,
    }

    if projected_spectra is not None:
        output_tensors[
            "projected_spectra"
        ] = projected_spectra

    for name, tensor in output_tensors.items():
        if not torch.isfinite(
            tensor.real
        ).all():
            raise RuntimeError(
                f"{name} contains non-finite real values."
            )

        if not torch.isfinite(
            tensor.imag
        ).all():
            raise RuntimeError(
                f"{name} contains non-finite imaginary values."
            )

    if tuple(
        acquired_n_timepoints.shape
    ) != (batch_size,):
        raise RuntimeError(
            "Unexpected acquired_n_timepoints shape:\n"
            f"  expected: {(batch_size,)}\n"
            f"  found:    "
            f"{tuple(acquired_n_timepoints.shape)}"
        )

    if acquired_n_timepoints.device != device:
        raise RuntimeError(
            "acquired_n_timepoints is on "
            f"{acquired_n_timepoints.device}, "
            f"but expected {device}."
        )

    if torch.any(
        acquired_n_timepoints
        < config.acquisition.min_acquired_n_timepoints
    ):
        raise RuntimeError(
            "At least one sampled acquisition length is below "
            "the configured minimum."
        )

    if torch.any(
        acquired_n_timepoints
        > config.acquisition.max_acquired_n_timepoints
    ):
        raise RuntimeError(
            "At least one sampled acquisition length is above "
            "the configured maximum."
        )

    return AssembledSpectra(
        metabolites=metabolites,
        noise=noise,
        water_spectra=water_spectra,
        lipid_spectra=lipid_spectra,
        clean_mixture_spectra=(
            clean_mixture_spectra
        ),
        mixture_spectra=(
            mixture_spectra
        ),
        baseline_spectra=(
            baseline_spectra
        ),
        acquired_clean_metabolite_spectra=(
            acquired_metabolite_spectra
        ),
        acquired_n_timepoints=(
            acquired_n_timepoints
        ),
        projected_spectra=(
            projected_spectra
        ),
        water_scaling=(
            water_scaling.contiguous()
        ),
        lipid_scaling=(
            lipid_scaling.contiguous()
        ),
        water_subject_indices=(
            sampled
            .water_subject_indices
            .contiguous()
        ),
        lipid_subject_indices=(
            sampled
            .lipid_subject_indices
            .contiguous()
        ),
    )
