# src/walinet/training_data/distributions.py
# Fast vectorized mixture samplers without rejection loops or CUDA sync points.

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch

from walinet.config.schema_simulation import (
    LogUniformDistributionCfg,
    PositiveMixtureDistributionCfg,
    SymmetricMixtureDistributionCfg,
    UniformDistributionCfg,
)


# Fixed global mixture weights.
POSITIVE_NORMAL_WEIGHT = 0.5
POSITIVE_LOGNORMAL_WEIGHT = 0.5

SIGNED_NORMAL_WEIGHT = 0.5
SIGNED_POSITIVE_TAIL_WEIGHT = 0.25
SIGNED_NEGATIVE_TAIL_WEIGHT = 0.25


TensorParameter: TypeAlias = float | int | torch.Tensor
ShapeLike: TypeAlias = int | Sequence[int] | torch.Size
DistributionCfg: TypeAlias = (
    LogUniformDistributionCfg
    | PositiveMixtureDistributionCfg
    | SymmetricMixtureDistributionCfg
    | UniformDistributionCfg
)


# -----------------------------------------------------------------------------
# Lightweight preparation helpers
# -----------------------------------------------------------------------------


def validate_generator_device(
    *,
    generator: torch.Generator,
    device: torch.device | str,
) -> None:
    """Require generator and output tensors on the same device."""
    target_device = torch.device(device)
    generator_device = torch.device(generator.device)

    if generator_device.type != target_device.type:
        raise ValueError(
            "Generator and output tensors must use the same device type:\n"
            f"  generator: {generator_device}\n"
            f"  output:    {target_device}"
        )

    if (
        target_device.type == "cuda"
        and generator_device.index is not None
        and target_device.index is not None
        and generator_device.index != target_device.index
    ):
        raise ValueError(
            "Generator and output tensors must use the same CUDA device:\n"
            f"  generator: {generator_device}\n"
            f"  output:    {target_device}"
        )


def _normalize_shape(
    shape: ShapeLike,
) -> tuple[int, ...]:
    if isinstance(shape, int):
        normalized = (int(shape),)
    else:
        normalized = tuple(int(size) for size in shape)

    if not normalized or any(size <= 0 for size in normalized):
        raise ValueError(
            "Sampling shape must contain only dimensions > 0, "
            f"but found {normalized}."
        )

    return normalized


def _prepare_sampling_context(
    *,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[tuple[int, ...], torch.device]:
    if not dtype.is_floating_point:
        raise TypeError(
            "Distribution sampling requires a floating-point dtype, "
            f"but found {dtype}."
        )

    output_shape = _normalize_shape(shape)
    target_device = torch.device(device)

    validate_generator_device(
        generator=generator,
        device=target_device,
    )

    return output_shape, target_device


def _broadcast_parameter(
    value: TensorParameter,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    parameter_name: str,
) -> torch.Tensor:
    """
    Convert one scalar/tensor parameter and broadcast it to ``shape``.

    Numerical parameter validation belongs to the configuration/schema layer.
    Avoiding ``torch.any(...).item()`` checks here prevents a CPU/GPU
    synchronization on every simulated batch.
    """
    try:
        tensor = torch.as_tensor(
            value,
            device=device,
            dtype=dtype,
        )
        return torch.broadcast_to(
            tensor,
            shape,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            f"{parameter_name} cannot be broadcast to sampling shape "
            f"{shape}."
        ) from error


def _open_unit_uniform(
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Draw U(0, 1) while avoiding exact endpoints required by ``ndtri``.
    """
    values = torch.rand(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    finfo = torch.finfo(dtype)
    return values.clamp_(
        min=finfo.tiny,
        max=1.0 - finfo.eps,
    )


def _standard_normal_from_uniform(
    uniform: torch.Tensor,
) -> torch.Tensor:
    """Transform open-unit uniform samples into standard-normal samples."""
    return torch.special.ndtri(uniform)


def _enforce_strict_lower_bound(
    values: torch.Tensor,
    minimum: torch.Tensor,
) -> torch.Tensor:
    """Correct floating-point roundoff at an open lower boundary."""
    positive_infinity = torch.full(
        (),
        torch.inf,
        device=values.device,
        dtype=values.dtype,
    )
    smallest_valid = torch.nextafter(
        minimum,
        positive_infinity,
    )
    return torch.maximum(values, smallest_valid)


def _sample_lower_truncated_normal_from_uniform(
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    minimum: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    """
    Sample Normal(mean, std²) conditional on X > minimum via inverse CDF.

    The survival-function formulation avoids catastrophic cancellation from
    explicitly computing ``Phi(alpha) + u * (1 - Phi(alpha))`` when the lower
    truncation point lies far in the positive tail.
    """
    safe_std = torch.where(
        std > 0,
        std,
        torch.ones((), device=std.device, dtype=std.dtype),
    )

    alpha = (minimum - mean) / safe_std

    # P(Z > alpha). Multiplication by (1-u) samples uniformly from the
    # conditional survival-probability interval (0, P(Z > alpha)].
    survival_at_lower = torch.special.ndtr(-alpha)
    survival_probability = survival_at_lower * (1.0 - uniform)

    finfo = torch.finfo(uniform.dtype)
    survival_probability = survival_probability.clamp(
        min=finfo.tiny,
        max=1.0 - finfo.eps,
    )

    standard_values = -torch.special.ndtri(
        survival_probability
    )

    sampled = mean + std * standard_values

    # For deterministic components (std == 0), preserve the configured mean.
    sampled = torch.where(
        std == 0,
        mean,
        sampled,
    )

    # In exact arithmetic inverse-CDF sampling is strictly above the lower
    # bound. Float32 rounding can occasionally place a value one ULP below it.
    return _enforce_strict_lower_bound(
        sampled,
        minimum,
    )


# -----------------------------------------------------------------------------
# Primitive distributions
# -----------------------------------------------------------------------------


def sample_normal(
    *,
    mean: TensorParameter,
    std: TensorParameter,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``Normal(mean, std**2)``."""
    output_shape, target_device = _prepare_sampling_context(
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    mean_tensor = _broadcast_parameter(
        mean,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="normal mean",
    )
    std_tensor = _broadcast_parameter(
        std,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="normal std",
    )

    return (
        mean_tensor
        + std_tensor
        * torch.randn(
            output_shape,
            generator=generator,
            device=target_device,
            dtype=dtype,
        )
    ).contiguous()


def sample_truncated_normal(
    *,
    mean: TensorParameter,
    std: TensorParameter,
    minimum: TensorParameter,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_resampling_rounds: int = 1024,
) -> torch.Tensor:
    """
    Sample a lower-truncated normal distribution in one vectorized pass.

    ``max_resampling_rounds`` is retained only for API compatibility; inverse
    CDF sampling does not perform rejection rounds.
    """
    del max_resampling_rounds

    output_shape, target_device = _prepare_sampling_context(
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    mean_tensor = _broadcast_parameter(
        mean,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="truncated-normal mean",
    )
    std_tensor = _broadcast_parameter(
        std,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="truncated-normal std",
    )
    minimum_tensor = _broadcast_parameter(
        minimum,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="truncated-normal minimum",
    )

    uniform = _open_unit_uniform(
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        generator=generator,
    )

    return _sample_lower_truncated_normal_from_uniform(
        mean=mean_tensor,
        std=std_tensor,
        minimum=minimum_tensor,
        uniform=uniform,
    ).contiguous()


def sample_lognormal(
    *,
    log_mu: TensorParameter,
    log_sigma: TensorParameter,
    minimum: TensorParameter = 0.0,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_resampling_rounds: int = 1024,
) -> torch.Tensor:
    """
    Sample ``exp(N(log_mu, log_sigma**2))`` above ``minimum``.

    The truncation is performed in latent log-space with inverse CDF sampling.
    ``max_resampling_rounds`` is retained only for API compatibility.
    """
    del max_resampling_rounds

    output_shape, target_device = _prepare_sampling_context(
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    log_mu_tensor = _broadcast_parameter(
        log_mu,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="lognormal log_mu",
    )
    log_sigma_tensor = _broadcast_parameter(
        log_sigma,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="lognormal log_sigma",
    )
    minimum_tensor = _broadcast_parameter(
        minimum,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="lognormal minimum",
    )

    negative_infinity = torch.full(
        (),
        -torch.inf,
        device=target_device,
        dtype=dtype,
    )
    latent_minimum = torch.where(
        minimum_tensor > 0,
        torch.log(minimum_tensor),
        negative_infinity,
    )

    uniform = _open_unit_uniform(
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        generator=generator,
    )

    latent_values = _sample_lower_truncated_normal_from_uniform(
        mean=log_mu_tensor,
        std=log_sigma_tensor,
        minimum=latent_minimum,
        uniform=uniform,
    )

    values = torch.exp(latent_values)
    values = _enforce_strict_lower_bound(
        values,
        minimum_tensor,
    )
    return values.contiguous()


def sample_uniform(
    *,
    minimum: TensorParameter,
    maximum: TensorParameter,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample a continuous uniform distribution on ``[minimum, maximum)``."""
    output_shape, target_device = _prepare_sampling_context(
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    minimum_tensor = _broadcast_parameter(
        minimum,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="uniform minimum",
    )
    maximum_tensor = _broadcast_parameter(
        maximum,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="uniform maximum",
    )

    random_values = torch.rand(
        output_shape,
        generator=generator,
        device=target_device,
        dtype=dtype,
    )

    return (
        minimum_tensor
        + (maximum_tensor - minimum_tensor) * random_values
    ).contiguous()


# -----------------------------------------------------------------------------
# Shared calibrated mixture families
# -----------------------------------------------------------------------------


def sample_positive_mixture_parameters(
    *,
    normal_mean: TensorParameter,
    normal_std: TensorParameter,
    log_mu: TensorParameter,
    log_sigma: TensorParameter,
    minimum: TensorParameter,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_resampling_rounds: int = 1024,
) -> torch.Tensor:
    """
    Sample the shared positive mixture model:

        0.5 * TruncatedNormal(normal_mean, normal_std)
        +
        0.5 * LogNormal(log_mu, log_sigma)

    Both components use the same lower bound. Sampling is fully vectorized and
    uses no rejection loop or host-side CUDA synchronization.
    """
    del max_resampling_rounds

    output_shape, target_device = _prepare_sampling_context(
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    normal_mean_tensor = _broadcast_parameter(
        normal_mean,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="positive-mixture normal mean",
    )
    normal_std_tensor = _broadcast_parameter(
        normal_std,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="positive-mixture normal std",
    )
    log_mu_tensor = _broadcast_parameter(
        log_mu,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="positive-mixture log_mu",
    )
    log_sigma_tensor = _broadcast_parameter(
        log_sigma,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="positive-mixture log_sigma",
    )
    minimum_tensor = _broadcast_parameter(
        minimum,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="positive-mixture minimum",
    )

    component_selector = torch.rand(
        output_shape,
        generator=generator,
        device=target_device,
        dtype=dtype,
    )
    quantile = _open_unit_uniform(
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        generator=generator,
    )

    normal_values = _sample_lower_truncated_normal_from_uniform(
        mean=normal_mean_tensor,
        std=normal_std_tensor,
        minimum=minimum_tensor,
        uniform=quantile,
    )

    negative_infinity = torch.full(
        (),
        -torch.inf,
        device=target_device,
        dtype=dtype,
    )
    latent_minimum = torch.where(
        minimum_tensor > 0,
        torch.log(minimum_tensor),
        negative_infinity,
    )

    lognormal_latent = _sample_lower_truncated_normal_from_uniform(
        mean=log_mu_tensor,
        std=log_sigma_tensor,
        minimum=latent_minimum,
        uniform=quantile,
    )
    lognormal_values = torch.exp(lognormal_latent)
    lognormal_values = _enforce_strict_lower_bound(
        lognormal_values,
        minimum_tensor,
    )

    values = torch.where(
        component_selector < POSITIVE_NORMAL_WEIGHT,
        normal_values,
        lognormal_values,
    )
    return _enforce_strict_lower_bound(
        values,
        minimum_tensor,
    ).contiguous()


def sample_positive_mixture(
    *,
    distribution: PositiveMixtureDistributionCfg,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_resampling_rounds: int = 1024,
) -> torch.Tensor:
    return sample_positive_mixture_parameters(
        normal_mean=distribution.normal.mean,
        normal_std=distribution.normal.std,
        log_mu=distribution.lognormal.log_mu,
        log_sigma=distribution.lognormal.log_sigma,
        minimum=distribution.minimum,
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
        max_resampling_rounds=max_resampling_rounds,
    )


def sample_log_uniform(
    *,
    distribution: LogUniformDistributionCfg,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample values uniformly in logarithmic space."""
    log_values = sample_uniform(
        minimum=torch.log(
            torch.tensor(distribution.minimum, dtype=dtype, device=device)
        ),
        maximum=torch.log(
            torch.tensor(distribution.maximum, dtype=dtype, device=device)
        ),
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return torch.exp(log_values).contiguous()


def sample_symmetric_mixture_parameters(
    *,
    center: TensorParameter,
    normal_std: TensorParameter,
    tail_log_mu: TensorParameter,
    tail_log_sigma: TensorParameter,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_resampling_rounds: int = 1024,
) -> torch.Tensor:
    """
    Sample the signed mixture model:

        0.50 * Normal(center, normal_std)
        + 0.25 * (center + LogNormal(tail_log_mu, tail_log_sigma))
        + 0.25 * (center - LogNormal(tail_log_mu, tail_log_sigma))

    One selector determines normal core versus either tail. A second random
    tensor supplies the conditional quantile, so the complete mixture requires
    only two random tensors and no rejection loop.
    """
    del max_resampling_rounds

    output_shape, target_device = _prepare_sampling_context(
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    center_tensor = _broadcast_parameter(
        center,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="symmetric-mixture center",
    )
    normal_std_tensor = _broadcast_parameter(
        normal_std,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="symmetric-mixture normal std",
    )
    tail_log_mu_tensor = _broadcast_parameter(
        tail_log_mu,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="symmetric-mixture tail log_mu",
    )
    tail_log_sigma_tensor = _broadcast_parameter(
        tail_log_sigma,
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        parameter_name="symmetric-mixture tail log_sigma",
    )

    component_selector = torch.rand(
        output_shape,
        generator=generator,
        device=target_device,
        dtype=dtype,
    )
    quantile = _open_unit_uniform(
        shape=output_shape,
        device=target_device,
        dtype=dtype,
        generator=generator,
    )

    standard_values = _standard_normal_from_uniform(
        quantile
    )

    normal_values = (
        center_tensor
        + normal_std_tensor * standard_values
    )

    tail_magnitudes = torch.exp(
        tail_log_mu_tensor
        + tail_log_sigma_tensor * standard_values
    )

    tail_signs = torch.where(
        component_selector
        < (
            SIGNED_NORMAL_WEIGHT
            + SIGNED_POSITIVE_TAIL_WEIGHT
        ),
        torch.ones((), device=target_device, dtype=dtype),
        -torch.ones((), device=target_device, dtype=dtype),
    )

    tail_values = center_tensor + tail_signs * tail_magnitudes

    return torch.where(
        component_selector < SIGNED_NORMAL_WEIGHT,
        normal_values,
        tail_values,
    ).contiguous()


def sample_symmetric_mixture(
    *,
    distribution: SymmetricMixtureDistributionCfg,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_resampling_rounds: int = 1024,
) -> torch.Tensor:
    return sample_symmetric_mixture_parameters(
        center=distribution.center,
        normal_std=distribution.normal_std,
        tail_log_mu=distribution.lognormal_tail.log_mu,
        tail_log_sigma=distribution.lognormal_tail.log_sigma,
        shape=shape,
        device=device,
        dtype=dtype,
        generator=generator,
        max_resampling_rounds=max_resampling_rounds,
    )


def sample_distribution(
    *,
    distribution: DistributionCfg,
    shape: ShapeLike,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_resampling_rounds: int = 1024,
) -> torch.Tensor:
    """Dispatch one validated simulation distribution configuration."""
    if isinstance(distribution, UniformDistributionCfg):
        return sample_uniform(
            minimum=distribution.minimum,
            maximum=distribution.maximum,
            shape=shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    if isinstance(distribution, LogUniformDistributionCfg):
        return sample_log_uniform(
            distribution=distribution,
            shape=shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    if isinstance(
        distribution,
        PositiveMixtureDistributionCfg,
    ):
        return sample_positive_mixture(
            distribution=distribution,
            shape=shape,
            device=device,
            dtype=dtype,
            generator=generator,
            max_resampling_rounds=max_resampling_rounds,
        )

    if isinstance(
        distribution,
        SymmetricMixtureDistributionCfg,
    ):
        return sample_symmetric_mixture(
            distribution=distribution,
            shape=shape,
            device=device,
            dtype=dtype,
            generator=generator,
            max_resampling_rounds=max_resampling_rounds,
        )

    raise TypeError(
        "Unsupported distribution configuration type: "
        f"{type(distribution).__name__}."
    )
