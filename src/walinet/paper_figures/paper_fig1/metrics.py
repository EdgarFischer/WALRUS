"""Per-spectrum metrics for nuisance-removal evaluation."""

from __future__ import annotations

import torch


NUISANCE_RESIDUAL_METRIC = "residual_l2_over_nuisance_l2"
METABOLITE_RESIDUAL_METRIC = "residual_l2_over_clean_metabolite_l2"
REFERENCE_RATIO_METRIC = "nuisance_l2_over_clean_metabolite_l2"
PREDICTION_METRICS = (NUISANCE_RESIDUAL_METRIC, METABOLITE_RESIDUAL_METRIC)
ALL_METRICS = (*PREDICTION_METRICS, REFERENCE_RATIO_METRIC)


def channels_to_complex(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[1] < 2:
        raise ValueError(f"Expected (B, >=2, T), got {tuple(value.shape)}")
    return torch.complex(value[:, 0, :], value[:, 1, :])


def relative_l2(
    numerator_value: torch.Tensor,
    denominator_value: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    numerator = torch.sum(torch.abs(numerator_value).square(), dim=-1)
    denominator = torch.sum(torch.abs(denominator_value).square(), dim=-1)
    return torch.sqrt(numerator / denominator.clamp_min(eps))


def calculate_errors(
    *,
    true_nuisance: torch.Tensor,
    predicted_nuisance: torch.Tensor,
    clean_metabolite: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return two residual metrics and the unprocessed nuisance reference.

    All three metrics use the acquired, clean simulated components. The two
    prediction metrics share the residual numerator ``n_hat - n_true``.
    """
    truth = channels_to_complex(true_nuisance)
    prediction = channels_to_complex(predicted_nuisance)
    metabolite = (
        channels_to_complex(clean_metabolite)
        if not torch.is_complex(clean_metabolite)
        else clean_metabolite
    )
    if metabolite.ndim != 2 or metabolite.shape != truth.shape:
        raise ValueError(
            f"clean_metabolite shape {tuple(metabolite.shape)} != {tuple(truth.shape)}"
        )
    residual = prediction - truth
    return (
        relative_l2(residual, truth),
        relative_l2(residual, metabolite),
        relative_l2(truth, metabolite),
    )

