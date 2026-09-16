"""Batched HSVD water estimation on CPU or CUDA."""

from __future__ import annotations

import numpy as np
import torch

from walinet.training_data.water_removal import hsvd


def _water_from_components(
    fid: np.ndarray,
    *,
    sampling_frequency_hz: float,
    n_components: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> np.ndarray:
    """Run the established NumPy/SciPy HSVD path for one FID."""
    frequencies, _, basis, amplitudes = hsvd(
        y=fid,
        fs=sampling_frequency_hz,
        k=n_components,
    )
    selected = (frequencies >= min_frequency_hz) & (
        frequencies <= max_frequency_hz
    )
    return basis[:, selected] @ amplitudes[selected]


def _estimate_cpu(
    fids: np.ndarray,
    *,
    sampling_frequency_hz: float,
    n_components: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> np.ndarray:
    result = np.empty_like(fids)
    for index, fid in enumerate(fids):
        result[index] = _water_from_components(
            fid,
            sampling_frequency_hz=sampling_frequency_hz,
            n_components=n_components,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
        )
    return result


def _estimate_cuda_batch(
    fids: torch.Tensor,
    *,
    sampling_frequency_hz: float,
    n_components: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> torch.Tensor:
    """Vectorized equivalent of ``water_removal.hsvd`` for one CUDA batch."""
    batch_size, n_timepoints = fids.shape
    hankel_size = n_timepoints // 2
    if hankel_size < 2:
        raise ValueError("FIDs are too short for HSVD")
    if not 0 < n_components <= hankel_size:
        raise ValueError(
            f"n_components must be in 1..{hankel_size}, got {n_components}"
        )

    # Reproduce scipy.linalg.hankel(y[:L], y[L:N]) exactly. SciPy ignores
    # r[0] and assumes it equals c[-1]. In the established caller it is
    # actually y[L], so y[L] is skipped: anti-diagonals up to L-1 use
    # y[i+j], and all later anti-diagonals use y[i+j+1].
    row_count = hankel_size
    column_count = n_timepoints - hankel_size
    indices = (
        torch.arange(row_count, device=fids.device)[:, None]
        + torch.arange(column_count, device=fids.device)[None, :]
    )
    indices = indices + (indices >= hankel_size)
    hankel = fids[:, indices]

    u, _, _ = torch.linalg.svd(hankel, full_matrices=False)
    uk = u[:, :, :n_components]
    uk_top = uk[:, 1:, :]
    uk_bottom = uk[:, :-1, :]
    z_prime = torch.linalg.pinv(uk_bottom, rtol=1e-15) @ uk_top

    _, eigenvectors = torch.linalg.eig(z_prime)
    diagonalized = (
        torch.linalg.inv(eigenvectors) @ z_prime @ eigenvectors
    )
    q = torch.log(torch.diagonal(diagonalized, dim1=-2, dim2=-1))
    sample_interval = 1.0 / sampling_frequency_hz
    dampings = q.real / sample_interval
    dampings = torch.where(dampings > 10, 10, dampings)
    frequencies = q.imag / (2 * torch.pi * sample_interval)

    time = (
        torch.arange(n_timepoints, device=fids.device, dtype=fids.real.dtype)
        * sample_interval
    )
    exponents = dampings + 2j * torch.pi * frequencies
    basis = torch.exp(time[None, :, None] * exponents[:, None, :])
    amplitudes = (
        torch.linalg.pinv(basis, rtol=1e-15) @ fids.unsqueeze(-1)
    ).squeeze(-1)

    selected = (frequencies >= min_frequency_hz) & (
        frequencies <= max_frequency_hz
    )
    return torch.sum(
        basis * amplitudes[:, None, :] * selected[:, None, :], dim=-1
    )


def estimate_water_hsvd(
    fids: np.ndarray,
    *,
    sampling_frequency_hz: float,
    n_components: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
    device: str | torch.device = "cuda",
    batch_size: int = 256,
    cuda_dtype: torch.dtype = torch.complex64,
) -> np.ndarray:
    """Estimate water in one or more FIDs using HSVD.

    Parameters
    ----------
    fids
        Complex array with shape ``(..., n_timepoints)``.
    device
        ``"cuda"`` (the default), a concrete CUDA device such as
        ``"cuda:1"``, or ``"cpu"``. CPU uses the established scalar
        NumPy/SciPy implementation; CUDA uses a vectorized PyTorch version.
    batch_size
        Number of FIDs per CUDA decomposition batch.
    cuda_dtype
        Complex precision used on CUDA. ``torch.complex64`` is fastest;
        ``torch.complex128`` is useful for strict CPU/GPU validation.
    """
    array = np.asarray(fids)
    if array.ndim < 1:
        raise ValueError("fids must have at least one dimension")
    if not np.iscomplexobj(array):
        raise TypeError("fids must be complex-valued")
    if not np.isfinite(array).all():
        raise ValueError("fids contain NaN or Inf")
    if sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be positive")
    if min_frequency_hz > max_frequency_hz:
        raise ValueError("min_frequency_hz must not exceed max_frequency_hz")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    original_shape = array.shape
    flat = np.ascontiguousarray(array.reshape(-1, original_shape[-1]))
    requested_device = torch.device(device)
    if requested_device.type == "cpu":
        output = _estimate_cpu(
            flat,
            sampling_frequency_hz=sampling_frequency_hz,
            n_components=n_components,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
        )
    elif requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA HSVD was requested, but torch.cuda.is_available() is false. "
                "Pass device='cpu' to use the reference implementation."
            )
        if cuda_dtype not in (torch.complex64, torch.complex128):
            raise ValueError("cuda_dtype must be torch.complex64 or torch.complex128")
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(flat), batch_size):
                tensor = torch.as_tensor(
                    flat[start : start + batch_size],
                    dtype=cuda_dtype,
                    device=requested_device,
                )
                water = _estimate_cuda_batch(
                    tensor,
                    sampling_frequency_hz=sampling_frequency_hz,
                    n_components=n_components,
                    min_frequency_hz=min_frequency_hz,
                    max_frequency_hz=max_frequency_hz,
                )
                chunks.append(water.cpu().numpy())
        output = np.concatenate(chunks, axis=0) if chunks else np.empty_like(flat)
    else:
        raise ValueError("device must be 'cpu', 'cuda', or a concrete CUDA device")

    return output.astype(array.dtype, copy=False).reshape(original_shape)
