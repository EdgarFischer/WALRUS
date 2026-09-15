import logging
import numpy as np
import torch


DEFAULT_TRACE_TOLERANCE = 1e-4


def _resolve_method(method: str) -> str:
    if method == "auto":
        return "gpu" if torch.cuda.is_available() else "eigenvalue"
    if method not in {"eigenvalue", "gpu", "legacy"}:
        raise ValueError(
            "method must be 'auto', 'eigenvalue', 'gpu', or 'legacy'."
        )
    return method


def diag_mean(beta: float, M: np.ndarray) -> float:
    """Mean absolute diagonal of (I + beta * M)^-1."""
    identity = np.eye(M.shape[0], dtype=M.dtype)
    rem_op = np.linalg.inv(identity + beta * M)
    return float(np.mean(np.abs(np.diag(rem_op))))


def find_beta_bisect(
    lipid_rf: np.ndarray,
    target: float = 0.938,
    tol: float = DEFAULT_TRACE_TOLERANCE,
    max_iter: int = 60,
    method: str = "auto",
    gpu_index: int = 0,
):
    """
    Bracketing/bisection for beta such that
    abs(diag_mean(beta) - target) <= tol.

    CAUTION: Do not change target=0.938 unless you know what you are doing,
    because WALINET was trained with this value.
    """
    method = _resolve_method(method)

    if method == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "method='gpu' requires an available CUDA GPU."
            )
        if not 0 <= gpu_index < torch.cuda.device_count():
            raise ValueError(
                f"gpu_index={gpu_index} is unavailable; found "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
        device = torch.device(f"cuda:{gpu_index}")
        lipid_gpu = torch.as_tensor(
            lipid_rf,
            dtype=torch.complex128,
            device=device,
        )
        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            matrix_gpu = lipid_gpu.mH @ lipid_gpu
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        eigenvalues_gpu = torch.linalg.eigvalsh(matrix_gpu)
        evaluate = lambda beta: float(
            torch.mean(1.0 / (1.0 + beta * eigenvalues_gpu)).item()
        )
        M = matrix_gpu.cpu().numpy()
    else:
        M = lipid_rf.conj().T @ lipid_rf

    # Analytic endpoint: diag_mean(0) = 1 exactly.
    if target == 1.0:
        return 0.0, M

    if method == "legacy":
        evaluate = lambda beta: diag_mean(beta, M)
    elif method == "eigenvalue":
        eigenvalues = np.linalg.eigvalsh(M)
        evaluate = lambda beta: float(
            np.mean(1.0 / (1.0 + beta * eigenvalues))
        )

    beta_low = 0.0
    beta_high = 1e-10

    g_high = evaluate(beta_high)

    while g_high > target:
        beta_high *= 2.0
        g_high = evaluate(beta_high)

        if beta_high > 1e12:
            raise RuntimeError(
                "Bracket search failed. Target may be larger than diag_mean(0)."
            )

    for _ in range(max_iter):
        beta_mid = 0.5 * (beta_low + beta_high)
        g_mid = evaluate(beta_mid)

        if abs(g_mid - target) <= tol:
            return beta_mid, M

        if g_mid > target:
            beta_low = beta_mid
        else:
            beta_high = beta_mid

    raise RuntimeError("Desired tolerance was not reached.")


def compute_lipid_projection_operator(
    spectra: np.ndarray,
    lipid_mask: np.ndarray,
    max_n_iter: int = 60,
    target: float = 0.938,
    tol: float = DEFAULT_TRACE_TOLERANCE,
    method: str = "auto",
    gpu_index: int = 0,
):
    """
    Args:
        spectra: (X, Y, Z, T) complex spectra, already FFT-transformed.
        lipid_mask: (X, Y, Z) boolean or 0/1 lipid mask.

    Returns:
        LipidProj_Operator_ff: (T, T)
    """
    method = _resolve_method(method)
    T = spectra.shape[-1]

    data_rf = spectra.reshape(-1, T)
    lipid_rf = data_rf[lipid_mask.flatten() > 0, :]

    # For target 1, beta=0, rem_op=I, and therefore the lipid projection
    # I-rem_op is analytically the exact zero operator.  Returning it directly
    # avoids replacing this well-defined endpoint by a tolerance-level value.
    if target == 1.0:
        lipid_proj_operator_ff = np.zeros((T, T), dtype=lipid_rf.dtype)
        msg = "Achieved diag-mean = 1.0000 at beta = 0.00e+00 (analytic)"
        logging.info(msg)
        print(msg)
        return lipid_proj_operator_ff

    beta, M = find_beta_bisect(
        lipid_rf,
        target=target,
        tol=tol,
        max_iter=max_n_iter,
        method=method,
        gpu_index=gpu_index,
    )

    if method == "gpu":
        device = torch.device(f"cuda:{gpu_index}")
        matrix_gpu = torch.as_tensor(M, device=device)
        identity_gpu = torch.eye(T, dtype=matrix_gpu.dtype, device=device)
        rem_op_gpu = torch.linalg.inv(identity_gpu + beta * matrix_gpu)
        lipid_proj_operator_ff = (
            identity_gpu - rem_op_gpu
        ).cpu().numpy().astype(
            lipid_rf.dtype,
            copy=False,
        )
        rem_op = rem_op_gpu.cpu().numpy()
    else:
        identity = np.eye(T, dtype=M.dtype)
        rem_op = np.linalg.inv(identity + beta * M)
        lipid_proj_operator_ff = identity - rem_op

    if method == "legacy":
        fac = diag_mean(beta, M)
    else:
        fac = float(np.mean(np.abs(np.diag(rem_op))))
    msg = f"Achieved diag-mean = {fac:.4f} at beta = {beta:.2e}"
    logging.info(msg)
    print(msg)

    return lipid_proj_operator_ff
