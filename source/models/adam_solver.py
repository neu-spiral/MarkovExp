from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

import torch

from source.utils.misc import round_n_vector


def project_onto_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    if z <= 0:
        return np.zeros_like(v)
    v_pos = np.maximum(v, 0.0)
    s = v_pos.sum()
    if s <= z + 1e-12:
        return v_pos

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * (np.arange(1, v.size + 1)) > (cssv - z))[0]
    if rho_idx.size == 0:
        theta = 0.0
    else:
        rho = rho_idx[-1]
        theta = (cssv[rho] - z) / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def build_M(pi: np.ndarray, Sigma: np.ndarray, A: np.ndarray, n: int, sigma: float) -> np.ndarray:
    """
    Generic: M(pi) = (n/sigma^2) * A^T diag(pi) A + Sigma^{-1}.
    - Linear case: A = X_pool (p x d_x)
    - Nonlinear case: A = G(beta_hat) (p x d_beta)
    """
    AtW_A = A.T @ (A * pi[:, None])  # A^T diag(pi) A
    return (n / (sigma ** 2)) * AtW_A + np.linalg.inv(Sigma)


def logdet_and_grad(
    pi: np.ndarray,
    Sigma: np.ndarray,
    A: np.ndarray,
    n: int,
    sigma: float,
    batch_idx: Optional[np.ndarray] = None,
    unbiased_scale: bool = True,
) -> Tuple[float, np.ndarray]:
    p, dA = A.shape
    M = build_M(pi, Sigma, A, n, sigma)

    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        raise np.linalg.LinAlgError("M is not SPD; check Sigma and inputs.")

    if batch_idx is None:
        B = np.arange(p)
    else:
        B = np.asarray(batch_idx, dtype=int)

    AB = A[B, :]                   # (|B|, dA)
    Y = np.linalg.solve(M, AB.T)   # (dA, |B|)

    qforms = np.einsum("ij,ij->i", AB, Y.T)

    grad = np.zeros(p, dtype=float)
    scale = n / (sigma ** 2)
    if batch_idx is None:
        grad[:] = scale * qforms
    else:
        if unbiased_scale and len(B) > 0:
            grad[B] = (p / len(B)) * scale * qforms
        else:
            grad[B] = scale * qforms

    return float(logdet), grad


def compute_G_from_model(model, X_pool: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(np.asarray(X_pool)).float()
    beta_t = torch.from_numpy(np.asarray(beta_hat).reshape(-1)).float()
    H_t = model.jacobian_wrt_beta(X_t, beta_t)
    G = H_t.detach().cpu().numpy()
    return G


def ADAMOptimize(
    Sigma: np.ndarray,
    X_pool: np.ndarray,
    beta_hat: np.ndarray,
    model,
    n: int,
    sigma: float,
    *,
    max_iter: int = 500,
    batch_size: Optional[int] = None,
    step_size: float = 0.05,
    tol: float = 1e-5,
    patience: int = 20,
    seed: Optional[int] = None,
    optimizer: str = "adam",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Projected (stochastic) gradient ascent over {pi >= 0, sum pi <= 1}.
    - Linear: model is None OR beta_hat is None -> uses A = X_pool.
    - Nonlinear: uses A = G(beta_hat) from model.jacobian_wrt_beta.
    Returns (n_k_rounded, best_logdet, pi_best).
    """
    rng = np.random.default_rng(seed)
    p = X_pool.shape[0]

    # Choose design matrix A
    if beta_hat is None or model is None:
        A = np.asarray(X_pool, dtype=float)            # (p, d_x)
    else:
        A = compute_G_from_model(model, X_pool, beta_hat)  # (p, d_beta)

    pi = np.full(p, 1.0 / p, dtype=float)

    m = np.zeros_like(pi)
    v = np.zeros_like(pi)

    best_obj = -np.inf
    best_pi = pi.copy()
    no_improve = 0
    last_obj = None

    for t in range(1, max_iter + 1):
        if batch_size is None or batch_size >= p:
            batch_idx = None
        else:
            batch_idx = rng.choice(p, size=batch_size, replace=False)

        obj, grad = logdet_and_grad(pi, Sigma, A, n, sigma, batch_idx=batch_idx, unbiased_scale=True)

        # Ascent step
        if optimizer.lower() == "adam":
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            step = step_size * m_hat / (np.sqrt(v_hat) + eps)
            pi_new = pi + step
        else:
            eta = step_size / np.sqrt(t)
            pi_new = pi + eta * grad

        pi = project_onto_simplex(pi_new, z=1.0)

        if obj > best_obj + 1e-12:
            best_obj = obj
            best_pi = pi.copy()
            no_improve = 0
        else:
            no_improve += 1

        if last_obj is not None:
            rel_impr = (obj - last_obj) / (abs(last_obj) + 1.0)
            if rel_impr < tol and no_improve >= patience:
                if verbose:
                    print(f"[ADAM] Early stop at iter {t} obj={obj:.6f}")
                break
        last_obj = obj

        if verbose and (t % 50 == 0 or t == 1):
            print(f"[ADAM] it={t:4d} obj={obj:.6f} sum(pi)={pi.sum():.4f}")

    pi_best = best_pi
    n_k = n * pi_best
    n_k_rounded = round_n_vector(n_k, n)

    # Final objective
    sign, final_obj = np.linalg.slogdet(build_M(pi_best, Sigma, A, n, sigma))
    if sign <= 0:
        raise np.linalg.LinAlgError("Final M not SPD at best iterate.")

    return n_k_rounded, float(final_obj)
