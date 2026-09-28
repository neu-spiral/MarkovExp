import numpy as np
import os
import sys
import shutil
import copy
from scipy.stats import ortho_group
from scipy.linalg import expm
from sklearn.model_selection import train_test_split
import torch

from source.models.model import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def create_data(d, dbeta, p, n, sigmaw, sigma, covconstant, stages, path, num_experiments, random_pool, data_type, dataset):
    save_path = os.path.join(path, f"d{d}_dbeta{dbeta}_p{p}_n{n}_sigmaw{sigmaw}_sigma{sigma}_covconstant{covconstant}_stages{stages}_experiments{num_experiments}_random_pool{random_pool}.npz")
    if not os.path.exists(save_path):
        print(f"\nCreating data file at {save_path}\n")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if data_type == "real":
            if dataset == "beijing":
                X = np.load(os.path.join("dataset", f"BeijingPM25_X_y.npz"))['X']
                y = np.load(os.path.join("dataset", f"BeijingPM25_X_y.npz"))['y']
            elif dataset == "sp500":
                X = np.load(os.path.join("dataset", f"SP500_X_y.npz"))['X']
                y = np.load(os.path.join("dataset", f"SP500_X_y.npz"))['y']
            elif dataset == "bike":
                X = np.load(os.path.join("dataset", f"BikeSharing_X_y.npz"))['X']
                y = np.load(os.path.join("dataset", f"BikeSharing_X_y.npz"))['y']
            elif dataset == "california":
                X = np.load(os.path.join("dataset", f"CaliforniaHousing_X_y.npz"))['X']
                y = np.load(os.path.join("dataset", f"CaliforniaHousing_X_y.npz"))['y']
            elif dataset == "atnt_dl":
                X = np.load(os.path.join("dataset", f"ATNT_DL_X_y.npz"))['X']
                y = np.load(os.path.join("dataset", f"ATNT_DL_X_y.npz"))['y']
            elif dataset == "atnt_ul":
                X = np.load(os.path.join("dataset", f"ATNT_UL_X_y.npz"))['X']
                y = np.load(os.path.join("dataset", f"ATNT_UL_X_y.npz"))['y']
            y_pool = np.empty((num_experiments, stages, round(y.shape[1] / 1.25)))  # Placeholder for y_pool
            y_pool_test = np.empty((num_experiments, stages, round(y.shape[1] / 5)))  # Placeholder for y_test
            X_pool = np.empty((num_experiments, stages, round(X.shape[1] / 1.25), d))  # Placeholder for X_pool
            X_pool_test = np.empty((num_experiments, stages, round(X.shape[1] / 5), d))  # Placeholder for X_test
        else:
            X_pool = np.empty((num_experiments, stages, p, d))  # Placeholder for X_pool
            X_pool_test = np.empty((num_experiments, stages, round(n/4), d))  # Placeholder for X_test
        e_noise = np.empty((num_experiments, stages, n))  # Placeholder for e_noise
        test_noise = np.empty((num_experiments, stages, round(n/4)))  # Placeholder for test_noise
        w_noise = np.empty((num_experiments, stages, dbeta))  # Placeholder for w_noise
        beta_0 = np.empty((num_experiments, dbeta))
        cov_0 = np.empty((num_experiments, dbeta, dbeta))
        beta_1 = np.empty((num_experiments, dbeta))
        beta_ast = np.empty((num_experiments, dbeta))

        for i in range(num_experiments):
            if data_type == "simulation":
                a = np.random.randn(p, d) if not random_pool else None
                for s in range(stages):
                    if random_pool:
                        X_pool[i, s] = np.random.randn(p, d)
                    else:
                        X_pool[i, s] = copy.deepcopy(a)

                    X_pool_test[i, s] = np.random.randn(round(n/4), d)
            else:
                for s in range(stages):
                    X_total = copy.deepcopy(X[s])
                    y_total = copy.deepcopy(y[s])
                    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, shuffle=False)
                    X_pool[i, s] = X_train
                    X_pool_test[i, s] = X_test
                    y_pool[i, s] = y_train
                    y_pool_test[i, s] = y_test

            e_noise[i] = np.random.randn(stages, n) * sigma
            test_noise[i] = np.random.randn(stages, round(n/4)) * sigma
            w_noise[i] = np.random.randn(stages, dbeta) * sigmaw

            beta_0[i] = np.random.randn(dbeta)

            beta_ast[i] = np.random.randn(dbeta)

            cov_0[i] = covconstant * np.eye(dbeta)

            beta_1[i] = np.random.multivariate_normal(beta_0[i], cov_0[i])
        
        if data_type == "simulation":
            np.savez(os.path.join(save_path), X_pool=X_pool, X_test=X_pool_test, e_noise=e_noise, test_noise=test_noise, w_noise=w_noise, beta_0=beta_0, cov_0=cov_0, beta_1=beta_1, beta_ast=beta_ast)
        else:
            np.savez(os.path.join(save_path), X_pool=X_pool, X_test=X_pool_test, y_pool=y_pool, y_test=y_pool_test, e_noise=e_noise, test_noise=test_noise, w_noise=w_noise, beta_0=beta_0, cov_0=cov_0, beta_1=beta_1, beta_ast=beta_ast)
    else:
        print(f"\nData file already exists at {save_path}, skipping creation.\n")

def simulate_true_beta_drift(true_beta, w_k, rho, beta_ast):
    """
    Simulate the drift of the true beta over stages.
    """
    return rho * true_beta + (1-rho) * beta_ast + w_k

def fischer_information(pi, X_pool):
    """Compute the Fisher information matrix B(π)"""
    return X_pool.T @ np.diag(pi) @ X_pool

def _pool_jacobian_G(model, X_pool: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(np.asarray(X_pool)).float()
    beta_t = torch.from_numpy(np.asarray(beta_hat).reshape(-1)).float()
    G_t = model.jacobian_wrt_beta(X_t, beta_t)
    return G_t.detach().cpu().numpy()

def objective_fractional_stage(pi, Sigma, X_pool, n, sigma, beta_hat, model):
    Sigma_inv = np.linalg.inv(Sigma)

    if beta_hat is None:
        # Linear
        B_pi = fischer_information(pi, X_pool)
    else:
        if model is None:
            raise ValueError("Nonlinear objective requested (beta_hat not None) but model is None.")
        G = _pool_jacobian_G(model, X_pool, beta_hat)   # (p, d_beta)
        B_pi = G.T @ (G * pi[:, None])                  # (d_beta, d_beta)

    M = (n / (sigma**2)) * B_pi + Sigma_inv
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        raise np.linalg.LinAlgError("M not SPD in objective_fractional_stage.")
    return float(logdet)


def objective_regular_stage(X_pool, n_k, sigma, Sigma, beta_hat, model):
    n_k = np.asarray(n_k).reshape(-1)
    Sigma_inv = np.linalg.inv(Sigma)

    if beta_hat is None:
        X = np.asarray(X_pool)
        B_ni = X.T @ (X * n_k[:, None])                 # X^T diag(n_k) X
    else:
        if model is None:
            raise ValueError("Nonlinear objective requested (beta_hat not None) but model is None.")
        G = _pool_jacobian_G(model, X_pool, beta_hat)   # (p, d_beta)
        B_ni = G.T @ (G * n_k[:, None])                 # G^T diag(n_k) G

    M = (1.0 / (sigma**2)) * B_ni + Sigma_inv
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        raise np.linalg.LinAlgError("M not SPD in objective_regular_stage.")
    return float(logdet)

def round_n_vector(n_vector, n):
    """
    Round a vector of numbers to the nearest integers, ensuring the sum equals n.
    """
    while sum(np.round(n_vector).astype(int)) - n != 0:
        diff = sum(np.round(n_vector).astype(int)) - n
        max_index = np.argmax(np.abs(np.round(n_vector).astype(int) - n_vector))

        if diff > 0:
            n_vector[max_index] -= 1
        elif diff < 0:
            n_vector[max_index] += 1
        n_vector[max_index] = np.round(n_vector[max_index])
        if n_vector[max_index] < 0:
            n_vector[max_index] = 0
    
    return np.round(n_vector).astype(int)

def collect_data(X_pool, n_k):
    """
    Select data from X_pool based on the allocation in n_k.
    """
    indices = np.repeat(np.arange(X_pool.shape[0]), n_k)
    # print(f"Collecting data with indices: {indices}")
    selected_data = X_pool[indices, :]
    
    return selected_data

def collect_real_labels(y_pool, n_k):
    """
    Select data from y_pool based on the allocation in n_k.
    """
    indices = np.repeat(np.arange(y_pool.shape[0]), n_k)
    # print(f"Collecting data with indices: {indices}")
    selected_data = y_pool[indices]

    return selected_data

def collect_labels(X_k, true_beta, e_k, model):
    """
    Collect labels for the selected data X_k based on the true beta and noise epsilon.
    """
    if model is None:
        # Linear case: y = X beta + noise
        y_k = X_k @ true_beta + e_k
    else:
        X_t = torch.from_numpy(np.asarray(X_k)).float()
        beta_t = torch.from_numpy(np.asarray(true_beta).reshape(-1)).float()

        with torch.no_grad():
            y_pred = model.forward(X_t, beta_t)  # (n_k,)

        y_k = y_pred.cpu().numpy() + e_k

    return y_k

def estimate_beta_star_cumulative(
    model,
    X_cum: np.ndarray,
    y_cum: np.ndarray,
    beta_star_hat_prev: np.ndarray = None,
    *,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    standardize: bool = True,
    grad_clip: float = 1.0,
    huber_delta: float = 0.0,
    init_from_model: bool = True,
):

    model.eval()

    # ---- numpy -> torch
    X_np = np.asarray(X_cum, dtype=np.float32)
    y_np = np.asarray(y_cum, dtype=np.float32).reshape(-1)

    if standardize:
        X_mu = X_np.mean(axis=0, keepdims=True)
        X_std = X_np.std(axis=0, keepdims=True) + 1e-8
        y_mu = y_np.mean()
        y_std = y_np.std() + 1e-8
        X_np = (X_np - X_mu) / X_std
        y_np = (y_np - y_mu) / y_std

    X_t = torch.from_numpy(X_np).to(device)
    y_t = torch.from_numpy(y_np).to(device)

    d_beta = model.num_params()

    if beta_star_hat_prev is not None:
        beta0 = np.asarray(beta_star_hat_prev, dtype=np.float32).reshape(-1)
        assert beta0.size == d_beta, f"beta_star_hat_prev len {beta0.size} != d_beta {d_beta}"
        beta = torch.tensor(beta0, device=device, requires_grad=True)
    else:
        if init_from_model:
            beta0 = torch.from_numpy(model.flatten_beta()).float().to(device)
            assert beta0.numel() == d_beta
            beta = beta0.clone().detach().requires_grad_(True)
        else:
            beta = torch.zeros(d_beta, dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.AdamW([beta], lr=lr, weight_decay=weight_decay)

    N = X_t.shape[0]
    bs = min(batch_size, N)

    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            X_b = X_t[idx]
            y_b = y_t[idx]

            y_pred = model.forward(X_b, beta)  # (bs,)
            resid = (y_b - y_pred)

            if huber_delta and huber_delta > 0:
                per_sample = torch.nn.functional.smooth_l1_loss(
                    y_pred, y_b, reduction="none"
                )
            else:
                per_sample = resid ** 2

            loss = per_sample.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()

            assert beta.grad is not None and torch.isfinite(beta.grad).all()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([beta], max_norm=float(grad_clip))

            opt.step()

            epoch_loss += loss.item() * X_b.shape[0]

    beta_star_hat = beta.detach().cpu().numpy()
    return beta_star_hat



def track_results(results, beta_hat_update, beta_hat_predict, beta_ast, beta_ast_est, true_beta, X_pool, X_test, y_test, obj, n_k, Sigma_update, Sigma_predict, sigma, model):
    """
    Track the results of the simulation.
    """
    beta_error = np.linalg.norm(beta_hat_update - true_beta)
    results['beta_error'].append(beta_error)

    # --- Prediction ---
    if model is None:
        # Linear case
        y_pred = X_test @ beta_hat_update
    else:
        # Nonlinear case: y = h(X, beta)
        model.eval()
        X_t = torch.from_numpy(np.asarray(X_test)).float()
        beta_t = torch.from_numpy(np.asarray(beta_hat_update).reshape(-1)).float()
        with torch.no_grad():
            y_pred = model.forward(X_t, beta_t).cpu().numpy()

    pred_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    results['pred_rmse'].append(pred_rmse)

    results['objective'].append(obj)

    beta_ast_est_diff = np.linalg.norm(beta_ast_est - beta_ast)
    results['beta_ast_est_diff'].append(beta_ast_est_diff)

    if model is not None:
        model.eval()
        X_t = torch.from_numpy(np.asarray(X_test)).float()
        beta_ast_est_t = torch.from_numpy(np.asarray(beta_ast_est).reshape(-1)).float()
        beta_ast_t = torch.from_numpy(np.asarray(beta_ast).reshape(-1)).float()

        beta_hat_t = torch.from_numpy(np.asarray(beta_hat_update).reshape(-1)).float()
        true_beta_t = torch.from_numpy(np.asarray(true_beta).reshape(-1)).float()
        with torch.no_grad():
            y_pred_ast_est = model.forward(X_t, beta_ast_est_t)
            y_pred_ast = model.forward(X_t, beta_ast_t)

            y_pred_hat = model.forward(X_t, beta_hat_t)
            y_pred_true = model.forward(X_t, true_beta_t)
        diff_ast_est = np.linalg.norm((y_pred_ast_est - y_pred_ast).detach().cpu().numpy())
        results['nonlinear_beta_ast_est_diff'].append(diff_ast_est)
        
        diff_beta_hat = np.linalg.norm((y_pred_hat - y_pred_true).detach().cpu().numpy())
        results['nonlinear_beta_error'].append(diff_beta_hat)

    # return results