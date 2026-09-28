import numpy as np
import copy
from scipy.optimize import minimize
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from source.utils.misc import round_n_vector, objective_fractional_stage, objective_regular_stage


def _slogdet_spd(M: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(M)
    if sign <= 0:
        raise np.linalg.LinAlgError("Matrix not SPD for logdet.")
    return float(val)


def _B_from_A(pi: np.ndarray, A: np.ndarray) -> np.ndarray:
    """B(pi) = A^T diag(pi) A"""
    pi = np.asarray(pi).reshape(-1)
    return A.T @ (A * pi[:, None])


def _G_from_model(model, X_pool: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(np.asarray(X_pool)).float()
    beta_t = torch.from_numpy(np.asarray(beta_hat).reshape(-1)).float()
    H_t = model.jacobian_wrt_beta(X_t, beta_t)   # expected (p, dbeta)
    return H_t.detach().cpu().numpy()


class ScipyMinimize:
    def __init__(
        self,
        d: int,
        dbeta: int,
        p: int,
        rho: float,
        sigmaw: float,
        sigma: float,
        covconstant: float,
        n: int,
        stages: int,
        X_pool,
        X_pool_full,
        Sigma: np.ndarray,
        beta_hat,
        model=None,
        steady_max_iter: int = 200,
        steady_tol: float = 1e-7,
        reg_eps: float = 1e-8,
    ):
        self.d = int(d)               # feature dim
        self.dbeta = int(dbeta)       # parameter dim
        self.p = int(p)
        self.rho = float(rho)
        self.sigmaw = float(sigmaw)
        self.sigma = float(sigma)
        self.covconstant = float(covconstant)
        self.n = int(n)
        self.stages = int(stages)
        self.X_pool = X_pool
        self.X_pool_full = X_pool_full
        self.Sigma = Sigma
        self.beta_hat = beta_hat
        self.model = model
        self.steady_max_iter = steady_max_iter
        self.steady_tol = steady_tol
        self.reg_eps = reg_eps

        self._A_cache = None
        self._A_cache_id = None


    def _is_nonlinear(self) -> bool:
        return self.beta_hat is not None

    def _dim_design(self) -> int:
        return self.dbeta if self._is_nonlinear() else self.d

    def _get_A(self, X_pool: np.ndarray) -> np.ndarray:
        """
        Linear:    A = X_pool            shape (p, d)
        Nonlinear: A = G_pool(beta_hat)  shape (p, dbeta)
        """
        key = (id(X_pool), self._is_nonlinear(), id(self.beta_hat))
        if self._A_cache_id == key and self._A_cache is not None:
            return self._A_cache

        if not self._is_nonlinear():
            A = np.asarray(X_pool, dtype=float)
            if A.shape[1] != self.d:
                raise ValueError(f"Linear mode expects X_pool shape (p,{self.d}), got {A.shape}.")
        else:
            if self.model is None:
                raise ValueError("Nonlinear mode (beta_hat not None) requires model.")
            A = _G_from_model(self.model, X_pool, self.beta_hat)
            if A.shape[1] != self.dbeta:
                raise ValueError(f"Nonlinear mode expects G shape (p,{self.dbeta}), got {A.shape}.")

        self._A_cache = A
        self._A_cache_id = key
        return A

    def _check_Sigma_shape(self) -> None:
        dM = self._dim_design()
        if self.Sigma.shape != (dM, dM):
            raise ValueError(
                f"Sigma shape mismatch. Expected {(dM,dM)} "
                f"({'nonlinear dbeta' if self._is_nonlinear() else 'linear d'}), "
                f"got {self.Sigma.shape}."
            )

    def solve(self, selection_method: str):

        initial_guess = np.ones(self.p) / self.p
        constraints = {'type': 'ineq', 'fun': lambda pi: 1.0 - np.sum(pi)}
        bounds = [(0.0, 1.0)] * self.p

        if selection_method == 'stagewise_optimal':
            result = minimize(
                self.objective_fractional_minimize,
                initial_guess,
                constraints=constraints,
                bounds=bounds,
                method="SLSQP",
                options={"maxiter": 500, "ftol": 1e-9, "disp": False},
            )
        elif selection_method == 'stationary_optimal':
            result = minimize(
                self.objective_steady_state_minimize,
                initial_guess,
                constraints=constraints,
                bounds=bounds,
                method="SLSQP",
                options={"maxiter": 500, "ftol": 1e-9, "disp": False},
            )
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")

        pi_star = result.x
        n_k = round_n_vector(self.n * pi_star, self.n)

        return n_k, -float(result.fun)


    def objective_fractional_minimize(self, pi: np.ndarray) -> float:
        A = self._get_A(self.X_pool)
        B_pi = _B_from_A(pi, A)

        Sigma_inv = np.linalg.inv(self.Sigma)
        dM = self._dim_design()
        M = (self.n / (self.sigma**2)) * B_pi + Sigma_inv + self.reg_eps * np.eye(dM)

        return -_slogdet_spd(M)

    def compute_steady_Sigma_minimize(self, pi: np.ndarray, X_pool: np.ndarray) -> np.ndarray:
        A = self._get_A(X_pool)
        B_pi = _B_from_A(pi, A)

        dM = self._dim_design()
        I = np.eye(dM)
        Q = (self.sigmaw**2) * I

        Sigma_xi = self.covconstant * I

        for _ in range(self.steady_max_iter):
            M = (self.rho**2) * Sigma_xi + Q
            M_inv = np.linalg.inv(M + self.reg_eps * I)

            Sigma_new = np.linalg.inv(
                M_inv + (self.n / (self.sigma**2)) * B_pi + self.reg_eps * I
            )

            if np.linalg.norm(Sigma_new - Sigma_xi, ord="fro") < self.steady_tol:
                Sigma_xi = Sigma_new
                break
            Sigma_xi = Sigma_new

        return Sigma_xi

    def objective_steady_state_minimize(self, pi: np.ndarray) -> float:
        dM = self._dim_design()

        if self.X_pool_full is None:
            Sigma_bar = self.compute_steady_Sigma_minimize(pi, self.X_pool)
            return _slogdet_spd(Sigma_bar + self.reg_eps * np.eye(dM))
        else:
            total = 0.0
            for X_pool_sim in self.X_pool_full:
                Sigma_bar = self.compute_steady_Sigma_minimize(pi, X_pool_sim)
                total += _slogdet_spd(Sigma_bar + self.reg_eps * np.eye(dM))
            return total / len(self.X_pool_full)
