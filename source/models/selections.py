import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from scipy.optimize import minimize
import cvxpy as cp
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from source.utils.misc import *
from source.models.frank_wolfe import *
from source.models.scipy_minimize import *
from source.models.adam_solver import ADAMOptimize

class Selections:
    def __init__(self, d, dbeta, p, n, rho, sigmaw, sigma, covconstant, stages, model):
        self.d = d
        self.dbeta = dbeta
        self.p = p
        self.n = n
        self.rho = rho
        self.sigmaw = sigmaw
        self.sigma = sigma
        self.covconstant = covconstant
        self.stages = stages
        self.model = model
    
    def random_design(self):
        """ Random design """

        n_k = np.zeros(self.p, dtype=int)
        for _ in range(self.n):
            idx = np.random.randint(0, self.p)
            n_k[idx] += 1
        return n_k

    def cvxpy_design(self, Sigma, X_pool, beta_hat):
        pi = cp.Variable(self.p, nonneg=True)
        Sigma_inv = np.linalg.inv(Sigma)

        if beta_hat is None:
            # Define the optimization variables

            B_pi = cp.sum([pi[i] * np.outer(X_pool[i], X_pool[i]) for i in range(self.p)])
            M = cp.multiply(self.n / self.sigma**2, B_pi) + np.linalg.inv(Sigma)

            # Define the objective function
            objective = cp.Maximize(cp.log_det(M))
            constraints = [cp.sum(pi) <= 1, pi >= 0]

            # Define the problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)

            # print(pi.value)

            n_k = self.n * pi.value
            n_k_rounded = round_n_vector(n_k, self.n)
        
        else:
            if getattr(self, "model", None) is None:
                raise ValueError("Nonlinear design requested but self.model is None.")

            # Torch conversion
            X_t = torch.from_numpy(X_pool).float()
            beta_t = torch.from_numpy(np.asarray(beta_hat).reshape(-1)).float()

            H_t = self.model.jacobian_wrt_beta(X_t, beta_t)  # (p_k, d) torch
            G = H_t.detach().cpu().numpy()                   # (p_k, d)

            scaled_G = cp.multiply(pi[:, None], G)
            B_pi = G.T @ scaled_G                                # (d, d)
            
            M = (self.n / (self.sigma**2)) * B_pi + Sigma_inv

            objective = cp.Maximize(cp.log_det(M))
            constraints = [cp.sum(pi) <= 1, pi >= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)

            n_k = self.n * pi.value
            n_k_rounded = round_n_vector(n_k, self.n)

        return n_k_rounded, problem.value

    def greedy_infogain_design(self, Sigma, X_pool, beta_hat):
        if beta_hat is None:
            V = np.asarray(X_pool, dtype=float)                       # (p, d)
        else:
            if getattr(self, "model", None) is None:
                raise ValueError("Nonlinear greedy design requested but self.model is None.")
            X_t = torch.from_numpy(np.asarray(X_pool)).float()
            beta_t = torch.from_numpy(np.asarray(beta_hat).reshape(-1)).float()
            V = self.model.jacobian_wrt_beta(X_t, beta_t).detach().cpu().numpy()  # (p, d_beta)

        s2 = self.sigma ** 2
        M_inv = np.asarray(Sigma, dtype=float).copy()
        n_k = np.zeros(self.p, dtype=int)

        for _ in range(self.n):
            # information gain / leverage for each candidate: v_i^T M_inv v_i
            scores = np.einsum('ij,jk,ik->i', V, M_inv, V)
            j = int(np.argmax(scores))
            n_k[j] += 1
            v = V[j]
            Minv_v = M_inv @ v
            denom = s2 + float(v @ Minv_v)
            # Sherman-Morrison rank-1 update for M += (1/s2) v v^T
            M_inv = M_inv - np.outer(Minv_v, Minv_v) / denom

        objective = objective_regular_stage(X_pool, n_k, self.sigma, Sigma, beta_hat, self.model)
        return n_k, objective

    def scipy_design(self, selection_method, Sigma, X_pool, X_pool_full, beta_hat):
        SC = ScipyMinimize(self.d, self.dbeta, self.p, self.rho, self.sigmaw, self.sigma, self.covconstant, self.n, self.stages, X_pool, X_pool_full, Sigma, beta_hat, self.model)
        return SC.solve(selection_method)

    def FW_design(self, selection_method, Sigma, X_pool):
        FW = FrankWolfe(self.d, self.p, self.rho, self.sigmaw, self.sigma, self.covconstant, self.n, self.stages, X_pool)
        return FW.solve(selection_method, Sigma)

    def ADAM_design(
        self,
        Sigma: np.ndarray,
        X_pool: np.ndarray,
        beta_hat: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        n_k, obj = ADAMOptimize(
            Sigma=Sigma,
            X_pool=X_pool,
            beta_hat=beta_hat,
            model=self.model,
            n=self.n,
            sigma=self.sigma,
            **kwargs,
        )
        return n_k, obj

    def conduct_selection(self, selection_method, algorithm, Sigma, X_pool, X_pool_full, beta_hat):
        # Select data
        if algorithm == 'random':
            n_k = self.random_design()
            objective = objective_regular_stage(X_pool, n_k, self.sigma, Sigma, beta_hat, self.model)
        elif algorithm == 'cvxpy':
            n_k, objective = self.cvxpy_design(Sigma, X_pool, beta_hat)
        elif algorithm == 'scipy':
            n_k, objective = self.scipy_design(selection_method, Sigma, X_pool, X_pool_full, beta_hat)
        elif algorithm == 'fw':
            n_k, objective = self.FW_design(selection_method, Sigma, X_pool)
        elif algorithm == 'adam':
            n_k, objective = self.ADAM_design(Sigma, X_pool, beta_hat)
        elif algorithm == 'greedy':
            n_k, objective = self.greedy_infogain_design(Sigma, X_pool, beta_hat)
        else:
            raise ValueError(f"Unknown selection algorithm: {algorithm}")
        return n_k, objective

    


