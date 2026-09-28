import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import os
import sys
import copy
from scipy.linalg import solve_sylvester

from scipy.optimize import minimize_scalar, check_grad

plt.style.use('tableau-colorblind10')
# plt.style.use('seaborn-v0_8-colorblind')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from source.utils.misc import *
# from source.utils.eval import *

class FrankWolfe:
    def __init__(self, d, p, rho, sigmaw, sigma, covconstant, n, stages, X_pool):
        self.d = d
        self.p = p
        self.rho = rho
        self.sigmaw = sigmaw
        self.sigma = sigma
        self.covconstant = covconstant
        self.n = n
        self.stages = stages
        self.X_pool = X_pool

    def solve(self, selection_method, Sigma, max_iter=1000, tol=1e-6):
        """Frank-Wolfe algorithm implementation"""
        # Initialize at the center of the simplex
        beta_k = np.ones(self.p) / self.p

        if selection_method == 'stationary_optimal':
            obj_history = [self.objective_steady_state(beta_k)]
        elif selection_method == 'stagewise_optimal':
            obj_history = [objective_fractional_stage(beta_k, Sigma, self.X_pool, self.n, self.sigma)]
        
        for k in range(max_iter):
            if selection_method == 'stationary_optimal':
                grad = self.gradient_steady(beta_k)
            elif selection_method == 'stagewise_optimal':
                grad = self.gradient_stage(beta_k, Sigma)
            
            # Solve linear subproblem
            s_k = self.solve_linear_subproblem(grad)

            gap = grad @ (s_k - beta_k)

            if gap <= tol:
                break
                
            # Exact line search
            if selection_method == 'stationary_optimal':
                gamma_k = self.exact_line_search_steady(beta_k, s_k)
            elif selection_method == 'stagewise_optimal':
                gamma_k = self.exact_line_search_stage(beta_k, s_k, Sigma)


            # Update solution
            beta_k = (1 - gamma_k) * beta_k + gamma_k * s_k

            if selection_method == 'stationary_optimal':
                obj_history.append(self.objective_steady_state(beta_k))
            elif selection_method == 'stagewise_optimal':
                obj_history.append(objective_fractional_stage(beta_k, Sigma, self.X_pool, self.n, self.sigma))
        
        # Round to integer allocations
        n_k = self.n * beta_k
        n_k = round_n_vector(n_k, self.n)

        if selection_method == 'stationary_optimal':
            return n_k, self.objective_steady_state(beta_k)
        elif selection_method == 'stagewise_optimal':
            return n_k, objective_fractional_stage(beta_k, Sigma, self.X_pool, self.n, self.sigma)
        
    def gradient_stage(self, pi, Sigma):
        """Compute the gradient of the objective function"""
        B_pi = fischer_information(pi, self.X_pool)
        M = (self.n / self.sigma**2) * B_pi + np.linalg.inv(Sigma)
        M_inv = np.linalg.inv(M)

        return (self.n / self.sigma**2) * np.einsum('ij, jk, ik -> i', self.X_pool, M_inv, self.X_pool)

    def gradient_steady(self, pi):
        """Compute gradient of G(π) = log det[Σ_ξ(π)] for MAXIMIZATION."""    

        S = self.compute_steady_Sigma(pi)
        M = np.linalg.inv(self.rho**2 * S + self.sigmaw**2 * np.eye(self.d))
        S_inv = np.linalg.inv(S)

        grad = np.zeros_like(pi)
        for j in range(len(pi)):
            xj = self.X_pool[j].reshape(-1,1)
            C = (self.n / self.sigma**2) * (xj @ xj.T)
            
            # Solve Delta + rho^2 M Delta M = C
            dS_dpi = self.solve_two_sided(self.rho**2 * M, M, C)
            
            grad[j] = np.trace(S_inv @ dS_dpi)
        return grad

    def solve_two_sided(self, A1, A2, C, tol=1e-12, max_iter=1000):
        X = np.zeros_like(C)
        for _ in range(max_iter):
            X_new = C - A1 @ X @ A2
            if np.linalg.norm(X_new - X, 'fro') < tol:
                return X_new
            X = X_new
        return X
        
    def compute_steady_Sigma(self, pi, max_iter=100, tol=1e-6):
        """Compute Σ_ξ using fixed-point iteration."""
        XtX = self.X_pool.T @ np.diag(pi) @ self.X_pool  # X^T X
        Sigma_xi = np.eye(self.d)  # Initial guess

        for _ in range(max_iter):
            # print(f"Iteration {_}:")
            M = self.rho**2 * Sigma_xi + self.sigmaw**2 * np.eye(self.d)
            M_inv = np.linalg.inv(M)
            new_Sigma_xi = M_inv + (self.n / self.sigma**2) * XtX

            if np.linalg.norm(new_Sigma_xi - Sigma_xi) < tol:
                break
            Sigma_xi = copy.deepcopy(new_Sigma_xi)

        return Sigma_xi

    def solve_linear_subproblem(self, grad):
        """Solve the linear subproblem to find the descent direction"""
        # Find the index of the maximum gradient component
        i = np.argmax(grad)
        
        # Create a one-hot vector at the maximum gradient position
        s = np.zeros(self.p)
        s[i] = 1.0
        
        return s
    
    def exact_line_search_stage(self, beta_k, s_k, Sigma):
        """Perform exact line search to find optimal step size"""
        def f(gamma):
            new_pi = (1 - gamma) * beta_k + gamma * s_k
            return -objective_fractional_stage(new_pi, Sigma, self.X_pool, self.n, self.sigma)
        
        res = minimize_scalar(f, bounds=(0, 1), method='bounded')
        return res.x

    def exact_line_search_steady(self, beta_k, s_k):
        """Perform exact line search to find optimal step size for steady state"""
        def f(gamma):
            new_pi = (1 - gamma) * beta_k + gamma * s_k
            return -self.objective_steady_state(new_pi)

        res = minimize_scalar(f, bounds=(0, 1), method='bounded')
        return res.x
    
    def objective_steady_state(self, pi):
        """Compute the objective function G(π) for steady state"""
        # Compute Σ_ξ using fixed-point iteration
        Sigma_xi = self.compute_steady_Sigma(pi)
        return np.log(np.linalg.det(Sigma_xi))
    
