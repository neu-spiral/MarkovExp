import numpy as np
import os
import sys
from scipy.stats import ortho_group
from scipy.linalg import expm
from source.utils.misc import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class Kalman_filter:
    def __init__(self, d, dbeta, p, rho, sigmaw, sigma, model):
        self.d = d
        self.dbeta = dbeta
        self.p = p
        self.rho = rho
        self.sigmaw = sigmaw
        self.sigma = sigma
        self.model = model

    def update(self, X_k, y_k, beta_hat, Sigma):
        if self.model is None:
            S = X_k @ Sigma @ X_k.T + self.sigma**2 * np.eye(X_k.shape[0])
            K = Sigma @ X_k.T @ np.linalg.inv(S)
            beta_hat_new = beta_hat + K @ (y_k - X_k @ beta_hat)
            Sigma_new = (np.eye(self.dbeta) - K @ X_k) @ Sigma
        else:
            # Convert to torch
            X_t = torch.from_numpy(np.asarray(X_k)).float()
            y_t = torch.from_numpy(np.asarray(y_k).reshape(-1)).float()
            beta_t = torch.from_numpy(np.asarray(beta_hat).reshape(-1)).float()

            # Predicted measurement: h(X_k, beta_hat)
            with torch.no_grad():
                y_pred = self.model.forward(X_t, beta_t)  # (n_k,)
            innovation = (y_t - y_pred).cpu().numpy()     # (n_k,)

            # Observation matrix H_k = dh/dbeta at beta_hat
            H_t = self.model.jacobian_wrt_beta(X_t, beta_t)     # (n_k, d) torch
            H = H_t.detach().cpu().numpy()

            # Kalman gain
            # S = H Sigma H^T + sigma^2 I
            n_k = H.shape[0]
            S = H @ Sigma @ H.T + (self.sigma**2) * np.eye(n_k)

            # K = Sigma H^T S^{-1}
            K = (Sigma @ H.T) @ np.linalg.inv(S)

            # State update
            beta_hat_new = beta_hat + K @ innovation

            # Cov update
            Sigma_new = (np.eye(self.dbeta) - K @ H) @ Sigma

        return beta_hat_new, Sigma_new

    def predict(self, beta_hat, Sigma, beta_ast_est):
        beta_hat_new = self.rho * beta_hat + (1 - self.rho) * beta_ast_est
        Sigma_new = self.rho**2 * Sigma + self.sigmaw**2 * np.eye(self.dbeta)

        return beta_hat_new, Sigma_new