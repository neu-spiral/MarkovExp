import numpy as np
import os
import sys
import copy
import torch
from source.models.kalman import Kalman_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class Estimations:
    def __init__(self, d, dbeta, p, rho, sigmaw, sigma, beta_0, cov_0, data_type, model, forgetting=0.98):
        self.d = d
        self.dbeta = dbeta
        self.p = p
        self.rho = rho
        self.sigmaw = sigmaw
        self.sigma = sigma
        self.beta_0 = beta_0
        self.cov_0 = cov_0
        self.data_type = data_type
        self.model = model
        self.forgetting = forgetting
        self.KF = Kalman_filter(d, dbeta, p, rho, sigmaw, sigma, model)

    def update_phase(self, estimation_method, X_k, y_k, beta_hat_predict, Sigma_predict):
        if estimation_method == 'no_kf':
            # Bayesian update without Kalman filter
            if self.model is None:
                beta_hat_update = self.beta_0 + self.cov_0 @ X_k.T @ np.linalg.inv(X_k @ self.cov_0 @ X_k.T + self.sigma**2 * np.eye(X_k.shape[0])) @ (y_k - X_k @ self.beta_0)
            else:
                # Convert to torch
                X_t = torch.from_numpy(np.asarray(X_k)).float()
                y_t = torch.from_numpy(np.asarray(y_k).reshape(-1)).float()
                beta_t = torch.from_numpy(np.asarray(self.beta_0).reshape(-1)).float()

                # Predicted measurement: h(X_k, beta_0)
                with torch.no_grad():
                    y_pred = self.model.forward(X_t, beta_t)  # (n_k,)
                innovation = (y_t - y_pred).cpu().numpy()     # (n_k,)

                H_t = self.model.jacobian_wrt_beta(X_t, beta_t)     # (n_k, d) torch
                H = H_t.detach().cpu().numpy()

                # Kalman gain
                S = H @ self.cov_0 @ H.T + (self.sigma**2) * np.eye(H.shape[0])
                K = (self.cov_0 @ H.T) @ np.linalg.inv(S)

                # State update
                beta_hat_update = self.beta_0 + K @ innovation
            
            return beta_hat_update, copy.deepcopy(Sigma_predict)
        elif estimation_method == 'kf':
            beta_hat_update, Sigma_update = self.KF.update(X_k, y_k, beta_hat_predict, Sigma_predict)
            return beta_hat_update, Sigma_update

        elif estimation_method == 'rls_ff':
            if self.model is None:
                # Linear case
                X = np.asarray(X_k, dtype=float)
                y = np.asarray(y_k, dtype=float).reshape(-1)
                S = X @ Sigma_predict @ X.T + self.sigma**2 * np.eye(X.shape[0])
                K = Sigma_predict @ X.T @ np.linalg.inv(S)
                beta_hat_update = np.asarray(beta_hat_predict, dtype=float).reshape(-1) + K @ (y - X @ np.asarray(beta_hat_predict, dtype=float).reshape(-1))
                Sigma_update = (np.eye(self.dbeta) - K @ X) @ Sigma_predict
            else:
                # Nonlinear case: linearize measurement via Jacobian (EKF-style update)
                X_t = torch.from_numpy(np.asarray(X_k)).float()
                y_t = torch.from_numpy(np.asarray(y_k).reshape(-1)).float()
                beta_t = torch.from_numpy(np.asarray(beta_hat_predict).reshape(-1)).float()
                with torch.no_grad():
                    y_pred = self.model.forward(X_t, beta_t)
                innovation = (y_t - y_pred).cpu().numpy()
                H = self.model.jacobian_wrt_beta(X_t, beta_t).detach().cpu().numpy()
                S = H @ Sigma_predict @ H.T + (self.sigma**2) * np.eye(H.shape[0])
                K = (Sigma_predict @ H.T) @ np.linalg.inv(S)
                beta_hat_update = np.asarray(beta_hat_predict).reshape(-1) + K @ innovation
                Sigma_update = (np.eye(self.dbeta) - K @ H) @ Sigma_predict
            # Symmetrize for numerical stability
            Sigma_update = 0.5 * (Sigma_update + Sigma_update.T)
            return beta_hat_update, Sigma_update

        elif estimation_method == 'sgd':
            # SGD based estimation with multiple epochs + convergence
            if self.data_type == 'simulation':
                learning_rate = 0.1
            else:
                learning_rate = 1e-7

            # ---- SGD controls ----
            max_epochs = 200
            tol_rel_impr = 1e-6
            tol_grad = 1e-6
            patience = 10
            min_epochs = 1

            eps = 1e-12

            if self.model is None:
                # ---------- Linear case ----------
                X = np.asarray(X_k)
                y = np.asarray(y_k).reshape(-1)
                beta = np.asarray(beta_hat_predict).reshape(-1).copy()

                prev_loss = np.inf
                bad_epochs = 0

                n = X.shape[0]

                for epoch in range(max_epochs):
                    # residuals, loss
                    r = y - X @ beta
                    loss = 0.5 * float(np.mean(r ** 2))

                    grad = -(X.T @ r) / n
                    grad_norm = float(np.linalg.norm(grad))

                    # SGD step
                    beta_new = beta - learning_rate * grad

                    # convergence checks
                    rel_impr = (prev_loss - loss) / (abs(prev_loss) + eps) if np.isfinite(prev_loss) else np.inf

                    if epoch + 1 >= min_epochs:
                        if (rel_impr >= 0 and rel_impr < tol_rel_impr) or (grad_norm < tol_grad):
                            bad_epochs += 1
                        else:
                            bad_epochs = 0

                        if bad_epochs >= patience:
                            beta = beta_new
                            break

                    beta = beta_new
                    prev_loss = loss

                beta_hat_update = beta

            else:
                X_t = torch.from_numpy(np.asarray(X_k)).float()
                y_t = torch.from_numpy(np.asarray(y_k).reshape(-1)).float()

                self.model.load_beta(beta_hat_predict)

                prev_loss = float("inf")
                bad_epochs = 0

                self.model.train()

                for epoch in range(max_epochs):
                    self.model.zero_grad(set_to_none=True)

                    y_pred = self.model.forward(X_t)  # no beta arg
                    loss_t = 0.5 * torch.mean((y_t - y_pred.reshape(-1)) ** 2)

                    loss_t.backward()

                    with torch.no_grad():
                        sq = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                sq += float(torch.sum(p.grad ** 2))
                        grad_norm = float(np.sqrt(sq))

                    loss = float(loss_t.detach().cpu().item())
                    rel_impr = (prev_loss - loss) / (abs(prev_loss) + eps) if np.isfinite(prev_loss) else np.inf

                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p -= learning_rate * p.grad

                    # convergence checks
                    if epoch + 1 >= min_epochs:
                        flat = ((rel_impr >= 0 and rel_impr < tol_rel_impr) or (grad_norm < tol_grad))
                        if flat:
                            bad_epochs += 1
                        else:
                            bad_epochs = 0

                        if bad_epochs >= patience:
                            break

                    prev_loss = loss

                self.model.eval()
                beta_hat_update = self.model.flatten_beta()

            return beta_hat_update, copy.deepcopy(self.cov_0)  # No update to Sigma in SGD method
        else:
            raise ValueError(f"Unknown estimation method for update phase: {estimation_method}")
        
        
    def predict_phase(self, estimation_method, beta_hat_update, Sigma_update, beta_ast_est):
        if estimation_method == 'no_kf':
            return copy.deepcopy(beta_hat_update), copy.deepcopy(Sigma_update)  # No prediction in no_kf method
        elif estimation_method == 'kf':
            # Update parameters using steady state estimation
            beta_hat_predict, Sigma_predict = self.KF.predict(beta_hat_update, Sigma_update, beta_ast_est)
            return beta_hat_predict, Sigma_predict
        elif estimation_method == 'rls_ff':
            # Forgetting-factor RLS time update
            gamma = max(float(self.forgetting), 1e-8)
            beta_hat_predict = copy.deepcopy(beta_hat_update)
            Sigma_predict = copy.deepcopy(Sigma_update) / gamma
            return beta_hat_predict, Sigma_predict
        elif estimation_method == 'sgd':
            return copy.deepcopy(beta_hat_update), copy.deepcopy(Sigma_update)
        else:
            raise ValueError(f"Unknown estimation method for predict phase: {estimation_method}")