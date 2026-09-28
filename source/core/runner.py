import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

from source.utils.misc import *
from source.models.selections import *
from source.models.estimations import *
from source.models.model import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# plt.style.use('tableau-colorblind10')

class MarkovianExperimental:
    def __init__(self, d, dbeta, p, n, rho, sigmaw, sigma, model_name, covconstant, stages, data_path, results_path, selection_method, estimation_method, algorithm, setting, num_experiments, random_pool, exper, data_type, system, rls_gamma=0.98):
        self.d = d
        self.dbeta = dbeta
        self.p = p
        self.n = n
        self.rho = rho
        self.sigmaw = sigmaw
        self.sigma = sigma
        self.model_name = model_name
        self.covconstant = covconstant
        self.stages = stages
        self.selection_method = selection_method
        self.estimation_method = estimation_method
        self.algorithm = algorithm
        self.setting = setting
        self.num_experiments = num_experiments
        self.random_pool = random_pool
        self.exper = exper
        self.data_type = data_type
        self.system = system
        self.rls_gamma = rls_gamma  # forgetting factor for 'rls_ff' estimation baseline

        self.load_data_path = os.path.join(data_path, f"d{d}_dbeta{dbeta}_p{p}_n{n}_sigmaw{sigmaw}_sigma{sigma}_covconstant{covconstant}_stages{stages}_experiments{num_experiments}_random_pool{random_pool}.npz")
        self.save_results_path = os.path.join(results_path, f"d{d}_dbeta{dbeta}_p{p}_n{n}_rho{rho}_sigmaw{sigmaw}_sigma{sigma}_covconstant{covconstant}_stages{stages}_experiments{num_experiments}_random_pool{random_pool}/{selection_method}_{algorithm}_{estimation_method}_{setting}_{exper}.npz")


        self.beta_0_general = np.load(self.load_data_path)['beta_0']
        self.cov_0_general = np.load(self.load_data_path)['cov_0']
        self.X_pool_general = np.load(self.load_data_path)['X_pool']
        self.X_test_general = np.load(self.load_data_path)['X_test']
        self.e_noise_general = np.load(self.load_data_path)['e_noise']
        self.test_noise_general = np.load(self.load_data_path)['test_noise']
        self.w_noise_general = np.load(self.load_data_path)['w_noise']
        self.beta_1_general = np.load(self.load_data_path)['beta_1']
        self.beta_ast_general = np.load(self.load_data_path)['beta_ast']
        if self.data_type == "real":
            self.y_pool_general = np.load(self.load_data_path)['y_pool']
            self.y_test_general = np.load(self.load_data_path)['y_test']


        self.results = {
            'pred_rmse': [],
            'beta_error': [],
            'nonlinear_beta_error': [],
            'objective': [],
            'n_k': [],
            'Sigma_diff': [],
            'Sigma': [],
            'beta_ast_est_diff': [],
            'nonlinear_beta_ast_est_diff': [],
        }

        os.makedirs(os.path.dirname(self.save_results_path), exist_ok=True)

    def run_med(self):
        """
        Run the Markovian experimental design.
        """

        print(f"\n\n\n\nRunning experiment {self.exper}/{self.num_experiments}")
        print(f"\nResult will be saved to {self.save_results_path}")
        print()

        beta_0 = self.beta_0_general[self.exper]
        cov_0 = self.cov_0_general[self.exper]
        X_pool = self.X_pool_general[self.exper]
        X_test = self.X_test_general[self.exper]
        print(f"X_pool shape: {X_pool.shape}, X_test shape: {X_test.shape}")
        e_noise = self.e_noise_general[self.exper]
        test_noise = self.test_noise_general[self.exper]
        w_noise = self.w_noise_general[self.exper]
        beta_1 = self.beta_1_general[self.exper]
        beta_ast = self.beta_ast_general[self.exper]
        if self.data_type == "real":
            y_pool = self.y_pool_general[self.exper]
            y_test_all = self.y_test_general[self.exper]
            print(f"y_pool shape: {y_pool.shape}, y_test shape: {y_test_all.shape}")
        true_beta = np.zeros((self.stages, self.dbeta))
        beta_hat_update = np.zeros((self.stages, self.dbeta))
        beta_hat_predict = np.zeros((self.stages, self.dbeta))
        Sigma_update = np.zeros((self.stages, self.dbeta, self.dbeta))
        Sigma_predict = np.zeros((self.stages, self.dbeta, self.dbeta))
        y_k = np.zeros((self.stages, self.n))
        if self.data_type == "real":
            y_test = np.zeros((self.stages, X_test.shape[1]))
        else:
            y_test = np.zeros((self.stages, X_test.shape[1]))
        X_k = np.zeros((self.stages, self.n, self.d))
        objective_all = np.zeros(self.stages)
        n_k_all = np.zeros((self.stages, self.p)).astype(int)

        if self.system == 'linear':
            model = None
            selections = Selections(self.d, self.dbeta, self.p, self.n, self.rho, self.sigmaw, self.sigma, self.covconstant, self.stages, None)
            estimations = Estimations(self.d, self.dbeta, self.p, self.rho, self.sigmaw, self.sigma, beta_0, cov_0, self.data_type, None, forgetting=self.rls_gamma)
        else:
            if self.model_name == "mlp":
                model = MLPMeasurement(
                            input_dim=self.d,
                            hidden_dims=(2,2,2),
                            activation='tanh'
                        )
            elif self.model_name == "cnn":
                model = CNNMeasurement(
                            input_dim=self.d,
                            channels=(3, 5),
                            kernel_sizes=(3, 5),
                            activation='tanh',
                            use_bias=True,
                            pool_out_len=1
                        )
            elif self.model_name == "resnet":
                model = ResNetMeasurement(
                            input_dim=self.d,
                            channels=(2, 3),
                            kernel_sizes=(3, 3),
                            activation='tanh',
                            use_bias=True,
                            pool_out_len=1
                        )
            model.eval()
            selections = Selections(self.d, self.dbeta, self.p, self.n, self.rho, self.sigmaw, self.sigma, self.covconstant, self.stages, model)
            estimations = Estimations(self.d, self.dbeta, self.p, self.rho, self.sigmaw, self.sigma, beta_0, cov_0, self.data_type, model, forgetting=self.rls_gamma)

        beta_ast_est = np.zeros((self.stages, self.dbeta))

        if self.setting == 'known':
            for c in range(self.stages):
                beta_ast_est[c] = beta_ast
        else:
            cumulative_data = np.zeros((self.n*self.stages, self.d))
            cumulative_labels = np.zeros(self.n*self.stages)

        if self.selection_method == 'stationary_optimal' and self.system == 'linear':
            if self.random_pool == True:
                n_k, objective = selections.conduct_selection(self.selection_method, self.algorithm, None, None, X_pool, None)
            else:
                n_k, objective = selections.conduct_selection(self.selection_method, self.algorithm, None, X_pool[0], None, None)

            for c in range(self.stages):
                n_k_all[c] = n_k
                X_k[c] = collect_data(X_pool[c], n_k)

        for k in range(self.stages):
            print(f"Stage {k+1}/{self.stages}")

            # Predict the next state
            if k == 0:
                true_beta[k] = copy.deepcopy(beta_1)
                beta_hat_predict[k], Sigma_predict[k] = copy.deepcopy(beta_0), copy.deepcopy(cov_0)
            else:
                true_beta[k] = simulate_true_beta_drift(true_beta[k-1], w_noise[k-1], self.rho, beta_ast)
                beta_hat_predict[k], Sigma_predict[k] = estimations.predict_phase(self.estimation_method, beta_hat_update[k-1], Sigma_update[k-1], beta_ast_est[k-1])
            if not self.selection_method == 'stationary_optimal':
                if self.system == 'linear':
                    n_k_all[k], objective = selections.conduct_selection(self.selection_method, self.algorithm, Sigma_predict[k], X_pool[k], None, None)
                else:
                    n_k_all[k], objective = selections.conduct_selection(self.selection_method, self.algorithm, Sigma_predict[k], X_pool[k], None, beta_hat_predict[k])
                X_k[k] = collect_data(X_pool[k], n_k_all[k])
            elif self.selection_method == 'stationary_optimal' and self.system == 'nonlinear':
                if self.setting == 'known':
                    n_k_all[k], _ = selections.conduct_selection(self.selection_method, self.algorithm, Sigma_predict[k], X_pool[k], None, beta_ast)
                else:
                    n_k_all[k], _ = selections.conduct_selection(self.selection_method, self.algorithm, Sigma_predict[k], X_pool[k], None, beta_ast_est[k-1] if k > 0 else beta_0)
                
                objective = objective_regular_stage(X_pool[k], n_k_all[k], self.sigma, Sigma_predict[k], beta_hat_predict[k], model)
                X_k[k] = collect_data(X_pool[k], n_k_all[k])

            # Simulate observations
            if self.data_type == "simulation":
                if self.system == 'linear':
                    y_k[k] = collect_labels(X_k[k], true_beta[k], e_noise[k], None)
                    y_test[k] = collect_labels(X_test[k], true_beta[k], test_noise[k], None)
                else:
                    y_k[k] = collect_labels(X_k[k], true_beta[k], e_noise[k], model)
                    y_test[k] = collect_labels(X_test[k], true_beta[k], test_noise[k], model)
            else:        
                #add very small noise
                y_k[k] = collect_real_labels(y_pool[k], n_k_all[k]) + 1e-1 * np.random.randn(*collect_real_labels(y_pool[k], n_k_all[k]).shape)
                y_test[k] = y_test_all[k] + 1e-1 * np.random.randn(*y_test_all[k].shape)


            objective_all[k] = objective

            # Update the Kalman filter
            beta_hat_update[k], Sigma_update[k] = estimations.update_phase(self.estimation_method, X_k[k], y_k[k], beta_hat_predict[k], Sigma_predict[k])

            if self.setting == 'unknown':
                if self.system == 'linear':
                    cumulative_data[k*self.n:(k+1)*self.n] = X_k[k]
                    cumulative_labels[k*self.n:(k+1)*self.n] = y_k[k]
                    beta_ast_est[k] = np.linalg.inv(cumulative_data[:(k+1)*self.n].T @ cumulative_data[:(k+1)*self.n] + 1e-6 * np.eye(cumulative_data.shape[1])) @ cumulative_data[:(k+1)*self.n].T @ cumulative_labels[:(k+1)*self.n]
                else:
                    # print("\nEstimating beta_ast_est with nonlinear model...")
                    fit = estimate_beta_star_cumulative(
                        model=model,
                        X_cum=X_k[k],
                        y_cum=y_k[k],
                    )
                    beta_ast_est[k] = fit

            track_results(self.results, beta_hat_update[k], beta_hat_predict[k], beta_ast, beta_ast_est[k], 
                          true_beta[k], X_pool[k], X_test[k], y_test[k], objective_all[k], 
                          n_k_all[k], Sigma_update[k], Sigma_predict[k], self.sigma, model)
            if self.system == 'linear':
                print(
                    f"Stage {k+1} | "
                    f"RMSE={self.results['pred_rmse'][-1]} | "
                    f"BetaErr={self.results['beta_error'][-1]} | "
                    f"Obj={self.results['objective'][-1]} | "
                    f"BetaAstEstDiff={self.results['beta_ast_est_diff'][-1]}"
                )
            else:
                print(
                    f"Stage {k+1} | "
                    f"RMSE={self.results['pred_rmse'][-1]} | "
                    f"NL-BetaErr={self.results['nonlinear_beta_error'][-1]} | "
                    f"Obj={self.results['objective'][-1]} | "
                    f"NL-BetaAstEstDiff={self.results['nonlinear_beta_ast_est_diff'][-1]}"
    )

    def save_results(self):
        """
        Save the results of the experiment.
        """
        np.savez(self.save_results_path, **self.results)
        print(f"Results saved to {self.save_results_path}")