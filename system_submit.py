import os
import time
import datetime
import shutil
from subprocess import Popen
from source.models.model import *
from source.utils.misc import *

os.makedirs("logs", exist_ok=True)

blocking = True

system = "linear"
# system = "nonlinear"

model_name = "none"
# model_name = "mlp"
# model_name = "cnn"
# model_name = "resnet"

remove_all_folders = False  # Set to True to remove all folders before running

num_experiments = 100 # Number of experiments to run
random_pool = True  # Use random pool for data generation
use_stationary = True

data_type = "simulation"  # "simulation" or "real"
# data_type = "real"  # "simulation" or "real"
dataset = "beijing"  # "beijing" or "sp500" or "bike" or "california" or "atnt_dl" or "atnt_ul"

d = 30
p = 40
n = 50
rho_all = [0.9]
# rho_all = [0.06, 0.12, 0.23, 0.45, 0.9]
sigmaw_all = [0.9]
# sigmaw_all = [0.12, 0.23, 0.45, 0.9, 1.8]
sigma_all = [0.9]
# sigma_all = [0.12, 0.23, 0.45, 0.9, 1.8]
covconstant = 50.0
stages = 2000

### Beijing PM2.5 dataset
# d = 7
# p = 19
# n = 30
# rho_all = [0.92] # Linear
# sigmaw_all = [0.98] # Linear
# sigma_all = [80.29] # Linear
# # rho_all = [0.93] # Nonlinear
# # sigmaw_all = [0.02] # Nonlinear
# # sigma_all = [134.48] # Nonlinear
# covconstant = 50.0
# stages = 1826

### S&P 500 dataset
# d = 6
# p = 16
# n = 25
# rho_all = [0.66] # Linear
# sigmaw_all = [0.22] # Linear
# sigma_all = [0.78] # Linear
# # rho_all = [0.91] # Nonlinear
# # sigmaw_all = [0.02] # Nonlinear
# # sigma_all = [0.78] # Nonlinear
# covconstant = 50.0
# stages = 62

### Bike Sharing Demand dataset
# d = 8
# p = 19
# n = 30
# rho_all = [0.93] # Linear
# sigmaw_all = [29.08] # Linear
# sigma_all = [156.98] # Linear
# # rho_all = [0.95] # Nonlinear
# # sigmaw_all = [0.03] # Nonlinear
# # sigma_all = [261.75] # Nonlinear
# covconstant = 50.0
# stages = 724

### California Housing dataset
# d = 6
# p = 40
# n = 55
# rho_all = [0.94] # Linear
# sigmaw_all = [0.01] # Linear
# sigma_all = [0.79] # Linear
# # rho_all = [0.91] # Nonlinear
# # sigmaw_all = [0.02] # Nonlinear
# # sigma_all = [1.21] # Nonlinear
# covconstant = 50.0
# stages = 412

### ATNT Downlink dataset
# d = 12
# p = 16
# n = 25
# rho_all = [0.53] # Linear
# sigmaw_all = [684.94] # Linear
# sigma_all = [183.58] # Linear
# # rho_all = [0.94] # Nonlinear
# # sigmaw_all = [0.04] # Nonlinear
# # sigma_all = [363.75] # Nonlinear
# covconstant = 50.0
# stages = 335

### ATNT Uplink dataset
# d = 8
# p = 16
# n = 25
# rho_all = [0.74] # Linear
# sigmaw_all = [644.75] # Linear
# sigma_all = [13.32] # Linear
# # rho_all = [0.91] # Nonlinear
# # sigmaw_all = [0.03] # Nonlinear
# # sigma_all = [20.04] # Nonlinear
# covconstant = 50.0
# stages = 328


# -----------------------------
# Model dimension
# -----------------------------
if system == "linear":
    d_beta = d
else:
    if model_name == "mlp":
        model = MLPMeasurement(
                    input_dim=d,
                    hidden_dims=(2,2,2),
                    activation='tanh'
                )
    elif model_name == "cnn":
        model = CNNMeasurement(
                    input_dim=d,
                    channels=(3, 5),
                    kernel_sizes=(3, 5),
                    activation='tanh',
                    use_bias=True,
                    pool_out_len=1
                )
    elif model_name == "resnet":
        model = ResNetMeasurement(
                    input_dim=d,
                    channels=(2, 3),
                    kernel_sizes=(3, 3),
                    activation='tanh',
                    use_bias=True,
                    pool_out_len=1
                )
    model.eval()
    d_beta = model.num_params()

# -----------------------------
# Experiment pairs
# -----------------------------
if use_stationary:
    experiment_pairs = [
        ["non_optimal", "random", "no_kf", None],
        ["non_optimal", "random", "sgd", None],
        ["non_optimal", "random", "kf", "unknown"],
        ["non_optimal", "random", "kf", "known"],
        ["non_optimal", "random", "rls_ff", None],
        ["stagewise_optimal", "cvxpy", "no_kf", None],
        ["stagewise_optimal", "cvxpy", "sgd", None],
        ["stagewise_optimal", "cvxpy", "kf", "unknown"],
        ["stagewise_optimal", "cvxpy", "kf", "known"],
        ["stagewise_optimal", "cvxpy", "rls_ff", None],
        ["stationary_optimal", "fw", "kf", "unknown"],
        ["stationary_optimal", "fw", "kf", "known"],
        ["adam_optimal", "adam", "no_kf", None],
        ["adam_optimal", "adam", "sgd", None],
        ["adam_optimal", "adam", "kf", "unknown"],
        ["adam_optimal", "adam", "kf", "known"],
        ["adam_optimal", "adam", "rls_ff", None],
        ["greedy_optimal", "greedy", "no_kf", None],
        ["greedy_optimal", "greedy", "sgd", None],
        ["greedy_optimal", "greedy", "kf", "unknown"],
        ["greedy_optimal", "greedy", "kf", "known"],
        ["greedy_optimal", "greedy", "rls_ff", None],
        ]
else:
    experiment_pairs = [
        ["non_optimal", "random", "no_kf", None],
        ["non_optimal", "random", "sgd", None],
        ["non_optimal", "random", "kf", "unknown"],
        ["non_optimal", "random", "kf", "known"],
        ["non_optimal", "random", "rls_ff", None],
        ["stagewise_optimal", "cvxpy", "no_kf", None],
        ["stagewise_optimal", "cvxpy", "sgd", None],
        ["stagewise_optimal", "cvxpy", "kf", "unknown"],
        ["stagewise_optimal", "cvxpy", "kf", "known"],
        ["stagewise_optimal", "cvxpy", "rls_ff", None],
        ["adam_optimal", "adam", "no_kf", None],
        ["adam_optimal", "adam", "sgd", None],
        ["adam_optimal", "adam", "kf", "unknown"],
        ["adam_optimal", "adam", "kf", "known"],
        ["adam_optimal", "adam", "rls_ff", None],
        ["greedy_optimal", "greedy", "no_kf", None],
        ["greedy_optimal", "greedy", "sgd", None],
        ["greedy_optimal", "greedy", "kf", "unknown"],
        ["greedy_optimal", "greedy", "kf", "known"],
        ["greedy_optimal", "greedy", "rls_ff", None],
        ]


data_path = os.path.join("data", system, data_type, model_name)
results_path = os.path.join("results", system, data_type, model_name)

figures_root = os.path.join("figures", system, data_type, model_name)
logs_root = os.path.join("logs", system, data_type, model_name)

# Ensure base directories exist
os.makedirs(data_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)
os.makedirs(logs_root, exist_ok=True)
os.makedirs(figures_root, exist_ok=True)

if remove_all_folders:
    print("Removing all folders...")
    for folder in [data_path, results_path, figures_root, logs_root]:
        try:
            shutil.rmtree(folder)
            print(f"Removed folder {folder}")
        except Exception as e:
            print(f"Error removing folder {folder}: {e}")

counter = 0
for rho in rho_all:
    for sigmaw in sigmaw_all:
        for sigma in sigma_all:
            special_name = (
                f"d{d}_dbeta{d_beta}_p{p}_n{n}_rho{rho}_sigmaw{sigmaw}_sigma{sigma}_"
                f"covconstant{covconstant}_stages{stages}_experiments{num_experiments}_random_pool{random_pool}"
            )
            data_name = (
                f"d{d}_dbeta{d_beta}_p{p}_n{n}_sigmaw{sigmaw}_sigma{sigma}_"
                f"covconstant{covconstant}_stages{stages}_experiments{num_experiments}_random_pool{random_pool}"
            )

            create_data(
                d, d_beta, p, n, sigmaw, sigma, covconstant, stages,
                data_path, num_experiments, random_pool, data_type, dataset
            )

            for selection, algorithm, estimation, setting in experiment_pairs:
                for exper in range(num_experiments):
                    exper_name = f"{selection}_{algorithm}_{estimation}_{setting}_{exper}"

                    save_results_path = os.path.join(
                        results_path,
                        special_name,
                        f"{exper_name}.npz",
                    )

                    if os.path.exists(save_results_path):
                        print(f"[skip, {datetime.datetime.now().strftime('%B %d, %Y, %H:%M:%S')}] Results exist: {save_results_path}")
                        continue

                    out_log = os.path.join(logs_root, special_name, f"{exper_name}.out")
                    err_log = os.path.join(logs_root, special_name, f"{exper_name}.err")

                    os.makedirs(os.path.dirname(out_log), exist_ok=True)

                    commands = (
                        f"--d={d} "
                        f"--dbeta={d_beta} "
                        f"--p={p} "
                        f"--n={n} "
                        f"--rho={rho} "
                        f"--sigmaw={sigmaw} "
                        f"--sigma={sigma} "
                        f"--model_name={model_name} "
                        f"--covconstant={covconstant} "
                        f"--stages={stages} "
                        f"--data_path={data_path} "
                        f"--results_path={results_path} "
                        f"--selection_method={selection} "
                        f"--estimation_method={estimation} "
                        f"--algorithm={algorithm} "
                        f"--setting={setting} "
                        f"--num_experiments={num_experiments} "
                        f"--random_pool={random_pool} "
                        f"--exper={exper} "
                        f"--data_type={data_type} "
                        f"--system={system}"
                    )

                    cmd_str = f"python -m source.core.master {commands}"
                    print(f"[submitter, {datetime.datetime.now().strftime('%B %d, %Y, %H:%M:%S')}] {save_results_path}")

                    if blocking:
                        os.system(f"{cmd_str} > {out_log} 2> {err_log}")
                    else:
                        Popen(
                            [
                                "sbatch",
                                f"--output={out_log}",
                                f"--error={err_log}",
                                "execute.sh",
                                cmd_str,
                            ],
                        )

                    time.sleep(0.01)
