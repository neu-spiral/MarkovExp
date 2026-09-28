import os
import sys
import time
import matplotlib.pyplot as plt
import argparse

from source.utils.misc import *
from source.core.runner import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
plt.style.use('tableau-colorblind10')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Markovian Experimental Setup')
    parser.add_argument('--d', type=int, default=10, help='Dimension of the state space')
    parser.add_argument('--dbeta', type=int, default=10, help='Dimension of the parameter space')
    parser.add_argument('--p', type=int, default=5, help='Number of pools')
    parser.add_argument('--n', type=int, default=100, help='Total number of samples per stage')
    parser.add_argument('--rho', type=float, default=0.9, help='Persistence factor')
    parser.add_argument('--sigmaw', type=float, default=0.1, help='Noise variance for the process')
    parser.add_argument('--sigma', type=float, default=0.1, help='Observation noise variance')
    parser.add_argument('--model_name', type=str, default='mlp', help='Model name for nonlinear system')
    parser.add_argument('--covconstant', type=float, default=1.0, help='Covariance constant')
    parser.add_argument('--stages', type=int, default=10, help='Number of stages in the experiment')
    parser.add_argument('--data_path', type=str, required=True, help='Path to load data from')
    parser.add_argument('--results_path', type=str, required=True, help='Path to save results to')
    parser.add_argument('--selection_method', type=str, default='random', help='Method for selecting samples')
    parser.add_argument('--estimation_method', type=str, default='predict_update', help='Method for estimating parameters')
    parser.add_argument('--algorithm', type=str, default='cvxpy', help='Algorithm to use')
    parser.add_argument('--setting', type=str, default='unknown', help='Setting of the experiment')
    parser.add_argument('--num_experiments', type=int, default=5, help='Number of experiments to run')
    parser.add_argument('--random_pool', type=str, default='yes', help='Use random pool for data generation')
    parser.add_argument('--exper', type=int, default=0, help='Experiment index for logging')
    parser.add_argument('--data_type', type=str, default='simulation', choices=['simulation', 'real'], help='Type of data to use')
    parser.add_argument('--system', type=str, default='linear', choices=['linear', 'nonlinear'], help='Type of system model')

    args = parser.parse_args()

    d = args.d
    dbeta = args.dbeta
    p = args.p
    n = args.n
    rho = args.rho
    sigmaw = args.sigmaw
    sigma = args.sigma
    model_name = args.model_name
    covconstant = args.covconstant
    stages = args.stages
    data_path = args.data_path
    results_path = args.results_path
    selection_method = args.selection_method
    estimation_method = args.estimation_method
    algorithm = args.algorithm
    setting = args.setting
    num_experiments = args.num_experiments
    random_pool = args.random_pool
    exper = args.exper
    data_type = args.data_type
    system = args.system

    start_time = time.time()

    # Initialize the experimental setup with the provided parameters
    experiment = MarkovianExperimental(d, dbeta, p, n, rho, sigmaw, sigma, model_name, covconstant, stages, data_path, results_path, selection_method, estimation_method, algorithm, setting, num_experiments, random_pool, exper, data_type, system)

    # Run the experiment
    experiment.run_med()

    # Save the results
    experiment.save_results()

    end_time = time.time()
    
    elapsed = end_time - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    print(f"Total execution time: {h}h {m}m {s}s")