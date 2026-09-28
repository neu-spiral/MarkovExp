# Markovian Experimental Design under Concept Drift

Official code for the paper:

**Markovian Experimental Design under Concept Drift** <br>
Ayberk Yarkın Yıldız, Lili Su, Carlee Joe-Wong, Edmund Yeh, and Stratis Ioannidis <br>
*NeurIPS 2026*

> We study how to optimally select training samples when the underlying model is subject to concept drift. A learner sequentially collects samples and re-estimates a model whose parameters drift according to mean-reverting Gaussian dynamics around an unknown mean fixed-point $\boldsymbol{\beta}_\ast$. In this setting, model posteriors are given by a Kalman filter, and D-optimal sample-selection policies, both myopic and in steady state, can be computed by solving convex optimization problems. The framework extends to non-linear models through an extended Kalman filter (EKF).

This repository contains the implementation of all design-selection and model-estimation methods in the paper, together with the scripts to run the synthetic and real-data experiments.

## Requirements

Install the dependencies in `requirements.txt`. The stage-wise optimal selection solves its convex program with [MOSEK](https://www.mosek.com/) through `cvxpy`, so a MOSEK license is required.

## Usage

All experiments are launched through `system_submit.py`. Set the configuration variables at the top of the script, then run:

```bash
python system_submit.py
```

The main configuration variables are:

| Variable | Description |
|---|---|
| `system` | `"linear"` or `"nonlinear"` (EKF-based) measurement model. |
| `model_name` | Model function $h$ in the non-linear setting: `"resnet"`, `"cnn"`, or `"mlp"`. It is not used by the linear model (e.g., set it to `"none"`), but it still appears in the output paths. |
| `data_type` | `"simulation"` for synthetic data or `"real"` for real-world datasets. |
| `dataset` | Real-world dataset: `"beijing"` (Beijing PM2.5), `"sp500"` (S&P 500), `"bike"` (Bike Sharing), `"california"` (California Housing), `"atnt_dl"` / `"atnt_ul"` (AT&T Downlink / Uplink). |
| `d`, `p`, `n` | Feature dimension $d$, number of candidate experiments per stage $p$, and number of samples collected per stage $n$. |
| `rho_all`, `sigmaw_all`, `sigma_all` | Lists of values for the drift parameter $\rho$, the drift noise $\sigma_{w}$, and the measurement noise $\sigma$; every combination is run. |
| `covconstant` | Scale $c$ of the prior covariance $\boldsymbol{\Sigma}_0 = c\,\mathbf{I}$. |
| `stages` | Number of stages $T$. |
| `num_experiments` | Number of independent runs (random seeds). |
| `random_pool` | If `True`, a new candidate set $\mathcal{X}_k$ is drawn at every stage. |
| `use_stationary` | If `True`, also run the steady-state (stationary) selection methods. |
| `remove_all_folders` | If `True`, remove previous results before running. |
| `blocking` | `True` runs the experiments locally; `False` submits them as SLURM jobs (see below). |

`system_submit.py` contains commented-out parameter presets for each real-world dataset, corresponding to the settings reported in the paper.

### Methods

Each experiment couples a design-selection method with a model-estimation method. The table below maps the method names used in the paper to their identifiers in the code.

| Method name | Code identifier |
|---|---|
| **Design selection** | |
| STG: stage-wise optimal selection (ours) | `stagewise_optimal` (`cvxpy`) |
| STS: stationary optimal selection (ours) | `stationary_optimal` (`fw`) |
| GKS: greedy Kalman information-gain selection | `greedy_optimal` (`greedy`) |
| ADM: Adam-based selection | `adam_optimal` (`adam`) |
| RND: random selection | `non_optimal` (`random`) |
| **Model estimation** | |
| AE: adaptive (Kalman filter / EKF) estimation, $\boldsymbol{\beta}_\ast$ estimated | `kf` (`unknown`) |
| AE-K: adaptive estimation with known $\boldsymbol{\beta}_\ast$ | `kf` (`known`) |
| RLS: forgetting-factor recursive least squares | `rls_ff` |
| SE: static estimation | `no_kf` |
| SGD: stochastic gradient descent estimation | `sgd` |

### Running on a SLURM cluster

To submit the experiments as SLURM jobs:

1. Set `blocking = False` in `system_submit.py`.
2. Create a bash file named `execute.sh` and set the partition (`#SBATCH --partition=...`), the resources, the module to load, and the path to your environment's Python interpreter (`PY=...`).
3. Run `python system_submit.py`. Each experiment is submitted via `sbatch execute.sh`.

### Outputs

Results are saved under `results/<system>/<data_type>/<model_name>/`, and the logs of each run under `logs/<system>/<data_type>/<model_name>/`.
