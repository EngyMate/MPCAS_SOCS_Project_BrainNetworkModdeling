import os.path

import numpy as np
from itertools import product

from matplotlib import pyplot as plt

import data_management
import Evaluation.evaluation_tools as et


def simulate_linear_model(C, alpha, beta, n_steps=1000, noise_std=1.0, seed=None):
    """
    Simulate the stochastic linear model:
        u(n+1) = A u(n) + xi(n)
    where A = (1-alpha)*I + beta*C
    """
    if seed is not None:
        np.random.seed(seed)

    N = C.shape[0]  # number of nodes
    I = np.eye(N)

    # Build A
    A = (1 - alpha) * I + beta * C

    # Normalize A to have unit norm for stability
    A = A / np.linalg.norm(A)
    # Initialize u
    u = np.zeros((n_steps, N))
    noise = np.random.normal(0, noise_std, size=(n_steps, N))

    # Generate stochastic time series

    for t in range(1, n_steps):
        u[t] = A @ u[t - 1] + noise[t]

    return u



def init_files():
    np.array([0]).tofile("optimization/slm_best_fitness.bin")
    np.array([0, 0, 0]).tofile("optimization/slm_best_param.bin")

def get_best():
    best_fitness = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\slm_best_fitness.bin", dtype=float)[0]
    best_alpha = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\slm_best_param.bin", dtype=float)[0]
    best_beta = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\slm_best_param.bin", dtype=float)[1]
    best_std = np.fromfile(
        "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\slm_best_param.bin",
        dtype=float)[2]

    print(f"best_fitness:{best_fitness}, best_alpha:{best_alpha}, best_beta:{best_beta}, best_std:{best_std}")
    return best_fitness, best_alpha, best_beta, best_std


def find_best_alpha_beta(individuals, C, true_time_series, alpha_range, beta_range,std_range,  n_steps=1000):
    """
    Grid search over alpha and beta to minimize L1 error with true FC
    """

    best_fitness = np.fromfile("slm_best_fitness.bin", dtype=float)[0]
    best_alpha = np.fromfile("slm_best_param.bin", dtype=float)[0]
    best_beta = np.fromfile("slm_best_param.bin", dtype=float)[1]
    best_std = np.fromfile("slm_best_param.bin", dtype=float)[2]
    FC_emp_list = []
    for fMRI in true_time_series:
        FC_emp_list.append(et.compute_fc(fMRI))

    for alpha, beta, noise_std in product(alpha_range, beta_range, std_range):
        print(f">>> |alpha:{alpha}| |beta:{beta}| |std{noise_std}|")
        fitness  = 0
        for i in range(individuals):
            u = simulate_linear_model(C[i], alpha, beta, n_steps=n_steps, noise_std=noise_std)
            FC_sim = et.compute_fc(u)
            r_p, p_p, r_s, p_s = et.fc_fc_corr(FC_emp_list[i], FC_sim)
            fitness += r_p

        if fitness > best_fitness:
            best_fitness = fitness
            np.array([float(best_fitness)]).tofile("slm_best_fitness.bin")

            best_alpha = alpha
            best_beta = beta
            best_std = noise_std
            print(f"New best found! best_fitness:{best_fitness}, alpha:{alpha}, beta:{beta}, noise_std:{noise_std}")
            np.array([float(alpha), float(beta), float(noise_std)]).tofile("slm_best_param.bin")

    return best_alpha, best_beta, best_std, best_fitness

if __name__ == "__main__":
    init_files()
    #get_best()

    """
    # --------------------------
    # Example usage
    # --------------------------
    # Example anatomical connectivity matrix (e.g., 5 nodes)
    C = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ], dtype=float)


    dti = data_management.load_DTI_data(0)
    fMRI = data_management.load_fMRI_data(0)
    # Suppose this is the "true" functional connectivity (for demonstration)
    FC_true = np.corrcoef(np.random.randn(5, 1000))

    # Define parameter ranges
    alpha_range = np.arange(-3, 3.1, 0.1)
    beta_range = np.arange(0, 6.1, 0.1)

    u = simulate_linear_model(dti, 1,6, n_steps=4800, noise_std=1)

    FC_emp = et.compute_fc(fMRI)
    FC_sim = et.compute_fc(u)

    # ------------------------------------------------------------------
    # Compute SC–FC correlations
    # ------------------------------------------------------------------
    pear_emp, p_emp, spear_emp, sp_emp, sc_vals, fc_emp_vals = et.sc_fc_corr(dti, FC_emp)
    pear_sim, p_sim, spear_sim, sp_sim, sc_vals_sim, fc_sim_vals = et.sc_fc_corr(dti, FC_sim)

    # Compute average correlations across individuals
    print("\n===== Empirical SC–FC =====")
    print("Pearson:  mean r =", pear_emp, " mean p =", p_emp)
    print("Spearman_values: mean r =", spear_emp, " mean p =", sp_emp)

    print("\n===== Simulated SC–FC =====")
    print("Pearson:  mean r =", pear_sim, " mean p =", p_sim)
    print("Spearman_values: mean r =", spear_sim, " mean p =", sp_sim)

    # ------------------------------------------------------------------
    # Scatterplot: Simulated FC vs Empirical FC
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 6))

    # Scatter: each point is a connection (across all individuals)
    plt.scatter(fc_sim_vals, fc_emp_vals, s=3, alpha=0.3, color='blue')

    # Linear fit
    coeff = np.polyfit(fc_sim_vals, fc_emp_vals, 1)  # y = m*x + b
    fit_line = np.polyval(coeff, fc_sim_vals)
    plt.plot(fc_sim_vals, fit_line, color='red', linewidth=2, label=f'Slope: {coeff[0]:.8f}')

    plt.title("Stochastic linear simulated FC vs Empirical FC ")
    plt.xlabel("Simulated Functional Connectivity")
    plt.ylabel("Empirical Functional Connectivity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Run grid search
    #best_alpha, best_beta, best_FC, best_error = find_best_alpha_beta( C, FC_true, alpha_range, beta_range, n_steps=1000 )

    #print(f"Best alpha: {best_alpha}, Best beta: {best_beta}, L1 error: {best_error}")
"""