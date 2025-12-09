import os.path

import numpy as np
from itertools import product


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


def compute_functional_connectivity(u):
    """
    Compute functional connectivity as correlation matrix
    """
    return np.corrcoef(u)


def init_files():
    np.array([0]).tofile("optimization/slm_best_fitness.bin")
    np.array([0, 0]).tofile("optimization/slm_best_param.bin")

def get_best():
    best_fitness = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\slm_best_fitness.bin", dtype=float)[0]
    best_alpha = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\slm_best_param.bin", dtype=float)[0]
    best_beta = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\slm_best_param.bin", dtype=float)[1]
    print(f"best_fitness:{best_fitness}, best_alpha:{best_alpha}, best_beta:{best_beta}")
    return best_fitness, best_alpha, best_beta


def find_best_alpha_beta(individuals, C, true_time_series, alpha_range, beta_range, n_steps=1000, noise_std=1.0):
    """
    Grid search over alpha and beta to minimize L1 error with true FC
    """

    best_fitness = np.fromfile("slm_best_fitness.bin", dtype=float)[0]
    best_alpha = np.fromfile("slm_best_param.bin", dtype=float)[0]
    best_beta = np.fromfile("slm_best_param.bin", dtype=float)[1]

    for alpha, beta in product(alpha_range, beta_range):
        print(f"alpha:{alpha}, beta:{beta}")
        fitness  = 0
        for i in range(individuals):
            u = simulate_linear_model(C[i], alpha, beta, n_steps=n_steps, noise_std=noise_std)
            error = np.sum(np.power(u - true_time_series[i], 2))
            fitness += 1 / error

        if fitness > best_fitness:
            best_fitness = fitness
            np.array([best_fitness]).tofile("slm_best_fitness.bin")

            best_alpha, best_beta = alpha, beta
            np.array([best_alpha, best_beta]).tofile("slm_best_param.bin")

    return best_alpha, best_beta, best_fitness

if __name__ == "__main__":
    init_files()
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

    # Suppose this is the "true" functional connectivity (for demonstration)
    FC_true = np.corrcoef(np.random.randn(5, 1000))

    # Define parameter ranges
    alpha_range = np.arange(-3, 3.1, 0.1)
    beta_range = np.arange(0, 6.1, 0.1)

    # Run grid search
    #best_alpha, best_beta, best_FC, best_error = find_best_alpha_beta( C, FC_true, alpha_range, beta_range, n_steps=1000 )

    #print(f"Best alpha: {best_alpha}, Best beta: {best_beta}, L1 error: {best_error}")
