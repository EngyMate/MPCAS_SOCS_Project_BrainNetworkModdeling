import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm

import numpy as np
from scipy.linalg import expm

import numpy as np


def prepare_laplacian_eigendecomposition(C):
    """
    Precompute the eigen-decomposition of the symmetric normalized Laplacian.
    Returns evals, evecs, and D_inv_sqrt so that other functions avoid recomputing.
    """
    deg = np.sum(C, axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)

    L = np.eye(C.shape[0]) - D_inv_sqrt @ C @ D_inv_sqrt

    # Symmetric → use eigh
    evals, evecs = np.linalg.eigh(L)

    return evals, evecs


def nodewise_diffusion_timeseries_fast(evals, evecs, beta, times, u0=None):
    """
    Fast diffusion using precomputed eigenvalues/vectors.
    """
    N=evecs.shape[0]
    print(N)
    if u0 is None:
        u0 = np.random.normal(0, 0.5, N)

    # project initial condition into eigenbasis
    u0_hat = evecs.T @ u0

    out = []
    for t in times:
        exp_factor = np.exp(-beta * evals * t)  # size N
        u = evecs @ (exp_factor * u0_hat)
        out.append(u)

    return np.array(out)


def init_files():
    np.array([0]).tofile("optimization/dlm_best_fitness.bin")
    np.array([0]).tofile("optimization/dlm_best_param.bin")

def compute_best_beta_fast(individuals, C, actual_activity, beta_range, times, u0=None):
    """
    Fast version: diagonalizes L once, reuses for all β.
    """
    N = C[0].shape[0]

    best_fitness = np.fromfile("dlm_best_fitness.bin", dtype=float)[0]
    best_beta = np.fromfile("dlm_best_param.bin", dtype=float)[0]
    best_sim = None

    # For all β values, simulate cheaply
    for beta in beta_range:
        error = 0
        print(f"beta:{beta}")
        for i in range(individuals):
            # Precompute eigenvalues and eigenvectors once
            c_ind = C[i]
            evals, evecs = prepare_laplacian_eigendecomposition(c_ind)

            if u0 is None:
                u0 = np.random.normal(0, 0.5, N)
            # Pre-project u0 into eigenbasis (done once)
            u0_hat = evecs.T @ u0

            sim = []
            for t in times:
                exp_factor = np.exp(-beta * evals * t)
                u_t = evecs @ (exp_factor * u0_hat)
                sim.append(u_t)
            sim = np.array(sim)

            if sim.shape != actual_activity[i].shape:
                raise ValueError("Shape mismatch between model output and f_series")

            error += np.sum((actual_activity[i] - sim) ** 2)
            fitness = 1/error

        if fitness > best_fitness:
            best_beta = beta
            np.array([best_beta]).tofile("dlm_best_beta.bin")
            best_fitness = fitness
            np.array([best_fitness]).tofile("dlm_best_fitness.bin")
            best_sim = sim

    return best_beta, best_fitness, best_sim


def get_best():
    best_fitness = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\dlm_best_fitness.bin", dtype=float)[0]
    best_beta = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\dlm_best_param.bin", dtype=float)[0]
    print(f"best_fitness:{best_fitness}, best_beta:{best_beta}")
    return best_fitness, best_beta

# Example usage
if __name__ == "__main__":
    # Example symmetric connectome matrix
    C = np.array([[0, 1, 2],
                  [1, 0, 3],
                  [2, 3, 0]], dtype=float)

    beta = 0.2
    best_fitness, best_beta = get_best()
    evals, evecs = prepare_laplacian_eigendecomposition(C)
    times = np.arange(0, 4800)
    u = nodewise_diffusion_timeseries_fast(evals, evecs, 0.1, times)
    plt.plot(u)
    plt.show()

