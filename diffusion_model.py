import numpy as np
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


def compute_best_beta_fast(C, f_series, beta_range, times, u0=None):
    """
    Fast version: diagonalizes L once, reuses for all β.
    """
    N = C.shape[0]

    if u0 is None:
        u0 = np.random.normal(0, 0.5, N)

    # Precompute eigenvalues and eigenvectors once
    evals, evecs = prepare_laplacian_eigendecomposition(C)

    # Pre-project u0 into eigenbasis (done once)
    u0_hat = evecs.T @ u0

    best_beta = None
    best_error = np.inf
    best_sim = None

    # For all β values, simulate cheaply
    for beta in beta_range:
        sim = []
        for t in times:
            exp_factor = np.exp(-beta * evals * t)
            u_t = evecs @ (exp_factor * u0_hat)
            sim.append(u_t)
        sim = np.array(sim)

        if sim.shape != f_series.shape:
            raise ValueError("Shape mismatch between model output and f_series")

        error = np.sum((f_series - sim)**2)

        if error < best_error:
            best_beta = beta
            best_error = error
            best_sim = sim

    return best_beta, best_error, best_sim


# Example usage
if __name__ == "__main__":
    # Example symmetric connectome matrix
    C = np.array([[0, 1, 2],
                  [1, 0, 3],
                  [2, 3, 0]], dtype=float)

    beta = 0.2
    times = [1.0, 2.0, 5.0]  # multiple diffusion times

    results = nodewise_diffusion_timeseries_fast(C, beta, times)
    print(results)
