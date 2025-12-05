import numpy as np
from scipy.linalg import expm

import numpy as np
from scipy.linalg import expm

def nodewise_diffusion_timeseries(C, beta=0.1, times=[1.0], u0=None):
    """
    Simulate diffusion dynamics and return a time series (T × N).

    Parameters:
    C : NxN connectome matrix
    beta : diffusion rate
    times : list of time points
    u0 : initial activity (length N). If None, defaults to all ones.

    Returns:
    u_t : array of shape (T, N) containing node activity at each time.
    """
    N = C.shape[0]

    # Degree
    deg = np.sum(C, axis=1)

    # Safe D^(-1/2)
    deg_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_inv_sqrt = np.diag(deg_inv_sqrt)

    # Normalized Laplacian
    L = np.eye(N) - D_inv_sqrt @ C @ D_inv_sqrt

    # Initial condition
    if u0 is None:
        u0 = np.random.normal(0,0.5, N)

    u_t = []
    for t in times:
        kernel = expm(-beta * L * t)   # NxN
        u = kernel @ u0                # Nx1 → N
        u_t.append(u)

    return np.array(u_t)  # → shape (T, N)


def compute_best_beta(C, f_series, beta_range, times):
    best_beta = None
    best_error = np.inf
    best_sim = None

    for beta in beta_range:
        sim = nodewise_diffusion_timeseries(C, beta, times)

        # ensure same shape
        if sim.shape != f_series.shape:
            raise ValueError("Shape mismatch between model output and f_series")

        error = np.sum((f_series - sim)**2)

        if error < best_error:
            best_error = error
            best_beta = beta
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

    results = nodewise_diffusion_timeseries(C, beta, times)
    print(results)
