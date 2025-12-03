import numpy as np
from scipy.linalg import expm

def nodewise_diffusion(C, beta=0.1, times=[1.0]):
    """
    Simulate node-wise graph diffusion over multiple time points.

    Parameters:
    C : np.ndarray
        Connectome matrix (NxN)
    beta : float
        Diffusion rate
    times : list or np.ndarray
        List of time points to simulate

    Returns:
    Cf_time : dict
        Dictionary mapping time t to functional connectivity matrix Cf(t)
    """
    N = C.shape[0]

    # Degree matrix Δ
    delta = np.diag(np.sum(C, axis=1))

    # Normalized Laplacian L = I - Δ^(-1/2) C Δ^(-1/2)
    delta_inv_sqrt = np.linalg.inv(np.sqrt(delta))
    L = np.eye(N) - delta_inv_sqrt @ C @ delta_inv_sqrt

    Cf_time = {}
    for t in times:
        # Compute the matrix exponential
        Cf = expm(-beta * L * t)
        Cf_time[t] = Cf

    return Cf_time

# Example usage
if __name__ == "__main__":
    # Example symmetric connectome matrix
    C = np.array([[0, 1, 2],
                  [1, 0, 3],
                  [2, 3, 0]], dtype=float)

    beta = 0.2
    times = [1.0, 2.0, 5.0]  # multiple diffusion times

    results = nodewise_diffusion(C, beta, times)
    for t, Cf in results.items():
        print(f"\nFunctional connectivity Cf(t={t}):\n", Cf)
