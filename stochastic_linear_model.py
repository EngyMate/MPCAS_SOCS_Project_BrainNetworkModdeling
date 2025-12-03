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
    u = np.zeros((N, n_steps))

    # Generate stochastic time series
    for t in range(1, n_steps):
        xi = np.random.normal(0, noise_std, size=N)
        u[:, t] = A @ u[:, t - 1] + xi

    return u


def compute_functional_connectivity(u):
    """
    Compute functional connectivity as correlation matrix
    """
    return np.corrcoef(u)


def find_best_alpha_beta(C, FC_true, alpha_range, beta_range, n_steps=1000, noise_std=1.0):
    """
    Grid search over alpha and beta to minimize L1 error with true FC
    """
    best_error = np.inf
    best_alpha, best_beta = None, None
    best_FC = None

    for alpha, beta in product(alpha_range, beta_range):
        u = simulate_linear_model(C, alpha, beta, n_steps=n_steps, noise_std=noise_std)
        FC_sim = compute_functional_connectivity(u)
        error = np.sum(np.abs(FC_sim - FC_true))  # L1 error

        if error < best_error:
            best_error = error
            best_alpha, best_beta = alpha, beta
            best_FC = FC_sim

    return best_alpha, best_beta, best_FC, best_error

if __name__ == "__main__":
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
    best_alpha, best_beta, best_FC, best_error = find_best_alpha_beta(
        C, FC_true, alpha_range, beta_range, n_steps=1000
    )

    print(f"Best alpha: {best_alpha}, Best beta: {best_beta}, L1 error: {best_error}")
