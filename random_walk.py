import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import data_management
import Evaluation.evaluation_tools as et

def random_walk(N, alpha, n_steps=4800, noise_std=1.0):
    # Initialize u
    u = np.zeros((n_steps, N))
    noise = np.random.normal(0, noise_std, size=(n_steps, N))

    # Generate stochastic time series

    for t in range(1, n_steps):
        u[t] = alpha * u[t - 1] + noise[t]

    return u

def find_best_alpha_beta(individuals, C, true_time_series, alpha_range, std_rage, n_steps=4800):
    """
    Grid search over alpha and beta to minimize L1 error with true FC
    """

    best_fitness = np.fromfile("rwm_best_fitness.bin", dtype=float)[0]
    best_alpha = np.fromfile("rwm_best_param.bin", dtype=float)[0]
    best_std = np.fromfile("rwm_best_param.bin", dtype=float)[1]

    N = C[0].shape[0]

    for alpha, std in product(alpha_range, std_rage):
        print(f"alpha:{alpha}, std:{std}")
        fitness  = 0
        for i in range(individuals):
            u = random_walk(N, alpha, n_steps=n_steps, noise_std=std)
            FC_sim = et.compute_fc(u)
            FC_emp = et.compute_fc(true_time_series[i])

            pear_sim, p_sim, spear_sim, sp_sim, sc_vals_sim, fc_sim_vals = et.sc_fc_corr(C[i], FC_sim)
            r_p, p_p, r_s, p_s = et.fc_fc_corr(FC_emp, FC_sim)
            fitness += r_p

        if fitness > best_fitness:
            best_fitness = fitness
            np.array([float(best_fitness)]).tofile("rwm_best_fitness.bin")

            best_alpha = alpha
            best_std = std
            print(f"New best found! best_fitness:{best_fitness}, alpha:{alpha}, beta:{std}")
            np.array([float(alpha), float(std)]).tofile("rwm_best_param.bin")

    return best_alpha, best_std, best_fitness

def init_files():
    np.array([0]).tofile("optimization/rwm_best_fitness.bin")
    np.array([0, 0]).tofile("optimization/rwm_best_param.bin")


def get_best():
    best_fitness = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\rwm_best_fitness.bin", dtype=float)[0]
    best_alpha = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\rwm_best_param.bin", dtype=float)[0]
    best_std = np.fromfile("C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\rwm_best_param.bin", dtype=float)[1]
    print(f"best_fitness:{best_fitness}, best_alpha:{best_alpha}, best_std:{best_std}")
    return best_fitness, best_alpha, best_std


if __name__ == "__main__":
    init_files()
    """fMRI = data_management.load_fMRI_data(0)
    N = fMRI.shape[1]
    u = random_walk(N,0.1, noise_std=0.5)

    # Plot time series of node 1 (index 0)
    plt.figure(figsize=(10, 4))
    plt.plot(fMRI[:, 0], linewidth=1, label="Empirical")
    plt.plot(u[:, 0], linewidth=1, label="Sim")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Simulated Time Series (Node 1)")
    plt.tight_layout()
    plt.show()"""