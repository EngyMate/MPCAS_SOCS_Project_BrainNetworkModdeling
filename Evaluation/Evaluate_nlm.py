import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import data_management as dm
import nonLinear_model as nlm
import random
import evaluation_tools as et

# ---------------------------------------------------------
# --- FUNCTIONAL CONNECTIVITY COMPUTATION -----------------
# ---------------------------------------------------------

def compute_fc(timeseries):
    """
    timeseries shape = (timepoints, nodes)
    returns FC matrix (nodes x nodes)
    """
    # z-score for safety
    ts = (timeseries - timeseries.mean(axis=0)) / timeseries.std(axis=0)
    FC = np.corrcoef(ts, rowvar=False)
    return FC


def sc_fc_correlation(SC, FC):
    mask = np.triu(np.ones(SC.shape), k=1).astype(bool)
    sc_vals = SC[mask]
    fc_vals = FC[mask]

    # Fisher z-transform (optional)
    # fc_vals = np.arctanh(fc_vals)

    pear_r, pear_p = pearsonr(sc_vals, fc_vals)
    spear_r, spear_p = spearmanr(sc_vals, fc_vals)

    return pear_r, pear_p, spear_r, spear_p, sc_vals, fc_vals


# ---------------------------------------------------------
# --- CONTINUE FROM YOUR SCRIPT ---------------------------
# ---------------------------------------------------------

def sim():
    individuals = 100
    DTI_data = []
    fMRI_data = []

    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))  # SC matrix
        fMRI_data.append(dm.load_fMRI_data(i))  # fMRI timeseries

    #best_fitness, best_params = nlm.get_best()

    t_span = (0, 4800 * 1.2)
    t_eval = np.linspace(t_span[1] - 1000 * 1.2, t_span[1], 1000)

    files_to_use = [0, 1, 2, 3, 4, 5]


    for r in files_to_use:
        SC = DTI_data[r]

        num_nodes = SC.shape[0]
        start_time = time.time()

        V, W, Z, t_eval = nlm.simulate_network(
            num_nodes,
            t_span,
            t_eval,
            connectome_matrix=SC,
            noise_level=0.1
        )
        end_time = time.time()
        print(f"simmulation running time: {start_time - end_time}")

        # Choose one simulated variable as simulated fMRI (e.g., V)
        sim_fMRI = V.transpose()  # ensure shape (timepoints, nodes)
        sim_fMRI.tofile(f"nln_sim_{r}.bin")

def eval():
    individuals = 100
    DTI_data = []
    fMRI_data = []

    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))  # SC matrix
        fMRI_data.append(dm.load_fMRI_data(i))  # fMRI timeseries

    pear_emp = np.zeros(individuals)
    p_emp = np.zeros(individuals)
    spear_emp = np.zeros(individuals)
    sp_emp = np.zeros(individuals)

    pear_sim = np.zeros(individuals)
    p_sim = np.zeros(individuals)
    spear_sim = np.zeros(individuals)
    sp_sim = np.zeros(individuals)

    files_to_use = [0, 1, 2, 3, 4, 5]
    sim_fMRI = []

    sc_vals_all = []
    fc_emp_vals_all = []
    fc_sim_vals_all = []

    for r in files_to_use:
        sim_fMRI = np.fromfile(f"nln_sim_{r}.bin", dtype=float).reshape((1000, 246))

        SC = DTI_data[r]
        fMRI = fMRI_data[r]
        # -------------------------------
        # Compute FC (empirical + simulated)
        # -------------------------------

        FC_empirical = compute_fc(fMRI)
        FC_simulated = compute_fc(sim_fMRI)

        # ------------------------------------------------------------------
        # Compute SC–FC correlations
        # ------------------------------------------------------------------
        pear_emp[i], p_emp[i], spear_emp[i], sp_emp[i], sc_vals, fc_emp_vals = et.sc_fc_corr(SC, FC_empirical)
        pear_sim[i], p_sim[i], spear_sim[i], sp_sim[i], sc_vals_sim, fc_sim_vals = et.sc_fc_corr(SC, FC_simulated)

        # Store SC-FC values for plotting (optional: concatenate across individuals)
        sc_vals_all.extend(sc_vals)
        fc_emp_vals_all.extend(fc_emp_vals)
        fc_sim_vals_all.extend(fc_sim_vals)

        # Compute average correlations across individuals
    print("\n===== Empirical SC–FC =====")
    print("Pearson:  mean r =", pear_emp.mean(), " mean p =", p_emp.mean())
    print("Spearman: mean r =", spear_emp.mean(), " mean p =", sp_emp.mean())

    print("\n===== Simulated SC–FC =====")
    print("Pearson:  mean r =", pear_sim.mean(), " mean p =", p_sim.mean())
    print("Spearman: mean r =", spear_sim.mean(), " mean p =", sp_sim.mean())

    # ------------------------------------------------------------------
    # Scatterplot: Simulated FC vs Empirical FC
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 6))

    # Scatter: each point is a connection (across all individuals)
    plt.scatter(fc_sim_vals_all, fc_emp_vals_all, s=3, alpha=0.3, color='blue')

    # Linear fit
    coeff = np.polyfit(fc_sim_vals_all, fc_emp_vals_all, 1)  # y = m*x + b
    fit_line = np.polyval(coeff, fc_sim_vals_all)
    plt.plot(fc_sim_vals_all, fit_line, color='red', linewidth=2, label=f'Slope: {coeff[0]:.8f}')

    plt.title("Diffusion model FC vs Empirical FC ")
    plt.xlabel("Simulated Functional Connectivity")
    plt.ylabel("Empirical Functional Connectivity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sim()
    #eval()
